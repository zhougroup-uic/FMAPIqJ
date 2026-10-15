# mix_kernels.jl - real-mixture counterparts of mc_batch.jl's
# mc_batch_kernel! and transfer_batch.jl's transfer_batch_kernel!, added
# as NEW multiple-dispatch methods (same function names, distinguished by
# grid::GridSet vs grid::GridView) rather than edits to those files - see
# mc_core.jl's GridSet comment for why: this keeps the existing
# grid::GridView methods untouched, while the drivers get real per-type-pair
# energetics just by passing a GridSet instead of a GridView.
#
# The original GPU kernels used context-0 energetics for all particle pairs
# regardless of type. The CPU engine (simulation.c/simulation_gemc.c's
# get_ctx/pair_energy_idx) computes true per-type-pair energies; these
# kernels use the same selection logic (mc_core.jl's ctx_index, identical
# formula) on the GPU.

using CUDA
using Adapt
using StaticArrays
# mc_core.jl (GridSet, ctx_index, sum_pair_energy, calculate_pair_energy,
# bins_to_rot_idx, cwrap, lcg_next, as_vec3) and mc_batch.jl/
# transfer_batch.jl (BoxView, TransferBoxView) are included before this
# file by the top-level driver.

# Translation and rotation are independent trial moves (50/50 per step, one
# LCG draw), each tallied separately - same as the C GPU kernel
# (fmap_energy.cu mc_batch_body_mix) - so the driver can tune js and rs
# against their own acceptance ratios. js/rs come in as launch arguments
# (they change every adjustment window); box.result is
# [delta, n_accepted, try_trans, acc_trans, try_rot, acc_rot].
function mc_batch_kernel!(box1::BoxView, box2::BoxView, grid::GridSet,
                           batch_size::Int32, temperature::Float64,
                           n_active1::Int32, n_active2::Int32,
                           js1::Float64, rs1::Float64, js2::Float64, rs2::Float64)
    first = blockIdx().x == Int32(1)
    box = first ? box1 : box2
    js = first ? js1 : js2
    rs = first ? rs1 : rs2
    tid = threadIdx().x - Int32(1)

    pos3  = as_vec3(box.pos)
    bins3 = as_vec3(box.bins)

    sdata      = @cuStaticSharedMem(Float64, 256)
    s_lcg      = @cuStaticSharedMem(UInt64, 1)
    s_delta    = @cuStaticSharedMem(Float64, 1)
    s_cnt      = @cuStaticSharedMem(Float64, 5)   # nacc, try_t, acc_t, try_r, acc_r
    s_pid      = @cuStaticSharedMem(Int32, 1)
    s_active   = @cuStaticSharedMem(Int32, 1)
    s_is_trans = @cuStaticSharedMem(Int32, 1)
    s_type     = @cuStaticSharedMem(Int32, 1)
    s_cur      = @cuStaticSharedMem(SVector{3,Float64}, 1)
    s_cur_bins = @cuStaticSharedMem(SVector{3,Float64}, 1)
    s_cur_rot  = @cuStaticSharedMem(Int32, 1)
    s_trial    = @cuStaticSharedMem(SVector{3,Float64}, 1)
    s_trial_b  = @cuStaticSharedMem(SVector{3,Float64}, 1)
    s_trial_r  = @cuStaticSharedMem(Int32, 1)

    if tid == Int32(0)
        s_delta[1] = 0.0
        for k in 1:5
            s_cnt[k] = 0.0
        end
        s_lcg[1]   = box.lcg[1]
    end
    sync_threads()

    n = first ? n_active1 : n_active2   # compact: [0,n) active
    for _ in Int32(1):batch_size
        if tid == Int32(0)
            state = s_lcg[1]
            state, r = lcg_next(state)
            pid = unsafe_trunc(Int32, ceil(r * n) - 1.0)
            pid = pid < Int32(0) ? Int32(0) : (pid >= n ? n - Int32(1) : pid)   # n==0: unused
            s_pid[1] = pid
            active = n > Int32(0)
            s_active[1] = active ? Int32(1) : Int32(0)

            if active
                s_type[1] = @inbounds box.ptype[pid+1]
                cur  = pos3[pid+1]
                curb = bins3[pid+1]
                s_cur[1]      = cur
                s_cur_bins[1] = curb
                s_cur_rot[1]  = @inbounds box.rot_idx[pid + 1]

                state, rm = lcg_next(state)
                is_trans = rm < 0.5
                s_is_trans[1] = is_trans ? Int32(1) : Int32(0)
                if is_trans
                    s_cnt[2] += 1.0
                else
                    s_cnt[4] += 1.0
                end

                tp = MVector{3,Float64}(cur[1], cur[2], cur[3])
                tb = MVector{3,Float64}(curb[1], curb[2], curb[3])
                for d in 1:3
                    state, r1 = lcg_next(state)
                    if is_trans
                        tp[d] = cwrap(cur[d] + (r1 - 0.5) * js, box.box[d])
                    else
                        tb[d] = cwrap(curb[d] + (r1 - 0.5) * rs, Float64(grid.nbin))
                    end
                end
                trial = SVector(tp); trial_b = SVector(tb)
                s_trial[1] = trial
                s_trial_b[1] = trial_b
                s_trial_r[1] = is_trans ? s_cur_rot[1] :
                               bins_to_rot_idx(trial_b, grid.angle_indices, grid.nbin)
            end
            s_lcg[1] = state
        end
        sync_threads()

        active = s_active[1] != Int32(0)
        if active
            pid = s_pid[1]
            cur = s_cur[1]; cur_rot = s_cur_rot[1]
            trial = s_trial[1]; trial_rot = s_trial_r[1]
            self_type = s_type[1]

            delta_sum = sum_pair_energy_diff(trial, trial_rot, cur, cur_rot, pid, self_type, pos3,
                                              box.rot_idx, box.ptype, n, box.box, grid, tid, sdata)

            if tid == Int32(0)
                delta = delta_sum
                accept = delta <= 0.0
                if !accept
                    state = s_lcg[1]
                    state, r = lcg_next(state)
                    s_lcg[1] = state
                    accept = r < exp(-delta / temperature)
                end
                if accept
                    pid2 = s_pid[1]
                    pos3[pid2+1]  = s_trial[1]
                    bins3[pid2+1] = s_trial_b[1]
                    @inbounds box.rot_idx[pid2+1] = s_trial_r[1]
                    s_delta[1] += delta
                    s_cnt[1] += 1.0
                    if s_is_trans[1] != Int32(0)
                        s_cnt[3] += 1.0
                    else
                        s_cnt[5] += 1.0
                    end
                end
            end
        end
        sync_threads()
    end

    if tid == Int32(0)
        box.result[1] = s_delta[1]
        for k in 1:5
            box.result[k+1] = s_cnt[k]
        end
        box.lcg[1] = s_lcg[1]
    end
    return
end

# js/rs are the RAW step sizes (see StepTune in gemc_box.jl); the rotation
# kernel offset is (rand-0.5)*rs*ROT_DEFAULT, same as the C GPU call site
# (sim_gemc_gpu_run_steps_pair_async multiplies rs by rot_default=3.0).
const ROT_DEFAULT = 3.0
function mc_batch_launch!(box1::BoxView, box2::BoxView, grid::GridSet,
                           batch_size::Integer, temperature::Float64,
                           n_active1::Integer, n_active2::Integer,
                           js1::Float64, rs1::Float64, js2::Float64, rs2::Float64;
                           stream::CuStream=CUDA.stream())
    @cuda threads=256 blocks=2 stream=stream mc_batch_kernel!(box1, box2, grid, Int32(batch_size), temperature,
                                                               Int32(n_active1), Int32(n_active2),
                                                               js1, rs1 * ROT_DEFAULT, js2, rs2 * ROT_DEFAULT)
end

# Reservoir-fill insertion (port of fmap_energy.cu's reservoir_insert_batch_
# mix_kernel): closes a startup placement deficit by inserting into one box
# from an implicit zero-energy reservoir (accept if insertion energy <= 0,
# else Metropolis on exp(-beta*e)). Up to batch_size attempts in one launch,
# stopping once max_accepts succeed. Slots [n_active, committed) already hold
# their intended type in box.ptype (pre-assigned at init/top-up), so the
# inserted particle's type is read straight from slot t = n_active (compact
# slots: insert at the highest index). result = [delta_e, n_accepted,
# n_active, type0_delta].
function reservoir_insert_kernel!(
        box::TransferBoxView, grid::GridSet,
        batch_size::Int32, max_accepts::Int32, temperature::Float64,
        lcg::CuDeviceVector{UInt64,1}, result::CuDeviceVector{Float64,1},
        n_active_init::Int32, committed::Int32)

    tid = threadIdx().x - Int32(1)
    pos3  = as_vec3(box.pos)
    bins3 = as_vec3(box.bins)

    sdata   = @cuStaticSharedMem(Float64, 256)
    s_lcg   = @cuStaticSharedMem(UInt64, 1)
    s_de    = @cuStaticSharedMem(Float64, 1)
    s_nacc  = @cuStaticSharedMem(Float64, 1)
    s_t0d   = @cuStaticSharedMem(Float64, 1)
    s_nact  = @cuStaticSharedMem(Int32, 1)
    s_valid = @cuStaticSharedMem(Int32, 1)
    s_ptype = @cuStaticSharedMem(Int32, 1)
    s_trial = @cuStaticSharedMem(SVector{3,Float64}, 1)
    s_trial_b = @cuStaticSharedMem(SVector{3,Float64}, 1)
    s_trial_r = @cuStaticSharedMem(Int32, 1)
    s_ins   = @cuStaticSharedMem(Float64, 1)

    if tid == Int32(0)
        s_lcg[1] = lcg[1]
        s_de[1] = 0.0; s_nacc[1] = 0.0; s_t0d[1] = 0.0
        s_nact[1] = n_active_init
    end
    sync_threads()

    for _ in Int32(1):batch_size
        if tid == Int32(0)
            valid = Int32(0)
            t = s_nact[1]
            if t < committed && t < box.max_cap && Int32(round(s_nacc[1])) < max_accepts
                valid = Int32(1)
                s_ptype[1] = @inbounds box.ptype[t+1]
                state = s_lcg[1]
                tp = MVector{3,Float64}(0.0, 0.0, 0.0)
                tb = MVector{3,Float64}(0.0, 0.0, 0.0)
                for d in 1:3
                    state, ra = lcg_next(state)
                    tp[d] = ra * box.box[d]
                    state, rb = lcg_next(state)
                    tb[d] = rb * grid.nbin
                end
                trial_b = SVector(tb)
                s_trial[1] = SVector(tp)
                s_trial_b[1] = trial_b
                s_trial_r[1] = bins_to_rot_idx(trial_b, grid.angle_indices, grid.nbin)
                s_lcg[1] = state
            end
            s_valid[1] = valid
        end
        sync_threads()

        if s_valid[1] != Int32(0)
            t = s_nact[1]
            ins = sum_pair_energy(s_trial[1], s_trial_r[1], t, s_ptype[1], pos3, box.rot_idx,
                                   box.ptype, t, box.box, grid, tid, sdata)
            if tid == Int32(0)
                s_ins[1] = ins
                accept = ins <= 0.0
                if !accept
                    state = s_lcg[1]
                    state, r = lcg_next(state)
                    s_lcg[1] = state
                    accept = r < exp(-ins / temperature)
                end
                if accept
                    pos3[t+1]  = s_trial[1]
                    bins3[t+1] = s_trial_b[1]
                    @inbounds box.rot_idx[t+1] = s_trial_r[1]
                    s_nact[1] = t + Int32(1)
                    s_de[1] += ins
                    s_nacc[1] += 1.0
                    s_ptype[1] == Int32(0) && (s_t0d[1] += 1.0)
                end
            end
        end
        sync_threads()
    end

    if tid == Int32(0)
        result[1] = s_de[1]
        result[2] = s_nacc[1]
        result[3] = Float64(s_nact[1])
        result[4] = s_t0d[1]
        lcg[1] = s_lcg[1]
    end
    return
end

function reservoir_insert_launch!(box::TransferBoxView, grid::GridSet,
                                   batch_size::Integer, max_accepts::Integer, temperature::Float64,
                                   lcg::AbstractVector{UInt64}, result::AbstractVector{Float64},
                                   n_active::Integer, committed::Integer;
                                   stream::CuStream=CUDA.stream())
    @cuda threads=256 blocks=1 stream=stream reservoir_insert_kernel!(
        box, grid, Int32(batch_size), Int32(max_accepts), temperature, lcg, result,
        Int32(n_active), Int32(committed))
end

function transfer_batch_kernel!(
        box1::TransferBoxView, box2::TransferBoxView, grid::GridSet,
        batch_size::Int32, temperature::Float64,
        lcg::CuDeviceVector{UInt64,1}, result::CuDeviceVector{Float64,1},
        n_active1_init::Int32, n_active2_init::Int32)

    tid = threadIdx().x - Int32(1)
    pos3_1 = as_vec3(box1.pos); bins3_1 = as_vec3(box1.bins)
    pos3_2 = as_vec3(box2.pos); bins3_2 = as_vec3(box2.bins)

    sdata   = @cuStaticSharedMem(Float64, 256)
    s_lcg   = @cuStaticSharedMem(UInt64, 1)
    s_de1   = @cuStaticSharedMem(Float64, 1)
    s_de2   = @cuStaticSharedMem(Float64, 1)
    s_nacc  = @cuStaticSharedMem(Float64, 1)
    s_type0d1 = @cuStaticSharedMem(Float64, 1)
    s_type0d2 = @cuStaticSharedMem(Float64, 1)
    s_nactive1 = @cuStaticSharedMem(Int32, 1)
    s_nactive2 = @cuStaticSharedMem(Int32, 1)
    s_dir   = @cuStaticSharedMem(Int32, 1)
    s_valid = @cuStaticSharedMem(Int32, 1)
    s_s     = @cuStaticSharedMem(Int32, 1)
    s_t     = @cuStaticSharedMem(Int32, 1)
    s_acc   = @cuStaticSharedMem(Int32, 1)
    s_nold  = @cuStaticSharedMem(Int32, 1)
    s_ptype = @cuStaticSharedMem(Int32, 1)
    s_trial = @cuStaticSharedMem(SVector{3,Float64}, 1)
    s_trial_b = @cuStaticSharedMem(SVector{3,Float64}, 1)
    s_trial_r = @cuStaticSharedMem(Int32, 1)
    s_removal = @cuStaticSharedMem(Float64, 1)
    s_insertion = @cuStaticSharedMem(Float64, 1)

    if tid == Int32(0)
        s_lcg[1] = lcg[1]
        s_de1[1] = 0.0; s_de2[1] = 0.0; s_nacc[1] = 0.0
        s_type0d1[1] = 0.0; s_type0d2[1] = 0.0
        s_nactive1[1] = n_active1_init
        s_nactive2[1] = n_active2_init
    end
    sync_threads()

    for step in Int32(1):batch_size
        dir0 = (step - Int32(1)) & Int32(1)

        if tid == Int32(0)
            s_dir[1] = dir0
            to_box = dir0 == Int32(0) ? box2 : box1
            from_nactive = dir0 == Int32(0) ? s_nactive1[1] : s_nactive2[1]
            to_nactive = dir0 == Int32(0) ? s_nactive2[1] : s_nactive1[1]
            from_ptype = dir0 == Int32(0) ? box1.ptype : box2.ptype

            valid = Int32(0)
            if from_nactive > 0 && to_nactive < to_box.max_cap
                state = s_lcg[1]
                state, r = lcg_next(state)
                r_idx = unsafe_trunc(Int32, r * from_nactive)
                r_idx = r_idx >= from_nactive ? from_nactive - Int32(1) : r_idx
                begin
                    # compact slots: the r-th active particle IS slot r_idx;
                    # the new particle is appended at the highest slot.
                    valid = Int32(1)
                    s_s[1] = r_idx
                    s_ptype[1] = @inbounds from_ptype[r_idx+1]
                    s_t[1] = to_nactive

                    tp = MVector{3,Float64}(0.0, 0.0, 0.0)
                    tb = MVector{3,Float64}(0.0, 0.0, 0.0)
                    for d in 1:3
                        state, ra = lcg_next(state)
                        tp[d] = ra * to_box.box[d]
                        state, rb = lcg_next(state)
                        tb[d] = rb * grid.nbin
                    end
                    trial_b = SVector(tb)
                    s_trial[1] = SVector(tp)
                    s_trial_b[1] = trial_b
                    s_trial_r[1] = bins_to_rot_idx(trial_b, grid.angle_indices, grid.nbin)
                end
                s_lcg[1] = state
            end
            s_valid[1] = valid
        end
        sync_threads()

        valid = s_valid[1] != Int32(0)
        dir = s_dir[1]
        from_box = dir == Int32(0) ? box1 : box2
        from_pos3 = dir == Int32(0) ? pos3_1 : pos3_2
        from_ridx = dir == Int32(0) ? box1.rot_idx : box2.rot_idx
        from_ptype = dir == Int32(0) ? box1.ptype : box2.ptype
        if valid
            s_idx = s_s[1]
            pos_s = from_pos3[s_idx+1]
            ris = @inbounds from_ridx[s_idx+1]
            n_from = dir == Int32(0) ? s_nactive1[1] : s_nactive2[1]
            removal = sum_pair_energy(pos_s, ris, s_idx, s_ptype[1], from_pos3, from_ridx,
                                       from_ptype, n_from, from_box.box,
                                       grid, tid, sdata)
            if tid == Int32(0)
                s_removal[1] = removal
            end
        end
        sync_threads()

        to_box = dir == Int32(0) ? box2 : box1
        to_pos3 = dir == Int32(0) ? pos3_2 : pos3_1
        to_ridx = dir == Int32(0) ? box2.rot_idx : box1.rot_idx
        to_ptype = dir == Int32(0) ? box2.ptype : box1.ptype
        if valid
            t_idx = s_t[1]
            n_to = dir == Int32(0) ? s_nactive2[1] : s_nactive1[1]
            trial = s_trial[1]; trial_rot = s_trial_r[1]
            insertion = sum_pair_energy(trial, trial_rot, t_idx, s_ptype[1], to_pos3, to_ridx,
                                         to_ptype, n_to, to_box.box,
                                         grid, tid, sdata)
            if tid == Int32(0)
                s_insertion[1] = insertion
            end
        end
        sync_threads()

        if tid == Int32(0)
            s_acc[1] = Int32(0)
        end
        if tid == Int32(0) && s_valid[1] != Int32(0)
            dir = s_dir[1]
            from_box, to_box = dir == Int32(0) ? (box1, box2) : (box2, box1)
            n_from_active = Float64(dir == Int32(0) ? s_nactive1[1] : s_nactive2[1])
            n_to_active   = Float64(dir == Int32(0) ? s_nactive2[1] : s_nactive1[1])
            V_from = volume(from_box)
            V_to   = volume(to_box)
            beta = 1.0 / temperature
            ln_acc = log(V_to / V_from) + log(n_from_active / (n_to_active + 1.0)) -
                     beta * (s_insertion[1] - s_removal[1])
            accept = ln_acc >= 0.0
            if !accept
                state = s_lcg[1]
                state, r = lcg_next(state)
                s_lcg[1] = state
                accept = r < exp(ln_acc)
            end
            if accept
                tidx = s_t[1]
                to_ptype = dir == Int32(0) ? box2.ptype : box1.ptype
                to_pos3 = dir == Int32(0) ? pos3_2 : pos3_1
                to_bins3 = dir == Int32(0) ? bins3_2 : bins3_1
                to_ridx = dir == Int32(0) ? box2.rot_idx : box1.rot_idx
                # Removal from `from` is done block-wide below (shift down);
                # the insertion at the end of `to` touches a different box.
                s_acc[1] = Int32(1)
                s_nold[1] = dir == Int32(0) ? s_nactive1[1] : s_nactive2[1]
                to_pos3[tidx+1] = s_trial[1]
                to_bins3[tidx+1] = s_trial_b[1]
                @inbounds to_ridx[tidx+1] = s_trial_r[1]
                ptype = s_ptype[1]
                @inbounds to_ptype[tidx+1] = ptype

                if dir == Int32(0)
                    s_nactive1[1] -= Int32(1); s_nactive2[1] += Int32(1)
                    s_de1[1] -= s_removal[1]; s_de2[1] += s_insertion[1]
                    if ptype == Int32(0)
                        s_type0d1[1] -= 1.0; s_type0d2[1] += 1.0
                    end
                else
                    s_nactive2[1] -= Int32(1); s_nactive1[1] += Int32(1)
                    s_de2[1] -= s_removal[1]; s_de1[1] += s_insertion[1]
                    if ptype == Int32(0)
                        s_type0d2[1] -= 1.0; s_type0d1[1] += 1.0
                    end
                end
                s_nacc[1] += 1.0
            end
        end
        sync_threads()

        if s_acc[1] != Int32(0)
            if s_dir[1] == Int32(0)
                compact_remove_slot!(box1.pos, box1.bins, box1.rot_idx, box1.ptype,
                                      s_s[1], s_nold[1], tid)
            else
                compact_remove_slot!(box2.pos, box2.bins, box2.rot_idx, box2.ptype,
                                      s_s[1], s_nold[1], tid)
            end
        end
        sync_threads()
    end

    if tid == Int32(0)
        result[1] = s_de1[1]
        result[2] = s_de2[1]
        result[3] = Float64(batch_size)
        result[4] = s_nacc[1]
        result[5] = Float64(s_nactive1[1])
        result[6] = Float64(s_nactive2[1])
        result[7] = s_type0d1[1]
        result[8] = s_type0d2[1]
        lcg[1] = s_lcg[1]
    end
    return
end

function transfer_batch_launch!(box1::TransferBoxView, box2::TransferBoxView,
                                 grid::GridSet, batch_size::Integer, temperature::Float64,
                                 lcg::AbstractVector{UInt64}, result::AbstractVector{Float64},
                                 n_active1::Integer, n_active2::Integer;
                                 stream::CuStream=CUDA.stream())
    @cuda threads=256 blocks=1 stream=stream transfer_batch_kernel!(
        box1, box2, grid, Int32(batch_size), temperature, lcg, result,
        Int32(n_active1), Int32(n_active2))
end

# ---- NVT same-box insertion (single box) ----
#
# Like transfer_batch_kernel! with source == target box: a random active particle s is removed
# and re-inserted at a random position / orientation, appended at the highest slot (compact
# order-preserving removal, as in a transfer). N and V are unchanged, so
# ln_acc = -beta*(E_insert - E_removal). result = [delta_e, 0, batch_size, n_accepted, n_active, 0, 0, 0]
# (same layout as transfer_batch_kernel!'s result with box2 untouched).

# Energy of `point` (appended after every other slot) against slots [0,n) except `skip`.
@inline function sum_pair_energy_excl(point::SVector{3,Float64}, rot_idx::Int32, skip::Int32,
                                       self_type::Int32, pos3, ridx, ptype,
                                       n::Int32, box::SVector{3,Float64},
                                       grid::GridSet, tid::Int32, sdata)
    val = 0.0
    j = tid
    while j < n
        if j != skip
            pj = pos3[j+1]
            rij = @inbounds ridx[j+1]
            tj = @inbounds ptype[j+1]
            ctx = ctx_index(self_type, tj, grid.n_types) + 1
            val += calculate_pair_energy(pj, rij, point, rot_idx, box,
                        grid.nrot, grid.blens[ctx], grid.scl, grid.rcut2s[ctx], grid.elec_scale, grid.vdw_scale,
                        grid.energy_ctxs[ctx], grid.angle_mapping, grid.rotation_matrices)
        end
        j += Int32(256)
    end
    sdata[tid+1] = val
    sync_threads()
    s = Int32(128)
    while s > Int32(0)
        if tid < s
            sdata[tid+1] += sdata[tid+1+s]
        end
        sync_threads()
        s >>= Int32(1)
    end
    return sdata[1]
end

function nvt_insert_kernel!(
        box::TransferBoxView, grid::GridSet,
        batch_size::Int32, temperature::Float64,
        lcg::CuDeviceVector{UInt64,1}, result::CuDeviceVector{Float64,1},
        n_active::Int32)

    tid = threadIdx().x - Int32(1)
    pos3  = as_vec3(box.pos)
    bins3 = as_vec3(box.bins)

    sdata   = @cuStaticSharedMem(Float64, 256)
    s_lcg   = @cuStaticSharedMem(UInt64, 1)
    s_de    = @cuStaticSharedMem(Float64, 1)
    s_nacc  = @cuStaticSharedMem(Float64, 1)
    s_valid = @cuStaticSharedMem(Int32, 1)
    s_s     = @cuStaticSharedMem(Int32, 1)
    s_acc   = @cuStaticSharedMem(Int32, 1)
    s_ptype = @cuStaticSharedMem(Int32, 1)
    s_trial = @cuStaticSharedMem(SVector{3,Float64}, 1)
    s_trial_b = @cuStaticSharedMem(SVector{3,Float64}, 1)
    s_trial_r = @cuStaticSharedMem(Int32, 1)
    s_removal = @cuStaticSharedMem(Float64, 1)
    s_insertion = @cuStaticSharedMem(Float64, 1)

    if tid == Int32(0)
        s_lcg[1] = lcg[1]
        s_de[1] = 0.0; s_nacc[1] = 0.0
    end
    sync_threads()

    n = n_active
    for _ in Int32(1):batch_size
        if tid == Int32(0)
            valid = Int32(0)
            if n > 0
                state = s_lcg[1]
                state, r = lcg_next(state)
                r_idx = unsafe_trunc(Int32, r * n)
                r_idx = r_idx >= n ? n - Int32(1) : r_idx
                valid = Int32(1)
                s_s[1] = r_idx
                s_ptype[1] = @inbounds box.ptype[r_idx+1]
                tp = MVector{3,Float64}(0.0, 0.0, 0.0)
                tb = MVector{3,Float64}(0.0, 0.0, 0.0)
                for d in 1:3
                    state, ra = lcg_next(state)
                    tp[d] = ra * box.box[d]
                    state, rb = lcg_next(state)
                    tb[d] = rb * grid.nbin
                end
                trial_b = SVector(tb)
                s_trial[1] = SVector(tp)
                s_trial_b[1] = trial_b
                s_trial_r[1] = bins_to_rot_idx(trial_b, grid.angle_indices, grid.nbin)
                s_lcg[1] = state
            end
            s_valid[1] = valid
        end
        sync_threads()

        valid = s_valid[1] != Int32(0)
        if valid
            s_idx = s_s[1]
            removal = sum_pair_energy(pos3[s_idx+1], @inbounds(box.rot_idx[s_idx+1]), s_idx, s_ptype[1],
                                       pos3, box.rot_idx, box.ptype, n, box.box, grid, tid, sdata)
            insertion = sum_pair_energy_excl(s_trial[1], s_trial_r[1], s_idx, s_ptype[1],
                                              pos3, box.rot_idx, box.ptype, n, box.box, grid, tid, sdata)
            if tid == Int32(0)
                s_removal[1] = removal
                s_insertion[1] = insertion
            end
        end
        sync_threads()

        if tid == Int32(0)
            s_acc[1] = Int32(0)
            if valid
                ln_acc = -(s_insertion[1] - s_removal[1]) / temperature
                accept = ln_acc >= 0.0
                if !accept
                    state = s_lcg[1]
                    state, r = lcg_next(state)
                    s_lcg[1] = state
                    accept = r < exp(ln_acc)
                end
                if accept
                    s_acc[1] = Int32(1)
                    s_de[1] += s_insertion[1] - s_removal[1]
                    s_nacc[1] += 1.0
                end
            end
        end
        sync_threads()

        if s_acc[1] != Int32(0)
            ptype = s_ptype[1]; tr = s_trial[1]; trb = s_trial_b[1]; trr = s_trial_r[1]
            compact_remove_slot!(box.pos, box.bins, box.rot_idx, box.ptype, s_s[1], n, tid)
            sync_threads()
            if tid == Int32(0)
                pos3[n] = tr
                bins3[n] = trb
                @inbounds box.rot_idx[n] = trr
                @inbounds box.ptype[n] = ptype
            end
        end
        sync_threads()
    end

    if tid == Int32(0)
        result[1] = s_de[1]
        result[2] = 0.0
        result[3] = Float64(batch_size)
        result[4] = s_nacc[1]
        result[5] = Float64(n)
        result[6] = 0.0; result[7] = 0.0; result[8] = 0.0
        lcg[1] = s_lcg[1]
    end
    return
end

function nvt_insert_launch!(box::TransferBoxView, grid::GridSet, batch_size::Integer, temperature::Float64,
                            lcg::AbstractVector{UInt64}, result::AbstractVector{Float64},
                            n_active::Integer; stream::CuStream=CUDA.stream())
    @cuda threads=256 blocks=1 stream=stream nvt_insert_kernel!(
        box, grid, Int32(batch_size), temperature, lcg, result, Int32(n_active))
end
