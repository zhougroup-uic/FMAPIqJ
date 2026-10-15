# transfer_batch.jl - GEMC transfer kernel with particle-type tracking,
# the Julia counterpart of the reference CUDA engine's
# transfer_batch_multitype_ kernel. Always tracks particle type.
#
# Slots are kept compact ([0,n_active) all active, no mask): a transfer
# appends at the highest slot of the target box, and removal shifts the
# slots above the removed one down by one (order-preserving, since FMAP pair
# energy depends on slot-index order) - see compact_remove_slot! in mc_core.jl.

using CUDA
using Adapt
using StaticArrays
# mc_core.jl (lcg_next, calculate_pair_energy, bins_to_rot_idx, as_vec3,
# GridView) is included once by the top-level driver before this file.

# Parametrized over the array type - see GridView's comment in mc_core.jl.
struct TransferBoxView{VF<:AbstractVector{Float64}, VI<:AbstractVector{Int32}}
    pos::VF
    bins::VF
    rot_idx::VI
    ptype::VI     # particle type per slot
    box::SVector{3,Float64}
    max_cap::Int32
end
Adapt.@adapt_structure TransferBoxView   # see mc_core.jl's GridView comment

@inline volume(b::TransferBoxView) = b.box[1] * b.box[2] * b.box[3]

function transfer_batch_kernel!(
        box1::TransferBoxView, box2::TransferBoxView, grid::GridView,
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
        dir0 = (step - Int32(1)) & Int32(1)   # 0: 1->2, 1: 2->1

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

        # ---- Removal energy: source slot vs all active slots in 'from' ----
        # (shared sum_pair_energy primitive, see mc_core.jl). s_valid[1] is
        # the same value on every thread (shared mem), so `if valid` below
        # is a uniform branch - all 256 threads take it together or skip it
        # together, so calling a block-wide-reducing function inside it is
        # safe (no divergence/deadlock risk).
        valid = s_valid[1] != Int32(0)
        dir = s_dir[1]
        from_box = dir == Int32(0) ? box1 : box2
        from_pos3 = dir == Int32(0) ? pos3_1 : pos3_2
        from_ridx = dir == Int32(0) ? box1.rot_idx : box2.rot_idx
        if valid
            s_idx = s_s[1]
            pos_s = from_pos3[s_idx+1]
            ris = @inbounds from_ridx[s_idx+1]
            n_from = dir == Int32(0) ? s_nactive1[1] : s_nactive2[1]
            removal = sum_pair_energy(pos_s, ris, s_idx, from_pos3, from_ridx,
                                       n_from, from_box.box, grid, tid, sdata)
            if tid == Int32(0)
                s_removal[1] = removal
            end
        end
        sync_threads()

        # ---- Insertion energy: trial particle vs all active slots in 'to' ----
        to_box = dir == Int32(0) ? box2 : box1
        to_pos3 = dir == Int32(0) ? pos3_2 : pos3_1
        to_ridx = dir == Int32(0) ? box2.rot_idx : box1.rot_idx
        if valid
            t_idx = s_t[1]
            n_to = dir == Int32(0) ? s_nactive2[1] : s_nactive1[1]
            trial = s_trial[1]; trial_rot = s_trial_r[1]
            insertion = sum_pair_energy(trial, trial_rot, t_idx, to_pos3, to_ridx,
                                         n_to, to_box.box, grid, tid, sdata)
            if tid == Int32(0)
                s_insertion[1] = insertion
            end
        end
        sync_threads()

        # ---- GEMC Metropolis + commit ----
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
                                 grid::GridView, batch_size::Integer, temperature::Float64,
                                 lcg::AbstractVector{UInt64}, result::AbstractVector{Float64},
                                 n_active1::Integer, n_active2::Integer;
                                 stream::CuStream=CUDA.stream())
    @cuda threads=256 blocks=1 stream=stream transfer_batch_kernel!(
        box1, box2, grid, Int32(batch_size), temperature, lcg, result,
        Int32(n_active1), Int32(n_active2))
end
