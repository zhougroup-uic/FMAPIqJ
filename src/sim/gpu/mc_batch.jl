# mc_batch.jl - GEMC displacement kernel: box1+box2 as one launch (block 0
# = box1, block 1 = box2), the Julia counterpart of the reference CUDA
# engine's mc_batch_pair_kernel. box1/box2 are plain separate arguments
# (not an NTuple{N,...}) since only 2 boxes are ever needed.
#
# No mask: slots are kept compact ([0,n_active) all active), so every pick
# is an active particle. The active count of each box is passed per launch
# (it changes as transfers move particles between boxes).

using CUDA
using Adapt
using StaticArrays

# Parametrized over the array type - see GridView's comment in mc_core.jl
# for why (host-side CuArray fields, auto-converted to CuDeviceArray by
# `cudaconvert` when passed through `@cuda`).
struct BoxView{VF<:AbstractVector{Float64}, VI<:AbstractVector{Int32}, VU<:AbstractVector{UInt64}}
    pos::VF          # 3*n_particles, particle-major (as_vec3'd inside the kernel)
    bins::VF         # 3*n_particles
    rot_idx::VI      # n_particles
    ptype::VI        # n_particles; particle type per slot - unused by the
                      # single-context mc_batch_kernel! below (grid::GridView
                      # method, kept bit-identical to the C reference), read only by
                      # the mixture-aware method in mix_kernels.jl
                      # (grid::GridSet) - a harmless extra field for the
                      # former, so it stays present unconditionally rather
                      # than needing two BoxView types.
    box::SVector{3,Float64}
    n_particles::Int32   # array capacity, not the active count
    max_trans::Float64
    max_rot::Float64
    lcg::VU          # length 1
    result::VF       # length 2: [total_delta, n_accepted]
end
Adapt.@adapt_structure BoxView   # see mc_core.jl's GridView comment

# GridView is defined in mc_core.jl (shared with transfer_batch.jl) - the
# top-level driver includes mc_core.jl once before this file.

function mc_batch_kernel!(box1::BoxView, box2::BoxView, grid::GridView,
                           batch_size::Int32, temperature::Float64,
                           n_active1::Int32, n_active2::Int32)
    box = blockIdx().x == Int32(1) ? box1 : box2
    tid = threadIdx().x - Int32(1)   # 0-based, mirrors C's `int tid`

    pos3  = as_vec3(box.pos)
    bins3 = as_vec3(box.bins)

    sdata      = @cuStaticSharedMem(Float64, 256)
    s_lcg      = @cuStaticSharedMem(UInt64, 1)
    s_delta    = @cuStaticSharedMem(Float64, 1)
    s_naccept  = @cuStaticSharedMem(Float64, 1)
    s_pid      = @cuStaticSharedMem(Int32, 1)
    s_active   = @cuStaticSharedMem(Int32, 1)
    s_cur      = @cuStaticSharedMem(SVector{3,Float64}, 1)
    s_cur_bins = @cuStaticSharedMem(SVector{3,Float64}, 1)
    s_cur_rot  = @cuStaticSharedMem(Int32, 1)
    s_trial    = @cuStaticSharedMem(SVector{3,Float64}, 1)
    s_trial_b  = @cuStaticSharedMem(SVector{3,Float64}, 1)
    s_trial_r  = @cuStaticSharedMem(Int32, 1)

    if tid == Int32(0)
        s_delta[1]   = 0.0
        s_naccept[1] = 0.0
        s_lcg[1]     = box.lcg[1]
    end
    sync_threads()

    n = blockIdx().x == Int32(1) ? n_active1 : n_active2   # compact: [0,n) active
    for _ in Int32(1):batch_size
        if tid == Int32(0)
            state = s_lcg[1]
            state, r = lcg_next(state)
            pid = unsafe_trunc(Int32, ceil(r * n) - 1.0)
            pid = pid < Int32(0) ? Int32(0) : (pid >= n ? n - Int32(1) : pid)
            s_pid[1] = pid
            active = n > Int32(0)
            s_active[1] = active ? Int32(1) : Int32(0)

            # Trial generation (and its 6 LCG draws) is skipped only for an
            # empty box (n==0); every pick otherwise lands on an active slot.
            if active
                cur  = pos3[pid+1]
                curb = bins3[pid+1]
                s_cur[1]      = cur
                s_cur_bins[1] = curb
                s_cur_rot[1]  = @inbounds box.rot_idx[pid + 1]

                # Same interleaved trans,rot,trans,rot,trans,rot draw order
                # as the original per-dimension C loop, via a mutable
                # scratch (SVector itself is immutable) built up
                # component-by-component then frozen once at the end.
                tp = MVector{3,Float64}(0.0, 0.0, 0.0)
                tb = MVector{3,Float64}(0.0, 0.0, 0.0)
                for d in 1:3
                    state, r1 = lcg_next(state)
                    tp[d] = cwrap(cur[d] + (r1 - 0.5) * box.max_trans, box.box[d])
                    state, r2 = lcg_next(state)
                    tb[d] = cwrap(curb[d] + (r2 - 0.5) * box.max_rot, Float64(grid.nbin))
                end
                trial = SVector(tp); trial_b = SVector(tb)
                s_trial[1] = trial
                s_trial_b[1] = trial_b
                s_trial_r[1] = bins_to_rot_idx(trial_b, grid.angle_indices, grid.nbin)
            end
            s_lcg[1] = state
        end
        sync_threads()

        active = s_active[1] != Int32(0)
        if active
            pid = s_pid[1]
            cur = s_cur[1]; cur_rot = s_cur_rot[1]
            trial = s_trial[1]; trial_rot = s_trial_r[1]

            # Two single-value reductions (shared with transfer_batch.jl's
            # removal/insertion) instead of one fused per-neighbor
            # difference - see sum_pair_energy's comment in mc_core.jl for
            # why that's safe here specifically.
            sum_new = sum_pair_energy(trial, trial_rot, pid, pos3, box.rot_idx,
                                       n, box.box, grid, tid, sdata)
            sum_old = sum_pair_energy(cur, cur_rot, pid, pos3, box.rot_idx,
                                       n, box.box, grid, tid, sdata)

            if tid == Int32(0)
                delta = sum_new - sum_old
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
                    s_naccept[1] += 1.0
                end
            end
        end
        sync_threads()
    end

    if tid == Int32(0)
        box.result[1] = s_delta[1]
        box.result[2] = s_naccept[1]
        box.lcg[1] = s_lcg[1]
    end
    return
end

function mc_batch_launch!(box1::BoxView, box2::BoxView, grid::GridView,
                           batch_size::Integer, temperature::Float64,
                           n_active1::Integer, n_active2::Integer;
                           stream::CuStream=CUDA.stream())
    @cuda threads=256 blocks=2 stream=stream mc_batch_kernel!(box1, box2, grid, Int32(batch_size), temperature,
                                                               Int32(n_active1), Int32(n_active2))
end
