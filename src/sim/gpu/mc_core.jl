# mc_core.jl - device-level physics core, ported term-for-term from
# the reference CUDA engine's fmap_energy.cu (apply_pbc_gpu, to_grid_index_gpu,
# apply_rotation_gpu, calculate_pair_energy_gpu, lcg_next_dev).
#
# Positions/bins/box use StaticArrays.jl's SVector{3,Float64} instead of 3
# separate scalars/manual `arr[idx*3+1/2/3]` indexing - mirrors the vec3
# struct used in the C reference. `p1 - p2`,
# `sum(abs2, v)` etc. just work elementwise on SVector, so this collapses
# what were 3-line component-wise blocks into single expressions.

using CUDA
using Adapt
using StaticArrays

const TERN_MAX = Int8(127)   # tErn_MAX = (tErnElem)SCHAR_MAX, fmap_energy.cu:32
const TERN_SCL = 8.0         # tErn_SCL, fmap_energy.cu:33
# Cell format, the Julia counterpart of the C/CUDA -DTERN_BYTES switch
# (ext/tern.h): 2 (default) = [elec,vdw] bytes per cell, elec_scale/vdw_scale
# live; 1 = one pre-summed byte, scales ignored. Chosen at load time via the
# environment (TERN_BYTES=1 julia ...); a const, so the
# unused branch below is compiled away. Must match the grid files' format.
const TERN_BYTES = parse(Int, get(ENV, "TERN_BYTES", "2"))
TERN_BYTES in (1, 2) || error("TERN_BYTES must be 1 or 2, got $TERN_BYTES")

# Shared FMAP grid view - used by both mc_batch.jl (displacement) and
# transfer_batch.jl (transfer); kept here (not in either) so both files
# can just `include("mc_core.jl")` without redefining it twice.
struct GridView{V1<:AbstractVector{Int8}, V2<:AbstractVector{Int16}, V3<:AbstractVector{Float64}}
    energy_data::V1
    angle_indices::V2
    angle_mapping::V2
    rotation_matrices::V3
    nrot::Int32
    blen::Int32
    nbin::Int32
    scl::Float64
    rcut2::Float64
    elec_scale::Float64
    vdw_scale::Float64
end
Adapt.@adapt_structure GridView

# C: fmod(x, m); if (x < 0) x += m;  - Julia's `rem(x, m)` (default
# RoundToZero) is the same hardware fmod C uses, so this reproduces the
# two-step C logic bit-for-bit rather than relying on Julia's fused `mod`
# (which computes x - m*floor(x/m), a different algorithm that isn't
# guaranteed bit-identical to fmod+correction in every case).
@inline function cwrap(x::Float64, m::Float64)
    x = rem(x, m)
    return x < 0.0 ? x + m : x
end

@inline function apply_pbc(x::Float64, box_length::Float64)
    h = box_length * 0.5
    return x > h ? x - box_length : (x < -h ? x + box_length : x)
end
@inline apply_pbc3(d::SVector{3,Float64}, box::SVector{3,Float64}) =
    SVector(apply_pbc(d[1], box[1]), apply_pbc(d[2], box[2]), apply_pbc(d[3], box[3]))

# C: idx = (int)round(x); (idx<0) ? idx+grid_size : (idx==grid_size) ? 0 : idx
# C99's round() is round-half-AWAY-FROM-ZERO. Julia's round(x) default mode
# is RoundNearest (ties-to-even / banker's rounding) - NOT the same at
# exact .5 ties - so RoundNearestTiesAway must be passed explicitly to
# match C bit-for-bit (matters only at exact ties, but the port is
# checked for bit-identical results, not "close enough").
@inline function to_grid_index(x::Float64, grid_size::Int32)
    idx = unsafe_trunc(Int32, round(x, RoundNearestTiesAway))
    return idx < Int32(0) ? idx + grid_size : (idx == grid_size ? Int32(0) : idx)
end

# C: bins_to_rot_idx_dev - ai is angle_indices, flat nbin^3, 0-based.
@inline function bins_to_rot_idx(bins::SVector{3,Float64}, ai, nbin::Int32)
    bi = to_grid_index(bins[1], nbin)
    bj = to_grid_index(bins[2], nbin)
    bk = to_grid_index(bins[3], nbin)
    idx = (Int64(bi) * nbin + bj) * nbin + bk
    return Int32(@inbounds ai[idx + 1])
end

@inline function apply_rotation(rot, base::Int64, c::SVector{3,Float64})
    return SVector(
        rot[base+1]*c[1] + rot[base+2]*c[2] + rot[base+3]*c[3],
        rot[base+4]*c[1] + rot[base+5]*c[2] + rot[base+6]*c[3],
        rot[base+7]*c[1] + rot[base+8]*c[2] + rot[base+9]*c[3])
end

# C: calculate_pair_energy_gpu - nearest-cell FMAP lookup (no interpolation).
# energy_data is flat Int8, TERN_BYTES (2 or 1) entries/cell, 0-based offset
# a3*blen^3 + i*blen^2 + j*blen + k, entries at [off*2] (elec) / [off*2+1]
# (vdw) - tErn[2] flattened. angle_mapping is flat nrot*nrot, 0-based
# a1*nrot+a2. rotation_matrices is flat nrot*9, 0-based a1*9.
@inline function calculate_pair_energy(
        p1::SVector{3,Float64}, rot_idx1::Int32,
        p2::SVector{3,Float64}, rot_idx2::Int32,
        box::SVector{3,Float64},
        nrot::Int32, blen::Int32, scl::Float64, rcut2::Float64, elec_scale::Float64, vdw_scale::Float64,
        energy_data, angle_mapping, rotation_matrices)

    rel = apply_pbc3(p1 - p2, box)
    d2 = sum(abs2, rel)
    d2 > rcut2 && return 0.0

    a1 = rot_idx1
    a2 = rot_idx2
    a3 = Int32(@inbounds angle_mapping[Int64(a1)*nrot + a2 + 1])

    rotated = apply_rotation(rotation_matrices, Int64(a1) * 9, rel)

    h = blen ÷ Int32(2)
    i = unsafe_trunc(Int32, rotated[1]*scl + h + 0.5)
    j = unsafe_trunc(Int32, rotated[2]*scl + h + 0.5)
    k = unsafe_trunc(Int32, rotated[3]*scl + h + 0.5)

    (i < 0 || i >= blen || j < 0 || j >= blen || k < 0 || k >= blen) && return 0.0

    blen2 = Int64(blen) * blen
    blen3 = blen2 * blen
    off = Int64(a3)*blen3 + Int64(i)*blen2 + Int64(j)*blen + k

    if TERN_BYTES == 2
        e0 = @inbounds energy_data[off*2 + 1]
        e1 = @inbounds energy_data[off*2 + 2]
        e0 == TERN_MAX && return 9999.0
        return (Float64(e0) * elec_scale + Float64(e1) * vdw_scale) / TERN_SCL
    else
        e = @inbounds energy_data[off + 1]
        e == TERN_MAX && return 9999.0
        return Float64(e) / TERN_SCL   # elec_scale/vdw_scale ignored (=1)
    end
end

# Shared by mc_batch.jl (twice per step: trial sum, cur sum, subtracted)
# and transfer_batch.jl (twice per step: removal sum, insertion sum) - one
# point/rot_idx vs every other particle in a box, block-wide reduced. Slots
# are kept compact ([0,n) all active, no mask), so the loop runs 0..n-1.
#
# NOTE: the move energy is computed as two passes of this primitive (sum_new,
# then sum_old, subtracted after) rather than one fused sum_j(e_new_j - e_old_j).
# Mathematically equal, but not bit-identical in floating point (summation
# order differs from the original C mc_batch_body); the reference C copy in
# the test setup uses the same two-pass form.
@inline function sum_pair_energy(point::SVector{3,Float64}, rot_idx::Int32, self_idx::Int32,
                                  pos3, ridx, n::Int32, box::SVector{3,Float64},
                                  grid::GridView, tid::Int32, sdata)
    val = 0.0
    j = tid
    while j < n
        if j != self_idx
            pj = pos3[j+1]
            rij = @inbounds ridx[j+1]
            if j < self_idx
                val += calculate_pair_energy(pj, rij, point, rot_idx, box,
                            grid.nrot,grid.blen,grid.scl,grid.rcut2,grid.elec_scale,grid.vdw_scale,
                            grid.energy_data,grid.angle_mapping,grid.rotation_matrices)
            else
                val += calculate_pair_energy(point, rot_idx, pj, rij, box,
                            grid.nrot,grid.blen,grid.scl,grid.rcut2,grid.elec_scale,grid.vdw_scale,
                            grid.energy_data,grid.angle_mapping,grid.rotation_matrices)
            end
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

# GridSet: real-mixture FMAP contexts, one energy grid per sorted type
# pair (AA, AB, BB, ...), N = n_types*(n_types+1)/2. Only energy_data
# differs per pair in the on-disk convention (T0/{AA,AB,BB} each have
# byte-identical fmap.rot.zlib, only fmap.ve.zlib differs), so the rotation grid
# (angle_indices/angle_mapping/rotation_matrices/nrot/blen/nbin/scl/rcut2)
# and elec_scale (per-REPLICA temperature scaling, not per-context) are
# shared across all N contexts and loaded/stored once, exactly mirroring
# GridView's sharing pattern (see mk_grid_view's comment) - just extended
# from 1 energy grid to N.
#
# This is the type-aware counterpart to GridView/sum_pair_energy below,
# selected automatically via multiple dispatch (sum_pair_energy(...,
# grid::GridView, ...) vs sum_pair_energy(..., grid::GridSet, ...)) - the
# existing GridView/BoxView/mc_batch_kernel!/transfer_batch_kernel! methods
# are left untouched, and the GridSet path is used only by the
# mixture-aware kernels (see mix_kernels.jl).
struct GridSet{V1<:AbstractVector{Int8}, V2<:AbstractVector{Int16}, V3<:AbstractVector{Float64}, N}
    energy_ctxs::NTuple{N, V1}
    angle_indices::V2
    angle_mapping::V2
    rotation_matrices::V3
    nrot::Int32
    # blen (grid extent) is proportional to molecule size - AA/AB/BB
    # legitimately need different grid dimensions once each is generated
    # from its own PQR file(s) instead of one shared "AB" file copied to
    # all three, so this is a per-context tuple (like energy_ctxs), not a
    # single shared scalar. rcut2 is derived from blen (half the box
    # width) so it varies in lockstep and is also per-context. nrot/nbin/
    # scl stay genuinely invariant across contexts (angular resolution
    # depends only on ang.dat; scl=1/spacing, a fixed constant) and remain
    # single shared scalars.
    blens::NTuple{N, Int32}
    nbin::Int32
    scl::Float64
    rcut2s::NTuple{N, Float64}
    elec_scale::Float64
    vdw_scale::Float64
    n_types::Int32
end
Adapt.@adapt_structure GridSet

# simulation.c/simulation_gemc.c's get_ctx, ported term-for-term: sorted
# pair (i,j), i<=j, idx = i*(2*n_types-1-i)/2 + j - this is the CPU
# engine's real per-type-pair context selection (the ONLY place in this
# whole code base that does real mixture energetics; the original GPU
# kernels used ctx 0 for all particles, an approximation for cross-type
# pairs). 0-based in, 0-based out.
@inline function ctx_index(t1::Int32, t2::Int32, n_types::Int32)
    i = t1 < t2 ? t1 : t2
    j = t1 < t2 ? t2 : t1
    return i * (Int32(2)*n_types - Int32(1) - i) ÷ Int32(2) + j
end

# Type-aware counterpart to sum_pair_energy(..., grid::GridView, ...)
# above - identical reduction structure, but each neighbor's pairwise
# energy is looked up in ITS OWN sorted-type-pair context (selected via
# ctx_index), not unconditionally context 0. self_type is the type of the
# particle `point`/`rot_idx` belongs to (passed once per call since a
# single MC/transfer move never changes a particle's own type mid-step).
@inline function sum_pair_energy(point::SVector{3,Float64}, rot_idx::Int32, self_idx::Int32,
                                  self_type::Int32, pos3, ridx, ptype,
                                  n::Int32, box::SVector{3,Float64},
                                  grid::GridSet, tid::Int32, sdata)
    val = 0.0
    j = tid
    while j < n
        if j != self_idx
            pj = pos3[j+1]
            rij = @inbounds ridx[j+1]
            tj = @inbounds ptype[j+1]
            ctx = ctx_index(self_type, tj, grid.n_types) + 1
            edata = grid.energy_ctxs[ctx]
            blen  = grid.blens[ctx]
            rcut2 = grid.rcut2s[ctx]
            if j < self_idx
                val += calculate_pair_energy(pj, rij, point, rot_idx, box,
                            grid.nrot,blen,grid.scl,rcut2,grid.elec_scale,grid.vdw_scale,
                            edata,grid.angle_mapping,grid.rotation_matrices)
            else
                val += calculate_pair_energy(point, rot_idx, pj, rij, box,
                            grid.nrot,blen,grid.scl,rcut2,grid.elec_scale,grid.vdw_scale,
                            edata,grid.angle_mapping,grid.rotation_matrices)
            end
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

# Displacement-move energy change of one particle, summed the way the C kernel does
# (mc_batch_body_mix): every neighbour contributes e_new - e_old in ONE pass over the strided
# per-thread partition, then the same binary-tree reduction. Summing E_new and E_old
# separately and subtracting is equal mathematically but rounds differently, which changes
# accept/reject decisions at the last bit.
@inline function sum_pair_energy_diff(trial::SVector{3,Float64}, trial_rot::Int32,
                                       cur::SVector{3,Float64}, cur_rot::Int32, self_idx::Int32,
                                       self_type::Int32, pos3, ridx, ptype,
                                       n::Int32, box::SVector{3,Float64},
                                       grid::GridSet, tid::Int32, sdata)
    val = 0.0
    j = tid
    while j < n
        if j != self_idx
            pj = pos3[j+1]
            rij = @inbounds ridx[j+1]
            tj = @inbounds ptype[j+1]
            ctx = ctx_index(self_type, tj, grid.n_types) + 1
            edata = grid.energy_ctxs[ctx]
            blen  = grid.blens[ctx]
            rcut2 = grid.rcut2s[ctx]
            if j < self_idx
                e_new = calculate_pair_energy(pj, rij, trial, trial_rot, box,
                            grid.nrot,blen,grid.scl,rcut2,grid.elec_scale,grid.vdw_scale,
                            edata,grid.angle_mapping,grid.rotation_matrices)
                e_old = calculate_pair_energy(pj, rij, cur, cur_rot, box,
                            grid.nrot,blen,grid.scl,rcut2,grid.elec_scale,grid.vdw_scale,
                            edata,grid.angle_mapping,grid.rotation_matrices)
            else
                e_new = calculate_pair_energy(trial, trial_rot, pj, rij, box,
                            grid.nrot,blen,grid.scl,rcut2,grid.elec_scale,grid.vdw_scale,
                            edata,grid.angle_mapping,grid.rotation_matrices)
                e_old = calculate_pair_energy(cur, cur_rot, pj, rij, box,
                            grid.nrot,blen,grid.scl,rcut2,grid.elec_scale,grid.vdw_scale,
                            edata,grid.angle_mapping,grid.rotation_matrices)
            end
            val += e_new - e_old
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

# Order-preserving removal of slot `s` from a compact box of `n` particles:
# slots s+1..n-1 shift down by one (pair energy depends on slot-index order
# since FMAP is asymmetric, so a swap-with-last would change it). Block-wide:
# every thread must call it with the same arguments. Chunks of 256 slots are
# read into registers, synced, then written, so the in-place shift is safe.
# The caller must sync_threads() before reuse of anything this touches.
@inline function compact_remove_slot!(pos, bins, ridx, ptype, s::Int32, n::Int32, tid::Int32)
    base = s
    while base < n - Int32(1)
        k = base + tid
        live = k < n - Int32(1)
        p1 = 0.0; p2 = 0.0; p3 = 0.0
        b1 = 0.0; b2 = 0.0; b3 = 0.0
        r = Int32(0); t = Int32(0)
        if live
            o = (k + Int32(1)) * Int32(3)
            @inbounds begin
                p1 = pos[o+1]; p2 = pos[o+2]; p3 = pos[o+3]
                b1 = bins[o+1]; b2 = bins[o+2]; b3 = bins[o+3]
                r = ridx[k+2]; t = ptype[k+2]
            end
        end
        sync_threads()
        if live
            o = k * Int32(3)
            @inbounds begin
                pos[o+1] = p1; pos[o+2] = p2; pos[o+3] = p3
                bins[o+1] = b1; bins[o+2] = b2; bins[o+3] = b3
                ridx[k+1] = r; ptype[k+1] = t
            end
        end
        sync_threads()
        base += Int32(256)
    end
    return
end

# C: lcg_next_dev - state = (state*1103515245+12345) & 0x7fffffff; return
# state/0x7fffffff. Kept as UInt64 throughout (matches C's `unsigned long`
# on this platform, 64-bit) - masking with 0x7fffffff after the multiply
# means the multiply's upper bits never matter, so this is bit-identical
# to the C version regardless of the intermediate width.
@inline function lcg_next(state::UInt64)
    state = (state * UInt64(1103515245) + UInt64(12345)) & UInt64(0x7fffffff)
    return state, Float64(state) / Float64(0x7fffffff)
end

# Reinterprets a flat 3*n Float64 array (particle-major: x,y,z,x,y,z,...)
# as an n-element array of SVector{3,Float64} - one indexing op per
# particle instead of 3 (`v[pid+1]` instead of
# `v[pid*3+1], v[pid*3+2], v[pid*3+3]`). Works on both CuArray (host side)
# and CuDeviceArray (post-cudaconvert, inside a kernel) - same trick as
# the vec3 cast in the C reference, just Julia's built-in reinterpret
# instead of a C pointer cast.
@inline as_vec3(flat) = reinterpret(SVector{3,Float64}, flat)
