# gemc_box.jl - box management for the full production driver
# (main_gemc_gpu.jl): real FMAP grid loading (reads grid_dump's flat
# files - no zlib in Julia), particle initialization, and particle
# configuration save/load in production's exact text format
# (particle_fmap.c's save_to_file/load_from_line:
# "%16d%16.10f%16.10f%16.10f%16.10f%16.10f%16.10f%8d\n", id/pos/euler/type).
# nbin (orientation bins per rotation period) comes from the grid header, never hard-coded.

using StaticArrays
using CUDA
using Printf

# particle_fmap.c's euler_to_bins/bins_to_euler, ported term for term.
# cwrap (mc_core.jl) reproduces C's fmod+correction bit-for-bit, same as
# everywhere else - not required for correctness here, but keeping it
# consistent costs nothing.
@inline function euler_to_bins(euler::SVector{3,Float64}, nbin::Integer)
    a = cwrap(euler[1] + pi, 2pi)
    b1 = a * nbin / (2pi)
    b = clamp(euler[2], 0.0, pi)
    b2 = (1.0 - cos(b)) * nbin / 2.0
    g = cwrap(euler[3] + pi, 2pi)
    b3 = g * nbin / (2pi)
    return SVector(b1, b2, b3)
end

@inline function bins_to_euler(bins::SVector{3,Float64}, nbin::Integer)
    e1 = -pi + bins[1] / nbin * 2pi
    e2 = acos(clamp(1.0 - bins[2] / nbin * 2.0, -1.0, 1.0))
    e3 = -pi + bins[3] / nbin * 2pi
    return SVector(e1, e2, e3)
end

# ============================================================================
# REAL FMAP GRID (GridData populated by fmap_zlib_loader.jl's
# load_fmap_zlib, reading .zlib files directly - no intermediate dump file)
# ============================================================================

struct GridData
    nrot::Int32
    blen::Int32
    nbin::Int32
    scl::Float64
    angle_indices::Vector{Int16}
    angle_mapping::Vector{Int16}
    rotation_matrices::Vector{Float64}
    energy_data::AbstractVector{Int8}   # host Vector, or a CuVector when streamed straight to the GPU
end

# rcut = blen*0.5/scl, matching main_gemc_gpu.cu's own derivation (not the
# zlib header's separately-stored rcut2, which fmap_zlib_loader.jl doesn't
# even read for the same reason main_gemc_gpu.cu re-derives it instead).
# elec_scale is per-replica (temperature-dependent), so this is called once
# per replica sharing the SAME underlying CuArrays (cheap struct
# construction, not a GPU memory copy - see GridView's comment in
# mc_core.jl: elec_scale is baked into the struct, so N replicas need N
# lightweight GridView wrappers, not N copies of the multi-GB grid data).
function mk_grid_view(g::GridData, d_energy, d_aidx, d_amap, d_rotmat, elec_scale::Float64, vdw_scale::Float64=1.0)
    rcut = Float64(g.blen) * 0.5 / g.scl
    return GridView(d_energy, d_aidx, d_amap, d_rotmat, g.nrot, g.blen, g.nbin, g.scl, rcut^2, elec_scale, vdw_scale)
end

# GridSetData: real-mixture counterpart of GridData - n_ctx =
# n_types*(n_types+1)/2 energy grids (one per sorted type pair, AA/AB/BB/
# ...), sharing one rotation grid (angle_indices/angle_mapping/
# rotation_matrices/nrot/blen/nbin/scl) - see fmap_zlib_loader.jl's
# load_fmap_paths, which reads production's own fmap.paths file (2*n_ctx
# lines, rot/ve alternating per sorted pair - main_gemc_gpu.cu's exact
# convention) and verifies every context shares ctx 0's geometry, matching
# what main_gemc_gpu.cu itself assumes (it only ever reads nrot/blen/nbin/
# scl from paths[0], then creates all n_ctx GPU contexts with those same
# dimensions).
struct GridSetData
    nrot::Int32
    blens::Vector{Int32}   # length n_ctx - per-context grid extent, see GridSet's comment
    nbin::Int32
    scl::Float64
    angle_indices::Vector{Int16}
    angle_mapping::Vector{Int16}
    rotation_matrices::Vector{Float64}
    energy_ctxs::Vector{<:AbstractVector{Int8}}   # length n_ctx; CuVectors when streamed to the GPU (load_fmap_paths_gpu)
    n_types::Int32
end

function mk_grid_set(g::GridSetData, d_energy_ctxs::NTuple, d_aidx, d_amap, d_rotmat, elec_scale::Float64, vdw_scale::Float64=1.0)
    blens  = Tuple(g.blens)
    rcut2s = Tuple((Float64(b) * 0.5 / g.scl)^2 for b in g.blens)
    return GridSet(d_energy_ctxs, d_aidx, d_amap, d_rotmat, g.nrot, blens, g.nbin, g.scl,
                    rcut2s, elec_scale, vdw_scale, g.n_types)
end

# ============================================================================
# PHYSICAL BOX: bundles what BoxView (mc) and TransferBoxView (transfer)
# both need, so each physical box is set up once and viewed both ways -
# see as_boxview/as_transferboxview below. max_cap = N_total (every box
# gets full headroom, matching main_gemc_gpu.cu's
# sim_gemc_gpu_create_box(n_total, ...) - either box could end up holding
# every particle as the simulation progresses).
# ============================================================================

struct PhysBox
    pos::CuVector{Float64}
    bins::CuVector{Float64}
    rot_idx::CuVector{Int32}
    ptype::CuVector{Int32}
    lcg::CuVector{UInt64}
    mc_result::CuVector{Float64}   # length 6: [delta, n_accepted, try_t, acc_t, try_r, acc_r]
    box::SVector{3,Float64}
    max_cap::Int32
    max_trans::Float64
    max_rot::Float64
end

as_boxview(b::PhysBox) = BoxView(b.pos, b.bins, b.rot_idx, b.ptype, b.box, b.max_cap,
                                  b.max_trans, b.max_rot, b.lcg, b.mc_result)
as_transferboxview(b::PhysBox) = TransferBoxView(b.pos, b.bins, b.rot_idx, b.ptype,
                                                  b.box, b.max_cap)

# Adaptive step sizes for one box, same rule as simulation_gemc_gpu.cu's
# sim_gemc_gpu_adjust_step_sizes: after each adjustment window, js (rs) is
# shrunk x0.99 if its acceptance ratio is below target_ratio (never below its
# starting value js0/rs0), else grown x1.01 (never past js_max/rs_max, a soft
# cap - the multiply can overshoot it once). Counters reset after each use.
# Same conventions as the C GPU driver: js0 = max_trans*1.62, rs0 = max_rot,
# ceilings are the box edge and nbin/ROT_DEFAULT (trial offset (r-0.5)*js or
# *rs*ROT_DEFAULT, see mix_kernels.jl): js_max = box edge, rs_max = nbin/3.
mutable struct StepTune
    js::Float64
    rs::Float64
    js0::Float64
    rs0::Float64
    js_max::Float64
    rs_max::Float64
    ntry::Int64
    nacc::Int64
    nrtry::Int64
    nracc::Int64
end
StepTune(js0::Float64, rs0::Float64, js_max::Float64, rs_max::Float64) =
    StepTune(js0, rs0, js0, rs0, js_max, rs_max, 0, 0, 0, 0)

function adjust_step_sizes!(t::StepTune, target_ratio::Float64)
    if t.ntry > 0
        if t.nacc / t.ntry < target_ratio
            t.js > t.js0 && (t.js *= 0.99)
        else
            t.js < t.js_max && (t.js *= 1.01)
        end
        t.ntry = 0; t.nacc = 0
    end
    if t.nrtry > 0
        if t.nracc / t.nrtry < target_ratio
            t.rs > t.rs0 && (t.rs *= 0.99)
        else
            t.rs < t.rs_max && (t.rs *= 1.01)
        end
        t.nrtry = 0; t.nracc = 0
    end
    return
end

# One GEMC triple: box1's lcg field doubles as the transfer stage's shared
# RNG stream too (matches production: fmap_run_transfer_batch_multitype_
# async always reads pool1's own d_lcg_state, never pool2's - box2's lcg
# is used only for box2's own displacement moves). n_active tracks
# mutable population state across cycles (read back from transfer_result
# after each transfer_batch_launch! call).
mutable struct Triple
    box1::PhysBox
    box2::PhysBox
    transfer_lcg::CuVector{UInt64}      # == box1.lcg (same array, aliased)
    transfer_result::CuVector{Float64}  # length 8
    n_active1::Int32
    n_active2::Int32
    stream::CuStream   # own stream, so triples run concurrently rather
                        # than serializing on the default stream - this is
                        # what actually gets the multi-triple parallelism
                        # (at most 128 resident grids on this GPU class).
    # Cumulative stats for the per-triple log.<stage> file (see
    # main_gemc_gpu.cu's sim_gemc_gpu_get_stats: total_energy, n_moves,
    # n_accepted, n_transfer_attempts, n_transfer_accepted, n_type_a_active
    # - all running totals since this Triple was constructed, i.e. since
    # the start of the CURRENT stage, matching how a fresh GemcGpuBox's
    # counters start at 0 each time main_gemc_gpu.cu runs a stage).
    # e1/e2 are the box's real total energy (seeded via compute_total_energy
    # - a genuine O(N^2) pairwise sum over the placed/loaded configuration,
    # matching C's own compute_total_energy - never 0.0), then updated by
    # every accepted move's delta exactly like C's box->total_energy.
    e1::Float64
    e2::Float64
    n_moves1::Int64
    n_accepted1::Int64
    n_moves2::Int64
    n_accepted2::Int64
    n_transfer_attempts::Int64
    n_transfer_accepted::Int64
    nta1::Int32   # n_type_a_active (type-0 count) in box1
    nta2::Int32
    tune1::StepTune
    tune2::StepTune
    # Startup-deficit bookkeeping (see main_gemc_gpu.jl): slots
    # [n_active, committed) already hold their intended type in box.ptype;
    # `deficit` is how many particles short of N the triple still is
    # (closed by reservoir insertion instead of transfers while > 0).
    committed1::Int32
    committed2::Int32
    deficit::Int
    res_result1::CuVector{Float64}   # length 4, reservoir_insert_kernel! output
    res_result2::CuVector{Float64}
end

# ============================================================================
# DEVICE-SIDE ENERGY HELPERS (initial placement / total energy)
# ============================================================================
# The FMAP energy grids live on the GPU only (streamed there by
# load_fmap_paths_gpu, like the C GPU driver) - there is no host copy to run
# calculate_pair_energy against - so the start-up energy work (overlap checks
# while placing particles, the initial total energy) runs on the device too,
# through the same sum_pair_energy the MC kernels use.

# One block: energy of a trial particle at slot n (appended after the n
# already-placed ones, so every placed particle is the lower index / p1)
# against slots [0,n).
function ins_energy_kernel!(out, tp::SVector{3,Float64}, tridx::Int32, ttype::Int32,
                             pos, ridx, ptype, n::Int32, box::SVector{3,Float64}, grid::GridSet)
    tid = threadIdx().x - Int32(1)
    sdata = @cuStaticSharedMem(Float64, 256)
    e = sum_pair_energy(tp, tridx, n, ttype, as_vec3(pos), ridx, ptype, n, box, grid, tid, sdata)
    tid == Int32(0) && (out[1] = e)
    return
end

# Block i: sum over j<i of e(p_j, p_i) (lower index as p1) - the total energy
# is the sum over blocks.
function total_energy_kernel!(out, pos, ridx, ptype, n::Int32, box::SVector{3,Float64}, grid::GridSet)
    tid = threadIdx().x - Int32(1)
    i = Int32(blockIdx().x) - Int32(1)
    sdata = @cuStaticSharedMem(Float64, 256)
    pos3 = as_vec3(pos)
    e = sum_pair_energy(pos3[i+1], ridx[i+1], i, ptype[i+1], pos3, ridx, ptype, i, box, grid, tid, sdata)
    tid == Int32(0) && (out[i+1] = e)
    return
end

# Placed-particle state for the rejection placement, mirrored on the device.
mutable struct PlaceCtx
    pos::CuVector{Float64}
    ridx::CuVector{Int32}
    ptype::CuVector{Int32}
    out::CuVector{Float64}
    gs::GridSet
    n::Int
end

function PlaceCtx(gs::GridSet, cap::Int, pos::Vector{Float64}=Float64[],
                  ridx::Vector{Int32}=Int32[], ptype::Vector{Int32}=Int32[])
    d_pos = CUDA.zeros(Float64, max(3cap, 1)); d_ridx = CUDA.zeros(Int32, max(cap, 1))
    d_pt = CUDA.zeros(Int32, max(cap, 1))
    n = length(ridx)
    if n > 0
        copyto!(d_pos, 1, pos, 1, 3n); copyto!(d_ridx, 1, ridx, 1, n); copyto!(d_pt, 1, ptype, 1, n)
    end
    return PlaceCtx(d_pos, d_ridx, d_pt, CUDA.zeros(Float64, 1), gs, n)
end

function insertion_energy(pc::PlaceCtx, tp::SVector{3,Float64}, tridx::Int32, ttype::Int32,
                          box_len::SVector{3,Float64})
    @cuda threads=256 blocks=1 ins_energy_kernel!(pc.out, tp, tridx, ttype, pc.pos, pc.ridx, pc.ptype,
                                                   Int32(pc.n), box_len, pc.gs)
    return Array(pc.out)[1]
end

function push_placed!(pc::PlaceCtx, tp::SVector{3,Float64}, tridx::Int32, ttype::Int32)
    copyto!(pc.pos, 3pc.n + 1, Float64[tp[1], tp[2], tp[3]], 1, 3)
    copyto!(pc.ridx, pc.n + 1, Int32[tridx], 1, 1)
    copyto!(pc.ptype, pc.n + 1, Int32[ttype], 1, 1)
    pc.n += 1
    return
end

# ============================================================================
# RANDOM INITIALIZATION (stage 0)
# ============================================================================

# Type of the slots left over after the per-type counts (C: n_counts < n_types ? n_counts : n_types-1).
# N_TYPES is set by the driver once n_types is known.
const N_TYPES = Ref(1)
last_type(n_counts::Int) = Int32(n_counts < N_TYPES[] ? n_counts : N_TYPES[] - 1)

# Intended type of each of n consecutive slots from per-type counts: type 0
# for the first counts[1] slots, type 1 for the next counts[2], ..., then the
# last type fills the rest - the same rule random_particles/topup_particles
# use inline (and C's compute_intended_types).
function intended_types(counts::Vector{Int}, n::Int)
    it = Vector{Int32}(undef, n)
    idx = 1
    for (t, cnt) in enumerate(counts)
        for _ in 1:cnt
            idx > n && break
            it[idx] = Int32(t - 1)
            idx += 1
        end
    end
    last_t = last_type(length(counts))
    while idx <= n
        it[idx] = last_t
        idx += 1
    end
    return it
end

# Overlap-avoiding rejection placement - ports simulation_gemc_gpu.cu's
# place_particles_random term-for-term: for each new particle, try up to
# 1000 random positions/orientations, computing its pairwise insertion
# energy against every already-placed particle via ctx[0] (the AA
# context - matches C's box->ff_ctxs[0] usage here exactly; initial
# placement was never part of the run-time mixture-energetics bug/fix,
# C always used ctx[0] for this specific routine, not the real per-
# type-pair selection the MC/transfer kernels use). Accepts the first
# attempt whose summed insertion energy is below 999.0 (the same
# excluded-volume/overlap threshold C uses, comfortably below the
# tErn_MAX/elec_scale-independent 9999.0 sentinel). If no attempt
# succeeds within 1000 tries, stops placing further particles - matches
# C's own fallback exactly (skips C's subsequent topup retry machinery
# for simplicity; worst case is the same either way: fewer than
# n_active particles end up placed, which callers must handle - see
# main_gemc_gpu.jl's use of length(ridx) rather than the target count).
# At low/moderate density (most box lengths) this succeeds on the first
# or second attempt for every particle, so the extra O(placed) check per
# attempt costs little; only the smallest/densest box(es) actually pay
# for retries, exactly mirroring when C's own placement struggles.
#
# type_counts_total: [n_typeA, n_typeB] totals across BOTH boxes (matches
# main_gemc_gpu.cu's argv[16] convention). Each slot's INTENDED type is
# decided up front (first type_counts_box[1] slots are type 0, etc.) so
# the overlap check below can select the real per-sorted-type-pair
# context (ctx_index) for every pairwise insertion energy - matching what
# the (already-fixed) run-time MC/transfer kernels actually compute for
# that pair. An earlier version of this function checked every pair
# against ctx[0] only (matching C's own place_particles_random, which
# hardcodes ff_ctxs[0] the same way it does everywhere else) - that's
# fine for C, whose run-time kernels ALSO only ever use ctx[0] (internally
# consistent, if approximate); but once Julia's run-time was fixed to use
# real per-type-pair physics, ctx[0]-only placement became a genuine
# mismatch: a pair judged "safe" under the AA context can be a severe,
# unprotected overlap once evaluated as its real AB/BB pair (BB's
# excluded-volume fraction of the real production grid is ~6.9% vs AA's
# ~2.6% - not a rare edge case). If placement falls short (placed <
# n_active), later types simply get fewer/no particles (matches C's own
# truncation behavior), rather than shifting the shortfall proportionally
# onto every type.
function random_particles(state::UInt64, n_active::Int, box_len::SVector{3,Float64},
                           type_counts_box::Vector{Int}, grid::GridSetData, gs::GridSet)
    nbin = Int32(grid.nbin)
    pc = PlaceCtx(gs, n_active)   # placed particles, mirrored on the device

    intended_type = Vector{Int32}(undef, n_active)
    idx = 1
    for (t, cnt) in enumerate(type_counts_box)
        for _ in 1:cnt
            idx > n_active && break
            intended_type[idx] = Int32(t - 1)
            idx += 1
        end
    end
    last_t = last_type(length(type_counts_box))
    while idx <= n_active
        intended_type[idx] = last_t
        idx += 1
    end

    pos = Float64[]; bins = Float64[]; ridx = Int32[]; ptype = Int32[]
    placed = 0
    for i in 0:n_active-1
        self_type = intended_type[i+1]
        ok = false
        tp = SVector(0.0, 0.0, 0.0); tb = SVector(0.0, 0.0, 0.0); tridx = Int32(0)
        for attempt in 1:1000
            mp = MVector{3,Float64}(0.0, 0.0, 0.0)
            mb = MVector{3,Float64}(0.0, 0.0, 0.0)
            for d in 1:3
                state, r1 = lcg_next(state); mp[d] = r1 * box_len[d]
                state, r2 = lcg_next(state); mb[d] = r2 * nbin
            end
            tp = SVector(mp); tb = SVector(mb)
            tridx = bins_to_rot_idx(tb, grid.angle_indices, nbin)
            if placed == 0
                ok = true
                break
            end
            ins = insertion_energy(pc, tp, tridx, self_type, box_len)
            if ins < 999.0
                ok = true
                break
            end
        end
        if !ok
            println("Warning: Failed to place particle $i, continuing with $placed")
            break
        end
        push!(pos, tp[1], tp[2], tp[3])
        push!(bins, tb[1], tb[2], tb[3])
        push!(ridx, tridx)
        push!(ptype, self_type)
        push_placed!(pc, tp, tridx, self_type)
        placed += 1
    end

    return pos, bins, ptype, ridx, state
end

# Ports C's sim_gemc_gpu_topup (via place_particles_random_mix) - used at
# stage>0 continuation when a loaded box's particle count fell short of the
# nominal target (dense boxes' stage-0 random placement can legitimately
# fail to fit every particle within 1000 attempts - see random_particles
# above; without this, the shortfall would otherwise persist unchanged for
# every later stage, since Julia's driver had no self-correction at all
# until this was added). Extends pos/bins/ptype/ridx in place with newly
# placed particles: types for the NEW particles are decided up front from
# topup_counts (same intended_type-before-placement approach as
# random_particles), and the overlap check uses the real per-type-pair
# context for both new-new and new-existing pairs - so this does not
# reintroduce the placement-vs-runtime type mismatch bug random_particles
# was fixed for, nor the separate type-assignment-loop clobber bug found
# and fixed in simulation_gemc_gpu.cu's C topup (this function's
# intended_type loop threads its running index through all topup_counts
# types before the last-type catch-all, same as random_particles - it
# never had that bug to begin with).
function topup_particles(state::UInt64, pos::Vector{Float64}, bins::Vector{Float64},
                          ptype::Vector{Int32}, ridx::Vector{Int32},
                          target::Int, box_len::SVector{3,Float64},
                          topup_counts::Vector{Int}, grid::GridSetData, gs::GridSet)
    start = length(ridx)
    start >= target && return pos, bins, ptype, ridx, state

    nbin = Int32(grid.nbin)
    pc = PlaceCtx(gs, target, pos, ridx, ptype)   # existing particles on the device

    n_new = target - start
    intended_type = Vector{Int32}(undef, n_new)
    idx = 1
    for (t, cnt) in enumerate(topup_counts)
        for _ in 1:cnt
            idx > n_new && break
            intended_type[idx] = Int32(t - 1)
            idx += 1
        end
    end
    last_t = last_type(length(topup_counts))
    while idx <= n_new
        intended_type[idx] = last_t
        idx += 1
    end

    placed = start
    for k in 0:n_new-1
        self_type = intended_type[k+1]
        ok = false
        tp = SVector(0.0, 0.0, 0.0); tb = SVector(0.0, 0.0, 0.0); tridx = Int32(0)
        for attempt in 1:1000
            mp = MVector{3,Float64}(0.0, 0.0, 0.0)
            mb = MVector{3,Float64}(0.0, 0.0, 0.0)
            for d in 1:3
                state, r1 = lcg_next(state); mp[d] = r1 * box_len[d]
                state, r2 = lcg_next(state); mb[d] = r2 * nbin
            end
            tp = SVector(mp); tb = SVector(mb)
            tridx = bins_to_rot_idx(tb, grid.angle_indices, nbin)
            ins = insertion_energy(pc, tp, tridx, self_type, box_len)
            if ins < 999.0
                ok = true
                break
            end
        end
        if !ok
            println("Warning: Failed to top up particle $placed, continuing with $placed")
            break
        end
        push!(pos, tp[1], tp[2], tp[3])
        push!(bins, tb[1], tb[2], tb[3])
        push!(ridx, tridx)
        push!(ptype, self_type)
        push_placed!(pc, tp, tridx, self_type)
        placed += 1
    end

    return pos, bins, ptype, ridx, state
end

# Real per-type-pair total energy (all pairs, lower-index-as-p1 convention
# matching sum_pair_energy/C's energy_batch_kernel - see random_particles'
# ordering fix above for why this matters: calculate_pair_energy is not
# symmetric). Used to seed a Triple's e1/e2 with the actual starting
# energy of the placed/loaded configuration, matching what C's own
# compute_total_energy provides (never 0.0) - unlike C, this uses the
# real per-type-pair mixture contexts (ctx_index), consistent with the
# rest of this driver's run-time physics, not ctx[0] only. O(n^2) but
# n<=max_capacity (~few hundred) so this is milliseconds (one device launch), called
# once per triple per stage, never a hot-path cost.
function compute_total_energy(pos::Vector{Float64}, ridx::Vector{Int32}, ptype::Vector{Int32},
                               n::Int, gs::GridSet, box_len::SVector{3,Float64})
    n < 2 && return 0.0
    d_pos = CuArray(pos[1:3n]); d_ridx = CuArray(ridx[1:n]); d_pt = CuArray(ptype[1:n])
    d_out = CUDA.zeros(Float64, n)
    @cuda threads=256 blocks=n total_energy_kernel!(d_out, d_pos, d_ridx, d_pt, Int32(n), box_len, gs)
    return sum(Array(d_out))
end

function mk_phys_box(pos, bins, ptype, ridx, max_cap::Int32, box_dims::SVector{3,Float64},
                      lcg_seed::UInt64, max_trans::Float64, max_rot::Float64,
                      tail_types::Vector{Int32}=Int32[])
    n = length(ridx)
    h_pos = zeros(Float64, Int(max_cap) * 3)
    h_bins = zeros(Float64, Int(max_cap) * 3)
    h_ptype = zeros(Int32, Int(max_cap))
    h_ridx = zeros(Int32, Int(max_cap))
    h_pos[1:n*3] = pos
    h_bins[1:n*3] = bins
    h_ptype[1:n] = ptype
    # pre-assigned types of the deficit slots [n, n+length(tail_types))
    h_ptype[n+1:n+length(tail_types)] = tail_types
    h_ridx[1:n] = ridx
    lcg = CuArray([lcg_seed])
    return PhysBox(CuArray(h_pos), CuArray(h_bins), CuArray(h_ridx),
                   CuArray(h_ptype), lcg, CuArray(zeros(Float64, 6)),
                   box_dims, max_cap, max_trans, max_rot)
end

# ============================================================================
# SAVE / LOAD (production text format - traj.{stage}.{1,2}.txt)
# ============================================================================

# particle_fmap.c's save_to_file: "%16d%16.10f...%8d\n", sequential id
# starting at 1 (not the internal slot index), active slots only. Slots are
# compact, so the active ones are exactly [0,n_active). Returns the number
# of particles written.
function save_particles(path::String, box::PhysBox, n_active::Integer, nbin::Integer,
                        energy::Float64, js::Float64, rs::Float64)
    h_pos = Array(box.pos); h_bins = Array(box.bins)
    h_ptype = Array(box.ptype)
    n_written = 0
    open(path, "w") do io
        # header as simulation_gemc_gpu.cu: the adaptive step sizes are restored from it on load
        @printf(io, "# GPU GEMC mask: %d particles  energy: %.6f  js=%.10g rs=%.10g\n", n_active, energy, js, rs)
        idx = 1
        for slot in 0:Int(n_active)-1
            b = SVector(h_bins[slot*3+1], h_bins[slot*3+2], h_bins[slot*3+3])
            e = bins_to_euler(b, nbin)
            @printf(io, "%16d%16.10f%16.10f%16.10f%16.10f%16.10f%16.10f%8d\n",
                    idx, h_pos[slot*3+1], h_pos[slot*3+2], h_pos[slot*3+3],
                    e[1], e[2], e[3], h_ptype[slot+1])
            idx += 1
            n_written += 1
        end
    end
    return n_written
end

# particle_fmap.c's load_from_line: "%*d %lf %lf %lf %lf %lf %lf %d" - id
# column is ignored on load (re-sequenced into consecutive slots 0..n-1).
function load_particles(path::String, grid::Union{GridData,GridSetData})
    pos = Float64[]; bins = Float64[]; ptype = Int32[]; ridx = Int32[]
    js_rs = nothing
    for line in eachline(path)
        if startswith(line, '#')
            m = match(r"^# GPU GEMC mask: \d+ particles\s+energy: \S+\s+js=(\S+) rs=(\S+)", line)
            if m !== nothing
                js = tryparse(Float64, m.captures[1]); rs = tryparse(Float64, m.captures[2])
                (js !== nothing && rs !== nothing) && (js_rs = (js, rs))
            end
            continue
        end
        fields = split(line)
        length(fields) < 7 && continue
        x, y, z = parse(Float64, fields[2]), parse(Float64, fields[3]), parse(Float64, fields[4])
        ea, eb, ec = parse(Float64, fields[5]), parse(Float64, fields[6]), parse(Float64, fields[7])
        t = length(fields) >= 8 ? parse(Int32, fields[8]) : Int32(0)
        push!(pos, x, y, z)
        b = euler_to_bins(SVector(ea, eb, ec), grid.nbin)
        push!(bins, b[1], b[2], b[3])
        push!(ptype, t)
        push!(ridx, bins_to_rot_idx(b, grid.angle_indices, Int32(grid.nbin)))
    end
    return pos, bins, ptype, ridx, js_rs
end

# ============================================================================
# PER-TRIPLE LOG (matches main_gemc_gpu.cu's log.<stage> format: header,
# "Initial:" line, one "Cycle N:" line per sample window, "Final:" +
# timing + saved-paths at stage end). See gemc_box.jl's Triple comment for
# the one deliberate deviation (E1/E2 track energy change since stage
# start, not C's absolute total_energy).
# ============================================================================

count_type0(ptype::Vector{Int32}) = count(==(Int32(0)), ptype)

function open_triple_log(outdir::AbstractString, idx_stop::Int, N::Int, tr::Triple,
                          box_str::AbstractString, v1_str::AbstractString, label::AbstractString,
                          rpad::Int, seed::UInt64, n_replicas::Int, n_types::Int,
                          L1::Float64, L2::Float64, temperature::Float64,
                          elec_scale::Float64, vdw_scale::Float64, n_cycles::Int, sample_cycles::Int,
                          transfer_freq::Float64, target_ratio::Float64, adjust_cycles::Int)
    io = open("$outdir/log.$idx_stop", "w")
    n1_ini = Int(tr.n_active1); n2_ini = Int(tr.n_active2)
    @printf(io, "GPU Monte Carlo (n_types=%d)  L=%s V=%s R%0*d/%d\n",
            n_types, box_str, v1_str, rpad, seed, n_replicas)
    @printf(io, "N_total=%d  n1=%d n2=%d  L1=%.3f L2=%.3f  T=%.3f  elec_scale=%.4f  vdw_scale=%.4f  n_cycles=%d  sample=%d  transfer_freq=%.2f  target_ratio=%.2f  adjust_cycles=%d\n",
            N, n1_ini, n2_ini, L1, L2, temperature, elec_scale, vdw_scale,
            n_cycles, sample_cycles, transfer_freq, target_ratio, adjust_cycles)
    @printf(io, "Initial: N1=%d(%d) N2=%d(%d)  E1=%.2f E2=%.2f\n",
            n1_ini, tr.nta1, n2_ini, tr.nta2, tr.e1, tr.e2)
    print(io, "----------------------------------------\n")
    flush(io)
    return io
end

# main_gemc_gpu.cu's write_conf_frame_gpu: one snapshot frame, "# cycle=.. n=..
# energy=.." header then "id x y z e1 e2 e3 type" (%.6f) per active particle.
function write_conf_frame(io::IO, box::PhysBox, n_active::Integer, cycle::Integer, energy::Float64, nbin::Integer)
    h_pos = Array(box.pos); h_bins = Array(box.bins); h_ptype = Array(box.ptype)
    @printf(io, "# cycle=%d n=%d energy=%.6f\n", cycle, n_active, energy)
    for slot in 0:Int(n_active)-1
        b = SVector(h_bins[slot*3+1], h_bins[slot*3+2], h_bins[slot*3+3])
        e = bins_to_euler(b, nbin)
        @printf(io, "%d %.6f %.6f %.6f %.6f %.6f %.6f %d\n", slot + 1,
                h_pos[slot*3+1], h_pos[slot*3+2], h_pos[slot*3+3], e[1], e[2], e[3], h_ptype[slot+1])
    end
    flush(io)
end

function log_cycle!(io::IO, cycles_done::Int, tr::Triple, cps::Float64)
    n1 = Int(tr.n_active1); n2 = Int(tr.n_active2)
    ar1 = tr.n_moves1 > 0 ? tr.n_accepted1 / tr.n_moves1 : 0.0
    ar2 = tr.n_moves2 > 0 ? tr.n_accepted2 / tr.n_moves2 : 0.0
    tf_rate = tr.n_transfer_attempts > 0 ? tr.n_transfer_accepted / tr.n_transfer_attempts : 0.0
    @printf(io, "Cycle %8d: N1=%4d(%d) N2=%4d(%d)  E1=%10.2f E2=%10.2f  A1=%.3f A2=%.3f  TF=%.3f  speed=%.0f/s  js1=%.4g rs1=%.4g js2=%.4g rs2=%.4g\n",
            cycles_done, n1, tr.nta1, n2, tr.nta2, tr.e1, tr.e2, ar1, ar2, tf_rate, cps,
            tr.tune1.js, tr.tune1.rs, tr.tune2.js, tr.tune2.rs)
    flush(io)
end

function log_final!(io::IO, tr::Triple, total_time::Float64, n_cycles::Int,
                     traj1::String, traj2::String)
    n1 = Int(tr.n_active1); n2 = Int(tr.n_active2)
    ar1 = tr.n_moves1 > 0 ? tr.n_accepted1 / tr.n_moves1 : 0.0
    ar2 = tr.n_moves2 > 0 ? tr.n_accepted2 / tr.n_moves2 : 0.0
    tf_rate = tr.n_transfer_attempts > 0 ? tr.n_transfer_accepted / tr.n_transfer_attempts : 0.0
    print(io, "----------------------------------------\n")
    @printf(io, "Final: N1=%d(%d) N2=%d(%d)  E1=%.2f E2=%.2f  A1=%.3f A2=%.3f  TF=%.3f\n",
            n1, tr.nta1, n2, tr.nta2, tr.e1, tr.e2, ar1, ar2, tf_rate)
    @printf(io, "Time: %.2f s  Speed: %.0f cycles/s\n",
            total_time, total_time > 0.0 ? n_cycles / total_time : 0.0)
    @printf(io, "Saved: %s  %s\n", traj1, traj2)
    close(io)
end
