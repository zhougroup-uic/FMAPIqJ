# main_nvt_gpu.jl - NVT Monte Carlo of protein solutions on the GPU (CUDA.jl).
#
# One process runs all independent replicas ("triples") at once: every replica has its own box, RNG
# stream (seed + replica) and step-size adaptation; the displacement and insertion kernels of all
# replicas are launched together and synchronized once per sub-batch. A step is a displacement
# (translation or rotation) or, with probability transfer_freq, a same-box insertion (a particle is
# removed and re-inserted at a random position). Energies come from the FMAP grids (fmap.paths).
#
# Usage:
#   julia --project=<root> bin/simulation_nvt_gpu.jl N box_list TE_list n_cycles seed start stop \
#         sample_cycles fmap.paths [v1] [max_trans] [max_rot] [zratio] [transfer_freq] [target_ratio] \
#         [n_types] [type_counts] [ns_freq] [conf_freq] [adjust_cycles]
#   TE_list entries: T[:elec_scale[:vdw_scale[:label]]] (scales default 1.0); one replica per entry.
#   Defaults: max_trans/max_rot/zratio 1, transfer_freq 0.1, target_ratio 0.3, n_types 1,
#   ns_freq/conf_freq 0 (disabled), adjust_cycles 10 (0 = frozen step sizes). v1 is ignored (one box).
# Requires TERN_BYTES to match the grid (1 = summed channel, the pipeline default; 2 = [elec, vdw]).
using CUDA
using Printf
include("kernels.jl")
include("gemc_box.jl")
include("fmap_zlib_loader.jl")   # defines load_fmap_zlib/load_fmap_paths -
                                  # after gemc_box.jl since they construct
                                  # GridData/GridSetData (defined there)

function kbt_to_celsius(kbt::Float64)
    N_A = 6.022045000e23; k_B = 3.297623030e-24
    kelvin = kbt * 1000.0 / (N_A * k_B)
    return kelvin - 273.15
end

function parse_te_list(s::AbstractString)
    temps = Float64[]; escls = Float64[]; vscls = Float64[]; labels = String[]
    for tok in split(s, ",")
        parts = split(tok, ":")
        t = parse(Float64, parts[1])
        push!(temps, t)
        push!(escls, length(parts) >= 2 && !isempty(parts[2]) ? parse(Float64, parts[2]) : 1.0)
        push!(vscls, length(parts) >= 3 && !isempty(parts[3]) ? parse(Float64, parts[3]) : 1.0)
        push!(labels, length(parts) >= 4 ? String(parts[4]) : @sprintf("%.1f", kbt_to_celsius(t)))
    end
    return temps, escls, vscls, labels
end

function main()
    N          = parse(Int, ARGS[1])
    box_strs   = split(ARGS[2], ",")
    box_list   = [parse(Float64, x) for x in box_strs]
    temps, escls, vscls, labels = parse_te_list(ARGS[3])
    if TERN_BYTES == 1 && (any(!=(1.0), escls) || any(!=(1.0), vscls))
        @warn "elec_scale/vdw_scale requested but NOT applied: 1-byte tErn build (TERN_BYTES=1) has no separate channels; running as if both are 1.0"
    end
    n_cycles      = parse(Int, ARGS[4])
    seed          = parse(UInt64, ARGS[5])
    start_arg     = ARGS[6]
    stop_arg      = ARGS[7]
    sample_cycles = parse(Int, ARGS[8])
    fmap_paths    = ARGS[9]
    argf(i, d) = length(ARGS) >= i ? parse(Float64, ARGS[i]) : d
    v1            = argf(10, 0.5)
    max_trans     = argf(11, 1.0)
    max_rot       = argf(12, 1.0)
    zratio        = argf(13, 1.0)
    transfer_freq = clamp(argf(14, 0.1), 0.0, 1.0)
    target_ratio  = clamp(argf(15, 0.3), 0.0, 1.0)
    n_types       = max(1, length(ARGS) >= 16 ? parse(Int, ARGS[16]) : 1)
    N_TYPES[] = n_types
    type_counts_total = length(ARGS) >= 17 && !isempty(ARGS[17]) ? [parse(Int, x) for x in split(ARGS[17], ",")] : [N]
    ns_freq_cycles   = length(ARGS) >= 18 ? parse(Int, ARGS[18]) : 0
    conf_freq_cycles = length(ARGS) >= 19 ? parse(Int, ARGS[19]) : 0
    adjust_cycles    = length(ARGS) >= 20 ? max(parse(Int, ARGS[20]), 0) : 10   # 0 = step sizes frozen
    chunk_cycles     = adjust_cycles > 0 ? adjust_cycles : 10   # sub-batch length (same when frozen)
    v1 = 1.0   # NVT: everything in box 1

    n_boxes    = length(box_list)
    n_replicas = length(temps)
    total_pairs = n_boxes * n_replicas

    idx_stop  = parse(Int, replace(stop_arg, "traj." => ""))
    use_traj  = start_arg != "-"
    idx_start = use_traj ? parse(Int, replace(start_arg, "traj." => "")) : 0

    rpad = seed < 10 ? 1 : seed < 100 ? 2 : seed < 1000 ? 3 : 4
    rdir = "R" * lpad(string(seed), rpad, '0')

    println("Loading grid contexts from $fmap_paths (n_types=$n_types) ..."); flush(stdout)
    # Energy grids are inflated in chunks straight into device memory (like
    # the C GPU driver) - no uncompressed host copy is ever built.
    gridset = load_fmap_paths_gpu(fmap_paths, Int32(n_types))
    n_ctx = length(gridset.energy_ctxs)
    d_energy_ctxs = Tuple(gridset.energy_ctxs)
    d_aidx   = CuArray(gridset.angle_indices)
    d_amap   = CuArray(gridset.angle_mapping)
    d_rotmat = CuArray(gridset.rotation_matrices)
    # CuArrays are implicitly synchronized by default whenever they're
    # accessed from a different CUDA stream than they were last used on
    # (CUDA.jl's array.jl docstring for enable_synchronization!) - with
    # every one of the triples' streams reading these SAME shared grid
    # arrays, that turns into a full serialization chain across all
    # triples (empirically confirmed: 119 triples went from 38s down to
    # 0.006s after disabling this here - see mixbench/concurrencyProbe
    # notes). Safe to disable: these arrays are uploaded once and never
    # written to again for the rest of the run, so there is no actual
    # race for the safety synchronization to protect against.
    CUDA.enable_synchronization!.(d_energy_ctxs, false)
    CUDA.enable_synchronization!(d_aidx, false)
    CUDA.enable_synchronization!(d_amap, false)
    CUDA.enable_synchronization!(d_rotmat, false)
    println("Grid loaded: nrot=$(gridset.nrot) nbin=$(gridset.nbin) n_ctx=$n_ctx blens=$(gridset.blens)"); flush(stdout)
    grid_views = [mk_grid_set(gridset, d_energy_ctxs, d_aidx, d_amap, d_rotmat, escls[r], vscls[r]) for r in 1:n_replicas]

    # type_counts_total is totals across BOTH boxes (matches
    # main_gemc_gpu.cu's argv[16] convention) - split proportionally by v1,
    # clamped so the per-type box1+box2 split always sums to the total
    # (last type absorbs any rounding remainder, same spirit as
    # production's topup clamp, simplified since we place exactly rather
    # than top up after a possibly-short random placement).
    n1_ini = round(Int, N * v1)
    n2_ini = N - n1_ini
    tc1 = Int[]; tc2 = Int[]
    remaining1 = n1_ini
    for (k, tc) in enumerate(type_counts_total)
        t1 = k == length(type_counts_total) ? remaining1 : min(round(Int, tc * v1), remaining1)
        push!(tc1, t1); push!(tc2, tc - t1)
        remaining1 -= t1
    end

    println("Setting up $total_pairs triples (N=$N, $n_boxes boxes x $n_replicas replicas)..."); flush(stdout)
    triples = Vector{Triple}(undef, total_pairs)
    replica_of = Vector{Int}(undef, total_pairs)
    outdirs = Vector{String}(undef, total_pairs)
    trajpaths1 = Vector{String}(undef, total_pairs)
    trajpaths2 = Vector{String}(undef, total_pairs)
    logios = Vector{IO}(undef, total_pairs)
    ns_ios    = Vector{Union{IO,Nothing}}(nothing, total_pairs)
    conf_ios1 = Vector{Union{IO,Nothing}}(nothing, total_pairs)
    conf_ios2 = Vector{Union{IO,Nothing}}(nothing, total_pairs)
    next_ns   = fill(ns_freq_cycles, total_pairs)
    next_conf = fill(conf_freq_cycles, total_pairs)

    for b in 1:n_boxes
        L = box_list[b]
        L1 = L
        L2 = L
        boxdim1 = SVector(L1, L1, L1 * zratio)
        boxdim2 = SVector(L2, L2, L2 * zratio)
        for r in 1:n_replicas
            fi = (b - 1) * n_replicas + r
            replica_of[fi] = r
            # Per-box LCG streams, as in main_gemc_gpu.cu (unsigned int arithmetic): box 1 is
            # seeded `seed`, box 2 `seed + 12345`, whatever L, T or V1 - so R<seed> does not
            # depend on which other triples share the process. The host stream does the
            # placement; the device stream continues from where it stopped.
            # NVT driver: replica r (entry r of TE_list) is seeded seed + r - 1, so a TE_list of identical
            # temperatures with different labels is a set of independent replicas in one process.
            state1 = UInt64((seed + UInt64(r - 1)) % UInt32)
            state2 = UInt64((seed + UInt64(r - 1) + UInt64(12345)) % UInt32)

            basedir = "T$(labels[r])/N$(N)L$(box_strs[b])"
            outdir  = "$basedir/V$v1/$rdir"
            mkpath(outdir)
            outdirs[fi] = outdir
            trajpaths1[fi] = "$outdir/traj.$(idx_stop).1.txt"
            trajpaths2[fi] = "$outdir/traj.$(idx_stop).2.txt"

            if !use_traj
                pos1, bins1, ptype1, ridx1, state1 = random_particles(state1, n1_ini, boxdim1, tc1, gridset, grid_views[r])
                pos2, bins2, ptype2, ridx2, state2 = random_particles(state2, n2_ini, boxdim2, tc2, gridset, grid_views[r])
                # Placement may stop short (dense box): the unplaced slots
                # keep their intended type so reservoir insertion fills the
                # right species later (C: initialize_random's type commit).
                committed1 = n1_ini; committed2 = n2_ini
                tail1 = intended_types(tc1, n1_ini)[length(ridx1)+1:end]
                tail2 = intended_types(tc2, n2_ini)[length(ridx2)+1:end]
            else
                pos1, bins1, ptype1, ridx1, jsrs1 = load_particles("$outdir/traj.$(idx_start).1.txt", gridset)
                pos2, bins2, ptype2, ridx2, jsrs2 = load_particles("$outdir/traj.$(idx_start).2.txt", gridset)

                # Top up to N if the loaded configuration is short (e.g. a
                # crowded box's random placement fell short at an earlier
                # stage) - mirrors main_gemc_gpu.cu's own topup clamp:
                # split the deficit by distance from the nominal n1_ini/
                # n2_ini target, never topping up past N even if one box
                # already exceeds its nominal share.
                n1c = length(ridx1); n2c = length(ridx2)
                committed1 = n1c; committed2 = n2c   # raised below if topped up
                tail1 = Int32[]; tail2 = Int32[]
                deficit = N - (n1c + n2c)
                if deficit > 0
                    extra1 = clamp(n1_ini - n1c, 0, deficit)
                    extra2 = deficit - extra1
                    if extra1 > 0 || extra2 > 0
                        # Each type's remaining deficit is laid out along one virtual queue
                        # (extra_0 slots of type 0, then extra_1 of type 1, ...) and split at
                        # extra1 between the boxes (main_gemc_gpu.cu), so the per-box totals equal
                        # extra1/extra2 and the per-type totals equal extra_k exactly.
                        topup1 = zeros(Int, length(type_counts_total)); topup2 = zeros(Int, length(type_counts_total))
                        qpos = 0
                        for (k, tc) in enumerate(type_counts_total)
                            loaded_k = count(==(Int32(k - 1)), ptype1) + count(==(Int32(k - 1)), ptype2)
                            extra_k  = max(tc - loaded_k, 0)
                            qend = qpos + extra_k
                            hi1 = min(qend, extra1); hi1 > qpos && (topup1[k] += hi1 - qpos)
                            lo2 = max(qpos, extra1); hi2 = min(qend, extra1 + extra2); hi2 > lo2 && (topup2[k] += hi2 - lo2)
                            qpos = qend
                        end
                        println("[L=$(box_strs[b]) V=$v1 $rdir] Top-up box1+$extra1 box2+$extra2"); flush(stdout)
                        if extra1 > 0
                            pos1, bins1, ptype1, ridx1, state1 = topup_particles(state1, pos1, bins1, ptype1, ridx1,
                                                                                 n1c + extra1, boxdim1, topup1, gridset, grid_views[r])
                            committed1 = n1c + extra1
                            tail1 = intended_types(topup1, extra1)[length(ridx1)-n1c+1:end]
                        end
                        if extra2 > 0
                            pos2, bins2, ptype2, ridx2, state2 = topup_particles(state2, pos2, bins2, ptype2, ridx2,
                                                                                 n2c + extra2, boxdim2, topup2, gridset, grid_views[r])
                            committed2 = n2c + extra2
                            tail2 = intended_types(topup2, extra2)[length(ridx2)-n2c+1:end]
                        end
                    end
                end
            end

            box1 = mk_phys_box(pos1, bins1, ptype1, ridx1, Int32(N), boxdim1, state1, max_trans, max_rot, tail1)
            box2 = mk_phys_box(pos2, bins2, ptype2, ridx2, Int32(1), boxdim2, state2, max_trans, max_rot, tail2)
            e1_init = compute_total_energy(pos1, ridx1, ptype1, length(ridx1), grid_views[r], boxdim1)
            e2_init = compute_total_energy(pos2, ridx2, ptype2, length(ridx2), grid_views[r], boxdim2)
            tr = Triple(box1, box2, box1.lcg, CuArray(zeros(Float64, 8)),
                        Int32(length(ridx1)), Int32(length(ridx2)),
                        CuStream(),
                        e1_init, e2_init, 0, 0, 0, 0, 0, 0,
                        Int32(count_type0(ptype1)), Int32(count_type0(ptype2)),
                        StepTune(max_trans * 1.62, max_rot, boxdim1[1], Float64(gridset.nbin) / ROT_DEFAULT),
                        StepTune(max_trans * 1.62, max_rot, boxdim2[1], Float64(gridset.nbin) / ROT_DEFAULT),
                        Int32(committed1), Int32(committed2),
                        max(0, N - (length(ridx1) + length(ridx2))),
                        CuArray(zeros(Float64, 4)), CuArray(zeros(Float64, 4)))
            # restore the saved adaptive step sizes (traj header), as the C engine does on load
            if use_traj
                jsrs1 === nothing || ((tr.tune1.js, tr.tune1.rs) = jsrs1)
                jsrs2 === nothing || ((tr.tune2.js, tr.tune2.rs) = jsrs2)
            end
            triples[fi] = tr
            logios[fi] = open_triple_log(outdir, idx_stop, N, tr, box_strs[b], string(v1), labels[r],
                                          rpad, seed, n_replicas, n_types, L1, L2, temps[r], escls[r], vscls[r],
                                          n_cycles, sample_cycles, transfer_freq,
                                          target_ratio, adjust_cycles)
            if tr.deficit > 0
                println("[L=$(box_strs[b]) V=$v1 $rdir] reservoir_deficit=$(tr.deficit)"); flush(stdout)
                println(logios[fi], "reservoir_deficit=$(tr.deficit)"); flush(logios[fi])
            end
            if ns_freq_cycles > 0
                ns_ios[fi] = open("$outdir/Ns.$(idx_stop).txt", "w")
            end
            if conf_freq_cycles > 0
                conf_ios1[fi] = open("$outdir/conf.$(idx_stop).1", "w")
                conf_ios2[fi] = open("$outdir/conf.$(idx_stop).2", "w")
            end
        end
    end

    n_total_steps    = n_cycles * N
    sample_steps     = sample_cycles * N

    println("Running $n_cycles cycles, sample every $sample_cycles cycles..."); flush(stdout)
    t0 = time()
    done = 0
    adjust_chunk_steps = chunk_cycles * N
    while done < n_total_steps
        batch = min(sample_steps, n_total_steps - done)

        # Walk the sample window in adjust_cycles-cycle sub-batches and
        # adapt js/rs after each one (as the C GPU driver does), instead of
        # once per (often much coarser) sample window.
        sub_done = 0
        while sub_done < batch
            chunk = min(adjust_chunk_steps, batch - sub_done)
            sub_done += chunk
            disp_per_box = round(Int, chunk * (1.0 - transfer_freq) * 0.5)
            n_transfers  = chunk - 2 * disp_per_box

            # Launch every triple's displacement kernel on its OWN stream
            # before syncing any of them - this is what lets the GPU
            # actually run many triples concurrently (up to CUDA CC8.9's
            # 128-resident-grid limit) instead of
            # serializing on the default stream.
            for (fi, tr) in enumerate(triples)
                gv = grid_views[replica_of[fi]]
                temp = temps[replica_of[fi]]
                mc_batch_launch!(as_boxview(tr.box1), as_boxview(tr.box2), gv, disp_per_box, temp,
                                  tr.n_active1, tr.n_active2,
                                  tr.tune1.js, tr.tune1.rs, tr.tune2.js, tr.tune2.rs;
                                  stream=tr.stream)
            end
            CUDA.synchronize()

            # Triples still short of N (deficit > 0) skip the transfer
            # kernel this sub-batch and spend its transfer budget on
            # reservoir insertions instead (C driver: same split). The
            # window's budget is halved between the boxes, each box's share
            # of the remaining deficit is clamped to its own committed room,
            # and what one box can't use goes to its partner.
            did_transfer = falses(length(triples))
            did_res1 = falses(length(triples)); did_res2 = falses(length(triples))
            for (fi, tr) in enumerate(triples)
                gv = grid_views[replica_of[fi]]
                temp = temps[replica_of[fi]]
                if tr.deficit == 0
                    did_transfer[fi] = true
                    nvt_insert_launch!(as_transferboxview(tr.box1), gv, n_transfers, temp,
                                        tr.transfer_lcg, tr.transfer_result, tr.n_active1;
                                        stream=tr.stream)
                else
                    half  = max(n_transfers ÷ 2, 1)
                    want1 = (tr.deficit + 1) ÷ 2
                    want2 = tr.deficit - want1
                    room1 = max(0, Int(tr.committed1) - Int(tr.n_active1))
                    room2 = max(0, Int(tr.committed2) - Int(tr.n_active2))
                    if want1 > room1; want2 += want1 - room1; want1 = room1; end
                    if want2 > room2; want1 += want2 - room2; want2 = room2; end
                    want1 = min(want1, room1)
                    if want1 > 0
                        did_res1[fi] = true
                        reservoir_insert_launch!(as_transferboxview(tr.box1), gv, half, want1, temp,
                                                  tr.box1.lcg, tr.res_result1,
                                                  tr.n_active1, tr.committed1; stream=tr.stream)
                    end
                    if want2 > 0
                        did_res2[fi] = true
                        reservoir_insert_launch!(as_transferboxview(tr.box2), gv, half, want2, temp,
                                                  tr.box2.lcg, tr.res_result2,
                                                  tr.n_active2, tr.committed2; stream=tr.stream)
                    end
                end
            end
            CUDA.synchronize()

            for (fi, tr) in enumerate(triples)
                # [delta, n_accepted, try_t, acc_t, try_r, acc_r] this sub-batch
                hmc1 = Array(tr.box1.mc_result)
                hmc2 = Array(tr.box2.mc_result)
                tr.e1 += hmc1[1]; tr.n_accepted1 += round(Int, hmc1[2]); tr.n_moves1 += disp_per_box
                tr.e2 += hmc2[1]; tr.n_accepted2 += round(Int, hmc2[2]); tr.n_moves2 += disp_per_box
                for (t, h) in ((tr.tune1, hmc1), (tr.tune2, hmc2))
                    t.ntry  += round(Int, h[3]); t.nacc  += round(Int, h[4])
                    t.nrtry += round(Int, h[5]); t.nracc += round(Int, h[6])
                end

                if did_transfer[fi]
                    h = Array(tr.transfer_result)
                    tr.e1 += h[1]; tr.e2 += h[2]
                    tr.n_transfer_attempts += round(Int, h[3]); tr.n_transfer_accepted += round(Int, h[4])
                    # n_active is unchanged by a same-box insertion
                end
                # reservoir result = [delta_e, n_accepted, n_active, type0_delta]
                acc = 0
                if did_res1[fi]
                    hr = Array(tr.res_result1)
                    tr.e1 += hr[1]; acc += round(Int, hr[2])
                    tr.n_active1 = Int32(round(Int, hr[3])); tr.nta1 += round(Int32, hr[4])
                end
                if did_res2[fi]
                    hr = Array(tr.res_result2)
                    tr.e2 += hr[1]; acc += round(Int, hr[2])
                    tr.n_active2 = Int32(round(Int, hr[3])); tr.nta2 += round(Int32, hr[4])
                end
                if acc > 0
                    tr.deficit = max(0, tr.deficit - acc)
                    if tr.deficit == 0
                        println("[triple $fi] reservoir deficit closed"); flush(stdout)
                    end
                end

                if adjust_cycles > 0
                    adjust_step_sizes!(tr.tune1, target_ratio)
                    adjust_step_sizes!(tr.tune2, target_ratio)
                end
            end

            # Ns rows: one every ns_freq_cycles cycles, checked at every sub-batch boundary
            # (chunk_cycles apart), so Ns.<stage>.txt is not limited to the sample window.
            cyc_sub = div(done + sub_done, N)
            for fi in eachindex(triples)
                io = ns_ios[fi]
                io === nothing && continue
                tr = triples[fi]
                while cyc_sub >= next_ns[fi]
                    # C: "N1 N2" for n_types==1, "N1 N2 N1_0 N2_0" otherwise
                    if n_types > 1
                        println(io, tr.n_active1, " ", tr.n_active2, " ", tr.nta1, " ", tr.nta2)
                    else
                        println(io, tr.n_active1, " ", tr.n_active2)
                    end
                    next_ns[fi] += ns_freq_cycles
                end
            end
        end

        done += batch
        elapsed = time() - t0
        cycles_done = div(done, N)
        cps = elapsed > 0 ? cycles_done / elapsed : 0.0

        for (fi, tr) in enumerate(triples)
            log_cycle!(logios[fi], cycles_done, tr, cps)

            # Ns rows were written at the sub-batch boundaries above; conf_freq is quantized to
            # sample_cycles granularity (threshold-based, so a non-multiple ratio still fires at
            # the next window).
            ns_ios[fi] === nothing || flush(ns_ios[fi])
            if conf_ios1[fi] !== nothing && cycles_done >= next_conf[fi]
                write_conf_frame(conf_ios1[fi], tr.box1, tr.n_active1, cycles_done, tr.e1, gridset.nbin)
                write_conf_frame(conf_ios2[fi], tr.box2, tr.n_active2, cycles_done, tr.e2, gridset.nbin)
                next_conf[fi] += conf_freq_cycles
            end
        end

        @printf("Cycle %8d: speed=%.0f/s (%d/%d triples' worth of steps done)\n",
                cycles_done, cps, done, n_total_steps)
        flush(stdout)
    end

    println("Saving final configurations...")
    total_time = time() - t0
    for b in 1:n_boxes, r in 1:n_replicas
        fi = (b - 1) * n_replicas + r
        tr = triples[fi]
        n1 = save_particles(trajpaths1[fi], tr.box1, tr.n_active1, gridset.nbin, tr.e1, tr.tune1.js, tr.tune1.rs)
        n2 = save_particles(trajpaths2[fi], tr.box2, tr.n_active2, gridset.nbin, tr.e2, tr.tune2.js, tr.tune2.rs)
        log_final!(logios[fi], tr, total_time, n_cycles, trajpaths1[fi], trajpaths2[fi])
        for io in (ns_ios[fi], conf_ios1[fi], conf_ios2[fi])
            io === nothing || close(io)
        end
        println("[L=$(box_strs[b]) T=$(labels[r])] saved N1=$n1 N2=$n2")
    end
    println("Done.")
end

main()
