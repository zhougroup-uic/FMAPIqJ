# main_nvt_gpu.jl - NVT Monte Carlo of protein solutions on the GPU (native/libfmapsim.so).
#
# One process runs all independent replicas ("triples") at once: every replica has its own box, RNG
# stream (seed + replica) and step-size adaptation; the displacement and insertion kernels of all
# replicas are launched together and synchronized once per sub-batch. A step is a displacement
# (translation or rotation) or, with probability transfer_freq, a same-box insertion (a particle is
# removed and re-inserted at a random position). Energies come from the FMAP grids (fmap.paths).
#
# GPU numerics: native/libfmapsim.so (CUDA C, no CUDA.jl); this file is the run logic.
#
# Usage:
#   julia --project=<root> bin/simulation_nvt_gpu.jl N box_list TE_list n_cycles seed start stop \
#         sample_cycles fmap.paths [v1] [max_trans] [max_rot] [zratio] [transfer_freq] [target_ratio] \
#         [n_types] [type_counts] [ns_freq] [conf_freq] [adjust_cycles]
#   TE_list entries: T[:elec_scale[:vdw_scale[:label]]] (scales default 1.0); one replica per entry.
#   Defaults: max_trans/max_rot/zratio 1, transfer_freq 0.1, target_ratio 0.3, n_types 1,
#   ns_freq/conf_freq 0 (disabled), adjust_cycles 10 (0 = frozen step sizes). v1 is ignored (one box).
# Requires TERN_BYTES to match the grid (1 = summed channel, the pipeline default; 2 = [elec, vdw]).
using Printf
include("../rotsys.jl")           # rotation tables from rot.eul (plain pair table), Rotij
include("nvt_native.jl")          # libfmapsim.so: tables, boxes and moves on the GPU, host-side helpers

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
    gridset = load_fmap_paths_native(fmap_paths, n_types)
    n_ctx = n_types * (n_types + 1) ÷ 2
    println("Grid loaded: nrot=$(gridset.nrot) nbin=$(gridset.nbin) n_ctx=$n_ctx blens=$(gridset.blens)"); flush(stdout)

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
        boxdim1 = (L1, L1, L1 * zratio)
        boxdim2 = (L2, L2, L2 * zratio)
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
                pos1, bins1, ptype1, ridx1, state1 = random_particles(gridset, escls[r], vscls[r], state1, n1_ini, boxdim1, tc1)
                pos2, bins2, ptype2, ridx2, state2 = random_particles(gridset, escls[r], vscls[r], state2, n2_ini, boxdim2, tc2)
                # Placement may stop short (dense box): the unplaced slots
                # keep their intended type so reservoir insertion fills the
                # right species later (C: initialize_random's type commit).
                committed1 = n1_ini; committed2 = n2_ini
                tail1 = intended_types(tc1, n1_ini)[length(ridx1)+1:end]
                tail2 = intended_types(tc2, n2_ini)[length(ridx2)+1:end]
            else
                pos1, bins1, ptype1, ridx1, jsrs1 = load_particles("$outdir/traj.$(idx_start).1.txt", gridset)
                f2 = "$outdir/traj.$(idx_start).2.txt"                # box 2 is empty in the NVT driver; its file may be absent
                pos2, bins2, ptype2, ridx2, jsrs2 = isfile(f2) ? load_particles(f2, gridset) : (Float64[], Float64[], Int32[], Int32[], nothing)

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
                            pos1, bins1, ptype1, ridx1, state1 = topup_particles(gridset, escls[r], vscls[r], state1, pos1, bins1, ptype1, ridx1,
                                                                                 n1c + extra1, boxdim1, topup1)
                            committed1 = n1c + extra1
                            tail1 = intended_types(topup1, extra1)[length(ridx1)-n1c+1:end]
                        end
                        if extra2 > 0
                            pos2, bins2, ptype2, ridx2, state2 = topup_particles(gridset, escls[r], vscls[r], state2, pos2, bins2, ptype2, ridx2,
                                                                                 n2c + extra2, boxdim2, topup2)
                            committed2 = n2c + extra2
                            tail2 = intended_types(topup2, extra2)[length(ridx2)-n2c+1:end]
                        end
                    end
                end
            end

            box1 = NBox(gridset, N, boxdim1, temps[r], escls[r], vscls[r])
            box_set!(box1, pos1, bins1, [ptype1; tail1], ridx1, state1)
            e1_init = total_energy(box1, length(ridx1))
            e2_init = 0.0          # box 2 never holds a molecule in the NVT driver
            tr = Triple(box1, Int32(length(ridx1)), Int32(length(ridx2)),
                        e1_init, e2_init, 0, 0, 0, 0, 0, 0,
                        Int32(count_type0(ptype1)), Int32(count_type0(ptype2)),
                        StepTune(max_trans * 1.62, max_rot, boxdim1[1], Float64(gridset.nbin) / ROT_DEFAULT),
                        StepTune(max_trans * 1.62, max_rot, boxdim2[1], Float64(gridset.nbin) / ROT_DEFAULT),
                        Int32(committed1), Int32(committed2),
                        max(0, N - (length(ridx1) + length(ridx2))))
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

            # Launch every replica's displacement kernel (each box has its own stream) before waiting for any of them, so the
            # GPU runs the replicas concurrently.
            for tr in triples
                mc_launch!(tr.box1, tr.n_active1, disp_per_box, tr.tune1.js, tr.tune1.rs)
            end
            hmcs = [wait_box(tr.box1; mc = true)[1] for tr in triples]      # [delta, n_accepted, try_t, acc_t, try_r, acc_r]

            # Replicas still short of N (deficit > 0) skip the insertion move this sub-batch and spend its budget on reservoir
            # insertions instead (C driver: same split). Box 2 is always empty in the NVT driver (no committed room), so the whole
            # deficit goes to box 1.
            did_transfer = falses(length(triples))
            did_res1 = falses(length(triples))
            for (fi, tr) in enumerate(triples)
                if tr.deficit == 0
                    did_transfer[fi] = true
                    insert_launch!(tr.box1, tr.n_active1, n_transfers)
                else
                    half  = max(n_transfers ÷ 2, 1)
                    want1 = min(tr.deficit, max(0, Int(tr.committed1) - Int(tr.n_active1)))
                    if want1 > 0
                        did_res1[fi] = true
                        reservoir_launch!(tr.box1, tr.n_active1, tr.committed1, half, want1)
                    end
                end
            end

            for (fi, tr) in enumerate(triples)
                hmc1 = hmcs[fi]
                _, hins, hres = wait_box(tr.box1; ins = did_transfer[fi], res = did_res1[fi])
                tr.e1 += hmc1[1]; tr.n_accepted1 += round(Int, hmc1[2]); tr.n_moves1 += disp_per_box
                tr.n_moves2 += disp_per_box
                t = tr.tune1
                t.ntry  += round(Int, hmc1[3]); t.nacc  += round(Int, hmc1[4])
                t.nrtry += round(Int, hmc1[5]); t.nracc += round(Int, hmc1[6])

                if did_transfer[fi]
                    tr.e1 += hins[1]; tr.e2 += hins[2]
                    tr.n_transfer_attempts += round(Int, hins[3]); tr.n_transfer_accepted += round(Int, hins[4])
                    # n_active is unchanged by a same-box insertion
                end
                # reservoir result = [delta_e, n_accepted, n_active, type0_delta]
                acc = 0
                if did_res1[fi]
                    tr.e1 += hres[1]; acc += round(Int, hres[2])
                    tr.n_active1 = Int32(round(Int, hres[3])); tr.nta1 += round(Int32, hres[4])
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
                println(conf_ios2[fi], "# cycle=$cycles_done n=0 energy=", @sprintf("%.6f", tr.e2)); flush(conf_ios2[fi])
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
        pos, bins, _, ptype = box_get(tr.box1, tr.n_active1)
        n1 = save_particles(trajpaths1[fi], pos, bins, ptype, tr.n_active1, gridset.nbin, tr.e1, tr.tune1.js, tr.tune1.rs)
        n2 = save_particles(trajpaths2[fi], Float64[], Float64[], Int32[], 0, gridset.nbin, tr.e2, tr.tune2.js, tr.tune2.rs)
        log_final!(logios[fi], tr, total_time, n_cycles, trajpaths1[fi], trajpaths2[fi])
        for io in (ns_ios[fi], conf_ios1[fi], conf_ios2[fi])
            io === nothing || close(io)
        end
        println("[L=$(box_strs[b]) T=$(labels[r])] saved N1=$n1 N2=$n2")
        close_box(tr.box1)
    end
    println("Done.")
end

main()
