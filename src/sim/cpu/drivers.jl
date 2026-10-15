# drivers.jl - command line of simulation_nvt (single box, same-box insertion) on the CPU engine.
# Same arguments, files and log lines as the maeppi C driver (reference/maeppi).

# ---- shared helpers ----

"Stage number from the trailing '.<N>.' segment of a name (extract_stage_index / extract_index)."
function extract_stage_index(filename::AbstractString)
    result = 0
    for m in eachmatch(r"\.(\d+)(?=\.|$)", filename)
        result = parse(Int, m.captures[1])
    end
    return result
end

function read_fmap_paths(file::AbstractString, n_ctx::Int)
    isfile(file) || error("Cannot open $file")
    lines = readlines(file)
    length(lines) >= 2n_ctx ||
        error("fmap.paths needs $n_ctx rot/ve pairs ($(2n_ctx) lines)")
    rot = [String(rstrip(lines[2k-1], ['\r', '\n'])) for k in 1:n_ctx]
    ve = [String(rstrip(lines[2k], ['\r', '\n'])) for k in 1:n_ctx]
    return rot, ve
end

mutable struct ArgReader
    a::Vector{String}
end
argn(r::ArgReader, i, default, f) = length(r.a) >= i ? f(r.a[i]) : default

# ======================================================================= CPU NVT driver

"""
    main_nvt(args) -> exit code

./simulation_nvt N box_length T[:elec_scale][:vdw_scale] n_steps seed start stop sample_freq
    fmap.paths [tx] [rx] [zx] [insert_freq] [target_ratio] [n_types] [type_counts]
    [conf_freq] [adjust_cycles]

Single box (NVT). A step is a displacement (translation or rotation) or, with probability
`insert_freq` (default 0.1), a same-box insertion: a random particle is removed and re-inserted at a
random position (the GEMC transfer with source = target box). `start`/`stop` are single
configuration files ("-" = random start); `type_counts` are the box's per-type totals.
"""
function main_nvt(argv::Vector{String})
    if length(argv) < 9
        println("Usage: simulation_nvt N box_length T[:elec_scale][:vdw_scale] n_steps seed start stop sample_freq" *
                " fmap.paths [tx] [rx] [zx] [insert_freq] [target_ratio] [n_types] [type_counts]" *
                " [conf_freq] [adjust_cycles]")
        return 1
    end
    n_total = c_atoi(argv[1])
    if n_total < 2
        println(stderr, "Error: N must be >= 2"); return 1
    end
    box_length = c_atof(argv[2])
    te = split(argv[3], ':')
    temperature = c_atof(te[1])
    elec_scale = 1.0; vdw_scale = 1.0
    length(te) >= 2 && !isempty(te[2]) && (elec_scale = c_atof(te[2]))
    length(te) >= 3 && (vdw_scale = c_atof(te[3]))
    n_steps = c_atoi(argv[4])
    seed = UInt32(c_atoi(argv[5]) % UInt32)
    start_file = argv[6]; stop_file = argv[7]
    sample_freq = c_atoi(argv[8])
    r = ArgReader(argv)
    max_trans_param = argn(r, 10, 1.0, c_atof)
    max_rot_param = argn(r, 11, 1.0, c_atof)
    zratio = argn(r, 12, 1.0, c_atof)
    insert_freq = clamp(argn(r, 13, 0.1, c_atof), 0.0, 1.0)
    target_ratio = clamp(argn(r, 14, 0.3, c_atof), 0.0, 1.0)
    n_types = max(argn(r, 15, 1, c_atoi), 1)
    if n_types * (n_types + 1) ÷ 2 > MAX_N_CTXS
        println(stderr, "Error: n_types=$n_types exceeds MAX_N_CTXS=$MAX_N_CTXS"); return 1
    end
    n_ctx = n_types * (n_types + 1) ÷ 2
    counts_total = zeros(Int, MAX_N_CTXS)
    ct = length(argv) >= 16 ? parse_type_totals(argv[16]) : Int[]
    n_counts = length(ct); counts_total[1:n_counts] .= ct
    conf_freq_cycles = argn(r, 17, 0, c_atoi)
    conf_freq_steps = conf_freq_cycles * n_total
    adjust_cycles = max(argn(r, 18, 10, c_atoi), 0)       # 0 = step sizes frozen
    adjust_freq_steps = adjust_cycles * n_total

    local rot_paths, ve_paths
    try
        rot_paths, ve_paths = read_fmap_paths(argv[9], n_ctx)
    catch e
        println(stderr, "Error: ", sprint(showerror, e)); return 1
    end
    nrot0 = nbin0 = 0
    for k in 1:n_ctx
        local nrot_k, nbin_k
        try
            nrot_k, nbin_k = peek_rot_header(rot_paths[k])
        catch e
            println(stderr, sprint(showerror, e)); return 1
        end
        if k == 1
            nrot0, nbin0 = nrot_k, nbin_k
        elseif nrot_k != nrot0 || nbin_k != nbin0
            println(stderr, "Error: FMAP pair $(k-1) has nrot=$nrot_k nbin=$nbin_k, but pair 0 has nrot=$nrot0 nbin=$nbin0")
            return 1
        end
    end

    tern = tern_bytes_from_env()
    tern_check_scales(tern, elec_scale, vdw_scale)
    ctxs = FmapCtx[]
    for k in 1:n_ctx
        try
            push!(ctxs, with_scales(fmap_create(rot_paths[k], ve_paths[k]; tern = tern), elec_scale, vdw_scale))
        catch e
            println(stderr, "Error: failed to load FMAP pair $(k-1): ", sprint(showerror, e)); return 1
        end
    end

    box = create_box(n_total, box_length, temperature, max_trans_param, max_rot_param, seed, ctxs, n_types,
                     zratio, target_ratio)
    if start_file == "-"
        initialize_random!(box, n_total, counts_total, n_counts)
    elseif !load_configuration!(box, start_file, 0)
        println(stderr, "Error: failed to load $start_file"); return 1
    end
    if box.n_particles < n_total                      # top up a short configuration
        topup_counts = zeros(Int, MAX_N_CTXS)
        for k in 1:n_counts
            topup_counts[k] = max(counts_total[k] - get_type_count(box, k - 1), 0)
        end
        @printf("Appending %d particles\n", n_total - box.n_particles)
        topup!(box, n_total, topup_counts, n_counts) || (println(stderr, "Error: top-up failed"); return 1)
    end

    println("NVT Monte Carlo (n_types=$n_types)")
    @printf("  N=%d  L=%.3f  T=%.3f  insert_freq=%.2f  conf_freq=%d  target_ratio=%.2f  adjust_cycles=%d\n",
            n_total, box_length, temperature, insert_freq, conf_freq_cycles, target_ratio, adjust_cycles)
    println("----------------------------------------")

    conf_file = nothing
    if conf_freq_steps > 0
        path = (let s = findlast('/', stop_file); s === nothing ? "" : stop_file[1:s] end) * "conf.$(extract_stage_index(stop_file))"
        try conf_file = open(path, "w") catch; println(stderr, "Warning: cannot open $path, conf logging disabled") end
    end

    t0 = time_ns()
    for step in 0:n_steps-1
        if rand_unit!(box) < insert_freq
            nvt_insert!(box)
        else
            run_steps!(box, 1)
        end
        if adjust_freq_steps > 0 && (step + 1) % adjust_freq_steps == 0
            adjust_step_sizes!(box)
        end
        if conf_file !== nothing && (step + 1) % conf_freq_steps == 0
            @printf(conf_file, "# cycle=%d n=%d energy=%.6f\n", (step + 1) ÷ n_total, box.n_particles, box.total_energy)
            for slot in 1:box.n_particles
                c = particle_coords(box, slot)
                @printf(conf_file, "%d %.6f %.6f %.6f %.6f %.6f %.6f %d\n", slot, c[1], c[2], c[3], c[4], c[5], c[6],
                        box.particles[slot].type)
            end
        end
        if (step + 1) % sample_freq == 0
            ar = box.n_moves > 0 ? box.n_accepted / box.n_moves : 0.0
            ir = box.n_transfer_attempts > 0 ? box.n_transfer_accepted / box.n_transfer_attempts : 0.0
            @printf("Cycle %8d: N=%4d  E=%10.2f  A=%.3f  INS=%.3f  js=%.4g rs=%.4g\n",
                    (step + 1) ÷ n_total, box.n_particles, box.total_energy, ar, ir, box.js, box.rs)
        end
    end
    total_time = (time_ns() - t0) / 1e9
    println("----------------------------------------")
    @printf("Time: %.2f s  (%.0f cycles/s)\n", total_time, total_time > 0.0 ? n_steps / n_total / total_time : 0.0)
    conf_file === nothing || close(conf_file)
    save_configuration(box, stop_file)
    return 0
end
