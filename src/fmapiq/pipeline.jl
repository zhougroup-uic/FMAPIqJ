# pipeline.jl - FMAPIq pipeline command line (Julia replacement of fmapiq-pkg/pipeline.py).
#
#   stage 1  energy maps : fmappre -> fmap -> fmapstat -> fmap_pack_zlib        (src/grid)
#   stage 2  simulation  : n_reps NVT runs (same-box insertion engine, src/sim)
#   stage 3  scattering  : sim2iq pre + P(q) + one I(q) per configuration       (src/sim2iq)
#   then     statistics (Iqs.stat.txt), plot (iq_plot.svg) and optional packing.
#
# The tools run as separate julia processes of their own projects, so GPU grid generation
# (--gpu) and per-job thread counts work without a combined environment.
module FmapIq

using Printf
using Statistics
using DelimitedFiles
using JSON
using Tar
using CodecZlib

const ROOT = get(ENV, "FMAPIQ_ROOT", normpath(joinpath(@__DIR__, "..", "..")))
const BIN_DIR = joinpath(ROOT, "bin")
const DATA_DIR = joinpath(ROOT, "data")

# Ordered like DEFAULTS of pipeline.py
const DEFAULTS = Pair{Symbol,Any}[
    :n_proteins => 200, :box_size => 619.65, :ion => 0.016, :temp => 25, :nrot => 72,
    :escl => 1.0, :vscl => -1.0, :n_reps => 3, :n_cycles => 20000, :spacing => 1.0,
    # plot defaults
    :show_individual => true, :show_pq => true, :q_min => 0.01, :q_max => 0.4,
    :y_min => 0.01, :y_max => 1.5, :log_scale_x => true, :log_scale_y => false,
]
const PLOT_KEYS = (:show_individual, :show_pq, :q_min, :q_max, :y_min, :y_max, :log_scale_x, :log_scale_y)

# ---------------------------------------------------------------- tool invocation

julia_tool(script; threads = 1) =
    `$(Base.julia_exename()) --project=$ROOT -t $threads $(joinpath(BIN_DIR, script))`

"""
Run `cmd` in `dir`; `out` / `err` are file paths (or nothing = inherit). Success is returned.
File redirects go through `sh` (as in a shell script): a julia child whose stdout is a file opened
by a julia parent fails on `flush(stdout)` with "Permission denied".
"""
function run_in(cmd::Cmd, dir; out = nothing, err = nothing, env = Pair{String,String}[])
    e = merge(Dict{String,String}(ENV), Dict{String,String}(env))
    if out !== nothing || err !== nothing
        e["FMAPIQ_OUT"] = out === nothing ? "/dev/stdout" : String(out)
        e["FMAPIQ_ERR"] = err === nothing ? "/dev/stderr" : String(err)
        redirect = (out !== nothing && err !== nothing && out == err) ?
                   "exec \"\$@\" >\"\$FMAPIQ_OUT\" 2>&1" : "exec \"\$@\" >\"\$FMAPIQ_OUT\" 2>\"\$FMAPIQ_ERR\""
        cmd = `sh -c $redirect sh $(cmd.exec)`
    end
    return success(Cmd(cmd; dir = dir, env = e))
end

# ---------------------------------------------------------------- pipeline state

mutable struct Pipeline
    work_dir::Union{Nothing,String}
    pqr_file::Union{Nothing,String}
    temp::Union{Nothing,Float64}       # kbt from parms.txt (simulation "temperature")
    mass::Union{Nothing,Float64}
    params::Union{Nothing,String}
    configs::Vector{String}
    iq_data::Union{Nothing,NamedTuple}
    config::Dict{String,Any}
    threads::Int
    gpu::Bool
end

function Pipeline(work_dir = nothing; gpu = false)
    work_dir === nothing || (work_dir = abspath(work_dir))   # tools run with cwd = work_dir
    p = Pipeline(work_dir, nothing, nothing, nothing, nothing, String[], nothing,
                 Dict{String,Any}(String(k) => v for (k, v) in DEFAULTS),
                 max(Sys.CPU_THREADS ÷ 2, 1), gpu)
    work_dir === nothing || mkpath(work_dir)
    return p
end

function ensure_work_dir!(p::Pipeline)
    if p.work_dir === nothing
        mkpath("tmp")
        p.work_dir = abspath(mktempdir("tmp"; prefix = "maeppi_", cleanup = false))
    end
    mkpath(p.work_dir)
    return p.work_dir
end

config_path(p::Pipeline) = p.work_dir === nothing ? nothing : joinpath(ensure_work_dir!(p), "config.json")

function save_config(p::Pipeline)
    f = config_path(p)
    f === nothing && return
    try
        open(io -> JSON.print(io, p.config, 2), f, "w")
    catch e
        println("Warning: Could not save config: $e")
    end
end

function load_config!(p::Pipeline)
    f = config_path(p)
    (f !== nothing && isfile(f)) || return false
    try
        merge!(p.config, JSON.parsefile(f))
        return true
    catch e
        println("Warning: Could not load config: $e")
        return false
    end
end

function update_config!(p::Pipeline; kw...)
    for (k, v) in kw
        p.config[String(k)] = v
    end
    save_config(p)
end

function parse_params!(p::Pipeline)
    p.params === nothing && return
    for line in split(p.params, '\n')
        t = split(line)
        length(t) >= 3 || continue
        try
            occursin("kbt", line) && (p.temp = parse(Float64, t[3]))
            occursin("mass", line) && (p.mass = parse(Float64, t[3]))
        catch e
            println("Warning: Could not parse parameters: $e")
        end
    end
end

"Load existing results of the work directory (stage = nothing: all)."
function load_data!(p::Pipeline; stage = nothing)
    (p.work_dir !== nothing && isdir(p.work_dir)) || return false
    load_config!(p)
    try
        if stage in (nothing, 1)
            pf = joinpath(p.work_dir, "parms.txt")
            if isfile(pf)
                p.pqr_file = joinpath(p.work_dir, "subA.pqr")
                p.params = read(pf, String)
                parse_params!(p)
            end
        end
        if stage in (nothing, 2)
            p.configs = sort([basename(f) for f in readdir(p.work_dir; join = true)
                              if occursin(r"^config_.*\.dat$", basename(f))])
        end
        if stage in (nothing, 3)
            f = joinpath(p.work_dir, "Iqs.txt")
            if isfile(f)
                d = readdlm(f)
                if size(d, 1) > 1
                    p.iq_data = (q = d[:, 1], pq = d[:, 2], curves = d[:, 3:end], n_curves = size(d, 2) - 2)
                end
            end
        end
        return true
    catch e
        println("Load error: $e")
        return false
    end
end

# ---------------------------------------------------------------- stage 1

function stage1(p::Pipeline, pqr_file, ion, temp, escl, vscl, threads; nrot = 72)
    (pqr_file !== nothing || p.pqr_file !== nothing) || return "Error: No PQR file provided"
    update_config!(p; ion, temp, escl, vscl, nrot)
    try
        ensure_work_dir!(p)
        if p.pqr_file === nothing
            p.pqr_file = joinpath(p.work_dir, "subA.pqr")
            cp(pqr_file, p.pqr_file; force = true)
        end
        wd = p.work_dir
        ang = joinpath(DATA_DIR, @sprintf("sspNF%05d", nrot), "rot.eul")
        isfile(ang) || return "Error: orientation table not found: $ang"

        gbin(s) = joinpath(BIN_DIR, s)
        grid(script, args...; kw...) = run_in(`$(julia_tool(script; threads)) $(collect(args))`, wd; kw...)

        # fmap writes fmap.{rot,ve}.zlib directly (no ve/ directory). Grids of an earlier (python) run
        # (ve/tErn.*.mat.gz) are reused: only the packed format is added.
        have_zlib = isfile(joinpath(wd, "fmap.ve.zlib")) && isfile(joinpath(wd, "fmap.rot.zlib")) &&
                    isfile(joinpath(wd, "fmap.ern"))
        have_ve = isfile(joinpath(wd, "parms.txt")) && isfile(joinpath(wd, "fmap.ern")) &&
                  isdir(joinpath(wd, "ve")) && length(readdir(joinpath(wd, "ve"))) == nrot
        if have_zlib
            println("Energy maps found in $wd")
        elseif have_ve
            println("Energy maps found in $wd, packing only")
            grid("fmap_pack_zlib.jl", "parms.txt", "fmap") || return "Error: fmap_pack_zlib failed"
        else
            vscl_arg = vscl < 0.0 ? "-" : string(vscl)
            grid("fmappre.jl", "-o", "parms.txt", "subA.pqr", string(ion), string(temp), string(escl), vscl_arg) ||
                return "Error: fmappre failed"

            if nrot != 72
                open(f -> @printf(f, "%-12s : %16.0f\n", "nrot", nrot), joinpath(wd, "parms.txt"), "a")
            end

            env = p.gpu ? ["FMAPB2_GPU" => "1"] : ["FMAPB2_GPU" => "0"]
            grid("fmap.jl", "subA.vdw", "parms.txt", ang, "--zlib", "fmap"; out = joinpath(wd, "fmap.ern"),
                 err = joinpath(wd, "fmap.log"), env) || return "Error: fmap failed"

            grid("fmapstat.jl", "stats", "--ern", "fmap.ern") || return "Error: fmapstat failed"
        end
        write(joinpath(wd, "fmap.paths"), "fmap.rot.zlib\nfmap.ve.zlib\n")

        load_data!(p; stage = 1)
        return "Stage 1 Complete!\n$(p.params)"
    catch e
        return "Stage 1 Error: $(sprint(showerror, e))"
    end
end

# ---------------------------------------------------------------- stage 2

function run_single_simulation(p::Pipeline, sim_id, n_proteins, box_size, temp, n_cycles, sim_threads)
    config_file = @sprintf("config_%03d.dat", sim_id)
    n_steps = Int(n_cycles) * Int(n_proteins)
    # N box kT n_steps seed start stop sample_freq fmap.paths   (NVT with same-box insertion)
    cmd = `$(julia_tool("simulation_nvt.jl"; threads = sim_threads)) $n_proteins $box_size $temp $n_steps $sim_id - $config_file $n_steps fmap.paths`
    log = joinpath(p.work_dir, @sprintf("sim_%03d.log", sim_id))
    ok = run_in(cmd, p.work_dir; out = log, err = log)
    return ok && isfile(joinpath(p.work_dir, config_file)) ? config_file : nothing
end

# P(q) of the single protein does not depend on the trajectories: with --gpu a sim2iq_gpu_batch process is
# started before the NVT replicas, computes P(q) on the same GPU (start-up and kernel compilation overlap
# the simulation) and then waits on its stdin; stage 3 feeds it one I(q) job per configuration.
const PQ_PROC = Ref{Any}(nothing)       # (work_dir, ngrid, grid_spacing, process, log) of the waiting server

function sim2iq_grid(box_size, spacing)
    ngrid = Int(floor(box_size / spacing))
    return ngrid, box_size / ngrid
end

function start_pq(p::Pipeline, box_size, spacing)
    PQ_PROC[] = nothing
    wd = p.work_dir
    ngrid, gsp = sim2iq_grid(box_size, spacing)
    rm(joinpath(wd, "Pq.dat"); force = true)
    run_in(`$(julia_tool("sim2iqpre.jl")) -f subA.vdw -o subA.vol`, wd) || return false
    log = joinpath(wd, "sim2iq_gpu.log")
    cmd = pipeline(Cmd(`$(julia_tool("sim2iq_gpu_batch.jl")) subA.vol $ngrid $gsp 4 --serve`; dir = wd);
                   stdout = log, stderr = log)
    proc = open(cmd, "w")
    atexit(() -> process_running(proc) && kill(proc))
    println(proc, "Pq.dat=-"); flush(proc)
    PQ_PROC[] = (wd, ngrid, gsp, proc)
    return true
end

# GPU: all replicas in one process (one triple per replica, seeds 1..n_reps; sim/gpu/main_nvt_gpu.jl).
function stage2_gpu(p::Pipeline, n_reps, n_cycles, n_proteins, box_size)
    labels = [@sprintf("r%03d", i) for i in 0:n_reps-1]
    te = join(("$(p.temp):1.0:1.0:$l" for l in labels), ",")
    cmd = `$(julia_tool("simulation_nvt_gpu.jl")) $n_proteins $box_size $te $n_cycles 1 - traj.0 $n_cycles fmap.paths`
    log = joinpath(p.work_dir, "sim_gpu.log")
    run_in(cmd, p.work_dir; out = log, err = log) || return String[]
    configs = String[]
    for (i, l) in enumerate(labels)
        src = joinpath(p.work_dir, "T$l", "N$(n_proteins)L$(box_size)", "V1.0", "R1", "traj.0.1.txt")
        isfile(src) || continue
        dst = @sprintf("config_%03d.dat", i - 1)
        cp(src, joinpath(p.work_dir, dst); force = true)
        push!(configs, dst)
    end
    return configs
end

function stage2(p::Pipeline, n_reps, n_cycles, n_proteins, box_size, threads, sim_threads; spacing = 1.0)
    (p.work_dir !== nothing && isdir(p.work_dir)) || return "Error: Run Stage 1 first"
    isfile(joinpath(p.work_dir, "fmap.paths")) || return "Error: Run Stage 1 first (fmap.paths missing)"
    p.temp === nothing && return "Error: kbt not found in parms.txt"
    update_config!(p; n_reps, n_cycles, n_proteins, box_size)
    try
        if p.gpu
            start_pq(p, box_size, spacing)
            ok = length(stage2_gpu(p, n_reps, n_cycles, n_proteins, box_size))
            load_data!(p; stage = 2)
            return ok == n_reps ? "Stage 2 Complete!\nGenerated $ok configurations (GPU)" :
                                  "Stage 2 Error: only $ok of $n_reps configurations (see sim_gpu.log)"
        end
        max_jobs = max(Int(threads) ÷ Int(sim_threads), 1)
        ids = collect(0:n_reps-1)
        job(i) = run_single_simulation(p, i, n_proteins, box_size, p.temp, n_cycles, sim_threads)
        # The first job alone: it inflates the raw grid cache that all later jobs map (no racing writers).
        results = Union{Nothing,String}[job(ids[1])]
        append!(results, asyncmap(job, ids[2:end]; ntasks = max_jobs))
        load_data!(p; stage = 2)
        ok = count(!isnothing, results)
        return "Stage 2 Complete!\nGenerated $ok configurations"
    catch e
        return "Stage 2 Error: $(sprint(showerror, e))"
    end
end

# ---------------------------------------------------------------- stage 3

function stage3(p::Pipeline, box_size, spacing, threads)
    (p.work_dir !== nothing && !isempty(p.configs)) || return "Error: Run Stage 2 first"
    update_config!(p; spacing)
    try
        n_proteins = Int(p.config["n_proteins"])
        ngrid = Int(floor(box_size / spacing))
        grid_spacing = box_size / ngrid               # sim2iq takes the spacing, the python CLI the box length
        wd = p.work_dir
        pre = julia_tool("sim2iqpre.jl")
        calc = julia_tool("sim2iq.jl"; threads)
        # subA.vdw: centred coordinates (the trajectories rotate about the centre) with the original atom names
        pq_file = "Pq.dat"
        srv = PQ_PROC[]
        served = p.gpu && srv !== nothing && srv[1:3] == (wd, ngrid, grid_spacing) && process_running(srv[4])
        if p.gpu && served      # feed the waiting GPU process (P(q) already computed or being computed)
            proc = srv[4]
            for (i, c) in enumerate(p.configs)
                println(proc, @sprintf("iq_%03d.dat=%s", i - 1, c))
            end
            close(proc.in)
            wait(proc)
            PQ_PROC[] = nothing
            success(proc) || return "Error: GPU sim2iq failed (see sim2iq_gpu.log)"
        else
            run_in(`$pre -f subA.vdw -o subA.vol`, wd) || return "Error: sim2iq_pre failed"
            if p.gpu       # all runs in one julia/CUDA.jl process (sim2iq/gpu): P(q), then one I(q) per configuration
                jobs = [pq_file * "=-"; [@sprintf("iq_%03d.dat=%s", i - 1, c) for (i, c) in enumerate(p.configs)]]
                batch = julia_tool("sim2iq_gpu_batch.jl")
                log = joinpath(wd, "sim2iq_gpu.log")
                run_in(`$batch subA.vol $ngrid $grid_spacing 4 $jobs`, wd; out = log, err = log) ||
                    return "Error: GPU sim2iq failed (see sim2iq_gpu.log)"
            else
                run_in(`$calc subA.vol $ngrid $grid_spacing $pq_file -steps 4`, wd)
            end
        end
        isfile(joinpath(wd, pq_file)) || return "Error: P(q) calculation failed"

        iq_results = NamedTuple[]
        for (i, cfg) in enumerate(p.configs)
            out = @sprintf("iq_%03d.dat", i - 1)
            p.gpu || run_in(`$calc subA.vol $ngrid $grid_spacing $out -steps 4 -traj $cfg -txt`, wd)
            f = joinpath(wd, out)
            if isfile(f)
                try
                    d = readdlm(f)
                    push!(iq_results, (q = d[:, 1], iq = d[:, 2]))
                catch
                end
            end
        end
        isempty(iq_results) && return "Error: No I(q) data generated"

        pq = readdlm(joinpath(wd, pq_file))
        q = iq_results[1].q
        curves = [r.iq ./ n_proteins for r in iq_results if length(r.iq) == length(q)]
        isempty(curves) && return "Error: No valid I(q) curves"

        comb = hcat(q, pq[:, 2], reduce(hcat, curves))
        write_columns(joinpath(wd, "Iqs.txt"), comb)
        load_data!(p; stage = 3)
        return "Stage 3 Complete!\nCalculated I(q) for $(length(iq_results)) configurations"
    catch e
        return "Stage 3 Error: $(sprint(showerror, e))"
    end
end

function write_columns(path, M)
    open(path, "w") do io
        for i in axes(M, 1)
            println(io, join((@sprintf("%.6e", M[i, j]) for j in axes(M, 2)), " "))
        end
    end
end

# ---------------------------------------------------------------- statistics, plot, pack

function svg_plot(path, q, series, mean_y, err_y; qmin, qmax, ymin, ymax, logx, logy, title)
    W, H, L, R, T, B = 800, 500, 70, 20, 40, 55
    tx(x) = (v = logx ? log10(max(x, 1e-300)) : x;
             a = logx ? log10(qmin) : qmin; b = logx ? log10(qmax) : qmax; L + (v - a) / (b - a) * (W - L - R))
    ty(y) = (v = logy ? log10(max(y, 1e-300)) : y;
             a = logy ? log10(max(ymin, 1e-300)) : ymin; b = logy ? log10(ymax) : ymax; H - B - (v - a) / (b - a) * (H - T - B))
    clip(v) = clamp(v, -1e5, 1e5)
    poly(x, y, col, w, op) = begin
        pts = join((@sprintf("%.1f,%.1f", clip(tx(x[k])), clip(ty(y[k]))) for k in eachindex(x)), " ")
        @sprintf("<polyline fill=\"none\" stroke=\"%s\" stroke-width=\"%s\" opacity=\"%s\" points=\"%s\"/>", col, w, op, pts)
    end
    io = IOBuffer()
    println(io, "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"$W\" height=\"$H\" viewBox=\"0 0 $W $H\">")
    println(io, "<rect width=\"$W\" height=\"$H\" fill=\"white\"/>")
    println(io, "<defs><clipPath id=\"c\"><rect x=\"$L\" y=\"$T\" width=\"$(W-L-R)\" height=\"$(H-T-B)\"/></clipPath></defs>")
    println(io, "<rect x=\"$L\" y=\"$T\" width=\"$(W-L-R)\" height=\"$(H-T-B)\" fill=\"none\" stroke=\"#444\"/>")
    println(io, "<g clip-path=\"url(#c)\">")
    for s in series
        println(io, poly(s.x, s.y, s.color, s.width, s.opacity))
    end
    println(io, poly(q[2:end], mean_y[2:end], "#1f77b4", 2, 1))
    for k in 2:length(q)
        x = tx(q[k])
        println(io, @sprintf("<line x1=\"%.1f\" x2=\"%.1f\" y1=\"%.1f\" y2=\"%.1f\" stroke=\"#1f77b4\"/>",
                             x, x, clip(ty(mean_y[k] - err_y[k])), clip(ty(mean_y[k] + err_y[k]))))
    end
    println(io, "</g>")
    println(io, "<text x=\"$(W÷2)\" y=\"24\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"15\">$title</text>")
    println(io, "<text x=\"$(W÷2)\" y=\"$(H-12)\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"13\">q (1/Å)</text>")
    println(io, "<text x=\"16\" y=\"$(H÷2)\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"13\" transform=\"rotate(-90 16 $(H÷2))\">I(q) / n_proteins (normalized)</text>")
    for v in (qmin, qmax)
        println(io, @sprintf("<text x=\"%.1f\" y=\"%d\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"11\">%g</text>", tx(v), H - B + 16, v))
    end
    for v in (ymin, ymax)
        println(io, @sprintf("<text x=\"%d\" y=\"%.1f\" text-anchor=\"end\" font-family=\"sans-serif\" font-size=\"11\">%g</text>", L - 6, ty(v) + 4, v))
    end
    println(io, "</svg>")
    write(path, take!(io))
end

"Statistics file (Iqs.stat.txt) and plot (iq_plot.svg). Returns (plot_path or nothing, message)."
function create_plot(p::Pipeline; show_individual = true, show_pq = true, q_min = 0.01, q_max = 0.4,
                     y_min = 0.0, y_max = 1.2, log_scale_x = true, log_scale_y = false)
    p.iq_data === nothing && return nothing, "No I(q) data available"
    update_config!(p; show_individual, show_pq, q_min, q_max, y_min, y_max, log_scale_x, log_scale_y)
    try
        q, pq, curves = p.iq_data.q, p.iq_data.pq, p.iq_data.curves
        pqn = pq ./ pq[1]
        cn = curves ./ pq[1]
        mean_i = vec(mean(cn; dims = 2))
        std_err = vec(std(cn; dims = 2, corrected = false)) ./ sqrt(size(cn, 2))
        write_columns(joinpath(p.work_dir, "Iqs.stat.txt"), hcat(q, pqn, mean_i, std_err))

        series = NamedTuple[]
        show_pq && push!(series, (x = q, y = pqn, color = "#bbbbbb", width = 1, opacity = 0.7))
        if show_individual
            for i in 1:size(cn, 2)
                push!(series, (x = q[2:end], y = cn[2:end, i], color = "red", width = 0.5, opacity = 0.3))
            end
        end
        path = joinpath(p.work_dir, "iq_plot.svg")
        svg_plot(path, q, series, mean_i, std_err; qmin = q_min, qmax = q_max, ymin = y_min, ymax = y_max,
                 logx = log_scale_x, logy = log_scale_y,
                 title = "Small Angle Scattering Intensity I(q), mean ± SEM (n=$(p.iq_data.n_curves))")
        return path, "Plot created successfully"
    catch e
        return nothing, "Plot error: $(sprint(showerror, e))"
    end
end

function pack_results(p::Pipeline; skip_energy_maps = true, skip_raw_data = true)
    (p.work_dir !== nothing && isdir(p.work_dir)) || return nothing, "No work directory to pack"
    try
        name = basename(normpath(p.work_dir))
        archive = joinpath(p.work_dir, name * ".tar.gz")
        rm(archive; force = true)
        keep(path) = begin
            rel = relpath(path, p.work_dir)
            parts = splitpath(rel)
            (endswith(rel, ".tar.gz") || endswith(rel, ".tar.gz.part")) && return false
            skip_energy_maps && !isempty(parts) && parts[1] == "ve" && return false
            skip_raw_data && isfile(path) && endswith(rel, ".dat") && return false
            return true
        end
        tmp = archive * ".part"
        open(tmp, "w") do io
            gz = GzipCompressorStream(io)
            Tar.create(keep, p.work_dir, gz)
            close(gz)
        end
        mv(tmp, archive; force = true)
        return archive, @sprintf("Archive created: %s (%.1f MB)", basename(archive), filesize(archive) / 2^20)
    catch e
        return nothing, "Pack error: $(sprint(showerror, e))"
    end
end

# ---------------------------------------------------------------- command line

const HELP = """
usage: fmapiq [pqr_file] [options]

FMAPIq Pipeline - Small Angle Scattering Calculation

positional:
  pqr_file               Input PQR structure file (optional when the work dir has subA.pqr)

options:
  --work-dir DIR         Work directory (new or existing)
  --load DIR             Load existing work directory
  --pack                 Pack results into archive
  --skip-energy-maps     Skip energy maps when packing
  --skip-raw-data        Skip raw data (*.dat) when packing
  --threads N            Total threads (default: CPU cores / 2)
  --sim-threads N        Threads per simulation job (default: 2)
  --gpu                  Run all stages on the GPU: energy maps (CUDA.jl), NVT (sim/gpu), I(q) (sim2iq/gpu, CUDA.jl)
  --n-proteins N         number of proteins (default: 200)
  --box-size L           box edge length, Å (default: 619.65)
  --ion C                ionic strength, M (default: 0.016)
  --temp T               temperature, °C (default: 25)
  --nrot N               number of orientations (default: 72)
  --escl X               electrostatic scale (default: 1.0)
  --vscl X               vdw scale, negative = from mass (default: -1.0)
  --n-reps N             independent simulations (default: 3)
  --n-cycles N           cycles per simulation (default: 20000)
  --spacing X            I(q) grid spacing, Å (default: 1.0)
  -h, --help             show this help

Environment: TERN_BYTES (default 1) grid cell format; FMAPIQ_CACHE raw-grid cache directory;
  FMAPIQ_ROOT location of grid/, sim/, sim2iq/, data/.
"""

function parse_cli(argv)
    defaults = Dict(String(k) => v for (k, v) in DEFAULTS if !(k in PLOT_KEYS))
    a = Dict{String,Any}("pqr_file" => nothing, "work_dir" => nothing, "load" => nothing, "pack" => false,
                         "skip_energy_maps" => false, "skip_raw_data" => false, "threads" => nothing,
                         "sim_threads" => 2, "gpu" => false)
    merge!(a, defaults)
    i = 1
    while i <= length(argv)
        s = argv[i]
        if s in ("-h", "--help")
            print(HELP); return nothing
        elseif startswith(s, "--")
            key = replace(s[3:end], '-' => '_')
            val = nothing
            if occursin('=', key)
                key, val = split(key, '='; limit = 2)
            end
            haskey(a, key) || (println(stderr, "error: unrecognized argument: $s"); return missing)
            if a[key] isa Bool
                a[key] = true
            else
                if val === nothing
                    i + 1 <= length(argv) || (println(stderr, "error: $s expects a value"); return missing)
                    val = argv[i+1]; i += 1
                end
                T = a[key] === nothing ? (key in ("work_dir", "load") ? String : Int) : typeof(a[key])
                a[key] = T === String ? String(val) : tryparse(T, val)
                a[key] === nothing && (println(stderr, "error: invalid value for $s: $val"); return missing)
            end
        else
            a["pqr_file"] === nothing || (println(stderr, "error: unrecognized argument: $s"); return missing)
            a["pqr_file"] = s
        end
        i += 1
    end
    return a
end

stage_done(wd, kind) =
    wd !== nothing && (kind == 1 ? isfile(joinpath(wd, "parms.txt")) && isfile(joinpath(wd, "fmap.paths")) :
                       kind == 2 ? any(f -> occursin(r"^config_.*\.dat$", f), readdir(wd)) :
                                   isfile(joinpath(wd, "Iqs.txt")))

"Command line entry; returns the process exit code."
function main(argv::Vector{String} = copy(ARGS))
    args = parse_cli(argv)
    args === nothing && return 0
    args === missing && return 2
    # Grid cell format shared by the grid tools and the simulation: 1 = pre-summed byte (the format of
    # the python pipeline; escl/vscl act at grid generation), 2 = [elec, vdw] channels. Override with TERN_BYTES.
    get!(ENV, "TERN_BYTES", "1")

    p = Pipeline(something(args["work_dir"], args["load"], Some(nothing)); gpu = args["gpu"])
    args["threads"] === nothing || (p.threads = args["threads"])

    if p.work_dir !== nothing && isdir(p.work_dir)
        if load_data!(p)
            println("Loaded existing work directory: $(p.work_dir)")
            for k in 1:3
                println("Stage $k: ", stage_done(p.work_dir, k) ? "OK" : "Not completed")
            end
            if stage_done(p.work_dir, 3) && p.iq_data !== nothing
                println("I(q) datasets available: $(p.iq_data.n_curves)")
            end
            if args["load"] !== nothing && args["pqr_file"] === nothing
                if args["pack"]
                    f, msg = pack_results(p; skip_energy_maps = args["skip_energy_maps"], skip_raw_data = args["skip_raw_data"])
                    println(f !== nothing ? "Results packed: $f" : "Packing failed: $msg")
                end
                return 0
            end
        else
            println("Failed to load work directory: $(p.work_dir)")
            return 1
        end
    end

    pqr = args["pqr_file"]
    if pqr === nothing && p.work_dir !== nothing
        cand = joinpath(p.work_dir, "subA.pqr")
        if isfile(cand)
            pqr = cand
            println("Using existing PQR file: $pqr")
        end
    end
    pqr === nothing && (println("Error: PQR file required for new pipeline run"); return 1)
    isfile(pqr) || (println("Error: PQR file not found: $pqr"); return 1)

    if !stage_done(p.work_dir, 1)
        println("Running Stage 1: Energy map calculation...")
        r = stage1(p, pqr, args["ion"], args["temp"], args["escl"], args["vscl"], p.threads; nrot = args["nrot"])
        println(r)
        occursin("Error", r) && return 1
    else
        println("Stage 1: Already completed, skipping")
    end

    if !stage_done(p.work_dir, 2)
        println("Running Stage 2: Monte Carlo simulations...")
        r = stage2(p, args["n_reps"], args["n_cycles"], args["n_proteins"], args["box_size"], p.threads, args["sim_threads"]; spacing = args["spacing"])
        println(r)
        occursin("Error", r) && return 1
    else
        println("Stage 2: Already completed, skipping")
    end

    if !stage_done(p.work_dir, 3)
        println("Running Stage 3: I(q) intensity calculation...")
        r = stage3(p, args["box_size"], args["spacing"], p.threads)
        println(r)
        occursin("Error", r) && return 1
    else
        println("Stage 3: Already completed, skipping")
    end

    println("Pipeline completed successfully! Results in: $(p.work_dir)")
    path, msg = create_plot(p)
    println(path !== nothing ? "Plot saved: $path" : "Plot generation failed: $msg")

    if args["pack"]
        f, msg = pack_results(p; skip_energy_maps = args["skip_energy_maps"], skip_raw_data = args["skip_raw_data"])
        println(f !== nothing ? "Results packed: $f" : "Packing failed: $msg")
    end
    return 0
end

end # module
