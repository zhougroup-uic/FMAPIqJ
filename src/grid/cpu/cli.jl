# cli.jl - command lines of the python tools: fmappre, fmap (grid generation), fmap_pack_zlib.

function main_fmappre(argv::Vector{String})
    out = nothing
    args = String[]
    i = 1
    while i <= length(argv)
        if argv[i] == "-o" && i < length(argv)
            out = argv[i+1]; i += 2
        else
            push!(args, argv[i]); i += 1
        end
    end
    if length(args) < 5
        println("Usage: fmappre [-o output_file] A.pqr ion temp escl vscl [B.pqr]")
        println("Extra output: subA.vdw [subB.vdw]")
        println("vscl: use '-' to auto-calculate from mass")
        println("  -o output_file  : write parameters to file instead of stdout")
        return 1
    end
    pqr_b = length(args) > 5 ? args[6] : nothing
    if out === nothing
        fmappre(args[1], parse(Float64, args[2]), parse(Float64, args[3]), args[4], args[5]; pqr_b)
    else
        open(io -> fmappre(args[1], parse(Float64, args[2]), parse(Float64, args[3]), args[4], args[5]; pqr_b, out = io), out, "w")
    end
    return 0
end

"fmap subA.vdw parms.txt ang.dat [fnamepat] [subB.vdw]: one gzip'd tErn grid per orientation."
function main_fmap(argv::Vector{String})
    # --zlib PREFIX: write PREFIX.ve.zlib and PREFIX.rot.zlib directly (no ve/ directory, no fmap_pack_zlib)
    zprefix = nothing
    if (k = findfirst(==("--zlib"), argv)) !== nothing && k < length(argv)
        zprefix = argv[k+1]
        argv = [argv[i] for i in eachindex(argv) if i != k && i != k + 1]
    end
    level = parse(Int, get(ENV, "FMAP_GZ_LEVEL", "1"))
    if length(argv) < 3
        println(stderr, "Usage: fmap subA.vdw parms.txt ang.dat [fnamepat] [subB.vdw] [--zlib PREFIX]")
        return 1
    end
    subA, parms_file, ang_file = argv[1], argv[2], argv[3]
    fnamepat = length(argv) < 4 ? "ve/tErn.{:05d}.mat.gz" : argv[4]
    subB = length(argv) > 4 ? argv[5] : nothing
    occursin("{:05d}", fnamepat) || (println(stderr, "Invalid filename pattern: $fnamepat"); return 1)

    params = read_parameters(parms_file)
    local kbt, kappa, sdie, rscl, qscl, escl, vscl, rcut, dx, blen
    try
        kbt = params["kbt"]; kappa = params["kappa"]; sdie = params["sdie"]; rscl = params["rscl"]
        qscl = params["qscl"]; escl = params["escl"]; vscl = params["vscl"]; rcut = params["rcut"]
        dx = params["spacing"]; blen = Int(params["blen"])
    catch e
        e isa KeyError ? (println(stderr, "Missing required parameter: ", e.key); return 1) : rethrow()
    end
    @printf(stderr, "Grid: %dx%dx%d, spacing: %s A, rcut: %.0f A\n", blen, blen, blen, string(dx), rcut)
    sol = Solv(rcut, 0.0, kappa, sdie, kbt, escl, vscl)

    crds0 = read_vdw(subA)
    pros0 = subB === nothing ? copy(crds0) : read_vdw(subB)
    crds = [Atom(a.info, a.x, a.y, a.z, a.q * qscl, a.r * rscl, a.Asq, a.Bsq) for a in crds0]
    pros = [Atom(a.info, a.x, a.y, a.z, a.q, a.r * rscl, a.Asq, a.Bsq) for a in pros0]
    crds = scale_coordinates(crds, dx); pros = scale_coordinates(pros, dx)
    angles = read_angles(ang_file)
    @printf(stderr, "Loaded: %d crds, %d pros, %d rotations\n", length(crds), length(pros), length(angles))
    if zprefix === nothing
        d = dirname(fnamepat)
        isempty(d) || mkpath(d)
    end

    ctx = if get(ENV, "FMAPB2_GPU", "0") in ("", "0")
        SoftCtx(blen)
    else            # CUDA.jl + CUFFT (gpu/gpu.jl), same softgrd contract
        Base.include(FmapGen, joinpath(@__DIR__, "..", "gpu", "gpu.jl"))
        Base.invokelatest(make_ctx, blen, Val(true))
    end
    println(stderr, "Processing orientations...(check output file for progress)")
    if zprefix === nothing
        # per-orientation gzip files; compression of one orientation overlaps the next one's computation
        slots = Channel{Nothing}(max(1, Threads.nthreads() - 1))
        tasks = Task[]
        for (i, ang) in enumerate(angles)
            rotated = rotate_atoms(pros, euler_to_rotmat(ang[1], ang[2], ang[3]))
            grid = Base.invokelatest(softgrd, ctx, crds, rotated, dx, sol)
            fname = replace(fnamepat, "{:05d}" => @sprintf("%05d", i - 1))
            put!(slots, nothing)
            push!(tasks, Threads.@spawn begin
                try
                    open(fname, "w") do io
                        s = GzipCompressorStream(io; level)
                        write(s, grid)
                        close(s)
                    end
                finally
                    take!(slots)
                end
            end)
        end
        foreach(wait, tasks)
    else
        write_ve_zlib(zprefix, parms_file, angles, level) do i
            ang = angles[i]
            rotated = rotate_atoms(pros, euler_to_rotmat(ang[1], ang[2], ang[3]))
            Base.invokelatest(softgrd, ctx, crds, rotated, dx, sol)
        end
    end
    println()
    println(stderr, "Completed successfully!")
    return 0
end

function main_pack(argv::Vector{String})
    if isempty(argv) || argv[1] in ("-h", "--help")
        println("Usage: fmap_pack_zlib <config.txt> [output_prefix] [--level N]")
        println("  output_prefix  default: 'fmap'  -> fmap.rot.zlib, fmap.ve.zlib")
        return 0
    end
    prefix = "fmap"; level = 1
    i = 2
    while i <= length(argv)
        if argv[i] == "--level" && i < length(argv)
            level = parse(Int, argv[i+1]); i += 2
        elseif !startswith(argv[i], "--")
            prefix = argv[i]; i += 1
        else
            println("Unknown option: ", argv[i]); return 1
        end
    end
    pack(argv[1], prefix; level)
    return 0
end
