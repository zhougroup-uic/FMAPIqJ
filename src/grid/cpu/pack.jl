# pack.jl - fmap_pack_zlib: ve/tErn.*.mat.gz + rotation tables -> <prefix>.rot.zlib / <prefix>.ve.zlib
# (tools/fmap_pack_zlib.py; file layout in src/fmap_zlib.c / the simulation's grid loader).

const ROT_MAGIC_PACK = 0x524F5446      # "ROTF"
const VE_MAGIC_PACK  = 0x46454D41      # "FEMA"

struct PackConfig
    nrot::Int
    blen::Int
    nbin::Int
    spacing::Float64
    fpat_rot::String
end

function load_pack_config(filename::AbstractString)
    blen = 0; nrot = 0; spacing = 0.0
    for line in eachline(filename)
        line = strip(line)
        if startswith(line, "blen")
            blen = parse(Int, strip(split(line, ':')[2]))
        elseif startswith(line, "nrot")
            nrot = parse(Int, strip(split(line, ':')[2]))
        elseif startswith(line, "spacing")
            spacing = parse(Float64, strip(split(line, ':')[2]))
        end
    end
    nrot == 0 && (nrot = 72)
    spacing == 0.0 && (spacing = 0.6)
    cand = joinpath(data_dir(), @sprintf("sspNF%05d", nrot))
    isdir(cand) || error("cannot find rotation tables $cand (nrot=$nrot); only the sets under fmapb2/data/ are supported")
    return PackConfig(nrot, blen, 120, spacing, cand)
end

function read_raw(filename::AbstractString, nbytes::Integer)
    data = if endswith(filename, ".gz")
        open(io -> read(GzipDecompressorStream(io), nbytes), filename)
    else
        open(io -> read(io, nbytes), filename)
    end
    length(data) == nbytes || error("$filename: expected $nbytes bytes, got $(length(data))")
    return data
end

# [header][u64 compressed size][zlib stream]; the size is patched in after the stream is closed.
function write_zlb(f, path::AbstractString, header::Vector{UInt8}, raw_bytes::Integer; level::Integer = 6)
    io = open(path, "w")
    write(io, header)
    write(io, UInt64(0))
    s = ZlibCompressorStream(io; level = level)
    f(s)
    close(s)                                  # ends the zlib stream and closes the file
    csz = filesize(path) - length(header) - 8
    open(path, "r+") do io2
        seek(io2, length(header))
        write(io2, UInt64(csz))
    end
    @printf("  wrote %s: %.1f MB -> %.2f MB (%.1fx)\n", path, raw_bytes / 1e6, csz / 1e6, raw_bytes / max(1, csz))
end

"<prefix>.rot.zlib from the rotation-table blobs (ijk, ij2k, rot, rotm)."
function write_rot_zlib(cfg::PackConfig, prefix::AbstractString, blobs; level::Integer = 1)
    hdr = IOBuffer()
    write(hdr, UInt32(ROT_MAGIC_PACK), UInt32(1), Int32(cfg.nrot), Int32(cfg.nbin))
    write_zlb(prefix * ".rot.zlib", take!(hdr), sum(length, blobs); level) do s
        foreach(b -> write(s, b), blobs)
    end
end

"Read the four rotation tables of the set `cfg.fpat_rot` (gz files)."
function read_rot_blobs(cfg::PackConfig)
    files = (("angijk.dat.gz", cfg.nbin^3, 2), ("angij2k.dat.gz", cfg.nrot^2, 2),
             ("rot.dat.gz", cfg.nrot * 3, 8), ("rotm.dat.gz", cfg.nrot * 9, 8))
    return [read_raw(joinpath(cfg.fpat_rot, f), c * e) for (f, c, e) in files]
end

function pack(config_file::AbstractString, prefix::AbstractString = "fmap"; level::Integer = 1, tern::Integer = tern_bytes_from_env())
    cfg = load_pack_config(config_file)
    @printf("Config: nrot=%d blen=%d nbin=%d spacing=%s lsel=%d\n", cfg.nrot, cfg.blen, cfg.nbin, string(cfg.spacing), cfg.blen)

    files = (("ijk", "angijk.dat.gz", cfg.nbin^3, 2), ("ij2k", "angij2k.dat.gz", cfg.nrot^2, 2),
             ("rot", "rot.dat.gz", cfg.nrot * 3, 8), ("rotm", "rotm.dat.gz", cfg.nrot * 9, 8))
    blobs = Vector{UInt8}[]
    for (name, fname, count, esz) in files
        path = joinpath(cfg.fpat_rot, fname)
        d = read_raw(path, count * esz)
        push!(blobs, d)
        println("  loaded $name: $path ($(length(d)) bytes)")
    end
    write_rot_zlib(cfg, prefix, blobs; level)

    slice_bytes = cfg.blen^3 * tern
    hdr = IOBuffer()
    write(hdr, UInt32(VE_MAGIC_PACK), UInt32(1), Int32(cfg.nrot), Int32(cfg.blen), Float64(cfg.spacing))
    for i in 0:cfg.nrot-1
        isfile(@sprintf("ve/tErn.%05d.mat.gz", i)) || error(@sprintf("ern file missing: ve/tErn.%05d.mat.gz", i))
    end
    write_zlb(prefix * ".ve.zlib", take!(hdr), cfg.nrot * slice_bytes; level) do s
        # inflate the files on other threads, ahead of the (sequential) deflate stream
        ahead = max(1, Threads.nthreads() - 1)
        names = [@sprintf("ve/tErn.%05d.mat.gz", i) for i in 0:cfg.nrot-1]
        pending = Task[Threads.@spawn(read_raw(names[i], slice_bytes)) for i in 1:min(ahead, length(names))]
        for i in eachindex(names)
            d = fetch(popfirst!(pending))
            nxt = i + ahead
            nxt <= length(names) && push!(pending, Threads.@spawn(read_raw(names[nxt], slice_bytes)))
            write(s, d)
        end
    end
    println("  loaded ern: $(cfg.nrot) files x $slice_bytes bytes")
    return nothing
end

"""
    write_ve_zlib(make_grid, prefix, parms_file, angles, level)

Direct output of the energy maps: `make_grid(i)` (GPU/CPU, main thread) returns the grid of orientation
`i`; a writer task deflates the grids in order into `<prefix>.ve.zlib` while the next one is computed
(the zlib stream is sequential, so one thread deflates, the grid computation overlaps it).
Also writes `<prefix>.rot.zlib`. Same files as fmap + fmap_pack_zlib, without the ve/ directory.
"""
function write_ve_zlib(make_grid, prefix::AbstractString, parms_file::AbstractString, angles, level::Integer;
                       tern::Integer = tern_bytes_from_env())
    cfg = load_pack_config(parms_file)
    cfg.nrot == length(angles) || error("parms nrot=$(cfg.nrot) but $(length(angles)) orientations in the angle file")
    write_rot_zlib(cfg, prefix, read_rot_blobs(cfg); level)
    slice_bytes = cfg.blen^3 * tern
    hdr = IOBuffer()
    write(hdr, UInt32(VE_MAGIC_PACK), UInt32(1), Int32(cfg.nrot), Int32(cfg.blen), Float64(cfg.spacing))
    ch = Channel{Any}(3)
    writer = Threads.@spawn write_zlb(prefix * ".ve.zlib", take!(hdr), cfg.nrot * slice_bytes; level) do s
        for g in ch
            length(g) == slice_bytes || error("grid has $(length(g)) bytes, expected $slice_bytes")
            write(s, g)
        end
    end
    try
        for i in 1:cfg.nrot
            put!(ch, make_grid(i))
        end
    finally
        close(ch)
    end
    wait(writer)
    return nothing
end
