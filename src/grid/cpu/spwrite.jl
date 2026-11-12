# spwrite.jl - direct output of the energy maps as the spatially compressed per-rotation table (SPT=2p, src/spt/Spt.jl): <prefix>.ve.sp2p.

const VE_FILE_SUFFIX = ".ve.sp2p"

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
    isfile(joinpath(cand, "rot.eul")) || error("cannot find $cand/rot.eul (nrot=$nrot); only the sets under data/ are supported")
    return PackConfig(nrot, blen, 120, spacing, cand)
end

"""
    write_ve_sp2p(make_grid, prefix, parms_file, angles; tern)

`make_grid(i)` (GPU/CPU, main thread) returns the grid of orientation `i`; a writer task compresses the grids in order into
`<prefix>.ve.sp2p` while the next one is computed.
"""
function write_ve_sp2p(make_grid, prefix::AbstractString, parms_file::AbstractString, angles; tern::Integer = tern_bytes_from_env())
    cfg = load_pack_config(parms_file)
    cfg.nrot == length(angles) || error("parms nrot=$(cfg.nrot) but $(length(angles)) orientations in the angle file")
    slice_bytes = cfg.blen^3 * tern
    w = Spt.sp2p_open(prefix * VE_FILE_SUFFIX, cfg.nrot, cfg.blen, cfg.spacing; tern)
    ch = Channel{Any}(2)
    writer = Threads.@spawn begin
        for g in ch
            length(g) == slice_bytes || error("grid has $(length(g)) bytes, expected $slice_bytes")
            Spt.sp2p_add!(w, g)
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
    Spt.sp2p_close!(w)
    return nothing
end
