# grid.jl - FMAP grid files (fmap_zlib.c / cpu/fmap_energy.c).
#
#   rot.zlib: [16 B header: u32 magic 0x524F5446, u32, i32 nrot, i32 nbin][u64 csz][zlib(blob)]
#             blob = angle_indices[nbin^3 i16] angle_mapping[nrot^2 i16] euler[nrot*3 f64] rotm[nrot*9 f64]
#   ve.zlib:  [24 B header: u32 magic 0x46454D41, u32, i32 nrot, i32 blen, f64 spacing][u64 csz][zlib(blob)]
#             blob = energy[nrot*blen^3 * TERN_BYTES i8]
#   *.raw:    the same header followed directly by the inflated blob (no csz): what the CPU
#             engine mmaps. A .zlib ve grid is inflated once into a raw cache file shared by all
#             processes ($FMAPIQ_CACHE, else $TMPDIR, else /tmp).
using CodecZlib
using Mmap

const ROT_MAGIC    = 0x524F5446
const VE_MAGIC     = 0x46454D41
const ROT_HDR_SIZE = 16
const VE_HDR_SIZE  = 24
const FMAP_MAX_NROT = 40000
const FMAP_MAX_DIM  = 1024

fmap_geometry_ok(nrot, nbin, blen) =
    0 < nrot <= FMAP_MAX_NROT && 0 < nbin <= FMAP_MAX_DIM && 0 < blen <= FMAP_MAX_DIM

@inline rd_u32(h, o) = UInt32(h[o+1]) | UInt32(h[o+2]) << 8 | UInt32(h[o+3]) << 16 | UInt32(h[o+4]) << 24
@inline rd_i32(h, o) = reinterpret(Int32, rd_u32(h, o))
@inline rd_f64(h, o) = reinterpret(Float64, UInt64(rd_u32(h, o)) | UInt64(rd_u32(h, o + 4)) << 32)

is_zlib(path) = occursin(".zlib", path)

# ---- headers ----

"Rotation-table header of a rot.zlib / rot.raw: (nrot, nbin). Errors on a bad file."
function peek_rot_header(path::AbstractString)
    hdr = open(io -> read(io, ROT_HDR_SIZE), path, "r")
    length(hdr) == ROT_HDR_SIZE && rd_u32(hdr, 0) == ROT_MAGIC || error("FMAP: bad rot header in $path")
    nrot = Int(rd_i32(hdr, 8)); nbin = Int(rd_i32(hdr, 12))
    fmap_geometry_ok(nrot, nbin, 1) || error("FMAP: implausible nrot=$nrot nbin=$nbin in $path")
    return nrot, nbin
end

"Energy-grid header of a ve.zlib / ve.raw: (nrot, blen, spacing, header bytes)."
function peek_ve_header(path::AbstractString)
    hdr = open(io -> read(io, VE_HDR_SIZE), path, "r")
    length(hdr) == VE_HDR_SIZE && rd_u32(hdr, 0) == VE_MAGIC || error("FMAP: bad ve header in $path")
    nrot = Int(rd_i32(hdr, 8)); blen = Int(rd_i32(hdr, 12)); spacing = rd_f64(hdr, 16)
    (fmap_geometry_ok(nrot, 1, blen) && spacing > 0.0 && isfinite(spacing)) ||
        error("FMAP: implausible nrot=$nrot blen=$blen spacing=$spacing in $path")
    return nrot, blen, spacing, hdr
end

# ---- rotation tables (always read into memory) ----

struct RotTables
    nrot::Int
    nbin::Int
    aidx::Vector{Int16}    # angle_indices [nbin^3]
    amap::Vector{Int16}    # angle_mapping [nrot^2]
    rotm::Vector{Float64}  # rotation matrices [nrot*9]
end

function read_rot_tables(path::AbstractString)
    nrot, nbin = peek_rot_header(path)
    aidx = Vector{Int16}(undef, nbin^3)
    amap = Vector{Int16}(undef, nrot^2)
    skip_n = nrot * 3 * 8
    rotm = Vector{Float64}(undef, nrot * 9)
    open(path, "r") do io
        skip(io, ROT_HDR_SIZE)
        if is_zlib(path)
            csz = read(io, UInt64)
            filesize(path) >= ROT_HDR_SIZE + 8 + csz || error("FMAP: $path is truncated")
            s = ZlibDecompressorStream(io)
            try
                read!(s, aidx); read!(s, amap); read!(s, Vector{UInt8}(undef, skip_n)); read!(s, rotm)
            catch e
                e isa EOFError ? error("FMAP: $path inflates to less data than its header needs") : rethrow()
            end
            eof(s) || error("FMAP: $path inflates to more data than its header needs")
        else
            read!(io, aidx); read!(io, amap); skip(io, skip_n); read!(io, rotm)
        end
    end
    return RotTables(nrot, nbin, aidx, amap, rotm)
end

# ---- energy grid streaming (inflate) ----

"Bytes of the inflated ve blob for this grid and cell format."
ve_blob_bytes(nrot, blen, tern) = nrot * blen^3 * tern

"""
    stream_ve(f, path, nbytes; chunk=8MiB)

Inflate (or read, for a raw file) the ve blob and call `f(offset, buf::Vector{UInt8}, n)` for
each chunk, checking the blob is exactly `nbytes` long. The uncompressed grid never needs to be
in memory.
"""
function stream_ve(f, path::AbstractString, nbytes::Integer; chunk::Integer = 8 << 20)
    buf = Vector{UInt8}(undef, min(chunk, max(nbytes, 1)))
    open(path, "r") do io
        skip(io, VE_HDR_SIZE)
        src = io
        if is_zlib(path)
            csz = read(io, UInt64)
            filesize(path) >= VE_HDR_SIZE + 8 + csz || error("FMAP: $path is truncated")
            src = ZlibDecompressorStream(io)
        end
        off = 0
        while off < nbytes
            n = min(length(buf), nbytes - off)
            nr = readbytes!(src, buf, n)
            nr == n || error("FMAP: $path holds only $(off + nr) of the $nbytes bytes its header needs " *
                             "(wrong TERN_BYTES for this grid?)")
            f(off, buf, n)
            off += n
        end
        eof(src) || error("FMAP: $path holds more than the $nbytes bytes its header needs " *
                          "(wrong TERN_BYTES for this grid?)")
    end
    return nbytes
end

# ---- raw-grid cache (CPU engine) ----

function cache_dir()
    for v in ("FMAPIQ_CACHE", "TMPDIR")
        d = get(ENV, v, "")
        isempty(d) || return d
    end
    return "/tmp"
end

function cache_name(path::AbstractString, stem::AbstractString, tern::Integer)
    rp = realpath(path); st = stat(rp)
    h = hash((rp, st.size, st.mtime, tern))
    return joinpath(cache_dir(), @sprintf("fmapjl_%d_%016x_%s.raw", Int(ccall(:geteuid, Cuint, ())), h, stem))
end

"Path of a mmap-able ve raw file for `path` (inflating a .zlib into the cache the first time)."
function ve_raw_path(path::AbstractString, tern::Integer)
    is_zlib(path) || return String(path)
    nrot, blen, _, hdr = peek_ve_header(path)
    nbytes = ve_blob_bytes(nrot, blen, tern)
    raw = cache_name(path, "ve", tern)
    want = VE_HDR_SIZE + nbytes
    if isfile(raw) && !islink(raw) && filesize(raw) == want
        return raw
    end
    tmp = raw * ".tmp$(getpid())"
    println("FMAP: inflating $path -> $raw ($(round(nbytes / 1e6; digits = 1)) MB)"); try flush(stdout) catch end
    try
        open(tmp, "w") do out
            write(out, hdr)
            stream_ve(path, nbytes) do off, buf, n
                unsafe_write(out, pointer(buf), n)
            end
        end
        mv(tmp, raw; force = true)
    catch
        rm(tmp; force = true)
        rethrow()
    end
    return raw
end

# ---- CPU context ----

"One per-type-pair FMAP context for the CPU engine: shared rotation tables + this pair's energy grid."
struct FmapCtx
    nrot::Int
    blen::Int
    nbin::Int
    scl::Float64
    rcut2::Float64
    tern::Int
    elec_scale::Float64
    vdw_scale::Float64
    aidx::Vector{Int16}
    amap::Vector{Int16}
    rotm::Vector{Float64}
    energy::Vector{Int8}   # mmap'd (read-only), nrot*blen^3*tern cells
end

with_scales(c::FmapCtx, es, vs) =
    FmapCtx(c.nrot, c.blen, c.nbin, c.scl, c.rcut2, c.tern, Float64(es), Float64(vs),
            c.aidx, c.amap, c.rotm, c.energy)

"""
    fmap_create(rot_path, ve_path; tern) -> FmapCtx

Load one pair context (CPU engine). `.zlib` grids are inflated into the raw cache once and
mmap'd; `.raw` files are mmap'd directly.
"""
function fmap_create(rot_path::AbstractString, ve_path::AbstractString; tern::Integer = tern_bytes_from_env())
    rt = read_rot_tables(rot_path)
    vraw = ve_raw_path(ve_path, tern)
    nrot_v, blen, spacing, _ = peek_ve_header(vraw)
    nrot_v == rt.nrot || error("FMAP: nrot mismatch: rot=$(rt.nrot) ve=$nrot_v")
    fmap_geometry_ok(rt.nrot, rt.nbin, blen) || error("FMAP: implausible blen=$blen in $ve_path")
    nbytes = ve_blob_bytes(rt.nrot, blen, tern)
    filesize(vraw) >= VE_HDR_SIZE + nbytes ||
        error("FMAP: $vraw is truncated ($(filesize(vraw)) bytes, need $(VE_HDR_SIZE + nbytes); wrong TERN_BYTES?)")
    energy = open(vraw, "r") do io
        Mmap.mmap(io, Vector{Int8}, (nbytes,), VE_HDR_SIZE; shared = true)
    end
    scl = 1.0 / spacing
    rcut = blen * 0.5 * spacing
    println("FMAP: nrot=$(rt.nrot) blen=$blen nbin=$(rt.nbin) scl=$(@sprintf("%.4f", scl)) rcut=$(@sprintf("%.4f", rcut))")
    return FmapCtx(rt.nrot, blen, rt.nbin, scl, rcut * rcut, Int(tern), 1.0, 1.0,
                   rt.aidx, rt.amap, rt.rotm, energy)
end

# ---- hot path ----

@inline function fmap_pair_energy(c::FmapCtx, dx::Float64, dy::Float64, dz::Float64,
                                  r1::Int32, r2::Int32)
    # Fused multiply-add where gcc (-O3 -march=native) fuses the C code, in the same operand
    # order, so the CPU engine reproduces the default C build bit for bit. muladd is a hardware
    # FMA when the CPU has one (as -march=native), plain mul+add otherwise.
    d2 = muladd(dz, dz, muladd(dx, dx, dy * dy))
    d2 > c.rcut2 && return 0.0
    @inbounds begin
        a3 = Int(c.amap[Int(r1) * c.nrot + Int(r2) + 1])
        b = Int(r1) * 9
        R = c.rotm
        rx = muladd(R[b+3], dz, muladd(R[b+1], dx, R[b+2] * dy))
        ry = muladd(R[b+6], dz, muladd(R[b+4], dx, R[b+5] * dy))
        rz = muladd(R[b+9], dz, muladd(R[b+7], dx, R[b+8] * dy))
        blen = c.blen
        h = Float64(blen ÷ 2)
        i = unsafe_trunc(Int, muladd(rx, c.scl, h) + 0.5)
        j = unsafe_trunc(Int, muladd(ry, c.scl, h) + 0.5)
        k = unsafe_trunc(Int, muladd(rz, c.scl, h) + 0.5)
        (i < 0 || i >= blen || j < 0 || j >= blen || k < 0 || k >= blen) && return 0.0
        off = a3 * blen^3 + i * blen^2 + j * blen + k
        if c.tern == 2
            e0 = c.energy[2off+1]
            e0 == TERN_MAX && return 9999.0
            return muladd(Float64(e0), c.elec_scale, Float64(c.energy[2off+2]) * c.vdw_scale) / TERN_SCL
        else
            e = c.energy[off+1]
            e == TERN_MAX && return 9999.0
            return Float64(e) / TERN_SCL
        end
    end
end

@inline apply_pbc(x::Float64, L::Float64) = x > L * 0.5 ? x - L : (x < -L * 0.5 ? x + L : x)

"bins -> rotation index (fmap_bins_to_rot_idx)."
@inline function bins_to_rot_idx(c::FmapCtx, b1::Float64, b2::Float64, b3::Float64)
    n = Int64(c.nbin)
    bi = wrap_idx(c_round_int(b1), n); bj = wrap_idx(c_round_int(b2), n); bk = wrap_idx(c_round_int(b3), n)
    return Int32(@inbounds c.aidx[bi * n * n + bj * n + bk + 1])
end
