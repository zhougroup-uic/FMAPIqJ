# fmap_zlib_loader.jl - reads fmap.rot.zlib / fmap.ve.zlib directly: the ve grid is inflated in
# chunks straight into device memory (CodecZlib.jl), no intermediate raw file.
#
# Format (as written by fmap/fmap_pack_zlib and read by the C engine in reference/maeppi):
#   rot.zlib: [16-byte header][8-byte csz LE u64][zlib(rot_blob)]
#     header: u32 magic(0x524F5446) | u32 (unused) | i32 nrot | i32 nbin
#     rot_blob: angle_indices[nbin^3 i16] + angle_mapping[nrot^2 i16]
#               + euler_angles[nrot*3 f64] (skipped, unused downstream)
#               + rotation_matrices[nrot*9 f64]
#   ve.zlib:  [24-byte header][8-byte csz LE u64][zlib(ve_blob)]
#     header: u32 magic(0x46454D41) | u32 (unused) | i32 nrot | i32 blen
#             | f64 spacing  (scl = 1/spacing, rcut2 = (blen*0.5*spacing)^2)
#     ve_blob: energy_data[nrot*blen^3*TERN_BYTES i8]  (2: elec,vdw interleaved per cell; 1: summed)

using CodecZlib
using CUDA

const ROT_MAGIC = 0x524F5446
const VE_MAGIC   = 0x46454D41

function read_header_csz_blob(io::IO, hdr_size::Int)
    hdr = read(io, hdr_size)
    csz = read(io, UInt64)
    compressed = read(io, Int(csz))
    return hdr, transcode(ZlibDecompressor, compressed)
end

@inline rd_u32(h, off::Int) = UInt32(h[off+1]) | (UInt32(h[off+2])<<8) | (UInt32(h[off+3])<<16) | (UInt32(h[off+4])<<24)
@inline rd_i32(h, off::Int) = reinterpret(Int32, rd_u32(h, off))
@inline rd_f64(h, off::Int) = reinterpret(Float64, (UInt64(rd_u32(h,off)) | (UInt64(rd_u32(h,off+4))<<32)))

function load_fmap_zlib(rot_zlib::String, ve_zlib::String)
    rot_hdr, rot_blob = open(io -> read_header_csz_blob(io, 16), rot_zlib, "r")
    rd_u32(rot_hdr, 0) == ROT_MAGIC || error("bad rot magic in $rot_zlib")
    nrot = rd_i32(rot_hdr, 8)
    nbin = rd_i32(rot_hdr, 12)

    ve_hdr, ve_blob = open(io -> read_header_csz_blob(io, 24), ve_zlib, "r")
    rd_u32(ve_hdr, 0) == VE_MAGIC || error("bad ve magic in $ve_zlib")
    rd_i32(ve_hdr, 8) == nrot || error("nrot mismatch between $rot_zlib and $ve_zlib")
    blen = rd_i32(ve_hdr, 12)
    spacing = rd_f64(ve_hdr, 16)
    scl = 1.0 / spacing

    nbin3 = Int(nbin)^3
    nrot2 = Int(nrot)^2
    ijk_sz  = nbin3 * 2       # sizeof(tAng)=2 (Int16)
    ij2k_sz = nrot2 * 2
    rot_sz  = Int(nrot) * 3 * 8   # skipped euler_angles, still need the offset
    off = 0
    angle_indices = reinterpret(Int16, rot_blob[off+1:off+ijk_sz]); off += ijk_sz
    angle_mapping = reinterpret(Int16, rot_blob[off+1:off+ij2k_sz]); off += ij2k_sz
    off += rot_sz   # skip euler_angles (unused downstream, same as fmap_zlib.c's own callers)
    rotation_matrices = reinterpret(Float64, rot_blob[off+1:off+Int(nrot)*9*8])

    energy_data = reinterpret(Int8, ve_blob)

    return GridData(Int32(nrot), Int32(blen), Int32(nbin), scl,
                     Vector{Int16}(angle_indices), Vector{Int16}(angle_mapping),
                     Vector{Float64}(rotation_matrices), Vector{Int8}(energy_data))
end

# load_fmap_paths - reads a production fmap.paths file (main_gemc_gpu.cu's
# own convention: 2*n_ctx lines, rot.zlib/ve.zlib alternating, one pair per
# sorted type index (i,j), i<=j - matches simulation.c/simulation_gemc.c's
# get_ctx / mc_core.jl's ctx_index ordering) and builds a GridSetData with
# one energy grid per context. nrot/nbin/scl and the rotation tables
# (angle_indices/angle_mapping/rotation_matrices) are verified identical
# across all contexts and stored once - angular resolution depends only on
# ang.dat (shared by construction) and scl=1/spacing (a fixed constant), so
# a mismatch there is a real error, not an expected case. blen, however, IS
# allowed to differ per context: it's proportional to molecule size, so
# AA/AB/BB legitimately need different grid extents once each is generated
# from its own PQR file(s) - stored as a per-context Vector.
function load_fmap_paths(paths_file::String, n_types::Int32)
    n_ctx = Int(n_types) * (Int(n_types) + 1) ÷ 2
    lines = readlines(paths_file)
    length(lines) >= 2 * n_ctx || error("$paths_file needs $(2*n_ctx) lines (rot/ve x $n_ctx contexts), got $(length(lines))")

    grids = Vector{GridData}(undef, n_ctx)
    for k in 1:n_ctx
        rot_path = String(strip(lines[2k-1]))
        ve_path  = String(strip(lines[2k]))
        grids[k] = load_fmap_zlib(rot_path, ve_path)
    end

    g0 = grids[1]
    for k in 2:n_ctx
        gk = grids[k]
        (gk.nrot, gk.nbin, gk.scl) == (g0.nrot, g0.nbin, g0.scl) ||
            error("$paths_file: context $k angular resolution (nrot=$(gk.nrot) nbin=$(gk.nbin) scl=$(gk.scl)) " *
                  "differs from context 1 (nrot=$(g0.nrot) nbin=$(g0.nbin) scl=$(g0.scl)) - these must be shared")
        gk.angle_indices == g0.angle_indices || error("$paths_file: context $k angle_indices differs from context 1")
        gk.angle_mapping == g0.angle_mapping || error("$paths_file: context $k angle_mapping differs from context 1")
        gk.rotation_matrices == g0.rotation_matrices || error("$paths_file: context $k rotation_matrices differs from context 1")
    end

    return GridSetData(g0.nrot, Int32[g.blen for g in grids], g0.nbin, g0.scl,
                        g0.angle_indices, g0.angle_mapping, g0.rotation_matrices,
                        [g.energy_data for g in grids], n_types)
end

# ============================================================================
# GPU-ONLY LOADING (like main_gemc_gpu.cu's fmap_inflate_ve + upload callback)
# ============================================================================
# The ve grid (nrot*blen^3*TERN_BYTES bytes - several GB for the real production grids,
# x n_ctx for a mixture) is inflated in chunks straight into a device array:
# the uncompressed grid never exists in host memory, and neither does the
# whole compressed file (the file is streamed, not read(io, csz)'d). Only the
# small rotation tables stay on the host (needed to map bins -> rot index).
function stream_ve_to_device(ve_zlib::String, nrot::Int, blen::Int; chunk::Int=8 << 20)
    nbytes = nrot * blen^3 * TERN_BYTES   # tErn: (elec,vdw) = 2 bytes/cell, or 1 summed
    d = CuArray{Int8}(undef, nbytes)
    buf = Vector{Int8}(undef, chunk)
    open(ve_zlib, "r") do io
        skip(io, 24)      # header
        read(io, UInt64)  # csz; the rest of the file is the zlib stream
        s = ZlibDecompressorStream(io)
        off = 0
        while off < nbytes
            n = min(chunk, nbytes - off)
            GC.@preserve buf unsafe_read(s, Ptr{UInt8}(pointer(buf)), n)
            copyto!(d, off + 1, buf, 1, n)
            off += n
        end
        eof(s) || error("$ve_zlib: more data than nrot*blen^3*TERN_BYTES = $nbytes bytes")
    end
    return d
end

# Header-only read of a ve.zlib: (nrot, blen, scl).
function read_ve_header(ve_zlib::String)
    hdr = open(io -> read(io, 24), ve_zlib, "r")
    rd_u32(hdr, 0) == VE_MAGIC || error("bad ve magic in $ve_zlib")
    return Int(rd_i32(hdr, 8)), Int(rd_i32(hdr, 12)), 1.0 / rd_f64(hdr, 16)
end

# Counterpart of load_fmap_paths whose energy grids are device-resident
# CuVectors, streamed one context at a time.
function load_fmap_paths_gpu(paths_file::String, n_types::Int32)
    n_ctx = Int(n_types) * (Int(n_types) + 1) ÷ 2
    lines = readlines(paths_file)
    length(lines) >= 2 * n_ctx || error("$paths_file needs $(2*n_ctx) lines (rot/ve x $n_ctx contexts), got $(length(lines))")

    rot = nothing
    d_ctxs = CuVector{Int8}[]
    blens = Int32[]
    for k in 1:n_ctx
        rot_path = String(strip(lines[2k-1])); ve_path = String(strip(lines[2k]))
        rot_hdr, rot_blob = open(io -> read_header_csz_blob(io, 16), rot_path, "r")
        rd_u32(rot_hdr, 0) == ROT_MAGIC || error("bad rot magic in $rot_path")
        nrot = Int(rd_i32(rot_hdr, 8)); nbin = Int(rd_i32(rot_hdr, 12))
        ve_nrot, blen, scl = read_ve_header(ve_path)
        ve_nrot == nrot || error("nrot mismatch between $rot_path and $ve_path")

        nbin3 = nbin^3; nrot2 = nrot^2
        off = 0
        aidx = Vector{Int16}(reinterpret(Int16, rot_blob[off+1:off+nbin3*2])); off += nbin3*2
        amap = Vector{Int16}(reinterpret(Int16, rot_blob[off+1:off+nrot2*2])); off += nrot2*2
        off += nrot * 3 * 8   # euler angles, unused
        rotm = Vector{Float64}(reinterpret(Float64, rot_blob[off+1:off+nrot*9*8]))

        if rot === nothing
            rot = (nrot, nbin, scl, aidx, amap, rotm)
        else
            (nrot, nbin, scl) == rot[1:3] ||
                error("$paths_file: context $k angular resolution (nrot=$nrot nbin=$nbin scl=$scl) differs from context 1 - these must be shared")
            (aidx == rot[4] && amap == rot[5] && rotm == rot[6]) ||
                error("$paths_file: context $k rotation tables differ from context 1")
        end
        push!(blens, Int32(blen))
        push!(d_ctxs, stream_ve_to_device(ve_path, nrot, blen))
        println("  ctx $k: $(nrot * blen^3 * TERN_BYTES / 1e6) MB streamed to device"); flush(stdout)
    end
    CUDA.synchronize()
    nrot, nbin, scl, aidx, amap, rotm = rot
    return GridSetData(Int32(nrot), blens, Int32(nbin), scl, aidx, amap, rotm, d_ctxs, n_types)
end
