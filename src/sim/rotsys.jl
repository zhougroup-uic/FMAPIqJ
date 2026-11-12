# rotsys.jl - the rotation system of a simulation (CPU and GPU engines): the three tables
#
#   angle_indices  (angijk, nbin^3)  nearest sample of every (alpha, cos beta, gamma) bin
#   angle_mapping  (angij2k, nrot^2) nearest sample of R_i' R_j  (A2-A1), read at every pair evaluation
#   rotm           (nrot x 9)        inverse rotation matrices
#
# and how angle_mapping is held. `AMAP_FOLD` (alias `AMAP_MODE`) selects, for the same sample list:
#   0  plain 16-bit table (default), built from the Euler angles (symmetry-folded search, then expanded in memory); an old
#      rot.zlib of version 1 stores it and it is then read as it is.
#   1  the symmetry-folded, bit-packed table (about 7x smaller: 5.9 MB instead of 42.5 MB at 4608 samples) read in place,
#      P[i,j] = P[rep(i), s_g(j)] (4 loads). Same entries as mode 0 built from the Euler angles, so identical results.
#   2  no pair table at all: the index is looked up at every pair from q = R_i' R_j = M_i M_j' and the angijk table.
#      Approximate (4608 samples: same sample as the exact table for 88.6 % of the pairs, mean excess angle 0.09 deg,
#      worst 8 deg); for sample lists whose n^2 table does not fit.
# Mode 0/1 built from the Euler angles differ from the stored (exhaustive-search) table only at near-ties: ~0.6 % of the
# entries at 4608 samples, the returned sample is at most ~3e-4 deg farther than the nearest one.
# The Euler angles come from the rot.eul of the sample set (the pipeline copies it into the work directory and lists it in
# fmap.paths; any file that does not start with the rot.zlib magic is read as a rot.eul). rot.zlib files of earlier versions
# (1: with angle_mapping, 2: without) are still read; nothing writes them any more. The Rotij package (sim/rotij) does the searching.

include("rotij/Rotij.jl")

"Pair-table mode from the environment: 0 plain, 1 folded, 2 lookup (see above)."
function amap_mode_from_env()
    m = parse(Int, get(ENV, "AMAP_MODE", get(ENV, "AMAP_FOLD", "0")))
    m in (0, 1, 2) || error("AMAP_FOLD must be 0 (plain table), 1 (folded table) or 2 (lookup), got $m")
    return m
end

# angle_mapping[i*nrot + j]: plain table, folded table or no table (lookup); one concrete struct for CPU and GPU.
struct AMapC
    fold::Bool
    look::Bool
    dense::Vector{Int16}
    rrow::Vector{Int32}     # compact row (0-based) of the representative of i
    rg::Vector{UInt8}       # group element (1-based) with s_g(i) = representative
    PL::Vector{Int32}       # n x |G| (column-major), s_g(j) 0-based
    data::Vector{UInt64}    # bit-packed entries, row-major (representative row, column)
    n::Int
    nc::Int
    bits::Int
    rotm::Vector{Float64}   # look: inverse rotation matrices (9 per sample)
    aidx::Vector{Int16}     # look: angle_indices
    nbin::Int
end
AMapC(v::Vector{Int16}) = AMapC(false, false, v, Int32[], UInt8[], Int32[], UInt64[], 0, 0, 0, Float64[], Int16[], 0)

Base.:(==)(a::AMapC, b::AMapC) =
    a.fold == b.fold && a.look == b.look && a.dense == b.dense && a.rrow == b.rrow && a.rg == b.rg && a.PL == b.PL &&
    a.data == b.data && a.n == b.n && a.nc == b.nc && a.bits == b.bits && a.nbin == b.nbin

"Bytes the pair table occupies in memory (what a simulation holds beyond the bin table)."
amap_bytes(a::AMapC) = a.look ? 0 : a.fold ? sizeof(a.data) + sizeof(a.rrow) + sizeof(a.rg) + sizeof(a.PL) : sizeof(a.dense)

"Entry (0-based i, j) of a folded table given as arrays; shared by the CPU and the GPU kernels."
@inline function foldget(rrow, rg, PL, data, n::Int, nc::Int, bits::Int, i::Int, j::Int)
    gi = Int(@inbounds rg[i+1]); r = Int(@inbounds rrow[i+1])
    c = Int(@inbounds PL[j + 1 + (gi - 1) * n])
    pos = (r * nc + c) * bits
    wi = (pos >>> 6) + 1; off = pos & 63
    v = (@inbounds data[wi]) >>> off
    off + bits > 64 && (v |= (@inbounds data[wi+1]) << (64 - off))
    return Int(v & ((UInt64(1) << bits) - UInt64(1)))
end

# Approximate A2-A1 index without any pair table (port of Rotij.bin_lookup): q = R_i' R_j = M_i M_j' (M = rotm, row-major),
# Euler angles of q (alpha = atan2(q13, -q23), gamma = atan2(q31, q32), gamma = 0 at the poles), bins round / floor like the
# run-time orientation index (the beta bin centres of the angijk table are half a bin offset), then the angijk entry.
@inline function lookup_index(rotm, aidx, nd::Int, i::Int, j::Int)
    T = eltype(rotm)
    bi = 9 * i; bj = 9 * j
    @inline d3(r, c) = (@inbounds rotm[bi+3r+1] * rotm[bj+3c+1] + rotm[bi+3r+2] * rotm[bj+3c+2] + rotm[bi+3r+3] * rotm[bj+3c+3])
    q3 = d3(0, 2); q6 = d3(1, 2); q9 = d3(2, 2)
    if sqrt(q3 * q3 + q6 * q6) < T(1e-6)
        a = atan(d3(1, 0), d3(0, 0)); g = zero(T)
    else
        a = atan(q3, -q6); g = atan(d3(2, 0), d3(2, 1))
    end
    twopi = T(2) * T(pi)
    i1 = unsafe_trunc(Int, round((a + T(pi)) * T(nd) / twopi)); i1 = i1 >= nd ? i1 - nd : (i1 < 0 ? i1 + nd : i1)
    i3 = unsafe_trunc(Int, round((g + T(pi)) * T(nd) / twopi)); i3 = i3 >= nd ? i3 - nd : (i3 < 0 ? i3 + nd : i3)
    i2 = clamp(unsafe_trunc(Int, floor((one(T) - clamp(q9, -one(T), one(T))) * T(nd) / T(2))), 0, nd - 1)
    return Int(@inbounds aidx[(i1 * nd + i2) * nd + i3 + 1])
end

"angle_mapping[i, j] (0-based) from whichever form the table is held in."
@inline amget(a::AMapC, i::Int, j::Int, nrot::Int) =
    a.look ? lookup_index(a.rotm, a.aidx, a.nbin, i, j) :
    (a.fold ? foldget(a.rrow, a.rg, a.PL, a.data, a.n, a.nc, a.bits, i, j) : Int(@inbounds a.dense[i * nrot + j + 1]))
@inline amget(a::AbstractVector, i::Int, j::Int, nrot::Int) = Int(@inbounds a[i * nrot + j + 1])

struct RotTables
    nrot::Int
    nbin::Int
    aidx::Vector{Int16}    # angle_indices [nbin^3]
    amap::AMapC            # angle_mapping [nrot^2]
    rotm::Vector{Float64}  # inverse rotation matrices [nrot*9]
end

"""
    rot_tables(nrot, nbin, aidx, dense, eul, rotm; mode, what) -> RotTables

Assemble the rotation system. `aidx`, `dense` (the stored plain pair table), `eul` (3 x nrot Euler angles) and `rotm`
may each be `nothing`; whatever is missing is built from `eul` with Rotij. `dense` is used as it is in mode 0.
"""
function rot_tables(nrot::Int, nbin::Int, aidx, dense, eul, rotm; mode::Int = amap_mode_from_env(), what::AbstractString = "")
    t0 = time()
    nrot <= typemax(Int16) || error("FMAP: $nrot samples do not fit the 16-bit tables")
    S = nothing
    function needS()
        S === nothing || return S
        eul === nothing && error("FMAP: $what has neither the pair table nor the Euler angles")
        S = Rotij.RotSamples(eul)
        return S
    end
    if aidx === nothing
        A = Rotij.angijk(needS(); nbin = nbin)
        aidx = Vector{Int16}(undef, nbin^3)
        @inbounds for ai in 0:nbin-1, aj in 0:nbin-1, ak in 0:nbin-1     # file order (ai, aj, ak), ak fastest
            aidx[(ai * nbin + aj) * nbin + ak + 1] = Int16(A[ai+1, aj+1, ak+1])
        end
    end
    rotm === nothing && (rotm = vec(Rotij.rotm(needS())))
    how = ""
    if mode == 0 && dense !== nothing
        amap = AMapC(dense); how = "plain table as stored"
    elseif mode == 2
        amap = AMapC(false, true, Int16[], Int32[], UInt8[], Int32[], UInt64[], 0, 0, 0, rotm, aidx, nbin); how = "no pair table (lookup)"
    else
        T = Rotij.fold_angij(needS(); inv = true)
        if mode == 1
            amap = AMapC(true, false, Int16[], T.rrow, T.rg, vec(T.PL), T.data, nrot, T.nc, T.bits, Float64[], Int16[], 0)
            how = "folded table used directly"
        else
            d = Vector{Int16}(undef, nrot * nrot)
            Threads.@threads for i in 0:nrot-1
                @inbounds for j in 0:nrot-1
                    d[i * nrot + j + 1] = Int16(T[i+1, j+1])
                end
            end
            amap = AMapC(d); how = "folded search, expanded to the plain table"
        end
        how *= @sprintf(" (folded %.2f MB, plain %.1f MB)", Rotij.storage_bytes(T) / 1e6, 2 * nrot^2 / 1e6)
    end
    time() - t0 > 0.05 && println(@sprintf("FMAP: rotation system%s: nrot=%d nbin=%d, %s, %.1f s",
                                           isempty(what) ? "" : " from " * what, nrot, nbin, how, time() - t0))
    return RotTables(nrot, nbin, aidx, amap, rotm)
end

"Rotation system from a rot.eul (Euler ZXZ angles in rad, one sample per line)."
function rot_tables_from_eul(path::AbstractString; nbin::Int = parse(Int, get(ENV, "ROT_NBIN", "120")), mode::Int = amap_mode_from_env())
    S = Rotij.load_eul(path)
    return rot_tables(length(S), nbin, nothing, nothing, S.eul, nothing; mode, what = String(path))
end

const ROT_MAGIC_RS = 0x524F5446      # "ROTF"

"True if the file starts with the rot.zlib magic (otherwise it is taken as a rot.eul)."
is_rot_zlib_file(path::AbstractString) =
    open(io -> (h = read(io, 4); length(h) == 4 && (UInt32(h[1]) | UInt32(h[2]) << 8 | UInt32(h[3]) << 16 | UInt32(h[4]) << 24) == ROT_MAGIC_RS), path, "r")

"""
    rot_tables_from_blob(version, nrot, nbin, blob; mode, what)

`blob` = inflated rot.zlib body: angle_indices, [angle_mapping, version 1 only], euler (nrot*3 f64), rotm (nrot*9 f64).
"""
function rot_tables_from_blob(version::Integer, nrot::Int, nbin::Int, blob::AbstractVector{UInt8}; mode::Int = amap_mode_from_env(), what::AbstractString = "")
    off = 0
    take(::Type{T}, n) where {T} = (v = Vector{T}(reinterpret(T, blob[off+1:off+n*sizeof(T)])); off += n * sizeof(T); v)
    aidx = take(Int16, nbin^3)
    version == 1 || version == 2 || error("FMAP: $what: unknown rot.zlib version $version")
    dense = version == 1 ? take(Int16, nrot^2) : nothing
    eul = reshape(take(Float64, nrot * 3), 3, nrot)
    rotm = take(Float64, nrot * 9)
    off == length(blob) || error("FMAP: $what: rot blob has $(length(blob)) bytes, expected $off")
    return rot_tables(nrot, nbin, aidx, dense, eul, rotm; mode, what)
end
