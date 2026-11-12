# grid.jl - FMAP grid files (cpu/fmap_energy.c).
#
#   rot.eul:       Euler-angle list of the sample set; the rotation tables are built from it (see ../rotsys.jl)
#   fmap.ve.sp2p:  spatially compressed energy grid, one independent unit per rotation (see ../../spt/Spt.jl)
#   The energy grid is held as the SPT=1 block-sparse table (Spt.sparse1_table), never dense.

const FMAP_MAX_NROT = 40000
const FMAP_MAX_DIM  = 1024

fmap_geometry_ok(nrot, nbin, blen) =
    0 < nrot <= FMAP_MAX_NROT && 0 < nbin <= FMAP_MAX_DIM && 0 < blen <= FMAP_MAX_DIM

@inline rd_u32(h, o) = UInt32(h[o+1]) | UInt32(h[o+2]) << 8 | UInt32(h[o+3]) << 16 | UInt32(h[o+4]) << 24
@inline rd_i32(h, o) = reinterpret(Int32, rd_u32(h, o))
@inline rd_f64(h, o) = reinterpret(Float64, UInt64(rd_u32(h, o)) | UInt64(rd_u32(h, o + 4)) << 32)

# ---- headers ----

"Rotation-table header of a rot.eul: (nrot, nbin). Errors on a bad file."
function peek_rot_header(path::AbstractString)
    nrot = count(l -> !isempty(strip(l)), eachline(path))      # one sample per line, nbin from ROT_NBIN
    nbin = parse(Int, get(ENV, "ROT_NBIN", "120"))
    fmap_geometry_ok(nrot, nbin, 1) || error("FMAP: implausible nrot=$nrot nbin=$nbin in $path")
    return nrot, nbin
end

# ---- rotation tables (always read into memory) ----

# RotTables (aidx, amap, rotm) and the pair-table modes are in ../rotsys.jl.

# the contexts of a mixture share one rotation system: build it once per file
const ROT_TABLE_CACHE = Dict{Tuple{String,Int},RotTables}()

function read_rot_tables(path::AbstractString; mode::Int = amap_mode_from_env())
    key = (realpath(path), mode)
    haskey(ROT_TABLE_CACHE, key) && return ROT_TABLE_CACHE[key]
    return ROT_TABLE_CACHE[key] = read_rot_tables_file(path, mode)
end

read_rot_tables_file(path::AbstractString, mode::Int) = rot_tables_from_eul(path; mode)

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
    amap::AMapC
    rotm::Vector{Float64}
    energy::Vector{Int8}   # SPT=1 block-sparse table (Spt.sparse1_table), read-only
end

with_scales(c::FmapCtx, es, vs) =
    FmapCtx(c.nrot, c.blen, c.nbin, c.scl, c.rcut2, c.tern, Float64(es), Float64(vs),
            c.aidx, c.amap, c.rotm, c.energy)

"""
    fmap_create(rot_path, ve_path; tern) -> FmapCtx

Load one pair context (CPU engine): the rotation tables and the energy grid, a `.sp2p` file turned into the block-sparse lookup table
(SPT=1) directly from the compressed units, as on the GPU. No dense grid is made.
"""
function fmap_create(rot_path::AbstractString, ve_path::AbstractString; tern::Integer = tern_bytes_from_env())
    rt = read_rot_tables(rot_path)
    Spt.is_sp2p(ve_path) || error("FMAP: $ve_path is not a spatially compressed grid (fmap.ve.sp2p)")
    nrot_v, blen, spacing, ftern = Spt.read_sp2p_header(ve_path)
    nrot_v == rt.nrot || error("FMAP: nrot mismatch: rot=$(rt.nrot) ve=$nrot_v")
    fmap_geometry_ok(rt.nrot, rt.nbin, blen) || error("FMAP: implausible blen=$blen in $ve_path")
    ftern == tern || error("FMAP: $ve_path holds a TERN_BYTES=$ftern grid but TERN_BYTES=$tern is set")
    t0 = time()
    energy = Spt.sparse1_table(Spt.read_sp2p(ve_path))
    scl = 1.0 / spacing
    rcut = blen * 0.5 * spacing
    println("FMAP: nrot=$(rt.nrot) blen=$blen nbin=$(rt.nbin) scl=$(@sprintf("%.4f", scl)) rcut=$(@sprintf("%.4f", rcut)); " *
            "SPT=1 table $(round(length(energy) / 1e6, digits = 1)) MB in $(round(time() - t0, digits = 1)) s")
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
        a3 = amget(c.amap, Int(r1), Int(r2), c.nrot)
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
        e0, e1 = Spt.sp1_ve(c.energy, blen, c.nrot, i, j, k, a3)
        e0 == TERN_MAX && return 9999.0
        c.tern == 2 && return muladd(Float64(e0), c.elec_scale, Float64(e1) * c.vdw_scale) / TERN_SCL
        return Float64(e0) / TERN_SCL
    end
end

@inline apply_pbc(x::Float64, L::Float64) = x > L * 0.5 ? x - L : (x < -L * 0.5 ? x + L : x)

"bins -> rotation index (fmap_bins_to_rot_idx)."
@inline function bins_to_rot_idx(c::FmapCtx, b1::Float64, b2::Float64, b3::Float64)
    n = Int64(c.nbin)
    bi = wrap_idx(c_round_int(b1), n); bj = beta_idx(b2, n); bk = wrap_idx(c_round_int(b3), n)
    return Int32(@inbounds c.aidx[bi * n * n + bj * n + bk + 1])
end
