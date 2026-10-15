# mapping.jl - atom -> grid mapping (ext/cpu/map.c). Grids are Julia arrays of shape
# (l, l, l) indexed [k, j, i] (k fastest), the same memory layout as the C `grd[i][j][k]`.
#
# The spherical mapping (crowders) is parallelised over output x-planes, so no two threads write
# the same cell and the sums are deterministic (the C code used OpenMP atomics, whose order varies).

const FLT_MAX_EXP = 128.0       # float.h FLT_MAX_EXP, used as the excluded-volume offset / exp cutoff

# ---- radial functions (ext/rdf.h) ----

@inline rdf_vol(r2, rc2) = r2 > rc2 ? 0.0 : 1.0
@inline function rdf_debye(r2, kappa)
    r = sqrt(r2)
    return exp(-r * kappa) / r
end
@inline function rdf_vdw(r2, n)
    r6 = r2 * r2 * r2
    return n < 4.0 ? 1.0 / r6 : 1.0 / (r6 * r6)
end

@inline pbc(x::Int, mx::Int) = mod(x, mx)
@inline signI(x::Float64) = x < 0.0 ? -1 : 1

# ---- lattice weights for the 2nd-order charge spreading (data/lattice.txt) ----

struct Lattice
    ind::Matrix{Int32}       # (10, 1000)
    wght::Matrix{Float64}    # (10, 1000)
end

function load_lattice(path::AbstractString = joinpath(data_dir(), "lattice.txt"))
    isfile(path) || error("Cannot find lattice.txt ($path); set FMAPB2_DATA_PATH")
    ind = zeros(Int32, 10, 1000); wght = zeros(Float64, 10, 1000)
    toks = split(read(path, String))
    length(toks) >= 20000 || @warn "lattice.txt has fewer than 1000 entries" n = length(toks) ÷ 20
    for p in 1:min(1000, length(toks) ÷ 20), q in 1:10
        ind[q, p] = parse(Int32, toks[20(p-1)+2q-1])
        wght[q, p] = parse(Float64, toks[20(p-1)+2q])
    end
    return Lattice(ind, wght)
end

@inline scl_of(s::Float64, a::Int) = s
@inline scl_of(s::Vector{Float64}, a::Int) = @inbounds s[a]

# ---- spherical mapping of the crowder atoms: grd[pz,py,px] += coeff * rdf(r^2, scl) ----

"""
    map_R!(grd, atoms, coeffs, rups, rlow, dx, scl, rdf)

Add, for every atom, `coeff * rdf(r2, scl)` to all cells within `rup` (Angstrom) of it (r2 in
Angstrom^2, clamped below at `rlow`). Atom positions are in grid units.
"""
function map_R!(grd::Array{Float64,3}, atoms::Vector{Atom}, coeffs::Vector{Float64}, rups::Vector{Float64},
                rlow::Float64, dx::Float64, scl, rdf::F) where {F}
    l = size(grd, 1)
    rlow2 = rlow * rlow
    na = length(atoms)
    imin = Vector{Int}(undef, na); imax = similar(imin)
    jmin = similar(imin); jmax = similar(imin); kmin = similar(imin); kmax = similar(imin)
    for a in 1:na
        s = rups[a] / dx
        imin[a] = floor(Int, atoms[a].x - s); imax[a] = ceil(Int, atoms[a].x + s)
        jmin[a] = floor(Int, atoms[a].y - s); jmax[a] = ceil(Int, atoms[a].y + s)
        kmin[a] = floor(Int, atoms[a].z - s); kmax[a] = ceil(Int, atoms[a].z + s)
    end
    Threads.@threads :static for px in 0:l-1
        @inbounds for a in 1:na
            at = atoms[a]
            rup2 = rups[a] * rups[a]
            coeff = coeffs[a]
            i0 = imin[a] + mod(px - imin[a], l)
            for i in i0:l:imax[a]
                x = (i - at.x) * dx
                x2 = x * x
                x2 > rup2 && continue
                for j in jmin[a]:jmax[a]
                    y = (j - at.y) * dx
                    xy2 = x2 + y * y
                    xy2 > rup2 && continue
                    py = pbc(j, l)
                    for k in kmin[a]:kmax[a]
                        z = (k - at.z) * dx
                        r2 = xy2 + z * z
                        r2 > rup2 && continue
                        pz = pbc(k, l)
                        sc = scl_of(scl, a)
                        var = coeff * (r2 > rlow2 ? rdf(r2, sc) : rdf(rlow2, sc))
                        grd[pz+1, py+1, px+1] += var
                    end
                end
            end
        end
    end
    return grd
end

# ---- trilinear mapping of the probe atoms (Map2GrdL) ----

function map_L!(grd::Array{Float64,3}, atoms::Vector{Atom}, coeffs::Vector{Float64})
    l = size(grd, 1)
    @inbounds for a in eachindex(atoms)
        at = atoms[a]; coeff = coeffs[a]
        fa, ia = modf(at.x); fb, ib = modf(at.y); fc, ic = modf(at.z)
        px = pbc(trunc(Int, ia), l); py = pbc(trunc(Int, ib), l); pz = pbc(trunc(Int, ic), l)
        pxn = pbc(signI(fa) + px, l); pyn = pbc(signI(fb) + py, l); pzn = pbc(signI(fc) + pz, l)
        av = abs(fa); bv = abs(fb); cv = abs(fc)
        grd[pz+1, py+1, px+1] += coeff * (1.0 - av) * (1.0 - bv) * (1.0 - cv)
        grd[pzn+1, py+1, px+1] += coeff * (1.0 - av) * (1.0 - bv) * cv
        grd[pz+1, pyn+1, px+1] += coeff * (1.0 - av) * bv * (1.0 - cv)
        grd[pzn+1, pyn+1, px+1] += coeff * (1.0 - av) * bv * cv
        grd[pz+1, py+1, pxn+1] += coeff * av * (1.0 - bv) * (1.0 - cv)
        grd[pzn+1, py+1, pxn+1] += coeff * av * (1.0 - bv) * cv
        grd[pz+1, pyn+1, pxn+1] += coeff * av * bv * (1.0 - cv)
        grd[pzn+1, pyn+1, pxn+1] += coeff * av * bv * cv
    end
    return grd
end

# ---- 10-point lattice spreading (Map2GrdL2nd) ----

function map_L2nd!(grd::Array{Float64,3}, atoms::Vector{Atom}, coeffs::Vector{Float64}, lat::Lattice)
    l = size(grd, 1)
    @inbounds for a in eachindex(atoms)
        at = atoms[a]; coeff = coeffs[a]
        lx = floor(at.x); ly = floor(at.y); lz = floor(at.z)
        fx = at.x - lx; fy = at.y - ly; fz = at.z - lz
        pos = unsafe_trunc(Int, fx * 10.0) * 100 + unsafe_trunc(Int, fy * 10.0) * 10 + unsafe_trunc(Int, fz * 10.0)
        for q in 1:10
            n = Int(lat.ind[q, pos+1])
            i0 = n ÷ 16; j0 = (n - i0 * 16) ÷ 4; k0 = n - i0 * 16 - j0 * 4
            px = pbc(Int(lx) - 1 + i0, l); py = pbc(Int(ly) - 1 + j0, l); pz = pbc(Int(lz) - 1 + k0, l)
            grd[pz+1, py+1, px+1] += coeff * lat.wght[q, pos+1]
        end
    end
    return grd
end
