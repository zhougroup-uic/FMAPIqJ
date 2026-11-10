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

# ---- 10-point second-order spreading (Map2GrdL2nd), closed form ----
#
# Qin & Zhou, JCTC 2014, 10, 2824 (Fig. 3, Eq. 12). The 10 weights reproduce the charge, its first and its second moments exactly,
# so they are polynomials of the offset of the charge from its nearest grid point c. Per axis let s = +1 if the charge lies on the
# +side of c (-1 otherwise) and u, v, w in [0, 1/2] its distances to c along x, y, z. Weights (y and z by permuting u, v, w):
#   c                       1 - u^2 - v^2 - w^2 + uv + uw + vw
#   c - s_x ex   (far)      u(u-1)/2
#   c + s_x ex   (near)     u(u+1)/2 - u(v+w)
#   c + s_y ey + s_z ez     vw          (and the two other face diagonals)
# (this replaces the table data/lattice.txt of 1000 sub-grid positions, which was a 0.1-grid sampling of the same formulas)

"Nearest grid point offset (0 or 1 from floor), direction toward the charge (+1/-1) and the distance u of a fractional coordinate."
@inline function l2nd_axis(f::Float64)
    near = f < 0.5
    return near ? 0 : 1, near ? 1 : -1, near ? f : 1.0 - f
end

"Periodic neighbours of a grid index i0 already in 0:l-1: (toward the charge, away from it). A compare, not a mod."
@inline function l2nd_nbrs(i0::Int, s::Int, l::Int)
    up = i0 + 1 == l ? 0 : i0 + 1
    dn = i0 == 0 ? l - 1 : i0 - 1
    return s > 0 ? (up, dn) : (dn, up)
end

function map_L2nd!(grd::Array{Float64,3}, atoms::Vector{Atom}, coeffs::Vector{Float64})
    l = size(grd, 1)
    @inbounds for a in eachindex(atoms)
        at = atoms[a]; coeff = coeffs[a]
        lx = floor(at.x); ly = floor(at.y); lz = floor(at.z)
        cx, sx, u = l2nd_axis(at.x - lx); cy, sy, v = l2nd_axis(at.y - ly); cz, sz, w = l2nd_axis(at.z - lz)
        # wrap the nearest grid point once per axis; its two neighbours follow by a compare
        x0 = pbc(Int(lx) + cx, l); y0 = pbc(Int(ly) + cy, l); z0 = pbc(Int(lz) + cz, l)
        xn, xf = l2nd_nbrs(x0, sx, l); yn, yf = l2nd_nbrs(y0, sy, l); zn, zf = l2nd_nbrs(z0, sz, l)
        uv = u * v; uw = u * w; vw = v * w
        grd[z0+1, y0+1, x0+1] += coeff * (1.0 - u * u - v * v - w * w + uv + uw + vw)
        grd[z0+1, y0+1, xn+1] += coeff * (u * (u + 1.0) / 2 - uv - uw); grd[z0+1, y0+1, xf+1] += coeff * (u * (u - 1.0) / 2)
        grd[z0+1, yn+1, x0+1] += coeff * (v * (v + 1.0) / 2 - uv - vw); grd[z0+1, yf+1, x0+1] += coeff * (v * (v - 1.0) / 2)
        grd[zn+1, y0+1, x0+1] += coeff * (w * (w + 1.0) / 2 - uw - vw); grd[zf+1, y0+1, x0+1] += coeff * (w * (w - 1.0) / 2)
        grd[zn+1, yn+1, x0+1] += coeff * vw; grd[zn+1, y0+1, xn+1] += coeff * uw; grd[z0+1, yn+1, xn+1] += coeff * uv
    end
    return grd
end
