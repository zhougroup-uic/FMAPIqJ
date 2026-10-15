# softgrd.jl - one orientation's energy grid (ext/cpu/grd.c, score.c, fft.c).
#
# For each of vol / ele / vdw / v12 the crowder grid is built once (spherical mapping), kept as its
# Fourier transform, and cross-correlated with the probe grid of every orientation:
#   corr = irfft( FT(crowd) * conj(FT(probe)) ).

"Per-run state: FFT plans and the cached crowder transforms (the crowder atoms never change)."
mutable struct SoftCtx
    l::Int
    lat::Lattice
    tern::Int
    plan_f::Any
    plan_b::Any
    crowd::Dict{Symbol,Array{ComplexF64,3}}
    working::Array{Float64,3}
    lcEle::Array{Float64,3}
    lcVdw::Array{Float64,3}
    combined::Array{Float64,3}
    vol::BitArray{3}
    ft::Array{ComplexF64,3}
end

function SoftCtx(l::Int; tern::Int = tern_bytes_from_env(), lat::Lattice = load_lattice())
    FFTW.set_num_threads(Threads.nthreads())
    working = zeros(l, l, l)
    ft = Array{ComplexF64}(undef, l ÷ 2 + 1, l, l)
    plan_f = plan_rfft(working; flags = FFTW.ESTIMATE)
    plan_b = plan_irfft(ft, l; flags = FFTW.ESTIMATE)
    return SoftCtx(l, lat, tern, plan_f, plan_b, Dict{Symbol,Array{ComplexF64,3}}(),
                   working, zeros(l, l, l), zeros(l, l, l), zeros(l, l, l), falses(l, l, l), ft)
end

# ---- score.c ----

myEXP(x) = x > FLT_MAX_EXP ? 0.0 : exp(x)

function showscorenet(func, sum_, frac, u, u1)
    print(func, "\t")
    @printf("Sum: %16e Occ:%16.12f Ave:%16.8f Net:%16.8f\n", sum_, frac, u, u1)
end

"Print the Boltzmann-average summary of a grid (vRep); for `vol`, also (re)define the excluded-volume mask."
function vRep(vol::BitArray{3}, cutoff::Float64, grd::Array{Float64,3}, scl::Float64, name::AbstractString, kBT::Float64)
    l = size(grd, 1)
    dl3 = Float64(l)^3
    isvol = startswith(name, "vol")
    isvol && (vol .= grd[1:l, 1:l, 1:l] .> cutoff)
    sum_ = 0.0
    count = 0.0
    @inbounds for idx in eachindex(grd)
        if vol[idx]
            count += 1.0
        else
            sum_ += myEXP(grd[idx] * scl / (-kBT))
        end
    end
    if isvol
        u = -kBT * log(1.0 - count / dl3); un = 0.0
    else
        u = -kBT * log(sum_ / dl3); un = -kBT * log(sum_ / (dl3 - count))
    end
    showscorenet(name, sum_, count / dl3, u, un)
end

"grd1 = grd1*scl1 + grd2*scl2 outside the excluded volume, grd1*scl1 + FLT_MAX_EXP inside (AddTo1filt)."
function AddTo1filt!(grd1::Array{Float64,3}, scl1::Float64, grd2::Array{Float64,3}, scl2::Float64, vol::BitArray{3})
    @inbounds Threads.@threads for idx in eachindex(grd1)
        grd1[idx] = vol[idx] ? grd1[idx] * scl1 + FLT_MAX_EXP : grd1[idx] * scl1 + grd2[idx] * scl2
    end
    return grd1
end

# ---- the four convolutions ----

atom_coeffs(f, atoms) = Float64[f(a) for a in atoms]

# (kind, crowd atoms' mapping, probe atoms' mapping)
function crowd_grid!(g::Array{Float64,3}, kind::Symbol, atoms::Vector{Atom}, dx::Float64, sol::Solv)
    fill!(g, 0.0)
    if kind === :vol
        map_R!(g, atoms, ones(length(atoms)), Float64[a.r for a in atoms], 0.0, dx, Float64[a.r * a.r for a in atoms], rdf_vol)
    elseif kind === :ele
        map_R!(g, atoms, atom_coeffs(a -> a.q, atoms), fill(sol.Cut, length(atoms)), 1.0, dx, sol.kappa, rdf_debye)
    elseif kind === :vdw       # r6: -sqrt(B) / r^6
        map_R!(g, atoms, atom_coeffs(a -> -a.Bsq, atoms), fill(sol.Cut, length(atoms)), 1.0, dx, 3.0, rdf_vdw)
    else                       # :v12  +sqrt(A) / r^12
        map_R!(g, atoms, atom_coeffs(a -> a.Asq, atoms), fill(sol.Cut, length(atoms)), 1.0, dx, 6.0, rdf_vdw)
    end
end

function probe_grid!(g::Array{Float64,3}, kind::Symbol, atoms::Vector{Atom}, dx::Float64, sol::Solv, lat::Lattice)
    fill!(g, 0.0)
    if kind === :vol
        map_R!(g, atoms, ones(length(atoms)), Float64[a.r for a in atoms], 0.0, dx, Float64[a.r * a.r for a in atoms], rdf_vol)
    elseif kind === :ele
        map_L2nd!(g, atoms, atom_coeffs(a -> a.q, atoms), lat)
    elseif kind === :vdw
        map_L!(g, atoms, atom_coeffs(a -> a.Bsq, atoms))
    else
        map_L!(g, atoms, atom_coeffs(a -> a.Asq, atoms))
    end
end

"Cross-correlation of the cached crowder grid of `kind` with this orientation's probe grid, into ctx.working."
function correlate!(ctx::SoftCtx, kind::Symbol, crds::Vector{Atom}, pros::Vector{Atom}, dx::Float64, sol::Solv)
    g = ctx.working
    if !haskey(ctx.crowd, kind)
        crowd_grid!(g, kind, crds, dx, sol)
        ctx.crowd[kind] = ctx.plan_f * g
    end
    probe_grid!(g, kind, pros, dx, sol, ctx.lat)
    mul!(ctx.ft, ctx.plan_f, g)
    FC = ctx.crowd[kind]
    ft = ctx.ft
    @inbounds Threads.@threads for idx in eachindex(ft)
        ft[idx] = FC[idx] * conj(ft[idx])
    end
    mul!(g, ctx.plan_b, ft)
    return g
end

# ---- quantization (setdtErn) ----

const TERN_SCL_GEN = 8.0
const TERN_MAX_GEN = Int8(127)
@inline tern_round(x::Float64) = unsafe_trunc(Int32, round(x * TERN_SCL_GEN, RoundNearestTiesAway)) % Int8

# (separate functions per cell format: one variable bound in both branches of a threaded loop gets boxed)
function setdt2(vol::BitArray{3}, ele::Array{Float64,3}, vdw::Array{Float64,3})
    l = size(vol, 1); h = l ÷ 2
    out = Array{Int8}(undef, 2, l, l, l)
    Threads.@threads for i in 0:l-1
        ii = (i + h) % l
        @inbounds for j in 0:l-1
            jj = (j + h) % l
            for k in 0:l-1
                kk = (k + h) % l
                if vol[k+1, j+1, i+1]
                    out[1, kk+1, jj+1, ii+1] = TERN_MAX_GEN; out[2, kk+1, jj+1, ii+1] = TERN_MAX_GEN
                else
                    out[1, kk+1, jj+1, ii+1] = tern_round(ele[k+1, j+1, i+1])
                    out[2, kk+1, jj+1, ii+1] = tern_round(vdw[k+1, j+1, i+1])
                end
            end
        end
    end
    return out
end

function setdt1(vol::BitArray{3}, soft::Array{Float64,3})
    l = size(vol, 1); h = l ÷ 2
    out = Array{Int8}(undef, l, l, l)
    Threads.@threads for i in 0:l-1
        ii = (i + h) % l
        @inbounds for j in 0:l-1
            jj = (j + h) % l
            for k in 0:l-1
                kk = (k + h) % l
                out[kk+1, jj+1, ii+1] = vol[k+1, j+1, i+1] ? TERN_MAX_GEN : tern_round(soft[k+1, j+1, i+1])
            end
        end
    end
    return out
end

setdtErn(ctx::SoftCtx) = ctx.tern == 2 ? setdt2(ctx.vol, ctx.lcEle, ctx.lcVdw) : setdt1(ctx.vol, ctx.combined)

"""
    softgrd(ctx, crds, pros, dx, sol) -> tErn array

Energy grid of one orientation: (2, l, l, l) Int8 for TERN_BYTES=2 ([elec, vdw] per cell), (l, l, l)
for TERN_BYTES=1; the memory layout is the C `tErn[l][l][l]`. Prints the vol/ele/vdw/v12/v+e summary
lines of the C code to stdout.
"""
function softgrd(ctx::SoftCtx, crds::Vector{Atom}, pros::Vector{Atom}, dx::Float64, sol::Solv)
    fill!(ctx.lcEle, 0.0); fill!(ctx.lcVdw, 0.0); fill!(ctx.combined, 0.0)
    es = sol.escl * 332.0 / sol.sdie

    correlate!(ctx, :vol, crds, pros, dx, sol)
    vRep(ctx.vol, 0.001, ctx.working, 1.0, "vol", sol.kBT)
    flush(stdout)

    correlate!(ctx, :ele, crds, pros, dx, sol)
    AddTo1filt!(ctx.combined, 1.0, ctx.working, es, ctx.vol)
    AddTo1filt!(ctx.lcEle, 1.0, ctx.working, es, ctx.vol)
    vRep(ctx.vol, 0.001, ctx.working, es, "ele", sol.kBT)
    flush(stdout)

    correlate!(ctx, :vdw, crds, pros, dx, sol)
    AddTo1filt!(ctx.combined, 1.0, ctx.working, sol.vscl, ctx.vol)
    AddTo1filt!(ctx.lcVdw, 1.0, ctx.working, sol.vscl, ctx.vol)
    vRep(ctx.vol, 0.001, ctx.working, sol.vscl, "vdw", sol.kBT)
    flush(stdout)

    correlate!(ctx, :v12, crds, pros, dx, sol)
    AddTo1filt!(ctx.combined, 1.0, ctx.working, sol.vscl, ctx.vol)
    AddTo1filt!(ctx.lcVdw, 1.0, ctx.working, sol.vscl, ctx.vol)
    vRep(ctx.vol, 0.001, ctx.working, sol.vscl, "v12", sol.kBT)
    vRep(ctx.vol, 0.001, ctx.combined, 1.0, "v+e", sol.kBT)
    flush(stdout)

    return setdtErn(ctx)
end
