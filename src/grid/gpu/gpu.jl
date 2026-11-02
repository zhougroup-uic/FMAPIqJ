# gpu.jl - GPU grid generation (port of fmap/ext/gpu/*.cu) with CUDA.jl + CUFFT.
# Included into module FmapGen on demand (FMAPB2_GPU=1, see cli.jl); needs a working CUDA device.
#
# Same algorithm as the CPU code: crowder grids are built once and kept as their Fourier
# transform on the device, every orientation maps the probe atoms and cross-correlates. Atom
# mapping uses atomic adds (like the CUDA code), so the sums are not bit-reproducible; the
# quantized cells agree with the CPU code except where a value sits on a rounding boundary.
using CUDA
using CUDA.CUFFT

"All atoms of one list as device vectors (struct of arrays)."
struct DevAtoms
    x::CuVector{Float64}
    y::CuVector{Float64}
    z::CuVector{Float64}
    q::CuVector{Float64}
    r::CuVector{Float64}
    Asq::CuVector{Float64}
    Bsq::CuVector{Float64}
    n::Int
end

DevAtoms(a::Vector{Atom}) = DevAtoms(CuArray(Float64[t.x for t in a]), CuArray(Float64[t.y for t in a]),
                                     CuArray(Float64[t.z for t in a]), CuArray(Float64[t.q for t in a]),
                                     CuArray(Float64[t.r for t in a]), CuArray(Float64[t.Asq for t in a]),
                                     CuArray(Float64[t.Bsq for t in a]), length(a))

# ---- spherical mapping: block (bx, atom), threads stride over the atom's bounding box ----

function mapR_kernel!(grd, xs, ys, zs, coeffs, rups, scls, rlow::Float64, dx::Float64, l::Int32, rdf::F) where {F}
    a = blockIdx().y
    x0 = xs[a]; y0 = ys[a]; z0 = zs[a]
    rup = rups[a]; rup2 = rup * rup
    s = rup / dx
    imin = unsafe_trunc(Int32, floor(x0 - s)); imax = unsafe_trunc(Int32, ceil(x0 + s))
    jmin = unsafe_trunc(Int32, floor(y0 - s)); jmax = unsafe_trunc(Int32, ceil(y0 + s))
    kmin = unsafe_trunc(Int32, floor(z0 - s)); kmax = unsafe_trunc(Int32, ceil(z0 + s))
    nx = imax - imin + Int32(1); ny = jmax - jmin + Int32(1); nz = kmax - kmin + Int32(1)
    ncell = Int64(nx) * ny * nz
    coeff = coeffs[a]; sc = scls[a]; rlow2 = rlow * rlow
    idx = Int64(blockIdx().x - 1) * blockDim().x + threadIdx().x - 1
    stride = Int64(gridDim().x) * blockDim().x
    while idx < ncell
        k = Int32(idx % nz); j = Int32((idx ÷ nz) % ny); i = Int32(idx ÷ (Int64(ny) * nz))
        gi = imin + i; gj = jmin + j; gk = kmin + k
        x = (gi - x0) * dx; y = (gj - y0) * dx; z = (gk - z0) * dx
        r2 = x * x + y * y + z * z
        if r2 <= rup2
            px = mod(gi, l); py = mod(gj, l); pz = mod(gk, l)
            var = coeff * (r2 > rlow2 ? rdf(r2, sc) : rdf(rlow2, sc))
            CUDA.@atomic grd[pz+1, py+1, px+1] += var
        end
        idx += stride
    end
    return
end

function map_R!(grd::CuArray{Float64,3}, at::DevAtoms, coeffs, rups, rlow, dx, scls, rdf::F; maxrup::Float64) where {F}
    l = size(grd, 1)
    at.n == 0 && return grd
    side = Int(ceil(2 * maxrup / dx)) + 3
    ncell = side^3
    threads = 256
    bx = min(cld(ncell, threads), 4096)
    # gridDim.y is limited to 65535 atoms per launch
    for a0 in 1:65535:at.n
        a1 = min(a0 + 65534, at.n)
        rng = a0:a1
        @cuda threads = threads blocks = (bx, length(rng)) mapR_kernel!(grd, view(at.x, rng), view(at.y, rng), view(at.z, rng),
            view(coeffs, rng), view(rups, rng), view(scls, rng), rlow, dx, Int32(l), rdf)
    end
    return grd
end

# ---- trilinear mapping and 10-point lattice spreading: one thread per atom ----

function mapL_kernel!(grd, xs, ys, zs, coeffs, l::Int32)
    a = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    a > length(xs) && return
    coeff = coeffs[a]
    fa = xs[a] - trunc(xs[a]); fb = ys[a] - trunc(ys[a]); fc = zs[a] - trunc(zs[a])
    px = mod(unsafe_trunc(Int32, trunc(xs[a])), l); py = mod(unsafe_trunc(Int32, trunc(ys[a])), l); pz = mod(unsafe_trunc(Int32, trunc(zs[a])), l)
    pxn = mod((fa < 0.0 ? Int32(-1) : Int32(1)) + px, l)
    pyn = mod((fb < 0.0 ? Int32(-1) : Int32(1)) + py, l)
    pzn = mod((fc < 0.0 ? Int32(-1) : Int32(1)) + pz, l)
    av = abs(fa); bv = abs(fb); cv = abs(fc)
    CUDA.@atomic grd[pz+1, py+1, px+1] += coeff * (1.0 - av) * (1.0 - bv) * (1.0 - cv)
    CUDA.@atomic grd[pzn+1, py+1, px+1] += coeff * (1.0 - av) * (1.0 - bv) * cv
    CUDA.@atomic grd[pz+1, pyn+1, px+1] += coeff * (1.0 - av) * bv * (1.0 - cv)
    CUDA.@atomic grd[pzn+1, pyn+1, px+1] += coeff * (1.0 - av) * bv * cv
    CUDA.@atomic grd[pz+1, py+1, pxn+1] += coeff * av * (1.0 - bv) * (1.0 - cv)
    CUDA.@atomic grd[pzn+1, py+1, pxn+1] += coeff * av * (1.0 - bv) * cv
    CUDA.@atomic grd[pz+1, pyn+1, pxn+1] += coeff * av * bv * (1.0 - cv)
    CUDA.@atomic grd[pzn+1, pyn+1, pxn+1] += coeff * av * bv * cv
    return
end

function mapL2nd_kernel!(grd, xs, ys, zs, coeffs, lind, lwght, l::Int32)
    a = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    a > length(xs) && return
    coeff = coeffs[a]
    lx = floor(xs[a]); ly = floor(ys[a]); lz = floor(zs[a])
    fx = xs[a] - lx; fy = ys[a] - ly; fz = zs[a] - lz
    pos = unsafe_trunc(Int32, fx * 10.0) * Int32(100) + unsafe_trunc(Int32, fy * 10.0) * Int32(10) + unsafe_trunc(Int32, fz * 10.0)
    ix = unsafe_trunc(Int32, lx); iy = unsafe_trunc(Int32, ly); iz = unsafe_trunc(Int32, lz)
    for q in Int32(1):Int32(10)
        n = lind[q, pos+1]
        i0 = n ÷ Int32(16); j0 = (n - i0 * Int32(16)) ÷ Int32(4); k0 = n - i0 * Int32(16) - j0 * Int32(4)
        px = mod(ix - Int32(1) + i0, l); py = mod(iy - Int32(1) + j0, l); pz = mod(iz - Int32(1) + k0, l)
        CUDA.@atomic grd[pz+1, py+1, px+1] += coeff * lwght[q, pos+1]
    end
    return
end

function map_L!(grd, at::DevAtoms, coeffs)
    at.n == 0 && return grd
    @cuda threads = 128 blocks = cld(at.n, 128) mapL_kernel!(grd, at.x, at.y, at.z, coeffs, Int32(size(grd, 1)))
    return grd
end

function map_L2nd!(grd, at::DevAtoms, coeffs, lind, lwght)
    at.n == 0 && return grd
    @cuda threads = 128 blocks = cld(at.n, 128) mapL2nd_kernel!(grd, at.x, at.y, at.z, coeffs, lind, lwght, Int32(size(grd, 1)))
    return grd
end

# ---- per-run state ----

mutable struct GPUCtx
    l::Int
    tern::Int
    lind::CuMatrix{Int32}
    lwght::CuMatrix{Float64}
    plan_f::Any
    plan_b::Any
    crowd::Dict{Symbol,CuArray{ComplexF64,3}}
    working::CuArray{Float64,3}
    lcEle::CuArray{Float64,3}
    lcVdw::CuArray{Float64,3}
    combined::CuArray{Float64,3}
    vol::CuArray{Bool,3}
    ft::CuArray{ComplexF64,3}
    crds::Union{Nothing,DevAtoms}
end

function GPUCtx(l::Int; tern::Int = tern_bytes_from_env())
    lat = load_lattice()
    working = CUDA.zeros(Float64, l, l, l)
    ft = CuArray{ComplexF64}(undef, l ÷ 2 + 1, l, l)
    return GPUCtx(l, tern, CuArray(lat.ind), CuArray(lat.wght), plan_rfft(working), plan_irfft(ft, l),
                  Dict{Symbol,CuArray{ComplexF64,3}}(), working, CUDA.zeros(Float64, l, l, l),
                  CUDA.zeros(Float64, l, l, l), CUDA.zeros(Float64, l, l, l), CUDA.zeros(Bool, l, l, l), ft, nothing)
end

make_ctx(l::Int, ::Val{true}) = GPUCtx(l)

# ---- score.c on the device ----

dmyEXP(x) = x > FLT_MAX_EXP ? 0.0 : exp(x)

function vRep(ctx::GPUCtx, cutoff::Float64, grd, scl::Float64, name::AbstractString, kBT::Float64)
    l = ctx.l
    dl3 = Float64(l)^3
    isvol = startswith(name, "vol")
    isvol && (ctx.vol .= grd .> cutoff)
    count = Float64(sum(ctx.vol))
    sum_ = mapreduce((v, x) -> v ? 0.0 : dmyEXP(x * scl / (-kBT)), +, ctx.vol, grd)
    if isvol
        u = -kBT * log(1.0 - count / dl3); un = 0.0
    else
        u = -kBT * log(sum_ / dl3); un = -kBT * log(sum_ / (dl3 - count))
    end
    showscorenet(name, sum_, count / dl3, u, un)
end

AddTo1filt!(g1, scl1::Float64, g2, scl2::Float64, vol) =
    (g1 .= ifelse.(vol, g1 .* scl1 .+ FLT_MAX_EXP, g1 .* scl1 .+ g2 .* scl2); g1)

# ---- the four convolutions ----

function crowd_grid!(ctx::GPUCtx, kind::Symbol, at::DevAtoms, dx::Float64, sol::Solv)
    g = ctx.working; fill!(g, 0.0)
    n = at.n
    if kind === :vol
        map_R!(g, at, CUDA.ones(Float64, n), at.r, 0.0, dx, at.r .* at.r, rdf_vol; maxrup = maximum(at.r))
    elseif kind === :ele
        map_R!(g, at, at.q, CUDA.fill(sol.Cut, n), 1.0, dx, CUDA.fill(sol.kappa, n), rdf_debye; maxrup = sol.Cut)
    elseif kind === :vdw
        map_R!(g, at, -at.Bsq, CUDA.fill(sol.Cut, n), 1.0, dx, CUDA.fill(3.0, n), rdf_vdw; maxrup = sol.Cut)
    else
        map_R!(g, at, at.Asq, CUDA.fill(sol.Cut, n), 1.0, dx, CUDA.fill(6.0, n), rdf_vdw; maxrup = sol.Cut)
    end
end

function probe_grid!(ctx::GPUCtx, kind::Symbol, at::DevAtoms, dx::Float64)
    g = ctx.working; fill!(g, 0.0)
    if kind === :vol
        map_R!(g, at, CUDA.ones(Float64, at.n), at.r, 0.0, dx, at.r .* at.r, rdf_vol; maxrup = maximum(at.r))
    elseif kind === :ele
        map_L2nd!(g, at, at.q, ctx.lind, ctx.lwght)
    elseif kind === :vdw
        map_L!(g, at, at.Bsq)
    else
        map_L!(g, at, at.Asq)
    end
end

function correlate!(ctx::GPUCtx, kind::Symbol, crds::DevAtoms, pros::DevAtoms, dx::Float64, sol::Solv)
    g = ctx.working
    if !haskey(ctx.crowd, kind)
        crowd_grid!(ctx, kind, crds, dx, sol)
        ctx.crowd[kind] = ctx.plan_f * g
    end
    probe_grid!(ctx, kind, pros, dx)
    mul!(ctx.ft, ctx.plan_f, g)
    ctx.ft .= ctx.crowd[kind] .* conj.(ctx.ft)
    mul!(g, ctx.plan_b, ctx.ft)
    return g
end

# ---- quantization ----

gtern_round(x) = unsafe_trunc(Int32, round(x * 8.0, RoundNearestTiesAway)) % Int8

function setdtErn(ctx::GPUCtx)
    l = ctx.l; h = l ÷ 2
    q(g) = circshift(ifelse.(ctx.vol, Int8(127), gtern_round.(g)), (h, h, h))
    if ctx.tern == 2
        out = CuArray{Int8}(undef, 2, l, l, l)
        out[1, :, :, :] .= q(ctx.lcEle)
        out[2, :, :, :] .= q(ctx.lcVdw)
        return Array(out)
    else
        return Array(q(ctx.combined))
    end
end

"Same contract as the CPU softgrd: returns the host tErn array of one orientation."
function softgrd(ctx::GPUCtx, crds::Vector{Atom}, pros::Vector{Atom}, dx::Float64, sol::Solv)
    ctx.crds === nothing && (ctx.crds = DevAtoms(crds))
    dp = DevAtoms(pros)
    dc = ctx.crds
    fill!(ctx.lcEle, 0.0); fill!(ctx.lcVdw, 0.0); fill!(ctx.combined, 0.0)
    es = sol.escl * 332.0 / sol.sdie

    correlate!(ctx, :vol, dc, dp, dx, sol)
    vRep(ctx, 0.001, ctx.working, 1.0, "vol", sol.kBT); flush(stdout)

    correlate!(ctx, :ele, dc, dp, dx, sol)
    AddTo1filt!(ctx.combined, 1.0, ctx.working, es, ctx.vol)
    AddTo1filt!(ctx.lcEle, 1.0, ctx.working, es, ctx.vol)
    vRep(ctx, 0.001, ctx.working, es, "ele", sol.kBT); flush(stdout)

    correlate!(ctx, :vdw, dc, dp, dx, sol)
    AddTo1filt!(ctx.combined, 1.0, ctx.working, sol.vscl, ctx.vol)
    AddTo1filt!(ctx.lcVdw, 1.0, ctx.working, sol.vscl, ctx.vol)
    vRep(ctx, 0.001, ctx.working, sol.vscl, "vdw", sol.kBT); flush(stdout)

    correlate!(ctx, :v12, dc, dp, dx, sol)
    AddTo1filt!(ctx.combined, 1.0, ctx.working, sol.vscl, ctx.vol)
    AddTo1filt!(ctx.lcVdw, 1.0, ctx.working, sol.vscl, ctx.vol)
    vRep(ctx, 0.001, ctx.working, sol.vscl, "v12", sol.kBT)
    vRep(ctx, 0.001, ctx.combined, 1.0, "v+e", sol.kBT); flush(stdout)

    return setdtErn(ctx)
end

"Cells below `ecut` that are not excluded (same lines as the CPU method): the selection runs on the device."
function select_poses(io::IO, ctx::GPUCtx, ang, ecut::Real)
    mask = (.!ctx.vol) .& (ctx.combined .< ecut)
    idx = findall(mask)
    isempty(idx) && return 0
    vals = Array(ctx.combined[idx])
    ci = Array(idx)
    order = sortperm(ci; by = c -> (c[3], c[2], c[1]))          # C order: i slowest, k fastest
    for o in order
        c = ci[o]
        @printf(io, "%10.6f%10.6f%10.6f%8d%8d%8d%10.3f\n", ang[1], ang[2], ang[3], c[3] - 1, c[2] - 1, c[1] - 1, vals[o])
    end
    return length(order)
end

# ---- pair energies of listed poses (fmapdd / fmapdd.atm, cpu/pair.jl) ----

# One thread per (pose, atom): side 1 = ligand atom looping over the receptor, side 2 = receptor atom looping over the
# ligand (so the per-atom sums need no atomics and are reproducible). out[:, atom, pose] = (vol, ele, vdw), unscaled.
function pair_kernel!(out, side::Int, rec, lig, Rm, offs, p0::Int, np::Int, kap::Float64, rEup2::Float64, rup2::Float64,
                      rlow2::Float64, L::Float64)
    t = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    nr = length(rec.x); nl = length(lig.x)
    na = side == 1 ? nl : nr
    t > na * np && return
    a = (t - 1) % na + 1
    p = (t - 1) ÷ na + 1
    gp = p0 + p
    hL = 0.5 * L
    cut2 = max(rEup2, rup2)
    r11 = Rm[1, gp]; r12 = Rm[2, gp]; r13 = Rm[3, gp]; r21 = Rm[4, gp]; r22 = Rm[5, gp]; r23 = Rm[6, gp]
    r31 = Rm[7, gp]; r32 = Rm[8, gp]; r33 = Rm[9, gp]
    o1 = offs[1, gp]; o2 = offs[2, gp]; o3 = offs[3, gp]
    sv = 0.0; se = 0.0; sw = 0.0
    n = side == 1 ? nr : nl
    for b in 1:n
        i = side == 1 ? b : a
        j = side == 1 ? a : b
        px = lig.x[j]; py = lig.y[j]; pz = lig.z[j]
        lx = r11 * px + r12 * py + r13 * pz + o1
        ly = r21 * px + r22 * py + r23 * pz + o2
        lz = r31 * px + r32 * py + r33 * pz + o3
        dx = rec.x[i] - lx; dx = dx > hL ? dx - L : (dx < -hL ? dx + L : dx)
        dy = rec.y[i] - ly; dy = dy > hL ? dy - L : (dy < -hL ? dy + L : dy)
        dz = rec.z[i] - lz; dz = dz > hL ? dz - L : (dz < -hL ? dz + L : dz)
        d2 = dx * dx + dy * dy + dz * dz
        d2 > cut2 && continue
        rs = rec.r[i] + lig.r[j]
        v = d2 > rs * rs ? 0.0 : 1.0
        e = 0.0
        if d2 <= rEup2
            dd = sqrt(d2 > rlow2 ? d2 : rlow2)
            e = rec.q[i] * lig.q[j] * exp(-1.0 * dd * kap) / dd
        end
        w = 0.0
        d2m = d2 > rlow2 ? d2 : rlow2
        if d2m <= rup2
            d6 = d2m * d2m * d2m
            w = rec.Asq[i] * lig.Asq[j] / (d6 * d6) - rec.Bsq[i] * lig.Bsq[j] / d6
        end
        sv += v; se += e; sw += w
    end
    out[1, a, p] = sv; out[2, a, p] = se; out[3, a, p] = sw
    return
end

"GPU version of `pair_energies` (same result up to the order of the sums)."
function pair_energies_gpu(rec::Vector{Atom}, lig::Vector{Atom}, poses::Poses, ps::PairSys; atm::Bool = false, chunk::Int = 256)
    nr = length(rec); nl = length(lig); np = length(poses.ang)
    nt(d::DevAtoms) = (x = d.x, y = d.y, z = d.z, q = d.q, r = d.r, Asq = d.Asq, Bsq = d.Bsq)      # adaptable to the device
    dr = nt(DevAtoms(rec)); dl = nt(DevAtoms(lig))
    Rh = zeros(9, np); Oh = zeros(3, np)
    for p in 1:np
        a = poses.ang[p]; R = euler_to_rotmat(a[1], a[2], a[3]); k = poses.ijk[p]
        Rh[:, p] = [R[1][1], R[1][2], R[1][3], R[2][1], R[2][2], R[2][3], R[3][1], R[3][2], R[3][3]]
        Oh[:, p] = [wrap_ijk(k[c], poses.l) * poses.dx for c in 1:3]
    end
    Rm = CuArray(Rh); offs = CuArray(Oh)
    ern = zeros(4, np)
    accr = zeros(3, nr); accl = zeros(3, nl)
    out1 = CUDA.zeros(Float64, 3, nl, chunk)
    out2 = atm ? CUDA.zeros(Float64, 3, nr, chunk) : nothing
    thr = 128
    for p0 in 0:chunk:np-1
        nc = min(chunk, np - p0)
        @cuda threads = thr blocks = cld(nl * nc, thr) pair_kernel!(out1, 1, dr, dl, Rm, offs, p0, nc, ps.kap, ps.rEup2, ps.rup2, ps.rlow2, ps.L)
        s = Array(sum(view(out1, :, :, 1:nc); dims = 2))                  # 3 x 1 x nc
        for c in 1:nc
            p = p0 + c
            ern[2, p] = s[1, 1, c]; ern[3, p] = s[2, 1, c] * ps.es; ern[4, p] = s[3, 1, c] * ps.vscl
            ern[1, p] = ern[3, p] + ern[4, p]
        end
        if atm
            accl .+= Array(dropdims(sum(view(out1, :, :, 1:nc); dims = 3); dims = 3))
            @cuda threads = thr blocks = cld(nr * nc, thr) pair_kernel!(out2, 2, dr, dl, Rm, offs, p0, nc, ps.kap, ps.rEup2, ps.rup2, ps.rlow2, ps.L)
            accr .+= Array(dropdims(sum(view(out2, :, :, 1:nc); dims = 3); dims = 3))
        end
    end
    atm || return PairResult(ern, nothing, nothing)
    function fin(s)
        out = zeros(4, size(s, 2))
        out[1, :] = s[1, :]; out[2, :] = s[2, :] .* ps.es; out[3, :] = s[3, :] .* ps.vscl
        out[4, :] = out[2, :] .+ out[3, :]
        return out
    end
    return PairResult(ern, fin(accr), fin(accl))
end
