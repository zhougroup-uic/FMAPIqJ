# native.jl - GPU grid generation through native/libfmapgrid.so (CUDA C + cuFFT, see native/fmapgrid.cu).
# Included into module FmapGen on demand (FMAPB2_GPU=1, see cli.jl). Needs only the shared library and an NVIDIA
# driver: no CUDA.jl. The library keeps all device memory; Julia passes host arrays and gets the Int8 grid back.
using Libdl

const FG_NSTAT = 4
const FG_NAMES = ("vol", "ele", "vdw", "v12", "v+e")

function fg_libpath()
    p = get(ENV, "FMAPGRID_LIB", "")
    isempty(p) || return p
    return normpath(joinpath(@__DIR__, "..", "..", "..", "native", "libfmapgrid.so"))
end

const FG_HANDLE = Ref{Ptr{Cvoid}}(C_NULL)
function fg_sym(name::Symbol)
    if FG_HANDLE[] == C_NULL
        p = fg_libpath()
        isfile(p) || error("$p not found: build it with `module load CUDA && make -C native`, or set FMAPGRID_LIB")
        FG_HANDLE[] = Libdl.dlopen(p, Libdl.RTLD_NOW | Libdl.RTLD_LOCAL)
    end
    return Libdl.dlsym(FG_HANDLE[], name)
end
fg_error() = unsafe_string(ccall(fg_sym(:fg_error), Cstring, ()))
fg_check(rc::Integer) = rc < 0 ? error("libfmapgrid: ", fg_error()) : rc

mutable struct NativeCtx
    h::Ptr{Cvoid}
    l::Int
    tern::Int
    crowd_set::Bool
end

function NativeCtx(l::Int; tern::Int = tern_bytes_from_env(), device::Int = 0)
    h = ccall(fg_sym(:fg_open), Ptr{Cvoid}, (Cint, Cint, Cint), l, tern, device)
    h == C_NULL && error("libfmapgrid: ", fg_error())
    ctx = NativeCtx(h, l, tern, false)
    finalizer(close_ctx, ctx)
    return ctx
end

function close_ctx(ctx::NativeCtx)
    ctx.h == C_NULL && return nothing
    ccall(fg_sym(:fg_close), Cvoid, (Ptr{Cvoid},), ctx.h)
    ctx.h = C_NULL
    return nothing
end

make_native_ctx(l::Int) = NativeCtx(l)

# x y z q r Asq Bsq as seven consecutive arrays
function pack_atoms(a::Vector{Atom})
    n = length(a)
    v = Vector{Float64}(undef, 7n)
    @inbounds for (i, t) in enumerate(a)
        v[i] = t.x; v[n+i] = t.y; v[2n+i] = t.z; v[3n+i] = t.q; v[4n+i] = t.r; v[5n+i] = t.Asq; v[6n+i] = t.Bsq
    end
    return v
end

"Same contract as the CPU softgrd: returns the host tErn array of one orientation (prints the summary lines)."
function softgrd(ctx::NativeCtx, crds::Vector{Atom}, pros::Vector{Atom}, dx::Float64, sol::Solv)
    if !ctx.crowd_set
        fg_check(ccall(fg_sym(:fg_set_crowd), Cint, (Ptr{Cvoid}, Cint, Ptr{Float64}, Float64, Float64, Float64),
                       ctx.h, length(crds), pack_atoms(crds), dx, sol.Cut, sol.kappa))
        ctx.crowd_set = true
    end
    l = ctx.l
    out = ctx.tern == 2 ? Array{Int8}(undef, 2, l, l, l) : Array{Int8}(undef, l, l, l)
    stat = Vector{Float64}(undef, length(FG_NAMES) * FG_NSTAT)
    es = sol.escl * 332.0 / sol.sdie
    fg_check(ccall(fg_sym(:fg_softgrd), Cint, (Ptr{Cvoid}, Cint, Ptr{Float64}, Float64, Float64, Float64, Float64, Ptr{Int8}, Ptr{Float64}),
                   ctx.h, length(pros), pack_atoms(pros), dx, es, sol.vscl, sol.kBT, out, stat))
    for (k, name) in enumerate(FG_NAMES)
        s = stat[(k-1)*FG_NSTAT+1:k*FG_NSTAT]
        showscorenet(name, s[1], s[2], s[3], s[4])
    end
    flush(stdout)
    return out
end

"Cells below `ecut` that are not excluded (same lines as the CPU method), selected on the device."
function select_poses(io::IO, ctx::NativeCtx, ang, ecut::Real)
    l = ctx.l
    cap = 1 << 16
    local cell, val, n
    while true
        cell = Vector{Int32}(undef, cap); val = Vector{Float64}(undef, cap)
        n = fg_check(ccall(fg_sym(:fg_select), Cint, (Ptr{Cvoid}, Float64, Cint, Ptr{Int32}, Ptr{Float64}), ctx.h, ecut, cap, cell, val))
        n <= cap && break
        cap = n
    end
    n == 0 && return 0
    order = sortperm(view(cell, 1:n))                       # linear index k + l*(j + l*i): C order, i slowest
    for o in order
        c = Int(cell[o]); k = c % l; j = (c ÷ l) % l; i = c ÷ (l * l)
        @printf(io, "%10.6f%10.6f%10.6f%8d%8d%8d%10.3f\n", ang[1], ang[2], ang[3], i, j, k, val[o])
    end
    return n
end

# ---- pair energies of listed poses (fmapdd / fmapdd.atm, cpu/pair.jl) on the device ----
"GPU version of `pair_energies` (same result up to the order of the sums)."
function pair_energies_gpu(rec::Vector{Atom}, lig::Vector{Atom}, poses::Poses, ps::PairSys; atm::Bool = false, chunk::Int = 256)
    nr = length(rec); nl = length(lig); np = length(poses.ang)
    Rm = zeros(9, np); Om = zeros(3, np)
    for p in 1:np
        a = poses.ang[p]; R = euler_to_rotmat(a[1], a[2], a[3]); k = poses.ijk[p]
        Rm[:, p] = [R[1][1], R[1][2], R[1][3], R[2][1], R[2][2], R[2][3], R[3][1], R[3][2], R[3][3]]
        Om[:, p] = [wrap_ijk(k[c], poses.l) * poses.dx for c in 1:3]
    end
    sums = zeros(3, np); accr = zeros(3, nr); accl = zeros(3, nl)
    fg_check(ccall(fg_sym(:fg_pair), Cint,
                   (Cint, Ptr{Float64}, Cint, Ptr{Float64}, Cint, Ptr{Float64}, Ptr{Float64}, Float64, Float64, Float64, Float64, Float64, Cint,
                    Ptr{Float64}, Ptr{Float64}, Ptr{Float64}),
                   nr, pack_atoms(rec), nl, pack_atoms(lig), np, Rm, Om, ps.kap, ps.rEup2, ps.rup2, ps.rlow2, ps.L, atm ? 1 : 0, sums, accr, accl))
    ern = zeros(4, np)
    for p in 1:np
        ern[2, p] = sums[1, p]; ern[3, p] = sums[2, p] * ps.es; ern[4, p] = sums[3, p] * ps.vscl
        ern[1, p] = ern[3, p] + ern[4, p]
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
