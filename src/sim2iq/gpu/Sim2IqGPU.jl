# Sim2IqGPU - CUDA.jl version of the sim2iq C core (grid mapping, FFT, binning on the GPU).
# Host-side input (PQR, trajectory, arguments, output) is the CPU module ../cpu/Sim2Iq.jl.
module Sim2IqGPU

using CUDA
using CUDA.CUFFT
using Printf

include(joinpath(@__DIR__, "..", "cpu", "Sim2Iq.jl"))   # the host-side module
using .Sim2Iq
import .Sim2Iq: get_element_type, get_cromer_mann_params, QIPoint

export sim2iq_gpu_main, sim2iq_gpu_batch, calculate_saxs_data_gpu

# Cromer-Mann tables indexed by element type + 1 (0 unknown, 1 HOH, 2 H, 3 C, 4 N, 5 O, 6 P, 7 S)
function cm_tables()
    a = zeros(8, 4); b = zeros(8, 4); vdw = zeros(8); el = zeros(Int32, 8)
    for t in 1:7
        cm = get_cromer_mann_params(t)
        a[t+1, :] .= cm.a; b[t+1, :] .= cm.b; vdw[t+1] = cm.vdw_radius; el[t+1] = cm.electrons
    end
    return a, b, vdw, el
end

const NORM_ELECTRON_COUNT = Int32(1)
const NORM_VOLUME = Int32(2)
const NORM_NONE = Int32(3)

@inline function contribution(mode::Int32, r2::Float64, et::Int32, vol::Float64, B::Float64, rho0::Float64, cma, cmb)
    if mode == NORM_NONE
        return 1.0
    elseif mode == NORM_VOLUME
        Bt = 4.0 * pi * vol^(2.0 / 3.0)
        pre = 8.0 * rho0 * pi^1.5 * vol / Bt^1.5
        return pre * exp(-4.0 * pi * pi * r2 / Bt)
    end
    ff = 0.0
    for k in 1:4
        Bt = cmb[et+1, k] + B
        ff += (4.0 * pi / Bt)^1.5 * cma[et+1, k] * exp(-4.0 * pi * pi * r2 / Bt)
    end
    return ff
end

# One block (128 threads) per atom: reduce the total contribution, then add the normalised values.
function map_atoms_kernel!(grid, N::Int32, spacing::Float64, ax, ay, az, avol, aet,
                           mode::Int32, cutoff_offset::Float64, B::Float64, rho0::Float64,
                           cma, cmb, vdw, elec)
    i = blockIdx().x
    et = aet[i]
    et == Int32(0) && return nothing
    cutoff = if mode == NORM_ELECTRON_COUNT
        3.0
    elseif mode == NORM_VOLUME
        2.0 * vdw[et+1]
    else
        cutoff_offset + vdw[et+1]
    end
    x = ax[i]; y = ay[i]; z = az[i]; vol = avol[i]
    small = Int32(2) * unsafe_trunc(Int32, ceil(cutoff / spacing)) + Int32(1)
    ox = unsafe_trunc(Int32, floor(x / spacing)) - small ÷ Int32(2)
    oy = unsafe_trunc(Int32, floor(y / spacing)) - small ÷ Int32(2)
    oz = unsafe_trunc(Int32, floor(z / spacing)) - small ÷ Int32(2)
    rx = x / spacing - ox; ry = y / spacing - oy; rz = z / spacing - oz
    cut2 = cutoff * cutoff
    npts = small * small * small
    tid = threadIdx().x - Int32(1)

    red = CuStaticSharedArray(Float64, 32)
    tot = CuStaticSharedArray(Float64, 1)
    loc = 0.0
    p = tid
    while p < npts
        xi = p ÷ (small * small); yi = (p ÷ small) % small; zi = p % small
        dx = (xi - rx) * spacing; dy = (yi - ry) * spacing; dz = (zi - rz) * spacing
        r2 = dx * dx + dy * dy + dz * dz
        if r2 <= cut2
            loc += contribution(mode, r2, et, vol, B, rho0, cma, cmb)
        end
        p += Int32(128)
    end
    off = Int32(16)
    while off > Int32(0)
        loc += CUDA.shfl_down_sync(0xffffffff, loc, off)
        off >>= Int32(1)
    end
    if tid & Int32(31) == Int32(0)
        red[(tid >> Int32(5)) + Int32(1)] = loc
    end
    sync_threads()
    if tid == Int32(0)
        t = 0.0
        for w in 1:4
            t += red[w]
        end
        tot[1] = t
    end
    sync_threads()
    total = tot[1]
    total > 0.0 || return nothing
    nf = if mode == NORM_ELECTRON_COUNT
        elec[et+1] > Int32(0) ? elec[et+1] / total : 1.0 / total
    elseif mode == NORM_VOLUME
        vol * rho0 / total
    else
        1.0
    end
    p = tid
    NN = Int(N)
    while p < npts
        xi = p ÷ (small * small); yi = (p ÷ small) % small; zi = p % small
        dx = (xi - rx) * spacing; dy = (yi - ry) * spacing; dz = (zi - rz) * spacing
        r2 = dx * dx + dy * dy + dz * dz
        if r2 <= cut2
            c = contribution(mode, r2, et, vol, B, rho0, cma, cmb) * nf
            if c > 0.0
                fx = mod(Int(ox + xi), NN); fy = mod(Int(oy + yi), NN); fz = mod(Int(oz + zi), NN)
                CUDA.@atomic grid[1 + fx + fy * NN + fz * NN * NN] += Float32(c)
            end
        end
        p += Int32(128)
    end
    return nothing
end

# |F_vac*exp(B q^2/16pi^2) + F_other|^2 binned in spherical shells (shared-memory bins per block).
function bin_kernel!(g_sums, g_qs, g_cnt, vac, other, have_other::Bool, N::Int32, factor::Float64,
                     qmax::Float64, bw::Float64, nb::Int32, b_factor::Float64)
    s = CuDynamicSharedArray(Float64, 3 * Int(nb))     # [sums | q sums | counts]
    tid = threadIdx().x
    j = tid
    while j <= 3 * Int(nb)
        s[j] = 0.0
        j += blockDim().x
    end
    sync_threads()
    total = Int(N)^3
    half = Int(N) ÷ 2
    idx = (blockIdx().x - 1) * blockDim().x + tid - 1
    stride = gridDim().x * blockDim().x
    nbi = Int(nb)
    while idx < total
        if idx != 0
            i = idx % Int(N); jj = (idx ÷ Int(N)) % Int(N); k = idx ÷ (Int(N) * Int(N))
            qx = (i <= half ? i : i - Int(N)) * factor
            qy = (jj <= half ? jj : jj - Int(N)) * factor
            qz = (k <= half ? k : k - Int(N)) * factor
            q = sqrt(qx * qx + qy * qy + qz * qz)
            if q > 0.0 && q < qmax
                b = unsafe_trunc(Int, q / bw) + 1
                if b <= nbi
                    bs = b_factor > 0.0 ? exp(b_factor * (q / (4.0 * pi))^2) : 1.0
                    v = vac[idx+1]
                    re = Float64(real(v)) * bs; im = Float64(imag(v)) * bs
                    if have_other
                        o = other[idx+1]
                        re += Float64(real(o)); im += Float64(imag(o))
                    end
                    CUDA.@atomic s[b] += re * re + im * im
                    CUDA.@atomic s[nbi+b] += q
                    CUDA.@atomic s[2*nbi+b] += 1.0
                end
            end
        end
        idx += stride
    end
    sync_threads()
    j = tid
    while j <= nbi
        if s[2*nbi+j] > 0.0
            CUDA.@atomic g_sums[j] += s[j]
            CUDA.@atomic g_qs[j] += s[nbi+j]
            CUDA.@atomic g_cnt[j] += s[2*nbi+j]
        end
        j += blockDim().x
    end
    return nothing
end

function map_step!(grid, N, spacing, d, mode, cutoff_offset, B, rho0, cm)
    fill!(grid, 0.0f0)
    n = length(d.et)
    n == 0 && return grid
    @cuda threads=128 blocks=n map_atoms_kernel!(grid, Int32(N), spacing, d.x, d.y, d.z, d.vol, d.et,
                                                 Int32(mode), cutoff_offset, B, rho0, cm.a, cm.b, cm.vdw, cm.el)
    return grid
end

function calculate_saxs_data_gpu(structure::Sim2Iq.Structure, N::Int, spacing::Float64, steps::Int,
                                 cutoff_offset::Float64, se_w::Float64, ou_w::Float64, in_w::Float64, B::Float64)
    t0 = time(); el() = time() - t0
    a, b, vdw, elc = cm_tables()
    cm = (a = CuArray(a), b = CuArray(b), vdw = CuArray(vdw), el = CuArray(elc))
    d = (x = CuArray([at.x for at in structure]), y = CuArray([at.y for at in structure]),
         z = CuArray([at.z for at in structure]), vol = CuArray([at.vol for at in structure]),
         et = CuArray(Int32[get_element_type(at.element) for at in structure]))

    vac = CUDA.zeros(Float32, N, N, N)
    other = steps > 1 ? CUDA.zeros(Float32, N, N, N) : nothing
    work = steps > 1 ? CUDA.zeros(Float32, N, N, N) : nothing

    @printf("[%.3f s] Mapping atoms to grid...\n", el())
    map_step!(vac, N, spacing, d, NORM_ELECTRON_COUNT, 0.0, B, 0.334, cm)
    if steps > 1
        @printf("[%.3f s] Mapping solvent excluded volume...\n", el())
        map_step!(work, N, spacing, d, NORM_VOLUME, 0.0, B, 0.334, cm)
        other .= Float32.(Float64.(other) .+ Float64.(work) .* se_w)
    end
    dV = spacing^3
    for s in 3:steps
        @printf("[%.3f s] Mapping %s shell...\n", el(), s == 3 ? "outer" : "inner")
        map_step!(work, N, spacing, d, NORM_NONE, s == 3 ? cutoff_offset : 0.0, B, 0.334, cm)
        add = (s == 3 ? ou_w : in_w) * dV
        cnt = count(>(0.5f0), work)
        other .= ifelse.(work .> 0.5f0, Float32.(Float64.(other) .+ add), other)
        @printf("Volume: %.12e\n", cnt * dV)
    end
    work = nothing

    @printf("[%.3f s] Calculating FFTs...\n", el())
    fv = ComplexF32.(vac); vac = nothing
    CUDA.reclaim()
    plan = CUFFT.plan_fft!(fv)
    plan * fv
    fo = nothing
    if steps > 1
        fo = ComplexF32.(other); other = nothing
        plan * fo
    end
    CUDA.synchronize()

    @printf("[%.3f s] Binning intensity...\n", el())
    factor = 2π / (N * spacing)
    qmax = (N ÷ 2 - 1) * factor - 1e-8
    bw = factor - 1e-8
    nb = floor(Int, qmax / bw) + 1
    gs = CUDA.zeros(Float64, nb); gq = CUDA.zeros(Float64, nb); gc = CUDA.zeros(Float64, nb)
    other_arg = fo === nothing ? fv : fo
    threads = 256; blocks = 1024
    @cuda threads=threads blocks=blocks shmem=3 * nb * sizeof(Float64) bin_kernel!(
        gs, gq, gc, fv, other_arg, fo !== nothing, Int32(N), factor, qmax, bw, Int32(nb), B)
    CUDA.synchronize()

    v0 = Array(view(vec(fv), 1:1))[1]
    o0 = fo === nothing ? ComplexF32(0) : Array(view(vec(fo), 1:1))[1]
    re0 = Float64(real(v0) + real(o0)); im0 = Float64(imag(v0) + imag(o0))
    sums = Array(gs); qs = Array(gq); cnt = Array(gc)
    out = QIPoint[QIPoint(0.0, re0 * re0 + im0 * im0)]
    for i in 1:nb
        c = cnt[i]
        c > 0 || continue
        push!(out, QIPoint(i == 1 ? 0.0 : qs[i] / c, sums[i] / c))
    end
    @printf("[%.3f s] SAXS calculation complete.\n", el())
    return out
end

# Same command line as the C executable / Sim2Iq.sim2iq_main
function sim2iq_gpu_main(argv::Vector{String} = copy(ARGS))
    args = Sim2Iq.parse_args(argv)
    args === nothing && return 1
    st = Sim2Iq.read_pqr_file(args.input_file)
    if args.traj_file !== nothing
        println("Processing trajectory file: ", args.traj_file)
        st = Sim2Iq.process_trajectory(st, args.traj_file, args.traj_is_txt)
    end
    B = args.b_factor > 0.0 ? args.b_factor : Sim2Iq.u2B(0.25 * args.grid_spacing)
    @printf("Actual B-factor: %.6f\n", B)
    println("GPU: ", CUDA.name(CUDA.device()))
    r = calculate_saxs_data_gpu(st, args.grid_size, args.grid_spacing, args.steps, args.cutoff_offset,
                                args.se_weight, args.ou_weight, args.in_weight, B)
    Sim2Iq.write_output(r, args.output_file) || return 1
    return 0
end

# Many runs on one structure in one process (one start-up): argv = input.pqr grid_size grid_spacing steps
# followed by `output.dat=traj.txt` jobs (`traj` = "-" for the structure itself, e.g. P(q)).
function sim2iq_gpu_batch(argv::Vector{String} = copy(ARGS))
    length(argv) < 5 && (println(stderr, "usage: input.pqr grid_size grid_spacing steps out=traj ... | --serve (jobs `out=traj` from stdin)"); return 1)
    pqr, N, spacing, steps = argv[1], parse(Int, argv[2]), parse(Float64, argv[3]), parse(Int, argv[4])
    base = Sim2Iq.read_pqr_file(pqr)
    B = Sim2Iq.u2B(0.25 * spacing)
    rc = 0
    function run_job(job)
        out, traj = split(job, '='; limit = 2)
        st = traj == "-" ? base : Sim2Iq.process_trajectory(base, String(traj), true)
        r = calculate_saxs_data_gpu(st, N, spacing, steps, 2.8, -1.0, 0.011, -0.011, B)
        Sim2Iq.write_output(r, String(out)) || (rc = 1)
        GC.gc(); CUDA.reclaim()
    end
    if argv[5] == "--serve"       # keep the process (CUDA context, compiled kernels) and wait for jobs on stdin
        for line in eachline(stdin)
            job = strip(line)
            isempty(job) && continue
            run_job(job)
            println("done ", job); flush(stdout)
        end
    else
        foreach(run_job, argv[5:end])
    end
    return rc
end

end # module
