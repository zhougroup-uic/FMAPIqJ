# Sim2IqGPU - GPU I(q) through native/libfmapsim2iq.so (CUDA C + cuFFT, see native/fmapsim2iq.h); no CUDA.jl.
# Host-side input (PQR, trajectory, arguments, output) is the CPU module ../cpu/Sim2Iq.jl; the library does the grid mapping, the FFT and the
# binning on the device. A context (device buffers and FFT plan) is kept per grid size, so many runs in one process pay the set-up once.
module Sim2IqGPU

using Printf
using Libdl

include(joinpath(@__DIR__, "..", "cpu", "Sim2Iq.jl"))   # the host-side module
using .Sim2Iq
import .Sim2Iq: get_element_type, QIPoint

export sim2iq_gpu_main, sim2iq_gpu_batch, calculate_saxs_data_gpu

function libpath()
    p = get(ENV, "FMAPSIM2IQ_LIB", "")
    isempty(p) || return p
    return normpath(joinpath(@__DIR__, "..", "..", "..", "native", "libfmapsim2iq.so"))
end

const HANDLE = Ref{Ptr{Cvoid}}(C_NULL)
function sym(name::Symbol)
    if HANDLE[] == C_NULL
        p = libpath()
        isfile(p) || error("$p not found: build it with `module load CUDA && make -C native`, or set FMAPSIM2IQ_LIB")
        HANDLE[] = Libdl.dlopen(p, Libdl.RTLD_NOW | Libdl.RTLD_LOCAL)
    end
    return Libdl.dlsym(HANDLE[], name)
end
lasterror() = unsafe_string(ccall(sym(:fs2i_error), Cstring, ()))
check(rc::Integer) = rc < 0 ? error("libfmapsim2iq: ", lasterror()) : rc

const CTX = Dict{Int,Ptr{Cvoid}}()
function context(N::Int)
    get!(CTX, N) do
        h = ccall(sym(:fs2i_open), Ptr{Cvoid}, (Cint, Cint), N, 0)
        h == C_NULL && error("libfmapsim2iq: ", lasterror())
        h
    end
end
function release_contexts()
    for (_, h) in CTX
        ccall(sym(:fs2i_close), Cvoid, (Ptr{Cvoid},), h)
    end
    empty!(CTX)
end

function calculate_saxs_data_gpu(structure::Sim2Iq.Structure, N::Int, spacing::Float64, steps::Int,
                                 cutoff_offset::Float64, se_w::Float64, ou_w::Float64, in_w::Float64, B::Float64)
    t0 = time(); el() = time() - t0
    n = length(structure)
    x = Float64[a.x for a in structure]; y = Float64[a.y for a in structure]; z = Float64[a.z for a in structure]
    vol = Float64[a.vol for a in structure]; et = Int32[get_element_type(a.element) for a in structure]
    q = Vector{Float64}(undef, N ÷ 2 + 2); iq = similar(q)
    shell = zeros(2)
    c = context(N)
    @printf("[%.3f s] Mapping atoms to grid, FFT and binning on the GPU...\n", el())
    np = check(ccall(sym(:fs2i_iq), Cint,
                     (Ptr{Cvoid}, Cint, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Int32}, Float64, Cint, Float64,
                      Float64, Float64, Float64, Float64, Ptr{Float64}, Ptr{Float64}, Cint, Ptr{Float64}),
                     c, n, x, y, z, vol, et, spacing, steps, cutoff_offset, se_w, ou_w, in_w, B, q, iq, length(q), shell))
    steps >= 3 && @printf("Volume: %.12e\n", shell[1])
    steps >= 4 && @printf("Volume: %.12e\n", shell[2])
    @printf("[%.3f s] SAXS calculation complete.\n", el())
    return QIPoint[QIPoint(q[i], iq[i]) for i in 1:np]
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
    r = calculate_saxs_data_gpu(st, args.grid_size, args.grid_spacing, args.steps, args.cutoff_offset,
                                args.se_weight, args.ou_weight, args.in_weight, B)
    Sim2Iq.write_output(r, args.output_file) || return 1
    release_contexts()
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
    end
    if argv[5] == "--serve"       # keep the process (device context) and wait for jobs on stdin
        for line in eachline(stdin)
            job = strip(line)
            isempty(job) && continue
            run_job(job)
            println("done ", job); flush(stdout)
        end
    else
        foreach(run_job, argv[5:end])
    end
    release_contexts()
    return rc
end

end # module
