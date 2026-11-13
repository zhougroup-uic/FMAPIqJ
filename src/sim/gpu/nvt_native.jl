# nvt_native.jl - the NVT GPU engine through native/libfmapsim.so (CUDA C, see native/fmapsim.h); no CUDA.jl.
# The library owns the device state (tables, boxes, RNG, moves); this file is the host side: loading fmap.paths, the library calls, particle
# files, step-size adaptation and logs.
using Printf
using Libdl

# ---- the library ----
function fsim_libpath()
    p = get(ENV, "FMAPSIM_LIB", "")
    isempty(p) || return p
    return normpath(joinpath(@__DIR__, "..", "..", "..", "native", "libfmapsim.so"))
end
const FSIM_HANDLE = Ref{Ptr{Cvoid}}(C_NULL)
function fsim_sym(name::Symbol)
    if FSIM_HANDLE[] == C_NULL
        p = fsim_libpath()
        isfile(p) || error("$p not found: build it with `module load CUDA && make -C native`, or set FMAPSIM_LIB")
        FSIM_HANDLE[] = Libdl.dlopen(p, Libdl.RTLD_NOW | Libdl.RTLD_LOCAL)
    end
    return Libdl.dlsym(FSIM_HANDLE[], name)
end
fsim_error() = unsafe_string(ccall(fsim_sym(:fsim_error), Cstring, ()))
fsim_check(rc::Integer) = rc < 0 ? error("libfmapsim: ", fsim_error()) : rc

const ROT_DEFAULT = 3.0

# ---- tables: fmap.paths -> rotation system (rot.eul) + one fmap.ve.sp2p per type pair ----
struct NativeGrid
    tab::Ptr{Cvoid}
    nrot::Int
    nbin::Int
    scl::Float64
    n_types::Int
    blens::Vector{Int}
    angle_indices::Vector{Int16}
    tern::Int                  # cell format of the grids (read from the files): 1 summed channel, 2 [elec, vdw]
end

const ROT_FILE_CACHE = Dict{Tuple{String,Int},RotTables}()
function load_rot_file(path::String)
    mode = amap_mode_from_env()
    key = (realpath(path), mode)
    haskey(ROT_FILE_CACHE, key) && return ROT_FILE_CACHE[key]
    return ROT_FILE_CACHE[key] = rot_tables_from_eul(path; mode)
end

# spt: 2 = the sp2p blob read in place (default: 7x less device memory for the N=200 grid, about 3% slower), 1 = lookup table built on the device (fastest, larger);
# use_float: lookup arithmetic in float (same energies). Defaults from FMAP_SPT / FMAP_FLOAT.
function load_fmap_paths_native(paths_file::String, n_types::Int; device::Int = 0,
                                spt::Int = parse(Int, get(ENV, "FMAP_SPT", "2")),
                                use_float::Bool = get(ENV, "FMAP_FLOAT", "0") == "1")
    n_ctx = n_types * (n_types + 1) ÷ 2
    lines = readlines(paths_file)
    length(lines) >= 2 * n_ctx || error("$paths_file needs $(2*n_ctx) lines (rot/ve x $n_ctx contexts), got $(length(lines))")
    rt = nothing; ves = String[]
    for k in 1:n_ctx
        rot_path = String(strip(lines[2k-1])); push!(ves, String(strip(lines[2k])))
        rk = load_rot_file(rot_path)
        rt === nothing ? (rt = rk) : (rk.aidx == rt.aidx && rk.amap == rt.amap && rk.rotm == rt.rotm) ||
            error("$paths_file: context $k rotation tables differ from context 1")
    end
    (rt.amap.fold || rt.amap.look) && error("the native engine needs the plain pair table (AMAP_FOLD=0)")
    tab = GC.@preserve ves begin
        cptrs = Cstring[Base.unsafe_convert(Cstring, s) for s in ves]
        ccall(fsim_sym(:fsim_tab_open), Ptr{Cvoid},
              (Cint, Cint, Cint, Cint, Ptr{Int16}, Ptr{Int16}, Ptr{Float64}, Cint, Ptr{Cstring}, Cint, Cint),
              device, n_types, rt.nrot, rt.nbin, rt.aidx, rt.amap.dense, rt.rotm, n_ctx, cptrs, spt, use_float)
    end
    tab == C_NULL && error("libfmapsim: ", fsim_error())
    scl = ccall(fsim_sym(:fsim_tab_scl), Float64, (Ptr{Cvoid},), tab)
    blens = [ccall(fsim_sym(:fsim_tab_blen), Cint, (Ptr{Cvoid}, Cint), tab, c - 1) for c in 1:n_ctx]
    tern = Int(ccall(fsim_sym(:fsim_tab_tern), Cint, (Ptr{Cvoid},), tab))      # cell format, from the grid files
    return NativeGrid(tab, rt.nrot, rt.nbin, scl, n_types, Int.(blens), rt.aidx, tern)
end

rot_index(g::NativeGrid, b1::Float64, b2::Float64, b3::Float64) =
    Int32(ccall(fsim_sym(:fsim_rot_index), Cint, (Ptr{Cvoid}, Float64, Float64, Float64), g.tab, b1, b2, b3))

# ---- orientation <-> euler (particle_fmap.c's euler_to_bins/bins_to_euler) ----
cwrap(x::Float64, m::Float64) = (x = rem(x, m); x < 0.0 ? x + m : x)
function euler_to_bins(ea::Float64, eb::Float64, ec::Float64, nbin::Integer)
    b1 = cwrap(ea + pi, 2pi) * nbin / (2pi)
    b2 = (1.0 - cos(clamp(eb, 0.0, pi))) * nbin / 2.0
    b3 = cwrap(ec + pi, 2pi) * nbin / (2pi)
    return b1, b2, b3
end
bins_to_euler(b1, b2, b3, nbin::Integer) =
    (-pi + b1 / nbin * 2pi, acos(clamp(1.0 - b2 / nbin * 2.0, -1.0, 1.0)), -pi + b3 / nbin * 2pi)

# ---- a box (replica) ----
mutable struct NBox
    h::Ptr{Cvoid}
    cap::Int
    box::NTuple{3,Float64}
end

function NBox(g::NativeGrid, cap::Int, box::NTuple{3,Float64}, kT::Float64, es::Float64, vs::Float64)
    h = ccall(fsim_sym(:fsim_box_open), Ptr{Cvoid}, (Ptr{Cvoid}, Cint, Ptr{Float64}, Float64, Float64, Float64),
              g.tab, cap, collect(box), kT, es, vs)
    h == C_NULL && error("libfmapsim: ", fsim_error())
    return NBox(h, cap, box)
end
close_box(b::NBox) = (b.h == C_NULL || ccall(fsim_sym(:fsim_box_close), Cvoid, (Ptr{Cvoid},), b.h); b.h = C_NULL)

function box_set!(b::NBox, pos, bins, ptype::Vector{Int32}, ridx, lcg::UInt64)
    n = length(ridx)
    ptype_full = zeros(Int32, b.cap); ptype_full[1:length(ptype)] = ptype      # types of the reserved slots follow the active ones
    fsim_check(ccall(fsim_sym(:fsim_box_set), Cint, (Ptr{Cvoid}, Cint, Ptr{Float64}, Ptr{Float64}, Ptr{Int32}, Ptr{Int32}, UInt64),
                     b.h, n, pos, bins, Int32.(ridx), ptype_full, lcg))
end

function box_get(b::NBox, n::Integer)
    pos = zeros(3n); bins = zeros(3n); ridx = zeros(Int32, n); ptype = zeros(Int32, n); lcg = Ref(UInt64(0))
    fsim_check(ccall(fsim_sym(:fsim_box_get), Cint, (Ptr{Cvoid}, Cint, Ptr{Float64}, Ptr{Float64}, Ptr{Int32}, Ptr{Int32}, Ptr{UInt64}),
                     b.h, n, pos, bins, ridx, ptype, lcg))
    return pos, bins, ridx, ptype
end

function total_energy(b::NBox, n::Integer)
    e = Ref(0.0)
    fsim_check(ccall(fsim_sym(:fsim_total_energy), Cint, (Ptr{Cvoid}, Cint, Ptr{Float64}), b.h, n, e))
    return e[]
end

mc_launch!(b::NBox, n, batch, js, rs) =
    fsim_check(ccall(fsim_sym(:fsim_mc_launch), Cint, (Ptr{Cvoid}, Cint, Cint, Float64, Float64), b.h, n, batch, js, rs))
insert_launch!(b::NBox, n, batch) =
    fsim_check(ccall(fsim_sym(:fsim_insert_launch), Cint, (Ptr{Cvoid}, Cint, Cint), b.h, n, batch))
reservoir_launch!(b::NBox, n, committed, batch, max_accepts) =
    fsim_check(ccall(fsim_sym(:fsim_reservoir_launch), Cint, (Ptr{Cvoid}, Cint, Cint, Cint, Cint), b.h, n, committed, batch, max_accepts))
function wait_box(b::NBox; mc::Bool = false, ins::Bool = false, res::Bool = false)
    m = zeros(6); i = zeros(8); r = zeros(4)
    fsim_check(ccall(fsim_sym(:fsim_wait), Cint, (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}), b.h,
                     mc ? m : C_NULL, ins ? i : C_NULL, res ? r : C_NULL))
    return m, i, r
end

# ---- initial placement: the library places with overlap rejection; the types of the new slots are decided up front ----
const N_TYPES = Ref(1)
last_type(n_counts::Int) = Int32(n_counts < N_TYPES[] ? n_counts : N_TYPES[] - 1)
function intended_types(counts::Vector{Int}, n::Int)
    it = Vector{Int32}(undef, n)
    idx = 1
    for (t, cnt) in enumerate(counts)
        for _ in 1:cnt
            idx > n && break
            it[idx] = Int32(t - 1); idx += 1
        end
    end
    last_t = last_type(length(counts))
    while idx <= n
        it[idx] = last_t; idx += 1
    end
    return it
end

# Appends up to `n_new` particles of the types `types` to the n_exist ones; returns the extended arrays and the new RNG state.
function place_particles(g::NativeGrid, es::Float64, vs::Float64, state::UInt64, box::NTuple{3,Float64},
                         pos::Vector{Float64}, bins::Vector{Float64}, ridx::Vector{Int32}, ptype::Vector{Int32}, types::Vector{Int32})
    n_exist = length(ridx); n_new = length(types)
    p = zeros(3 * (n_exist + n_new)); b = zeros(3 * (n_exist + n_new)); r = zeros(Int32, n_exist + n_new); t = zeros(Int32, n_exist + n_new)
    p[1:3n_exist] = pos[1:3n_exist]; b[1:3n_exist] = bins[1:3n_exist]; r[1:n_exist] = ridx; t[1:n_exist] = ptype
    st = Ref(state)
    placed = fsim_check(ccall(fsim_sym(:fsim_place), Cint,
                              (Ptr{Cvoid}, Float64, Float64, Ptr{Float64}, Cint, Cint, Ptr{Int32}, Ptr{Float64}, Ptr{Float64}, Ptr{Int32},
                               Ptr{Int32}, Ptr{UInt64}),
                              g.tab, es, vs, collect(box), n_exist, n_new, types, p, b, r, t, st))
    n = n_exist + placed
    return p[1:3n], b[1:3n], t[1:n], r[1:n], st[]
end

function random_particles(g::NativeGrid, es, vs, state::UInt64, n_active::Int, box::NTuple{3,Float64}, type_counts::Vector{Int})
    types = intended_types(type_counts, n_active)
    pos, bins, ptype, ridx, state = place_particles(g, es, vs, state, box, Float64[], Float64[], Int32[], Int32[], types)
    length(ridx) < n_active && println("Warning: placed only $(length(ridx)) of $n_active particles")
    return pos, bins, ptype, ridx, state
end

function topup_particles(g::NativeGrid, es, vs, state::UInt64, pos, bins, ptype, ridx, target::Int, box::NTuple{3,Float64}, topup_counts::Vector{Int})
    start = length(ridx)
    start >= target && return pos, bins, ptype, ridx, state
    types = intended_types(topup_counts, target - start)
    pos, bins, ptype, ridx, state = place_particles(g, es, vs, state, box, pos, bins, ridx, ptype, types)
    length(ridx) < target && println("Warning: topped up only $(length(ridx) - start) of $(target - start) particles")
    return pos, bins, ptype, ridx, state
end

# ---- step-size adaptation (simulation_gemc_gpu.cu's sim_gemc_gpu_adjust_step_sizes) ----
mutable struct StepTune
    js::Float64
    rs::Float64
    js0::Float64
    rs0::Float64
    js_max::Float64
    rs_max::Float64
    ntry::Int64
    nacc::Int64
    nrtry::Int64
    nracc::Int64
end
StepTune(js0::Float64, rs0::Float64, js_max::Float64, rs_max::Float64) = StepTune(js0, rs0, js0, rs0, js_max, rs_max, 0, 0, 0, 0)

function adjust_step_sizes!(t::StepTune, target_ratio::Float64)
    if t.ntry > 0
        if t.nacc / t.ntry < target_ratio
            t.js > t.js0 && (t.js *= 0.99)
        else
            t.js < t.js_max && (t.js *= 1.01)
        end
        t.ntry = 0; t.nacc = 0
    end
    if t.nrtry > 0
        if t.nracc / t.nrtry < target_ratio
            t.rs > t.rs0 && (t.rs *= 0.99)
        else
            t.rs < t.rs_max && (t.rs *= 1.01)
        end
        t.nrtry = 0; t.nracc = 0
    end
    return
end

# One replica ("triple" of the GEMC code: box 1, an always empty box 2, and the counters of the log).
mutable struct Triple
    box1::NBox
    n_active1::Int32
    n_active2::Int32
    e1::Float64
    e2::Float64
    n_moves1::Int64
    n_accepted1::Int64
    n_moves2::Int64
    n_accepted2::Int64
    n_transfer_attempts::Int64
    n_transfer_accepted::Int64
    nta1::Int32
    nta2::Int32
    tune1::StepTune
    tune2::StepTune
    committed1::Int32
    committed2::Int32
    deficit::Int
end

# ---- particle files (production text format, traj.{stage}.{1,2}.txt) ----
function save_particles(path::String, pos, bins, ptype, n_active::Integer, nbin::Integer, energy::Float64, js::Float64, rs::Float64)
    open(path, "w") do io
        @printf(io, "# GPU GEMC mask: %d particles  energy: %.6f  js=%.10g rs=%.10g\n", n_active, energy, js, rs)
        for slot in 0:Int(n_active)-1
            e = bins_to_euler(bins[slot*3+1], bins[slot*3+2], bins[slot*3+3], nbin)
            @printf(io, "%16d%16.10f%16.10f%16.10f%16.10f%16.10f%16.10f%8d\n", slot + 1,
                    pos[slot*3+1], pos[slot*3+2], pos[slot*3+3], e[1], e[2], e[3], ptype[slot+1])
        end
    end
    return Int(n_active)
end

# "%*d %lf %lf %lf %lf %lf %lf %d": the id column is ignored (re-sequenced into slots 0..n-1)
function load_particles(path::String, g::NativeGrid)
    pos = Float64[]; bins = Float64[]; ptype = Int32[]; ridx = Int32[]
    js_rs = nothing
    for line in eachline(path)
        if startswith(line, '#')
            m = match(r"^# GPU GEMC mask: \d+ particles\s+energy: \S+\s+js=(\S+) rs=(\S+)", line)
            if m !== nothing
                js = tryparse(Float64, m.captures[1]); rs = tryparse(Float64, m.captures[2])
                (js !== nothing && rs !== nothing) && (js_rs = (js, rs))
            end
            continue
        end
        fields = split(line)
        length(fields) < 7 && continue
        push!(pos, parse(Float64, fields[2]), parse(Float64, fields[3]), parse(Float64, fields[4]))
        b = euler_to_bins(parse(Float64, fields[5]), parse(Float64, fields[6]), parse(Float64, fields[7]), g.nbin)
        push!(bins, b[1], b[2], b[3])
        push!(ptype, length(fields) >= 8 ? parse(Int32, fields[8]) : Int32(0))
        push!(ridx, rot_index(g, b[1], b[2], b[3]))
    end
    return pos, bins, ptype, ridx, js_rs
end

# ---- per-replica log (main_gemc_gpu.cu's log.<stage> format) ----
count_type0(ptype::Vector{Int32}) = count(==(Int32(0)), ptype)

function open_triple_log(outdir::AbstractString, idx_stop::Int, N::Int, tr::Triple,
                         box_str::AbstractString, v1_str::AbstractString, label::AbstractString,
                         rpad::Int, seed::UInt64, n_replicas::Int, n_types::Int,
                         L1::Float64, L2::Float64, temperature::Float64,
                         elec_scale::Float64, vdw_scale::Float64, n_cycles::Int, sample_cycles::Int,
                         transfer_freq::Float64, target_ratio::Float64, adjust_cycles::Int)
    io = open("$outdir/log.$idx_stop", "w")
    n1_ini = Int(tr.n_active1); n2_ini = Int(tr.n_active2)
    @printf(io, "GPU Monte Carlo (n_types=%d)  L=%s V=%s R%0*d/%d\n", n_types, box_str, v1_str, rpad, seed, n_replicas)
    @printf(io, "N_total=%d  n1=%d n2=%d  L1=%.3f L2=%.3f  T=%.3f  elec_scale=%.4f  vdw_scale=%.4f  n_cycles=%d  sample=%d  transfer_freq=%.2f  target_ratio=%.2f  adjust_cycles=%d\n",
            N, n1_ini, n2_ini, L1, L2, temperature, elec_scale, vdw_scale, n_cycles, sample_cycles, transfer_freq, target_ratio, adjust_cycles)
    @printf(io, "Initial: N1=%d(%d) N2=%d(%d)  E1=%.2f E2=%.2f\n", n1_ini, tr.nta1, n2_ini, tr.nta2, tr.e1, tr.e2)
    print(io, "----------------------------------------\n")
    flush(io)
    return io
end

# main_gemc_gpu.cu's write_conf_frame_gpu: "# cycle=.. n=.. energy=.." then "id x y z e1 e2 e3 type" (%.6f) per active particle
function write_conf_frame(io::IO, box::NBox, n_active::Integer, cycle::Integer, energy::Float64, nbin::Integer)
    pos, bins, _, ptype = box_get(box, n_active)
    @printf(io, "# cycle=%d n=%d energy=%.6f\n", cycle, n_active, energy)
    for slot in 0:Int(n_active)-1
        e = bins_to_euler(bins[slot*3+1], bins[slot*3+2], bins[slot*3+3], nbin)
        @printf(io, "%d %.6f %.6f %.6f %.6f %.6f %.6f %d\n", slot + 1, pos[slot*3+1], pos[slot*3+2], pos[slot*3+3], e[1], e[2], e[3], ptype[slot+1])
    end
    flush(io)
end

function log_cycle!(io::IO, cycles_done::Int, tr::Triple, cps::Float64)
    n1 = Int(tr.n_active1); n2 = Int(tr.n_active2)
    ar1 = tr.n_moves1 > 0 ? tr.n_accepted1 / tr.n_moves1 : 0.0
    ar2 = tr.n_moves2 > 0 ? tr.n_accepted2 / tr.n_moves2 : 0.0
    tf_rate = tr.n_transfer_attempts > 0 ? tr.n_transfer_accepted / tr.n_transfer_attempts : 0.0
    @printf(io, "Cycle %8d: N1=%4d(%d) N2=%4d(%d)  E1=%10.2f E2=%10.2f  A1=%.3f A2=%.3f  TF=%.3f  speed=%.0f/s  js1=%.4g rs1=%.4g js2=%.4g rs2=%.4g\n",
            cycles_done, n1, tr.nta1, n2, tr.nta2, tr.e1, tr.e2, ar1, ar2, tf_rate, cps, tr.tune1.js, tr.tune1.rs, tr.tune2.js, tr.tune2.rs)
    flush(io)
end

function log_final!(io::IO, tr::Triple, total_time::Float64, n_cycles::Int, traj1::String, traj2::String)
    n1 = Int(tr.n_active1); n2 = Int(tr.n_active2)
    ar1 = tr.n_moves1 > 0 ? tr.n_accepted1 / tr.n_moves1 : 0.0
    ar2 = tr.n_moves2 > 0 ? tr.n_accepted2 / tr.n_moves2 : 0.0
    tf_rate = tr.n_transfer_attempts > 0 ? tr.n_transfer_accepted / tr.n_transfer_attempts : 0.0
    print(io, "----------------------------------------\n")
    @printf(io, "Final: N1=%d(%d) N2=%d(%d)  E1=%.2f E2=%.2f  A1=%.3f A2=%.3f  TF=%.3f\n", n1, tr.nta1, n2, tr.nta2, tr.e1, tr.e2, ar1, ar2, tf_rate)
    @printf(io, "Time: %.2f s  Speed: %.0f cycles/s\n", total_time, total_time > 0.0 ? n_cycles / total_time : 0.0)
    @printf(io, "Saved: %s  %s\n", traj1, traj2)
    close(io)
end
