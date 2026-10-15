# stat.jl - fmapstat: Flyvbjerg-Petersen block statistics and B22 from the grids (tools/fmapstat.py).
#
# To keep the text output identical to the python tool this reproduces three python/numpy details:
#   * random.seed(2019); random.shuffle(list)           -> PyRandom (MT19937, init_by_array, getrandbits)
#   * numpy's pairwise summation (mean/var/average)     -> np_sum
#   * repr(float) formatting in the .b22 files          -> pyrepr

# ---- Python's Mersenne Twister ----

mutable struct PyRandom
    mt::Vector{UInt32}
    idx::Int
end

function init_genrand!(r::PyRandom, s::UInt32)
    r.mt[1] = s
    for i in 2:624
        p = r.mt[i-1]
        r.mt[i] = UInt32(1812433253) * (p ⊻ (p >> 30)) + UInt32(i - 1)
    end
    r.idx = 625          # mti = N + 1 (not yet generated)
end

function PyRandom(seed::Integer)
    r = PyRandom(zeros(UInt32, 624), 625)
    key = UInt32[]
    n = abs(seed)
    while n > 0
        push!(key, UInt32(n & 0xffffffff)); n >>= 32
    end
    isempty(key) && push!(key, UInt32(0))
    init_genrand!(r, UInt32(19650218))
    i = 2; j = 1                                   # 1-based versions of C's i=1, j=0
    klen = length(key)
    for _ in 1:max(624, klen)
        p = r.mt[i-1]
        r.mt[i] = (r.mt[i] ⊻ ((p ⊻ (p >> 30)) * UInt32(1664525))) + key[j] + UInt32(j - 1)
        i += 1; j += 1
        if i > 624
            r.mt[1] = r.mt[624]; i = 2
        end
        j > klen && (j = 1)
    end
    for _ in 1:623
        p = r.mt[i-1]
        r.mt[i] = (r.mt[i] ⊻ ((p ⊻ (p >> 30)) * UInt32(1566083941))) - UInt32(i - 1)
        i += 1
        if i > 624
            r.mt[1] = r.mt[624]; i = 2
        end
    end
    r.mt[1] = UInt32(0x80000000)
    r.idx = 625
    return r
end

function genrand_uint32!(r::PyRandom)
    mt = r.mt
    if r.idx >= 625
        mag(y) = (y & UInt32(1)) != 0 ? UInt32(0x9908b0df) : UInt32(0)
        for kk in 1:624-397
            y = (mt[kk] & UInt32(0x80000000)) | (mt[kk+1] & UInt32(0x7fffffff))
            mt[kk] = mt[kk+397] ⊻ (y >> 1) ⊻ mag(y)
        end
        for kk in 624-397+1:623
            y = (mt[kk] & UInt32(0x80000000)) | (mt[kk+1] & UInt32(0x7fffffff))
            mt[kk] = mt[kk+397-624] ⊻ (y >> 1) ⊻ mag(y)
        end
        y = (mt[624] & UInt32(0x80000000)) | (mt[1] & UInt32(0x7fffffff))
        mt[624] = mt[397] ⊻ (y >> 1) ⊻ mag(y)
        r.idx = 1
    end
    y = mt[r.idx]; r.idx += 1
    y ⊻= (y >> 11)
    y ⊻= (y << 7) & UInt32(0x9d2c5680)
    y ⊻= (y << 15) & UInt32(0xefc60000)
    y ⊻= (y >> 18)
    return y
end

function randbelow!(r::PyRandom, n::Int)
    k = 8sizeof(n) - leading_zeros(n)              # n.bit_length()
    x = genrand_uint32!(r) >> (32 - k)
    while x >= n
        x = genrand_uint32!(r) >> (32 - k)
    end
    return Int(x)
end

"random.shuffle(x) (in place)."
function py_shuffle!(r::PyRandom, x::AbstractVector)
    for i in length(x)-1:-1:1                       # python: i = len-1 .. 1 (0-based)
        j = randbelow!(r, i + 1)
        x[i+1], x[j+1] = x[j+1], x[i+1]
    end
    return x
end

# ---- numpy-compatible reductions ----

function pairwise_sum(a::AbstractVector{Float64}, lo::Int, n::Int)
    if n < 8
        res = 0.0
        @inbounds for i in 0:n-1
            res += a[lo+i]
        end
        return res
    elseif n <= 128
        @inbounds begin
            r = ntuple(j -> a[lo+j-1], 8)
            i = 8
            while i < n - (n % 8)
                r = ntuple(j -> r[j] + a[lo+i+j-1], 8)
                i += 8
            end
            res = ((r[1] + r[2]) + (r[3] + r[4])) + ((r[5] + r[6]) + (r[7] + r[8]))
            while i < n
                res += a[lo+i]; i += 1
            end
            return res
        end
    else
        n2 = n ÷ 2
        n2 -= n2 % 8
        return pairwise_sum(a, lo, n2) + pairwise_sum(a, lo + n2, n - n2)
    end
end

np_sum(a::AbstractVector{Float64}) = pairwise_sum(a, 1, length(a))
np_mean(a) = np_sum(a) / length(a)
function np_var(a::AbstractVector{Float64})
    m = np_mean(a)
    d = a .- m
    return np_sum(d .* d) / length(a)
end
np_std(a) = sqrt(np_var(a))

# ---- Flyvbjerg-Petersen blocking (the python module's functions) ----

function block_transformation(series::Vector{Float64})
    n = length(series) ÷ 2
    return [0.5 * (series[2i-1] + series[2i]) for i in 1:n]
end

function calculate_blocked_variances(series::Vector{Float64}; npmin::Int = 15)
    n_steps = length(series)
    var_(d, n) = np_var(d) / (n - 1)
    stderr_var(n) = sqrt(2.0 / (n - 1))
    output_var = Float64[var_(series, n_steps)]
    var_stderr = Float64[stderr_var(n_steps)]
    while n_steps > npmin
        series = block_transformation(series)
        n_steps = length(series)
        push!(output_var, var_(series, n_steps))
        push!(var_stderr, stderr_var(n_steps))
    end
    return output_var, var_stderr
end

function detect_fixed_point(fp_var::Vector{Float64}, fp_sev::Vector{Float64})
    n_trans = length(fp_var)
    left_index = 0; right_index = 0
    for i in 0:n_trans-2
        if abs(fp_var[i+2] - fp_var[i+1]) < fp_var[i+2] * fp_sev[i+2]
            left_index = i; break
        end
    end
    for i in n_trans-1:-1:0
        prev = i == 0 ? n_trans - 1 : i - 1          # numpy's fp_var[-1] for i == 0
        if abs(fp_var[i+1] - fp_var[prev+1]) < fp_var[prev+1] * fp_sev[prev+1]
            right_index = i; break
        end
    end
    if left_index >= 0 && right_index >= 0 && right_index >= left_index
        rng = left_index+1:right_index+1
        w = 1.0 ./ fp_sev[rng]
        return np_sum(fp_var[rng] .* w) / np_sum(w), true
    else
        return maximum(fp_var), false
    end
end

function fp_stderr(data::Vector{Float64})
    bv, bs = calculate_blocked_variances(data)
    var_mean, conv = detect_fixed_point(bv, bs)
    conv || @warn "Fixed point not found, returned value is a lower bound on the standard error"
    return sqrt(var_mean)
end

# ---- python float formatting ----

"repr(float) as python prints it."
function pyrepr(x::Float64)
    isnan(x) && return "nan"
    isinf(x) && return x > 0 ? "inf" : "-inf"
    x == 0.0 && return (signbit(x) ? "-0.0" : "0.0")
    local digits, e10
    for p in 1:17
        s = @sprintf("%.*e", p - 1, abs(x))
        if parse(Float64, s) == abs(x)
            m, e = split(s, 'e')
            digits = replace(m, "." => ""); e10 = parse(Int, e)
            break
        end
    end
    sign = x < 0 ? "-" : ""
    if -4 <= e10 < 16
        if e10 >= 0
            ip = e10 + 1
            d = rpad(digits, ip, '0')
            return sign * d[1:ip] * "." * (length(d) > ip ? d[ip+1:end] : "0")
        else
            return sign * "0." * "0"^(-e10 - 1) * digits
        end
    else
        mant = length(digits) > 1 ? digits[1] * "." * digits[2:end] : digits
        return sign * mant * "e" * (e10 < 0 ? "-" : "+") * lpad(string(abs(e10)), 2, '0')
    end
end

# ---- fmapstat modes ----

function read_stat_parms(filename::AbstractString)
    parms = Dict{String,Float64}()
    if !isfile(filename)
        println("Warning: $filename not found"); return parms
    end
    for line in eachline(filename)
        items = split(line, ":")
        length(items) >= 2 || continue
        key = strip(items[1])
        key in ("mass", "blen", "spacing", "nrot", "kbt") || continue
        toks = split(items[2])
        isempty(toks) && continue
        v = tryparse(Float64, toks[1])
        v === nothing || (parms[key] = v)
    end
    return parms
end

"The ern statistic column (3rd field) of the lines of `ern_type`, shuffled as python does."
function process_ern_data(ern_type::AbstractString, ern_file)
    files = ern_file === nothing ? ["fmap.ern", "fmap.ern.gz"] : [ern_file]
    data = Float64[]
    for fname in files
        isfile(fname) || continue
        io = endswith(fname, ".gz") ? GzipDecompressorStream(open(fname)) : open(fname)
        for line in eachline(io)
            if occursin(ern_type, line)
                parts = split(line)
                length(parts) >= 3 && push!(data, parse(Float64, parts[3]))
            end
        end
        close(io)
        break
    end
    isempty(data) && return nothing
    py_shuffle!(PyRandom(2019), data)
    return data
end

function compute_msf_stats(data::Vector{Float64})
    results = Tuple{Int,Float64,Float64,Float64}[]
    block = length(data) < 100 ? 2 : 100
    for i in block:block:length(data)+block-1
        sub = i <= length(data) ? data[1:i] : data
        push!(results, (length(sub), np_mean(sub), np_std(sub), fp_stderr(sub)))
    end
    return results
end

write_b22(io, msf, blen, conv, dx) = for (n, m, s, e) in msf
    println(io, n, " ", pyrepr(0.5 * ((blen^3 - m) * conv * libm_pow(dx, 3.0))), " ", pyrepr(0.5 * (s * conv * libm_pow(dx, 3.0))), " ",
            pyrepr(0.5 * (e * conv * libm_pow(dx, 3.0))))
end

function append_results(parms_file, results::Vector{Pair{String,Float64}})
    open(parms_file, "a") do f
        for (key, value) in results
            k = rpad(key, 12)
            if occursin("vr", key)
                write(f, k, " :", @sprintf("%16.3f", value), " angstrom\n")
            elseif occursin("vco", key) || occursin("vs", key)
                write(f, k, " :", @sprintf("%16.3f", value), " angstrom^3\n")
            elseif occursin("std", key)
                write(f, k, " :", @sprintf("%16.4f", value), " x 10^-4 mL mol / g^2\n")
            else
                write(f, k, " :", @sprintf("%16.3f", value), " x 10^-4 mL mol / g^2\n")
            end
        end
    end
end

function run_stats(; parms_file::String = "parms.txt", ern_file = nothing)
    parms = read_stat_parms(parms_file)
    if !haskey(parms, "mass") || !haskey(parms, "blen")
        println("Error: mass and blen must be defined in $parms_file"); return
    end
    mass = parms["mass"]; blen = Int(parms["blen"])
    conv = 6.022140857e+23 / 1e+27 / (mass^2) * 1000 * 10000
    dx = get(parms, "spacing", 0.6)
    results = Pair{String,Float64}[]
    rd = Dict{String,Float64}()
    nrot_detected = nothing
    for ern_type in ("v+e", "vol")
        data = process_ern_data(ern_type, ern_file)
        data === nothing && continue
        nrot_detected === nothing && (nrot_detected = length(data))
        msf = compute_msf_stats(data)
        isempty(msf) && continue
        open("fmap.ern.$ern_type.msf.b22", "w") do f
            write_b22(f, msf, blen, conv, dx)
        end
        n, m, s, e = msf[end]
        if ern_type == "vol"
            rd["FMAPvco"] = (blen^3 - m) * libm_pow(dx, 3.0)
            rd["FMAPexb22"] = 0.5 * ((blen^3 - m) * conv * libm_pow(dx, 3.0))
            rd["FMAPexb22std"] = 0.5 * (e * conv * libm_pow(dx, 3.0))
        else
            rd["FMAPvs"] = (blen^3 - m) * libm_pow(dx, 3.0)
            rd["FMAPb22"] = 0.5 * ((blen^3 - m) * conv * libm_pow(dx, 3.0))
            rd["FMAPb22std"] = 0.5 * (e * conv * libm_pow(dx, 3.0))
        end
    end
    haskey(rd, "FMAPvco") && (rd["FMAPvr"] = libm_pow(rd["FMAPvco"] / (32.0 / 3 * pi), 1.0 / 3.0) * 2.0)
    if nrot_detected !== nothing && !haskey(parms, "nrot")
        open(parms_file, "a") do f
            write(f, "nrot         : $nrot_detected\n")
        end
    end
    # python's dict order: the v+e group first, then vol, then vr
    order = ["FMAPvs", "FMAPb22", "FMAPb22std", "FMAPvco", "FMAPexb22", "FMAPexb22std", "FMAPvr"]
    append_results(parms_file, Pair{String,Float64}[k => rd[k] for k in order if haskey(rd, k)])
    return nothing
end

GetkBT(tempK) = (N_A = 6.022045000e+23; k_B = 3.297623030e-24; N_A * k_B * tempK / 1000.0)

"Histogram (256 bins, int8 value + 128) of one gzip'd tErn grid, with the combined elec+vdw value for 2-byte cells."
function ve_histogram(fname::AbstractString, blen::Int, tern::Int)
    data = open(io -> read(GzipDecompressorStream(io)), fname)
    length(data) == blen^3 * tern || error("$fname: expected $(blen^3 * tern) bytes, got $(length(data))")
    hist = zeros(Int, 256)
    if tern == 1
        @inbounds for b in data
            hist[Int(reinterpret(Int8, b))+129] += 1
        end
    else
        @inbounds for c in 1:blen^3
            e0 = Int(reinterpret(Int8, data[2c-1])); e1 = Int(reinterpret(Int8, data[2c]))
            v = (e0 == 127 || e1 == 127) ? 127 : clamp(e0 + e1, -128, 127)
            hist[v+129] += 1
        end
    end
    return hist
end

function run_stats_ve(; nrot = nothing, scl_file = nothing, ve_path::String = "ve", parms_file::String = "parms.txt")
    parms = read_stat_parms(parms_file)
    if !haskey(parms, "mass") || !haskey(parms, "blen")
        println("Error: mass and blen required in $parms_file"); return
    end
    if nrot === nothing
        haskey(parms, "nrot") || (println("Error: nrot must be provided via --nrot or defined in $parms_file"); return)
        nrot = Int(parms["nrot"])
    elseif !haskey(parms, "nrot")
        open(parms_file, "a") do f
            write(f, "nrot         : $nrot\n")
        end
    end
    mass = parms["mass"]; blen = Int(parms["blen"])
    conv = 6.022140857e+23 / 1e+27 / (mass^2) * 1000 * 10000
    dx = get(parms, "spacing", 0.6)
    tern = tern_bytes_from_env()
    tern_scl = 8.0
    hist_combined = zeros(Int, 256)
    histograms = Vector{Int}[]
    for i in 0:nrot-1
        fname = @sprintf("%s/tErn.%05d.mat.gz", ve_path, i)
        isfile(fname) || continue
        try
            h = ve_histogram(fname, blen, tern)
            push!(histograms, h); hist_combined .+= h
        catch
            continue
        end
    end
    if isempty(histograms) || sum(hist_combined) == 0
        println("No ve data found"); return
    end
    if scl_file !== nothing
        fp = scl_file == "-" ? stdin : open(scl_file)
        for line in eachline(fp)
            parts = split(line)
            length(parts) < 3 && continue
            escl = parse(Float64, parts[1]); vscl = parse(Float64, parts[2]); tempk = parse(Float64, parts[3]) + 273.15
            kbt = GetkBT(tempk)
            vol = 0.0
            for b in 0:255
                hist_combined[b+1] > 0 || continue
                v8 = b - 128
                v8 == 127 && continue
                vol += hist_combined[b+1] * exp(-(v8 / tern_scl * escl) / kbt)
            end
            vol /= nrot
            b22 = 0.5 * ((blen^3 - vol) * conv * libm_pow(dx, 3.0))
            @printf("%8.3f %8.3f %8.2f %16.3f\n", escl, vscl, tempk, b22)
        end
        scl_file == "-" || close(fp)
    else
        kbt = get(parms, "kbt", 0.5924)
        volumes = Float64[]; b20_counts = Int[]
        for h in histograms
            vol = 0.0
            push!(b20_counts, h[256])
            for b in 0:255
                h[b+1] > 0 || continue
                v8 = b - 128
                v8 == 127 && continue
                vol += h[b+1] * exp(-(v8 / tern_scl) / kbt)
            end
            push!(volumes, vol)
        end
        msf = compute_msf_stats(volumes)
        isempty(msf) && return
        open("ve.msf.b22", "w") do f
            write_b22(f, msf, blen, conv, dx)
        end
        n, m, s, e = msf[end]
        vs = (blen^3 - m) * libm_pow(dx, 3.0)
        b22 = 0.5 * ((blen^3 - m) * conv * libm_pow(dx, 3.0))
        b22std = 0.5 * (e * conv * libm_pow(dx, 3.0))
        b20msf = compute_msf_stats(Float64[c / blen^3 for c in b20_counts])
        open(parms_file, "a") do f
            write(f, "VEvs         :", @sprintf("%16.3f", vs), " angstrom^3\n")
            write(f, "VEb22        :", @sprintf("%16.3f", b22), " x 10^-4 mL mol / g^2\n")
            write(f, "VEb22std     :", @sprintf("%16.4f", b22std), " x 10^-4 mL mol / g^2\n")
            if !isempty(b20msf)
                _, mb, _, eb = b20msf[end]
                write(f, "VEb20        :", @sprintf("%16.3f", 0.5 * mb * blen^3 * conv * libm_pow(dx, 3.0)), " x 10^-4 mL mol / g^2\n")
                write(f, "VEb20std     :", @sprintf("%16.4f", 0.5 * eb * blen^3 * conv * libm_pow(dx, 3.0)), " x 10^-4 mL mol / g^2\n")
            end
        end
    end
    open("vehist.txt", "w") do f
        for b in 0:255
            write(f, @sprintf("%12.6f %12d\n", (b - 128) / tern_scl, hist_combined[b+1]))
        end
    end
    return nothing
end

function run_stats_mat(; nrot = nothing, mat_file = nothing, scl_file = nothing, parms_file::String = "parms.txt")
    parms = read_stat_parms(parms_file)
    if !haskey(parms, "mass") || !haskey(parms, "blen")
        println("Error: mass and blen required in $parms_file"); return
    end
    if nrot === nothing
        haskey(parms, "nrot") || (println("Error: nrot must be provided via --nrot or defined in $parms_file"); return)
        nrot = Int(parms["nrot"])
    elseif !haskey(parms, "nrot")
        open(parms_file, "a") do f
            write(f, "nrot         : $nrot\n")
        end
    end
    mass = parms["mass"]; blen = Int(parms["blen"])
    conv = 6.022140857e+23 / 1e+27 / (mass^2) * 1000 * 10000
    dx = get(parms, "spacing", 0.6)
    nbin = 4001; lowb = -40.0; dxbin = 0.02
    fname = mat_file !== nothing ? mat_file : isfile("mat.bin.gz") ? "mat.bin.gz" : isfile("mat.bin") ? "mat.bin" : nothing
    fname === nothing && (println("Error: mat.bin.gz or mat.bin not found"); return)
    isfile(fname) || (println("Error: $fname not found"); return)
    raw = endswith(fname, ".gz") ? open(io -> read(GzipDecompressorStream(io)), fname) : read(fname)
    mat = reshape(reinterpret(Float64, raw), nbin, nbin)'      # numpy (nbin, nbin) row-major
    if scl_file !== nothing
        fp = scl_file == "-" ? stdin : open(scl_file)
        for line in eachline(fp)
            parts = split(line)
            length(parts) < 3 && continue
            escl = parse(Float64, parts[1]); vscl = parse(Float64, parts[2]); tempk = parse(Float64, parts[3]) + 273.15
            kbt = GetkBT(tempk)
            vol = 0.0
            for ei in 0:nbin-1, vi in 0:nbin-1
                if mat[ei+1, vi+1] > 0.5
                    vol += exp(((dxbin * ei + lowb) * escl + (dxbin * vi + lowb) * vscl) / -kbt) * mat[ei+1, vi+1] / nrot
                end
            end
            @printf("%8.3f %8.3f %8.2f %16.3f\n", escl, vscl, tempk, 0.5 * ((blen^3 - vol) * conv * libm_pow(dx, 3.0)))
        end
        scl_file == "-" || close(fp)
    else
        kbt = get(parms, "kbt", 0.5924)
        weighted = 0.0; total = 0.0
        for ei in 0:nbin-1, vi in 0:nbin-1
            if mat[ei+1, vi+1] > 0.5
                weighted += exp(((dxbin * ei + lowb) + (dxbin * vi + lowb)) / -kbt) * mat[ei+1, vi+1] / nrot
                total += mat[ei+1, vi+1]
            end
        end
        open(parms_file, "a") do f
            write(f, "MATv         :", @sprintf("%16.3f", (blen^3 - weighted) * libm_pow(dx, 3.0)), " angstrom^3\n")
            write(f, "MATb22       :", @sprintf("%16.3f", 0.5 * ((blen^3 - weighted) * conv * libm_pow(dx, 3.0))), " x 10^-4 mL mol / g^2\n")
            write(f, "MATb20       :", @sprintf("%16.3f", 0.5 * (blen^3 - total / nrot) * conv * libm_pow(dx, 3.0)), " x 10^-4 mL mol / g^2\n")
        end
    end
    return nothing
end

"""
    main_fmapstat(args)

fmapstat <datafile> <col>                      FP standard error of one column
fmapstat stats [--ern [FILE] | --ve PATH | --mat [FILE]] [--nrot N] [--scl FILE] [--parm FILE]
"""
function main_fmapstat(argv::Vector{String})
    if length(argv) >= 2 && argv[1] != "stats"
        pos = tryparse(Int, argv[2])
        if pos !== nothing && isfile(argv[1])
            data = Float64[]
            for line in eachline(argv[1])
                (isempty(line) || line[1] == '#') && continue
                push!(data, parse(Float64, split(line)[pos+1]))
            end
            println(length(data), " ", pyrepr(np_mean(data)), " ", pyrepr(np_std(data)), " ", pyrepr(fp_stderr(data)))
            return 0
        end
    end
    if isempty(argv) || argv[1] != "stats"
        println("Usage: fmapstat stats [--ern [FILE] | --ve PATH | --mat [FILE]] [--nrot N] [--scl FILE] [--parm FILE]")
        println("       fmapstat <datafile> <col>")
        return 0
    end
    ern = nothing; ve = nothing; mat = nothing; nrot = nothing; scl = nothing; parm = "parms.txt"
    modes = 0
    i = 2
    optarg(i) = (i + 1 <= length(argv) && !startswith(argv[i+1], "--")) ? argv[i+1] : nothing
    while i <= length(argv)
        a = argv[i]
        if a == "--ern"
            modes += 1; ern = something(optarg(i), true); ern === true || (i += 1)
        elseif a == "--mat"
            modes += 1; mat = something(optarg(i), true); mat === true || (i += 1)
        elseif a == "--ve"
            modes += 1; ve = argv[i+1]; i += 1
        elseif a == "--nrot"
            nrot = parse(Int, argv[i+1]); i += 1
        elseif a == "--scl"
            scl = argv[i+1]; i += 1
        elseif a == "--parm"
            parm = argv[i+1]; i += 1
        else
            println(stderr, "Unknown option: $a"); return 1
        end
        i += 1
    end
    if modes > 1
        println("Error: Please specify only one of --ern, --ve, or --mat"); return 1
    elseif ve !== nothing
        endswith(ve, ".zip") && (println("Error: .zip input is not supported by the Julia fmapstat"); return 1)
        run_stats_ve(; nrot, scl_file = scl, ve_path = ve, parms_file = parm)
    elseif mat !== nothing
        run_stats_mat(; nrot, mat_file = mat === true ? nothing : mat, scl_file = scl, parms_file = parm)
    else
        run_stats(; parms_file = parm, ern_file = ern === nothing || ern === true ? nothing : ern)
    end
    return 0
end
