# ---- q-vectors (fft.c) ----
struct QVectors
    qx::Vector{Float64}
    qy::Vector{Float64}
    qz::Vector{Float64}
    n::Int
    spacing::Float64
end

function create_q_vectors(grid_size::Int, grid_spacing::Float64)
    factor = 2π / (grid_size * grid_spacing)
    half = grid_size ÷ 2
    q = [i <= half ? i * factor : (i - grid_size) * factor for i in 0:grid_size-1]
    return QVectors(q, copy(q), copy(q), grid_size, grid_spacing)
end

# In-place forward 3D complex FFT (single precision, FFTW_ESTIMATE as in C)
function calculate_fft!(A::Array{ComplexF32,3})
    FFTW.set_num_threads(Threads.nthreads())
    plan = plan_fft!(A; flags = FFTW.ESTIMATE)
    plan * A
    return A
end

# ---- binning (binning.c) ----
struct QIPoint
    q::Float64
    intensity::Float64
end

mutable struct BinData
    sums::Vector{Float64}
    counts::Vector{Int}
    q_sums::Vector{Float64}
    max_bins::Int
    q_bin_width::Float64
    qmax::Float64
    Iq0::Float64
end

function create_bin_data(qv::QVectors)
    qmax = qv.qx[qv.n÷2] - 1e-8                       # C: qx[box/2 - 1]
    q_bin_width = 2π / (qv.n * qv.spacing) - 1e-8
    max_bins = floor(Int, qmax / q_bin_width) + 1
    return BinData(zeros(max_bins), zeros(Int, max_bins), zeros(max_bins),
                   max_bins, q_bin_width, qmax, 0.0)
end

# Accumulate |F_vac*exp(B q²/16π²) + F_other|² in spherical shells
function accumulate_intensity!(bins::BinData, ft_vac::Array{ComplexF32,3},
                               ft_other::Union{Nothing,Array{ComplexF32,3}},
                               qv::QVectors, b_factor::Float64)
    re0 = Float64(real(ft_vac[1])) + (ft_other === nothing ? 0.0 : Float64(real(ft_other[1])))
    im0 = Float64(imag(ft_vac[1])) + (ft_other === nothing ? 0.0 : Float64(imag(ft_other[1])))
    bins.Iq0 = re0 * re0 + im0 * im0

    N = qv.n
    nb = bins.max_bins
    nt = Threads.nthreads()
    lsums = [zeros(nb) for _ in 1:nt]
    lqs = [zeros(nb) for _ in 1:nt]
    lcnt = [zeros(Int, nb) for _ in 1:nt]
    qmax = bins.qmax
    bw = bins.q_bin_width
    Threads.@threads :static for ti in 1:nt
        sums = lsums[ti]; qs = lqs[ti]; cnt = lcnt[ti]
        @inbounds for k in ti:nt:N
            qz2 = qv.qz[k]^2
            for j in 1:N
                qyz2 = qv.qy[j]^2 + qz2
                for i in 1:N
                    (i == 1 && j == 1 && k == 1) && continue
                    q = sqrt(qv.qx[i]^2 + qyz2)
                    if q > 0 && q < qmax
                        b = floor(Int, q / bw) + 1
                        if b <= nb
                            bs = b_factor > 0.0 ? exp(b_factor * (q / (4π))^2) : 1.0
                            v = ft_vac[i, j, k]
                            re = Float64(real(v)) * bs
                            im = Float64(imag(v)) * bs
                            if ft_other !== nothing
                                o = ft_other[i, j, k]
                                re += Float64(real(o)); im += Float64(imag(o))
                            end
                            sums[b] += re * re + im * im
                            qs[b] += q
                            cnt[b] += 1
                        end
                    end
                end
            end
        end
    end
    for ti in 1:nt
        bins.sums .+= lsums[ti]; bins.q_sums .+= lqs[ti]; bins.counts .+= lcnt[ti]
    end
    return bins
end

# Final I(q) list: I(0) first, then non-empty bins (first bin reports q = 0 as in C)
function create_final_data(bins::BinData)
    out = QIPoint[QIPoint(0.0, bins.Iq0)]
    for i in 1:bins.max_bins
        c = bins.counts[i]
        c > 0 || continue
        qm = i == 1 ? 0.0 : bins.q_sums[i] / c
        push!(out, QIPoint(qm, bins.sums[i] / c))
    end
    return out
end

function write_output(data::Vector{QIPoint}, filename::AbstractString)
    try
        open(filename, "w") do io
            for p in data
                @printf(io, "%.12e\t%.12e\n", p.q, p.intensity)
            end
        end
    catch
        println(stderr, "Error: Could not open output file: ", filename)
        return false
    end
    @printf("Wrote %d data points to %s\n", length(data), filename)
    return true
end
