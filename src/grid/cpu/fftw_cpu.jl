# fftw_cpu.jl - the FFT plans of the CPU energy-map context. Included into module FmapGen by SoftCtx on first use, so that the
# GPU path (cuFFT inside native/libfmapgrid.so) and the other tools do not load FFTW.
using FFTW

"Forward real and backward plans for an l^3 grid (`working`: real input, `ft`: its half-complex transform)."
function fft_plans(working::Array{Float64,3}, ft::Array{ComplexF64,3}, l::Int)
    FFTW.set_num_threads(Threads.nthreads())
    return plan_rfft(working; flags = FFTW.ESTIMATE), plan_irfft(ft, l; flags = FFTW.ESTIMATE)
end
