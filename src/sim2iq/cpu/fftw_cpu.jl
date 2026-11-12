# fftw_cpu.jl - the FFT of the CPU sim2iq. Included into module Sim2Iq on first use, so that the GPU path (cuFFT inside
# native/libfmapsim2iq.so, which reuses this module for the host-side input) does not load FFTW.
using FFTW

"In-place forward 3D complex FFT (single precision, FFTW_ESTIMATE as in C)."
function fft_forward!(A::Array{ComplexF32,3})
    FFTW.set_num_threads(Threads.nthreads())
    plan = plan_fft!(A; flags = FFTW.ESTIMATE)
    plan * A
    return A
end
