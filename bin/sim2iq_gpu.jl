#!/usr/bin/env julia
# GPU I(q) (native/libfmapsim2iq.so); same arguments as sim2iq.jl
include(joinpath(@__DIR__, "..", "src", "sim2iq", "gpu", "Sim2IqGPU.jl")); using .Sim2IqGPU
exit(sim2iq_gpu_main(copy(ARGS)))
