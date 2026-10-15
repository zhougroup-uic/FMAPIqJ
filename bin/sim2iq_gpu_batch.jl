#!/usr/bin/env julia
# GPU I(q) for many configurations in one process
include(joinpath(@__DIR__, "..", "src", "sim2iq", "gpu", "Sim2IqGPU.jl")); using .Sim2IqGPU
exit(sim2iq_gpu_batch(copy(ARGS)))
