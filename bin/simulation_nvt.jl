#!/usr/bin/env julia
# CPU NVT Monte Carlo: N box T n_steps seed start stop sample_freq fmap.paths ...
include(joinpath(@__DIR__, "..", "src", "sim", "cpu", "FmapSim.jl")); using .FmapSim
exit(FmapSim.main_nvt(ARGS))
