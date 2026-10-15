#!/usr/bin/env julia
# fmapstat: energy-map tool (same arguments as the python tool). julia --project=<root> bin/fmapstat.jl ...
include(joinpath(@__DIR__, "..", "src", "grid", "cpu", "FmapGen.jl")); using .FmapGen
exit(FmapGen.main_fmapstat(ARGS))
