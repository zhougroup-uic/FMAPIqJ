#!/usr/bin/env julia
# fmap: energy-map tool (same arguments as the python tool). julia --project=<root> bin/fmap.jl ...
include(joinpath(@__DIR__, "..", "src", "grid", "cpu", "FmapGen.jl")); using .FmapGen
exit(FmapGen.main_fmap(ARGS))
