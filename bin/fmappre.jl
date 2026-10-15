#!/usr/bin/env julia
# fmappre: energy-map tool (same arguments as the python tool). julia --project=<root> bin/fmappre.jl ...
include(joinpath(@__DIR__, "..", "src", "grid", "cpu", "FmapGen.jl")); using .FmapGen
exit(FmapGen.main_fmappre(ARGS))
