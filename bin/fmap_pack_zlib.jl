#!/usr/bin/env julia
# fmap_pack_zlib: energy-map tool (same arguments as the python tool). julia --project=<root> bin/fmap_pack_zlib.jl ...
include(joinpath(@__DIR__, "..", "src", "grid", "cpu", "FmapGen.jl")); using .FmapGen
exit(FmapGen.main_pack(ARGS))
