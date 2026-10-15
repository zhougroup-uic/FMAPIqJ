#!/usr/bin/env julia
# FMAPIq pipeline (see --help)
include(joinpath(@__DIR__, "..", "src", "fmapiq", "pipeline.jl")); using .FmapIq
exit(FmapIq.main(copy(ARGS)))
