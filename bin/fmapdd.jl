#!/usr/bin/env julia
# fmapdd: pair energies of listed poses (fmapdd / fmapdd.atm of the C code). julia --project=<root> bin/fmapdd.jl poses.txt parms.txt [--atm]
include(joinpath(@__DIR__, "..", "src", "grid", "cpu", "FmapGen.jl")); using .FmapGen
exit(FmapGen.main_fmapdd(ARGS))
