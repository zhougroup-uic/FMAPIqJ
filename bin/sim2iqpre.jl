#!/usr/bin/env julia
# PDB -> PQR/vol helper
include(joinpath(@__DIR__, "..", "src", "sim2iq", "cpu", "Sim2Iq.jl")); using .Sim2Iq
exit(sim2iqpre_main(copy(ARGS)))
