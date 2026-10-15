#!/usr/bin/env julia
# CPU I(q): input.pqr grid_size grid_spacing output.dat [-steps 4 -traj snap.txt -txt]
include(joinpath(@__DIR__, "..", "src", "sim2iq", "cpu", "Sim2Iq.jl")); using .Sim2Iq
exit(sim2iq_main(copy(ARGS)))
