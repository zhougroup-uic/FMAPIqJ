# FmapSim.jl - FMAP-based GEMC Monte Carlo for protein mixtures, pure Julia.
#
# Port of the C/CUDA engines (src/cpu/simulation_gemc.c + main_gemc.c, src/gpu/*gemc_gpu.cu).
#   cpu_gemc.jl : sequential CPU engine, same RNG stream / move order as simulation_gemc.c
#   grid.jl     : .sp2p / .raw FMAP grid loading (raw-grid cache + mmap for the CPU engine)
#   drivers.jl  : command lines of simulation_gemc and simulation_nvt (single box, same-box insertion)
# The GPU engine is native/libfmapsim.so (CUDA C) driven by ../gpu/main_nvt_gpu.jl.
module FmapSim

using Printf

include(joinpath(@__DIR__, "..", "..", "spt", "Spt.jl")); using .Spt
include("common.jl")
include("../rotsys.jl")   # rotation tables, pair-table modes (plain / folded / lookup), Rotij
include("grid.jl")
include("cpu_gemc.jl")
include("drivers.jl")

end # module
