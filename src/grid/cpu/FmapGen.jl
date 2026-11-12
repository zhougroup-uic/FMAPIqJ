# FmapGen.jl - FMAP grid generation in Julia (port of fmap/ext/cpu/*.c and the python tools).
#
#   atoms.jl    : ATOM/SOLV records, .vdw reading, parameter file, orientations
#   mapping.jl  : atom -> grid mapping (spherical Map2GrdR, trilinear Map2GrdL, 10-point second-order L2nd, closed form)
#   softgrd.jl  : per-orientation energy grid (FFT cross-correlation, vol/ele/vdw/v12), tErn quantization
#   pre.jl      : fmappre   (parms.txt + sub*.vdw from a PQR)
#   spwrite.jl  : the energy maps written as the spatially compressed per-rotation table <prefix>.ve.sp2p (../../spt/Spt.jl)
#   pair.jl     : low-energy poses (SelErn), pair energies and residue decomposition (fmapdd, fmapdd.atm, resern/data2bf/fmappdb)
#   cli.jl      : command-line entry points (same arguments as the python tools)
module FmapGen

using Printf
using LinearAlgebra
# FFTW is loaded on demand by the CPU energy-map context (fftw_cpu.jl); the GPU path (cuFFT in native/) does not need it

include(joinpath(@__DIR__, "..", "..", "spt", "Spt.jl")); using .Spt

include("atoms.jl")
include("mapping.jl")
include("softgrd.jl")
include("pre.jl")
include("spwrite.jl")
include("stat.jl")
include("pair.jl")
include("cli.jl")

end # module
