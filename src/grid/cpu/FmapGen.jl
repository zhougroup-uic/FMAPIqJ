# FmapGen.jl - FMAP grid generation in Julia (port of fmap/ext/cpu/*.c and the python tools).
#
#   atoms.jl    : ATOM/SOLV records, .vdw reading, parameter file, orientations
#   mapping.jl  : atom -> grid mapping (spherical Map2GrdR, trilinear Map2GrdL, 10-point lattice L2nd)
#   softgrd.jl  : per-orientation energy grid (FFT cross-correlation, vol/ele/vdw/v12), tErn quantization
#   pre.jl      : fmappre   (parms.txt + sub*.vdw from a PQR)
#   pack.jl     : fmap_pack_zlib (ve/tErn.*.mat.gz + rotation tables -> fmap.{rot,ve}.zlib)
#   cli.jl      : command-line entry points (same arguments as the python tools)
module FmapGen

using Printf
using LinearAlgebra
using FFTW
using CodecZlib

include("atoms.jl")
include("mapping.jl")
include("softgrd.jl")
include("pre.jl")
include("pack.jl")
include("stat.jl")
include("cli.jl")

end # module
