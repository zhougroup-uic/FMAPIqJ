"""
Sim2Iq — SAXS I(q) from atomic snapshots of protein solutions in a periodic box.

Julia port of the C core of sim2iq-pkg (ext/*.c).  File layout mirrors the C sources:

    cromer_mann.jl   cromer_mann_4gauss.h
    rdf.jl           rdf.c
    pqr_reader.jl    pqr_reader.c
    pd6d.jl          pd6d.c  (+ process_trajectory from main.c)
    grid_mapping.jl  grid_mapping.c
    binning.jl       fft.c (q-vectors) + binning.c
    saxs.jl          saxs_functions.c
    cli.jl           args.c + main.c
    pre.jl           sim2iq_pre.py (PDB -> PQR helper)
"""
module Sim2Iq

using FFTW
using Printf

include("cromer_mann.jl")
include("rdf.jl")
include("pqr_reader.jl")
include("pd6d.jl")
include("grid_mapping.jl")
include("binning.jl")
include("saxs.jl")
include("cli.jl")
include("pre.jl")

export Atom, Structure, read_pqr_file, process_trajectory, calculate_saxs_data_multi_step,
       u2B, write_output, sim2iq_main, sim2iqpre_main

end # module
