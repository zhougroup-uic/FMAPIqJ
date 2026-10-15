#!/usr/bin/env julia
# notebook.jl - register a Julia kernel for this repository and open the example notebooks.
#   julia tools/notebook.jl            install IJulia + the "Julia" kernel (name `julia`, the name Colab uses too), then launch Jupyter
#   julia tools/notebook.jl install    only install the kernel (use your own jupyter / VS Code / JupyterLab)
# The kernel starts Julia with --project=<this repository>, so the notebooks need no further setup.
# IJulia is installed into your default Julia environment (not into this repository's Project.toml).
using Pkg
const ROOT = normpath(joinpath(@__DIR__, ".."))
Pkg.activate()                                   # default (global) environment
Base.find_package("IJulia") === nothing && Pkg.add("IJulia")
using IJulia
kernel = IJulia.installkernel("Julia", "--project=$ROOT"; specname = "julia")
println("kernel installed: ", kernel)
if !("install" in ARGS)
    IJulia.notebook(; dir = joinpath(ROOT, "examples", "n200_test"))
end
