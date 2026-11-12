# native_grid.jl - libfmapgrid.so (CUDA C + cuFFT, no CUDA.jl) against the CPU softgrd on a reduced grid.
# Needs a GPU node and native/libfmapgrid.so:   julia --project=. test/native_grid.jl [blen] [norient]
using Test, Printf
include(joinpath(@__DIR__, "..", "src", "grid", "cpu", "FmapGen.jl")); using .FmapGen
const FG = FmapGen
using Random: MersenneTwister
Base.include(FG, joinpath(@__DIR__, "..", "src", "grid", "gpu", "native.jl"))

blen = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 160
norient = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 3
root = normpath(joinpath(@__DIR__, ".."))
tern = FG.tern_bytes_from_env()

mktempdir() do tmp
    cd(tmp) do
        cp(joinpath(root, "examples", "cenA.pqr"), "subA.pqr")
        open(f -> FG.fmappre("subA.pqr", 0.016, 25.0, "1.0", "-"; out = f), "parms.txt", "w")
        P = FG.read_parameters("parms.txt")
        dx = P["spacing"]
        sol = FG.Solv(P["rcut"], 0.0, P["kappa"], P["sdie"], P["kbt"], P["escl"], P["vscl"])
        a0 = FG.read_vdw("subA.vdw")
        crds = FG.scale_coordinates([FG.Atom(a.info, a.x, a.y, a.z, a.q * P["qscl"], a.r * P["rscl"], a.Asq, a.Bsq) for a in a0], dx)
        pros0 = [FG.Atom(a.info, a.x, a.y, a.z, a.q, a.r * P["rscl"], a.Asq, a.Bsq) for a in a0]
        pros = FG.scale_coordinates(pros0, dx)
        angles = FG.read_angles(joinpath(root, "data", "sspNF00072", "rot.eul"))
        @printf("blen %d (box %.1f A), %d atoms, tern %d\n", blen, blen * dx, length(crds), tern)

        cpu = FG.SoftCtx(blen); gpu = FG.make_native_ctx(blen)
        for ang in angles[1:norient]
            R = FG.euler_to_rotmat(ang...)
            rot = FG.rotate_atoms(pros, R)
            a = redirect_stdout(devnull) do; FG.softgrd(cpu, crds, rot, dx, sol); end
            b = redirect_stdout(devnull) do; FG.softgrd(gpu, crds, rot, dx, sol); end
            d = abs.(Int.(a) .- Int.(b))
            nbad = count(>(0), d)
            @printf("orientation %s: %d cells, differ %d (%.2e), max |diff| %d\n", string(ang), length(a), nbad, nbad / length(a), maximum(d))
            @test size(a) == size(b)
            @test maximum(d) <= 1
            @test nbad / length(a) < 1e-4
            @test count(==(Int8(127)), a) == count(==(Int8(127)), b)        # excluded volume identical
        end

        @testset "fmapdd pair energies, CPU vs GPU" begin
            ps = FG.pair_system(P, blen, dx)
            rng = MersenneTwister(7)
            np = 40
            angs = [(2pi * rand(rng) - pi, pi * rand(rng), 2pi * rand(rng) - pi) for _ in 1:np]
            ijks = [(rand(rng, 0:blen-1), rand(rng, 0:blen-1), rand(rng, 0:blen-1)) for _ in 1:np]
            poses = FG.Poses(blen, dx, "subA.vdw", "subA.vdw", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), angs, ijks, zeros(np))
            rc = FG.pair_energies(a0, a0, poses, ps; atm = true)
            rg = FG.pair_energies_gpu(a0, a0, poses, ps; atm = true)
            @test isapprox(rc.ern, rg.ern; rtol = 1e-9, atol = 1e-9)
            @test isapprox(rc.rec, rg.rec; rtol = 1e-9, atol = 1e-9)
            @test isapprox(rc.lig, rg.lig; rtol = 1e-9, atol = 1e-9)
            @printf("fmapdd: %d poses, max |CPU - GPU| = %.2e (ern scale %.1f)\n", np, maximum(abs.(rc.ern .- rg.ern)), maximum(abs.(rc.ern)))
        end
    end
end
