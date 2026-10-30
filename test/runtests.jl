# Port of sim2iq-pkg/tests/test.sh: pre-processing round trip + full I(q) against reference.
using Test
include(joinpath(@__DIR__, "..", "src", "sim2iq", "cpu", "Sim2Iq.jl")); using .Sim2Iq

include(joinpath(@__DIR__, "..", "src", "sim", "cpu", "FmapSim.jl"))

@testset "beta-bin index mapping" begin
    n = 120
    bi(b) = FmapSim.beta_idx(b, Int64(n))
    @test bi(5.0) == 5 && bi(5.99) == 5 && bi(60.3) == 60
    @test bi(0.49) == 0 && bi(0.5) == 0 && bi(0.99) == 0 && bi(1.0) == 1
    @test bi(Float64(n) - 0.5) == n - 1 && bi(prevfloat(Float64(n))) == n - 1
    @test bi(Float64(n)) == n - 1                                           # beta = pi
    b = FmapSim.euler_to_bins(0.0, Float64(pi), 0.0, n)
    @test b[2] <= n && bi(b[2]) == n - 1
    @test all(0 <= bi(x) < n for x in range(0, n, length=1001))
end

const D = joinpath(@__DIR__, "data")

function readdat(f)
    rows = [parse.(Float64, split(l)) for l in eachline(f) if !isempty(strip(l))]
    return reduce(hcat, rows)'
end

mktempdir() do tmp
    @testset "input processing (sim2iqpre)" begin
        out = joinpath(tmp, "input.pqr")
        @test sim2iqpre_main(["-f", joinpath(D, "input.pqr"), "-o", out]) == 0
        @test read(out, String) == read(joinpath(D, "input.pqr"), String)
    end
    @testset "simulation output (sim2iq)" begin
        out = joinpath(tmp, "sim2iq.dat")
        N, L = 620, 619.65     # grid points, box length (Å) as in test.sh
        @test sim2iq_main(["$(joinpath(D, "input.pqr"))", "$N", "$(L / N)", out,
                           "-steps", "4", "-traj", joinpath(D, "snap.txt"), "-txt"]) == 0
        a, b = readdat(out), readdat(joinpath(D, "sim2iq.dat"))
        @test size(a) == size(b)
        @test all(abs.(a .- b) .<= 1e-6 .* max.(abs.(a), abs.(b)))
    end
end
