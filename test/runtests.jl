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

@testset "rotation system: plain / folded / lookup pair tables (rot.eul and rot.zlib v1/v2)" begin
    eul = joinpath(@__DIR__, "..", "data", "sspNF00072", "rot.eul")
    n = 72
    t0 = FmapSim.rot_tables_from_eul(eul; mode = 0)
    t1 = FmapSim.rot_tables_from_eul(eul; mode = 1)
    t2 = FmapSim.rot_tables_from_eul(eul; mode = 2)
    get(t, i, j) = FmapSim.amget(t.amap, i, j, n)
    @test all(get(t0, i, j) == get(t1, i, j) for i in 0:n-1, j in 0:n-1)          # folded == expanded, entry for entry
    @test all(0 <= get(t2, i, j) < n for i in 0:n-1, j in 0:n-1)                   # lookup: a valid sample, approximate
    @test count(get(t2, i, j) == get(t0, i, j) for i in 0:n-1, j in 0:n-1) > 0.8 * n^2
    @test FmapSim.amap_bytes(t1.amap) < FmapSim.amap_bytes(t0.amap) && FmapSim.amap_bytes(t2.amap) == 0
    @test t0.rotm == t1.rotm == t2.rotm && t0.aidx == t2.aidx
    # a version 1 rot.zlib (stored table) is used as it is in mode 0; version 2 (no table) gives the folded-search table
    blob(dense) = vcat(reinterpret(UInt8, t0.aidx), dense === nothing ? UInt8[] : reinterpret(UInt8, dense),
                       reinterpret(UInt8, vec(FmapSim.Rotij.load_eul(eul).eul)), reinterpret(UInt8, t0.rotm))
    stored = Int16[(i * 7 + j * 3) % n for i in 0:n-1 for j in 0:n-1]
    r1 = FmapSim.rot_tables_from_blob(1, n, 120, blob(stored); mode = 0)
    @test r1.amap.dense == stored
    r2 = FmapSim.rot_tables_from_blob(2, n, 120, blob(nothing); mode = 0)
    @test r2.amap.dense == t0.amap.dense
    r3 = FmapSim.rot_tables_from_blob(1, n, 120, blob(stored); mode = 1)           # a stored table is ignored when not used
    @test all(get(r3, i, j) == get(t0, i, j) for i in 0:n-1, j in 0:n-1)
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
