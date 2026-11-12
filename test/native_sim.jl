# native_sim.jl - libfmapsim.so (CUDA C, no CUDA.jl) against the CPU engine on a real grid.
#   julia --project=. test/native_sim.jl <work dir with rot.eul, fmap.ve.sp2p, fmap.paths>     (needs a GPU node; TERN_BYTES as the grid)
# 1. total energy of random placements: the library's sum equals the CPU engine's sum of pair energies (same lookup, same pair order);
# 2. a short run (displacement + same-box insertion moves): the energy tracked from the move deltas equals the total energy recomputed
#    on the device and by the CPU engine at the end, and every molecule keeps a valid rotation index.
using Test, Printf
include(joinpath(@__DIR__, "..", "src", "sim", "cpu", "FmapSim.jl")); using .FmapSim
include(joinpath(@__DIR__, "..", "src", "sim", "rotsys.jl"))
include(joinpath(@__DIR__, "..", "src", "sim", "gpu", "nvt_native.jl"))

wd = length(ARGS) >= 1 ? ARGS[1] : error("usage: native_sim.jl workdir")
paths = joinpath(wd, "fmap.paths")
cd(wd)
g = load_fmap_paths_native(paths, 1)
@printf("grid: nrot %d nbin %d scl %.4f blen %d\n", g.nrot, g.nbin, g.scl, g.blens[1])
tern = TERN_BYTES
c = FmapSim.fmap_create("rot.eul", "fmap.ve.sp2p"; tern)          # CPU engine, same file
L = 619.65; N = 200; kT = 0.5924; box = (L, L, L)

function cpu_total(pos, ridx)
    n = length(ridx); e = 0.0
    for i in 1:n-1, j in i+1:n
        dx = FmapSim.apply_pbc(pos[3i-2] - pos[3j-2], L); dy = FmapSim.apply_pbc(pos[3i-1] - pos[3j-1], L); dz = FmapSim.apply_pbc(pos[3i] - pos[3j], L)
        e += FmapSim.fmap_pair_energy(c, dx, dy, dz, ridx[i], ridx[j])
    end
    return e
end

@testset "native engine" begin
    pos, bins, ptype, ridx, st = random_particles(g, 1.0, 1.0, UInt64(1), N, box, [N])
    @test length(ridx) == N
    @test all(0 .<= ridx .< g.nrot)
    @test all(ridx .== [Int32(FmapSim.bins_to_rot_idx(c, bins[3i-2], bins[3i-1], bins[3i])) for i in 1:N])     # rotation index == CPU engine's
    b = NBox(g, N, box, kT, 1.0, 1.0)
    box_set!(b, pos, bins, ptype, ridx, st)
    e0 = total_energy(b, N)
    ec = cpu_total(pos, ridx)
    @printf("placement: E native %.6f, CPU %.6f\n", e0, ec)
    @test isapprox(e0, ec; atol = 1e-6, rtol = 1e-9)
    # moves
    e = e0; nacc = 0
    for it in 1:20
        mc_launch!(b, N, 200, 3.0 * 1.62, 1.0)
        m, _, _ = wait_box(b; mc = true)
        e += m[1]; nacc += round(Int, m[2])
        insert_launch!(b, N, 40)
        _, i, _ = wait_box(b; ins = true)
        e += i[1]
    end
    p2, b2, r2, t2 = box_get(b, N)
    ed = total_energy(b, N); ec2 = cpu_total(p2, r2)
    @printf("after moves: tracked %.6f, device %.6f, CPU %.6f (accepted %d)\n", e, ed, ec2, nacc)
    @test nacc > 0
    @test isapprox(ed, ec2; atol = 1e-6, rtol = 1e-9)
    @test isapprox(e, ed; atol = 1e-6 * max(1, abs(ed)), rtol = 1e-9)
    @test all(0 .<= r2 .< g.nrot) && all(t2 .== 0)
    @test all(0 .<= p2[1:3:end] .< L)
    close_box(b)
end
