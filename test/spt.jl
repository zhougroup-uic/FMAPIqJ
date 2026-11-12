# spt.jl - spatial compression (src/spt/Spt.jl): round trip of synthetic grids, any grid length, both cell formats.
#   julia --project=. -t 4 test/spt.jl [existing.fmap.ve.sp2p]     (a file of the GPU engines: its units must be reproduced byte for byte)
using Test, Random
include(joinpath(@__DIR__, "..", "src", "spt", "Spt.jl")); using .Spt

# a grid with zero, excluded-volume and mixed regions: a "protein" ball (127), a shell of varied values, zeros outside
function synth(B, tern, seed)
    rng = MersenneTwister(seed)
    d = zeros(Int8, tern * B^3)
    c = (B - 1) / 2
    for i in 0:B-1, j in 0:B-1, k in 0:B-1
        r = sqrt((i - c)^2 + (j - c)^2 + (k - c)^2)
        cell = (i * B + j) * B + k
        if r < B * 0.18
            tern == 2 ? (d[2cell+1] = 127; d[2cell+2] = 127) : (d[cell+1] = 127)
        elseif r < B * 0.4
            v = Int8(round(20 * exp(-(r - B * 0.18) / 4) * (1 + 0.3 * sin(i + 2j))) + rand(rng, -1:1))
            tern == 2 ? (d[2cell+1] = v; d[2cell+2] = Int8(rand(rng, -6:2))) : (d[cell+1] = v)
        end
    end
    return d
end

@testset "Spt round trip" begin
    for B in (16, 22, 25, 36), tern in (1, 2)
        nrot = 3
        mktempdir() do tmp
            path = joinpath(tmp, "t.sp2p")
            grids = [synth(B, tern, 10B + r) for r in 1:nrot]
            w = sp2p_open(path, nrot, B, 0.6; tern)
            foreach(g -> sp2p_add!(w, g), grids)
            sp2p_close!(w; quiet = true)
            @test is_sp2p(path)
            @test read_sp2p_header(path) == (nrot, B, 0.6, tern)
            f = read_sp2p(path)
            out = Vector{Int8}(undef, tern * B^3)
            for r in 0:nrot-1
                decode_rotation!(out, f, r)
                @test out == grids[r+1]
            end
            @test decode_all(f) == reduce(vcat, grids)
            for r in 0:nrot-1                                         # statistics on the units == statistics of the dense grid
                g = grids[r+1]; ref = zeros(Int, 256)
                for c in 0:B^3-1
                    e0 = Int(tern == 2 ? g[2c+1] : g[c+1]); e1 = tern == 2 ? Int(g[2c+2]) : 0
                    v = tern == 1 ? e0 : ((e0 == 127 || e1 == 127) ? 127 : clamp(e0 + e1, -128, 127))
                    ref[v+129] += 1
                end
                @test rotation_histogram(f, r) == ref
            end
            tbl = sparse1_table(f)                                    # SPT=1 lookup table, directly from the units
            ok = true
            for r in 0:nrot-1, i in 0:B-1, j in 0:B-1, k in 0:B-1
                e0, e1 = sp1_ve(tbl, B, nrot, i, j, k, r)
                c = (i * B + j) * B + k; g = grids[r+1]
                w0, w1 = tern == 2 ? (g[2c+1], g[2c+2]) : (g[c+1], Int8(0))
                ok &= e0 == w0 && (tern == 1 || e1 == w1)
            end
            @test ok
            @test filesize(path) < 0.6 * nrot * tern * B^3 + 4096 || B < 25      # compresses once the grid is not tiny
        end
    end
end

if length(ARGS) >= 1                       # a file written by the GPU engines' generator: same bytes after decode + compress
    @testset "existing file $(ARGS[1])" begin
        f = read_sp2p(ARGS[1])
        B = f.blen; @test B % 4 == 0
        out = Vector{Int8}(undef, f.tern * B^3)
        for r in unique([0, f.nrot ÷ 2, f.nrot - 1])
            decode_rotation!(out, f, r)
            u, o, nsl, pooln = compress_rotation(out, B, f.tern)
            uh = Spt.SP_PH + 16 * r
            bm = Int(Spt.rd32(f.blob, uh)); pool = Int(Spt.rd32(f.blob, uh + 12))
            ref = f.blob[bm+1:bm+length(u)]
            @test ref == u
        end
    end
end
