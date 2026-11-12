#!/usr/bin/env julia
# spt_stats.jl FILE.sp2p [--verify]: size and block statistics of a spatially compressed grid, and the size of the SPT=1 lookup table
# made from it. --verify: decode every rotation, compress it again and compare the unit with the file's (all rotations), time both.
include(joinpath(@__DIR__, "..", "src", "spt", "Spt.jl")); using .Spt
using Printf
path = ARGS[1]
f = read_sp2p(path)
B = f.blen; nrot = f.nrot; tern = f.tern; L = unit_layout(B); w = f.blob
@printf("%s: nrot %d, blen %d, spacing %.3f, cell format %d, file %.1f MB, dense grid %.1f MB (%d blocks of 4^3)\n",
        basename(dirname(path)) * "/" * basename(path), nrot, B, f.spacing, tern, filesize(path) / 1e6, nrot * tern * B^3 / 1e6, L.nblk)
cls = zeros(Int, 3); mixedany = falses(L.nblk); nonzeroany = falses(L.nblk)
for r in 0:nrot-1
    uh = SP_PH + 16r; bm = Int(Spt.rd32(w, uh))
    for blk in 0:L.nblk-1
        c = Int((Spt.rd32(w, bm + 4 * (blk >> 4)) >> ((blk & 15) << 1)) & 3)
        cls[c+1] += 1
        c == 2 && (mixedany[blk+1] = true)
        c != 0 && (nonzeroany[blk+1] = true)
    end
end
tot = nrot * L.nblk
@printf("per rotation: %.2f %% blocks all-zero, %.2f %% all-overlap, %.2f %% mixed (%.0f mixed blocks)\n",
        100 * cls[1] / tot, 100 * cls[2] / tot, 100 * cls[3] / tot, cls[3] / nrot)
nst = count(mixedany)
@printf("blocks mixed in some rotation (stored by SPT=1): %d = %.2f %%; non-zero in some rotation: %.2f %%\n", nst, 100 * nst / L.nblk, 100 * count(nonzeroany) / L.nblk)
mapb = (nrot + 3) >> 2
for cb in (2, 1)
    sz = 16 + 4 * L.nblk + mapb * L.nblk + nst * 64 * nrot * cb
    @printf("SPT=1 table, %d byte cells: %.1f MB (stored blocks %.1f MB, map+dir %.1f MB)\n", cb, sz / 1e6, nst * 64 * nrot * cb / 1e6, (4 + mapb) * L.nblk / 1e6)
end
function verify(f)
    B = f.blen; nrot = f.nrot; tern = f.tern; w = f.blob
    out = Vector{Int8}(undef, tern * B^3); ok = true; td = 0.0; tc = 0.0
    for r in 0:nrot-1
        td += @elapsed decode_rotation!(out, f, r)
        local u
        tc += @elapsed (u = compress_rotation(out, B, tern)[1])
        uh = SP_PH + 16r; bm = Int(Spt.rd32(w, uh))
        ok &= f.blob[bm+1:bm+length(u)] == u
    end
    @printf("verify (all %d rotations): decode -> compress reproduces the file's units: %s; decode %.1f s, compress %.1f s (%d threads)\n", nrot, ok, td, tc, Threads.nthreads())
end
"--verify" in ARGS && verify(f)
