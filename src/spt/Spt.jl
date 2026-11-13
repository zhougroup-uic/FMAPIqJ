# Spt.jl - spatial compression of the FMAP energy grids (SPT=2p, per-rotation block-sparse table).
#
# A grid of one orientation (rotation) is cut into 4x4x4 cell blocks. A block is all-zero (class 0), all-overlap (class 1, the excluded
# volume, elec byte 127) or mixed (class 2); only mixed blocks store data, as a small palette of the distinct (elec, vdw) pairs and one
# ceil(log2 nd)-bit palette index per cell. Every orientation is an independent unit, so a file can be written (and read) rotation by
# rotation. The table is exact: a decoded grid is cell for cell the grid that was compressed.
#
# Cells are 2 bytes (elec, vdw). A grid with a single summed channel (TERN_BYTES=1) is stored as (value, 0); the header's cell format
# field says which it was, and `decode_rotation!` hands the grid back in that format. A grid length that is not a multiple of 4 is
# padded with zero cells to whole blocks (the padding is never read: lookups outside the grid are zero before any table access).
#
# File fmap.ve.sp2p:  [48-byte header][blob]
#   header: "SPTV" | u32 version=1 | u32 mode=34 (2p) | i32 nrot | i32 blen | f64 spacing | u64 blob bytes | u32 cell format (1|2; 0 = 2)
#   blob:   [16 B: i32 nrot, i32 total mixed blocks, 0, 0] [unit table, 16 B per rotation: u32 offsets (in the blob) of the class bitmap,
#           the rank prefix, the slice offsets, the slice pool] [units, 16-byte aligned]
#   unit:   class bitmap (2 bits per block, u32 word = 16 blocks, block = (bi*nb + bj)*nb + bk) | rank prefix (u32 per 64 blocks: mixed
#           blocks before it) | slice offsets (u32 into the pool, one per mixed block, in block order) | pool of slices
#   slice:  [nd:u8][nd x (elec, vdw)][64 indices of ceil(log2 nd) bits, LSB first, cell in block = (di*4 + dj)*4 + dk]
# The layout of the unit is that of the block-sparse tables used by the GPU engines (a file of a multiple-of-4 grid is identical to theirs).
module Spt

using Printf

export Sp2pWriter, Sp2pFile, sp2p_open, sp2p_add!, sp2p_close!, read_sp2p, is_sp2p, read_sp2p_header,
       compress_rotation, decode_rotation!, decode_all, unit_layout, sparse1_table, sp1_ve, sp2p_blob, sp2p_ve, rotation_histogram, SP_FILE_HDR, SP_PH

const SP_MAGIC = b"SPTV"
const SP_FILE_HDR = 48
const SP_PH = 16                 # unit table starts here in the blob (16 bytes per rotation)

struct UnitLayout
    B::Int                       # grid length
    nb::Int                      # blocks per axis, cld(B, 4)
    nblk::Int
    nw::Int                      # class bitmap words
    ngrp::Int                    # rank prefix entries
end
unit_layout(B::Integer) = (nb = cld(Int(B), 4); nblk = nb^3; nw = cld(nblk, 16); UnitLayout(Int(B), nb, nblk, nw, cld(nw, 4)))

@inline bits_of(nd::Integer) = nd > 1 ? 32 - leading_zeros(UInt32(nd - 1)) : 0
@inline slice_bytes(nd::Integer) = 1 + 2 * nd + 8 * bits_of(nd)

# (elec << 8 | vdw) of a cell; cells of the padding are zero
@inline function cell_value(d::AbstractVector{Int8}, tern::Int, B::Int, i::Int, j::Int, k::Int)
    (i < B && j < B && k < B) || return UInt16(0)
    c = (i * B + j) * B + k
    tern == 2 ? (UInt16(reinterpret(UInt8, d[2c+1])) << 8) | UInt16(reinterpret(UInt8, d[2c+2])) :
                UInt16(reinterpret(UInt8, d[c+1])) << 8
end

# ---- compression of one rotation ----

"""
    compress_rotation(d, B, tern) -> (unit::Vector{UInt8}, offsets::NTuple{4,Int}, nmixed, poolbytes)

`d`: the grid of one rotation, cell-major (`tErn[i][j][k]`, k fastest), `tern` bytes per cell (2: elec, vdw; 1: one value).
`offsets` are those of the class bitmap, rank prefix, slice offset table and pool inside `unit`.
"""
function compress_rotation(d::AbstractVector{Int8}, B::Int, tern::Int)
    L = unit_layout(B); nb = L.nb; nblk = L.nblk
    length(d) == tern * B^3 || error("grid has $(length(d)) bytes, expected $(tern * B^3)")
    cmap = Vector{UInt8}(undef, nblk)
    Threads.@threads for blk in 0:nblk-1
        bk = blk % nb; bj = (blk ÷ nb) % nb; bi = blk ÷ (nb * nb)
        z = true; o = true
        @inbounds for di in 0:3, dj in 0:3, dk in 0:3
            v = cell_value(d, tern, B, bi * 4 + di, bj * 4 + dj, bk * 4 + dk)
            z &= v == 0x0000
            o &= (v >> 8) == 0x007f
        end
        @inbounds cmap[blk+1] = z ? 0x00 : (o ? 0x01 : 0x02)
    end
    words = zeros(UInt32, L.nw)
    for blk in 0:nblk-1
        @inbounds words[(blk >> 4) + 1] |= UInt32(cmap[blk+1]) << ((blk & 15) << 1)
    end
    mixed = findall(==(0x02), cmap)                       # 1-based block numbers, in block order
    nsl = length(mixed)
    pf = zeros(UInt32, L.ngrp)
    let r = 0
        for blk in 0:nblk-1
            (blk & 63) == 0 && (pf[(blk >> 6) + 1] = UInt32(r))
            @inbounds cmap[blk+1] == 0x02 && (r += 1)
        end
    end
    pal = Matrix{UInt16}(undef, 64, nsl); nds = Vector{UInt8}(undef, nsl)
    Threads.@threads for s in 1:nsl
        blk = mixed[s] - 1
        bk = blk % nb; bj = (blk ÷ nb) % nb; bi = blk ÷ (nb * nb)
        nd = 0
        @inbounds for cib in 0:63
            v = cell_value(d, tern, B, bi * 4 + (cib >> 4), bj * 4 + ((cib >> 2) & 3), bk * 4 + (cib & 3))
            f = false
            for p in 1:nd
                f |= pal[p, s] == v
            end
            f || (nd += 1; pal[nd, s] = v)
        end
        nds[s] = UInt8(nd)
    end
    offs = Vector{UInt32}(undef, nsl)
    pooln = 0
    for s in 1:nsl
        offs[s] = UInt32(pooln); pooln += slice_bytes(nds[s])
    end
    pool = zeros(UInt8, pooln + 4)
    Threads.@threads for s in 1:nsl
        blk = mixed[s] - 1
        bk = blk % nb; bj = (blk ÷ nb) % nb; bi = blk ÷ (nb * nb)
        off = Int(offs[s]); nd = Int(nds[s]); bits = bits_of(nd)
        @inbounds begin
            pool[off+1] = UInt8(nd)
            for p in 1:nd
                v = pal[p, s]; pool[off+2p] = UInt8(v >> 8); pool[off+2p+1] = UInt8(v & 0x00ff)
            end
            if nd > 1
                ib = off + 1 + 2 * nd
                for cib in 0:63
                    v = cell_value(d, tern, B, bi * 4 + (cib >> 4), bj * 4 + ((cib >> 2) & 3), bk * 4 + (cib & 3))
                    idx = 0
                    for p in 1:nd
                        idx = pal[p, s] == v ? p - 1 : idx
                    end
                    bp = cib * bits; by = ib + (bp >> 3); val = idx << (bp & 7)
                    pool[by+1] |= UInt8(val & 0xff)
                    val > 0xff && (pool[by+2] |= UInt8(val >> 8))
                end
            end
        end
    end
    o_bm = 0; o_pf = 4 * L.nw; o_of = o_pf + 4 * L.ngrp; o_pool = (o_of + 4 * nsl + 15) ÷ 16 * 16
    u = zeros(UInt8, o_pool + pooln + 4)
    copyto!(u, o_bm + 1, reinterpret(UInt8, words), 1, 4 * L.nw)
    copyto!(u, o_pf + 1, reinterpret(UInt8, pf), 1, 4 * L.ngrp)
    nsl > 0 && copyto!(u, o_of + 1, reinterpret(UInt8, offs), 1, 4 * nsl)
    copyto!(u, o_pool + 1, pool, 1, pooln + 4)
    return u, (o_bm, o_pf, o_of, o_pool), nsl, pooln
end

# ---- the file ----

mutable struct Sp2pWriter
    io::IO
    path::String
    nrot::Int
    blen::Int
    tern::Int
    spacing::Float64
    base::Int                    # next unit starts here (blob offset)
    tab::Vector{UInt32}
    nslt::Int
    poolt::Int
    t0::Float64
end

"Open `path` for `nrot` rotations of a `blen`^3 grid with `tern` bytes per cell; rotations are added in order with `sp2p_add!`."
function sp2p_open(path::AbstractString, nrot::Integer, blen::Integer, spacing::Real; tern::Integer = 2)
    tern in (1, 2) || error("tern must be 1 or 2")
    io = open(path, "w")
    first_unit = (SP_PH + 16 * nrot + 15) ÷ 16 * 16
    write(io, zeros(UInt8, SP_FILE_HDR + first_unit))              # headers and unit table, patched by sp2p_close!
    return Sp2pWriter(io, String(path), Int(nrot), Int(blen), Int(tern), Float64(spacing), first_unit, UInt32[], 0, 0, time())
end

function sp2p_add!(w::Sp2pWriter, grid::AbstractArray{Int8})
    d = vec(grid)
    length(w.tab) < 4 * w.nrot || error("sp2p_add!: all $(w.nrot) rotations written")
    u, o, nsl, pooln = compress_rotation(d, w.blen, w.tern)
    seek(w.io, SP_FILE_HDR + w.base); write(w.io, u)
    append!(w.tab, UInt32[w.base + o[1], w.base + o[2], w.base + o[3], w.base + o[4]])
    w.base += (length(u) + 15) ÷ 16 * 16; w.nslt += nsl; w.poolt += pooln
    return nothing
end

function sp2p_close!(w::Sp2pWriter; quiet::Bool = false)
    length(w.tab) == 4 * w.nrot || error("sp2p_close!: $(length(w.tab) ÷ 4) of $(w.nrot) rotations written")
    seek(w.io, SP_FILE_HDR + w.base - 1); write(w.io, UInt8(0))    # pad the last unit
    seek(w.io, 0)
    write(w.io, SP_MAGIC, UInt32(1), UInt32(2 + 32), Int32(w.nrot), Int32(w.blen), w.spacing, UInt64(w.base), UInt32(w.tern),
          zeros(UInt8, SP_FILE_HDR - 40))
    write(w.io, Int32[w.nrot, w.nslt, 0, 0]); seek(w.io, SP_FILE_HDR + SP_PH); write(w.io, w.tab)
    close(w.io)
    quiet || @printf(stderr, "SPT=2p: %d rotations, %d mixed blocks, pool %.1f MB, file %.1f MB -> %s (%.0f s)\n", w.nrot, w.nslt,
                     w.poolt / 1e6, (w.base + SP_FILE_HDR) / 1e6, w.path, time() - w.t0)
    return nothing
end

@inline rd32(b::AbstractVector{UInt8}, o::Int) = UInt32(b[o+1]) | UInt32(b[o+2]) << 8 | UInt32(b[o+3]) << 16 | UInt32(b[o+4]) << 24
@inline rd64(b::AbstractVector{UInt8}, o::Int) = UInt64(rd32(b, o)) | UInt64(rd32(b, o + 4)) << 32

is_sp2p(path::AbstractString) = isfile(path) && open(io -> (m = read(io, 4); length(m) == 4 && m == SP_MAGIC), path, "r")

"(nrot, blen, spacing, tern) of a sparse table file."
function read_sp2p_header(path::AbstractString)
    h = open(io -> read(io, SP_FILE_HDR), path, "r")
    (length(h) == SP_FILE_HDR && h[1:4] == SP_MAGIC) || error("$path is not a sparse table file (fmap.ve.sp2p)")
    rd32(h, 8) == UInt32(2 + 32) || error("$path: only the per-rotation table (SPT=2p) is supported, mode field $(rd32(h, 8))")
    tern = Int(rd32(h, 36)); tern == 0 && (tern = 2)
    return Int(reinterpret(Int32, rd32(h, 12))), Int(reinterpret(Int32, rd32(h, 16))), reinterpret(Float64, rd64(h, 20)), tern
end

struct Sp2pFile
    nrot::Int
    blen::Int
    spacing::Float64
    tern::Int
    blob::Vector{UInt8}          # the blob as the GPU engines hold it (offsets are relative to it)
end

function read_sp2p(path::AbstractString)
    nrot, blen, spacing, tern = read_sp2p_header(path)
    open(path, "r") do io
        h = read(io, SP_FILE_HDR)
        nbytes = Int(rd64(h, 28))
        blob = Vector{UInt8}(undef, nbytes); read!(io, blob)
        return Sp2pFile(nrot, blen, spacing, tern, blob)
    end
end

# ---- decoding ----

"""
    decode_rotation!(out, f, r)

Dense grid of rotation `r` (0-based) into `out` (`f.tern * blen^3` bytes, cell-major as it was compressed), overwriting it.
"""
function decode_rotation!(out::AbstractVector{Int8}, f::Sp2pFile, r::Integer)
    B = f.blen; tern = f.tern; L = unit_layout(B); nb = L.nb; w = f.blob
    length(out) == tern * B^3 || error("out has $(length(out)) bytes, expected $(tern * B^3)")
    fill!(out, Int8(0))
    uh = SP_PH + 16 * Int(r)
    bm = Int(rd32(w, uh)); ofo = Int(rd32(w, uh + 8)); pool = Int(rd32(w, uh + 12))
    nsl = 0
    for blk in 0:L.nblk-1
        c = (rd32(w, bm + 4 * (blk >> 4)) >> ((blk & 15) << 1)) & 0x00000003
        c == 0x00000000 && continue
        bk = blk % nb; bj = (blk ÷ nb) % nb; bi = blk ÷ (nb * nb)
        if c == 0x00000001
            for cib in 0:63
                i = bi * 4 + (cib >> 4); j = bj * 4 + ((cib >> 2) & 3); k = bk * 4 + (cib & 3)
                (i < B && j < B && k < B) || continue
                cell = (i * B + j) * B + k
                if tern == 2; out[2cell+1] = Int8(127); out[2cell+2] = Int8(127); else out[cell+1] = Int8(127); end
            end
        else
            off = pool + Int(rd32(w, ofo + 4 * nsl)); nsl += 1
            nd = Int(w[off+1]); bits = bits_of(nd)
            for cib in 0:63
                i = bi * 4 + (cib >> 4); j = bj * 4 + ((cib >> 2) & 3); k = bk * 4 + (cib & 3)
                (i < B && j < B && k < B) || continue
                idx = 0
                if nd > 1
                    bp = cib * bits; by = off + 1 + 2 * nd + (bp >> 3)
                    v = Int(w[by+1]) | (Int(w[by+2]) << 8)
                    idx = (v >> (bp & 7)) & ((1 << bits) - 1)
                end
                cell = (i * B + j) * B + k
                if tern == 2
                    out[2cell+1] = reinterpret(Int8, w[off+2+2idx]); out[2cell+2] = reinterpret(Int8, w[off+3+2idx])
                else
                    out[cell+1] = reinterpret(Int8, w[off+2+2idx])
                end
            end
        end
    end
    return out
end

"All rotations, `nrot * tern * blen^3` bytes (several GB for production grids; use `decode_rotation!` to stream)."
function decode_all(f::Sp2pFile)
    n = f.tern * f.blen^3
    out = Vector{Int8}(undef, f.nrot * n)
    Threads.@threads for r in 0:f.nrot-1
        decode_rotation!(view(out, r * n + 1:(r + 1) * n), f, r)
    end
    return out
end

# ---- statistics on the compressed units ----

"""
    rotation_histogram(f, r) -> Vector{Int}

Histogram (256 bins, bin `v + 129` for the Int8 value `v`) of the grid of rotation `r`, counted on the compressed unit without decoding it.
A cell counts as its elec value (TERN_BYTES=1) or as elec + vdw clamped to the Int8 range, 127 where either is excluded volume (2). The
padding cells of a grid length that is not a multiple of 4 are not counted.
"""
function rotation_histogram(f::Sp2pFile, r::Integer)
    B = f.blen; tern = f.tern; L = unit_layout(B); nb = L.nb; w = f.blob
    h = zeros(Int, 256)
    uh = SP_PH + 16 * Int(r)
    bm = Int(rd32(w, uh)); ofo = Int(rd32(w, uh + 8)); pool = Int(rd32(w, uh + 12))
    comb(e0::Int, e1::Int) = tern == 1 ? e0 : ((e0 == 127 || e1 == 127) ? 127 : clamp(e0 + e1, -128, 127))
    nsl = 0
    for blk in 0:L.nblk-1
        c = (rd32(w, bm + 4 * (blk >> 4)) >> ((blk & 15) << 1)) & 0x00000003
        bk = blk % nb; bj = (blk ÷ nb) % nb; bi = blk ÷ (nb * nb)
        nv = min(4, B - 4bi) * min(4, B - 4bj) * min(4, B - 4bk)          # cells of the block inside the grid
        if c == 0x00000000
            h[129] += nv
        elseif c == 0x00000001
            h[256] += nv
        else
            off = pool + Int(rd32(w, ofo + 4 * nsl)); nsl += 1
            nd = Int(w[off+1]); bits = bits_of(nd)
            for cib in 0:63
                i = bi * 4 + (cib >> 4); j = bj * 4 + ((cib >> 2) & 3); k = bk * 4 + (cib & 3)
                (i < B && j < B && k < B) || continue
                idx = 0
                if nd > 1
                    bp = cib * bits; by = off + 1 + 2 * nd + (bp >> 3)
                    v = Int(w[by+1]) | (Int(w[by+2]) << 8)
                    idx = (v >> (bp & 7)) & ((1 << bits) - 1)
                end
                e0 = Int(reinterpret(Int8, w[off+2+2idx])); e1 = Int(reinterpret(Int8, w[off+3+2idx]))
                h[comb(e0, e1)+129] += 1
            end
        end
    end
    return h
end

# ---- SPT=1: the lookup table of the engines, built from the 2p units without a dense grid ----
# table = [16 B: i32 number of stored blocks] [dir: i32 per block, index of its stored block or -1] [class map: ceil(nrot/4) bytes per block,
#          2 bits per orientation (0 all zero, 1 all overlap)] [stored blocks: [cell in block][orientation][elec, vdw]]
# A block is stored if some orientation has a mixed block there (the others need only their class); the GPU engine builds the same table
# (native/fmapsim.cu, load_sp2p).

"""
    sparse1_table(f) -> Vector{Int8}

The block-sparse lookup table (SPT=1) of all rotations of `f`, made directly from the compressed units.
"""
function sparse1_table(f::Sp2pFile)
    nrot = f.nrot; B = f.blen; w = f.blob
    nb = cld(B, 4); nblk = nb^3; mapb = (nrot + 3) >> 2
    sp_map = 16 + 4 * nblk
    sp_data = (sp_map + mapb * nblk + 15) ÷ 16 * 16
    cls = zeros(UInt8, nblk * mapb); stored = zeros(Int32, nblk)
    Threads.@threads for blk in 0:nblk-1
        any2 = false; wo = 4 * (blk >> 4); sh = (blk & 15) << 1
        for r in 0:nrot-1
            bm = Int(rd32(w, SP_PH + 16 * r))
            c = (rd32(w, bm + wo) >> sh) & 0x00000003
            any2 |= c == 0x00000002
            @inbounds cls[blk * mapb + (r >> 2) + 1] |= UInt8(c) << ((r & 3) << 1)
        end
        @inbounds stored[blk+1] = any2
    end
    inc = cumsum(stored); nst = Int(inc[end])
    dir = Vector{Int32}(undef, nblk); blkof = Vector{Int32}(undef, nst)
    for blk in 1:nblk
        if stored[blk] == 1
            dir[blk] = inc[blk] - 1; blkof[inc[blk]] = blk - 1
        else
            dir[blk] = -1
        end
    end
    sp = zeros(Int8, sp_data + nst * 64 * nrot * 2 + 4)
    copyto!(sp, 1, reinterpret(Int8, Int32[nst, 0, 0, 0]), 1, 16)
    copyto!(sp, 17, reinterpret(Int8, dir), 1, 4 * nblk)
    copyto!(sp, sp_map + 1, reinterpret(Int8, cls), 1, length(cls))
    cnt_mixed(x::UInt32) = count_ones((x >> 1) & ~x & 0x55555555)
    Threads.@threads for d in 0:nst-1
        blk = Int(blkof[d+1]); widx = blk >> 4; sh = (blk & 15) << 1
        for r in 0:nrot-1
            uh = SP_PH + 16 * r
            bm = Int(rd32(w, uh))
            word = rd32(w, bm + 4 * widx)
            c = (word >> sh) & 0x00000003
            if c == 0x00000001
                for cib in 0:63
                    o = sp_data + ((d * 64 + cib) * nrot + r) * 2
                    sp[o+1] = Int8(127); sp[o+2] = Int8(127)
                end
            elseif c == 0x00000002
                g = widx >> 2
                rk = Int(rd32(w, Int(rd32(w, uh + 4)) + 4 * g))
                for q in (g << 2):widx-1
                    rk += cnt_mixed(rd32(w, bm + 4 * q))
                end
                rk += cnt_mixed(word & ((UInt32(1) << sh) - 0x00000001))
                off = Int(rd32(w, uh + 12)) + Int(rd32(w, Int(rd32(w, uh + 8)) + 4 * rk))
                nd = Int(w[off+1]); bits = bits_of(nd)
                for cib in 0:63
                    idx = 0
                    if nd > 1
                        bp = cib * bits; by = off + 1 + 2 * nd + (bp >> 3)
                        v = Int(w[by+1]) | (Int(w[by+2]) << 8)
                        idx = (v >> (bp & 7)) & ((1 << bits) - 1)
                    end
                    o = sp_data + ((d * 64 + cib) * nrot + r) * 2
                    sp[o+1] = reinterpret(Int8, w[off+2+2idx]); sp[o+2] = reinterpret(Int8, w[off+3+2idx])
                end
            end
        end
    end
    return sp
end

"(elec, vdw) of cell (i, j, k) (0-based, inside the grid) of orientation a3 (0-based) from an SPT=1 table."
@inline function sp1_ve(w::Vector{Int8}, blen::Int, nrot::Int, i::Int, j::Int, k::Int, a3::Int)
    nb = (blen + 3) >> 2; nblk = nb * nb * nb; mapb = (nrot + 3) >> 2
    sp_map = 16 + 4 * nblk
    blk = ((i >> 2) * nb + (j >> 2)) * nb + (k >> 2)
    d = GC.@preserve w Int(unsafe_load(Ptr{Int32}(pointer(w) + 16 + 4 * blk)))
    if d >= 0
        sp_data = (sp_map + mapb * nblk + 15) ÷ 16 * 16
        off = sp_data + ((d * 64 + ((i & 3) * 4 + (j & 3)) * 4 + (k & 3)) * nrot + a3) * 2
        return (@inbounds w[off+1]), (@inbounds w[off+2])
    end
    mb = reinterpret(UInt8, @inbounds w[sp_map + blk * mapb + (a3 >> 2) + 1])
    c = (mb >> ((a3 & 3) << 1)) & 0x03
    return c == 0x00 ? (Int8(0), Int8(0)) : (Int8(127), Int8(127))
end

"""
    sp2p_blob(f) -> Vector{Int8}

The blob of a file as the in-place lookup `sp2p_ve` reads it (the layout the GPU engines use for SPT=2p), padded by 16 bytes.
"""
function sp2p_blob(f::Sp2pFile)
    w = zeros(Int8, length(f.blob) + 16)
    copyto!(w, 1, reinterpret(Int8, f.blob), 1, length(f.blob))
    return w
end

@inline ld32(p::Ptr{UInt8}, off::Int) = unsafe_load(Ptr{UInt32}(p + off))
@inline cnt_mixed(x::UInt32) = count_ones((x >> 1) & ~x & 0x55555555)

"""
    sp2p_ve(w, blen, i, j, k, a3) -> (elec, vdw)

Cell (i, j, k) (0-based, inside the grid) of orientation a3 (0-based), read in place from the blob of a `.sp2p` file (`sp2p_blob`): the class bitmap of the
4^3 blocks, then for a mixed block its rank among the mixed blocks and its palette slice. The same arithmetic as `fmap_sp2p_ve` in native/fmap_core.h.
"""
@inline function sp2p_ve(w::Vector{Int8}, blen::Int, i::Int, j::Int, k::Int, a3::Int)
    GC.@preserve w begin
        p = Ptr{UInt8}(pointer(w))
        nb = (blen + 3) >> 2
        blk = ((i >> 2) * nb + (j >> 2)) * nb + (k >> 2)
        uh = SP_PH + 16 * a3
        bm = Int(ld32(p, uh)); widx = blk >> 4; sh = (blk & 15) << 1
        word = ld32(p, bm + 4 * widx)
        c = (word >> sh) & 0x00000003
        c == 0x00000000 && return (Int8(0), Int8(0))
        c == 0x00000001 && return (Int8(127), Int8(127))
        g = widx >> 2
        r = Int(ld32(p, Int(ld32(p, uh + 4)) + 4 * g))
        for q in (g << 2):(widx - 1)
            r += cnt_mixed(ld32(p, bm + 4 * q))
        end
        r += cnt_mixed(word & ((UInt32(1) << sh) - 0x00000001))
        off = Int(ld32(p, uh + 12)) + Int(ld32(p, Int(ld32(p, uh + 8)) + 4 * r))
        nd = Int(unsafe_load(p + off)); idx = 0
        if nd > 1
            bits = 32 - leading_zeros(UInt32(nd - 1))
            bp = (((i & 3) * 4 + (j & 3)) * 4 + (k & 3)) * bits
            by = off + 1 + 2 * nd + (bp >> 3)
            v = Int(unsafe_load(p + by)) | (Int(unsafe_load(p + by + 1)) << 8)
            idx = (v >> (bp & 7)) & ((1 << bits) - 1)
        end
        pe = off + 1 + 2 * idx
        return (reinterpret(Int8, unsafe_load(p + pe)), reinterpret(Int8, unsafe_load(p + pe + 1)))
    end
end

end # module
