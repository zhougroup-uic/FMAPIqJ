# Symmetry-folded, bit-packed pair tables (included by Rotij.jl).
#
# The sample list is (to the 6-decimal rounding of rot.eul, ~1e-6 rad) invariant under a group of rotations:
#   left  g:  R_{s_g(i)} = g R_i        (HEALPix x psi: order 6 for the sequence samples)
#   right h:  R_{t_h(i)} = R_i h        (order 2)
# so, for the relative rotation tables  P[i,j] = nearest sample of R_i' R_j   (angij2k, inv=true)
#                                         Q[i,j] = nearest sample of R_i  R_j   (angij2a, inv=false)
#   P[s_g(i), s_g(j)] = P[i,j]
#   Q[s_g(i), j] = s_g(Q[i,j]),   Q[i, t_h(j)] = t_h(Q[i,j])
# Only the orbit representatives are stored (n^2/|G| entries, bit-packed to ceil(log2 n) bits); an entry is
# recovered with a few table lookups (O(1), no decoding of neighbours). Entries equal the full table except
# where two samples are equally near within the rounding of rot.eul (measured: <1% of entries at 4608, the
# returned sample is at most ~2e-4 degrees farther than the nearest one).

"""
    symmetries(S; tol=1e-4) -> (left, right)

Permutations of the samples induced by rotations that map the sample list onto itself within `tol` rad:
`left[g][i+1]` = 0-based `s_g(i)` with `R_{s_g(i)} = g R_i`, `right[h][i+1]` for `R_i h`. The identity is
first in each list. The candidates are g = R_k R_1' (left) and R_1' R_k (right), k = 1..n.
"""
function symmetries(S::RotSamples; tol::Float64=1e-4)
    n = length(S)
    sub = unique(round.(Int, range(1, n; length=min(n, 16))))
    R1 = getmat(S.R, 1)
    R1t = transpose9(R1)
    function search(left::Bool)
        found = Vector{Vector{Int32}}()
        for k in 1:n
            Rk = getmat(S.R, k)
            g = left ? mul9(Rk, R1t) : mul9(R1t, Rk)
            image(i) = left ? mul9(g, getmat(S.R, i)) : mul9(getmat(S.R, i), g)
            ok = all(sub) do i
                _, a = nearest(S, image(i))
                a < tol
            end
            ok || continue
            perm = Vector{Int32}(undef, n)
            seen = falses(n)
            good = true
            best = Best(); hint = 0
            for i in 1:n
                j, a = nearest(S, image(i); hint=hint, best=best)
                (a < tol && !seen[j]) || (good = false; break)
                seen[j] = true; perm[i] = j - 1; hint = j
            end
            good || continue
            k == 1 ? pushfirst!(found, perm) : push!(found, perm)
        end
        isempty(found) || found[1] == Int32.(0:n-1) || pushfirst!(found, Int32.(0:n-1))
        return found
    end
    return search(true), search(false)
end

inverse_perms(P::Vector{Vector{Int32}}) = [begin q = similar(p); q[Int.(p) .+ 1] .= Int32.(0:length(p)-1); q end for p in P]

# orbit representatives: rep[i] = min over group of image(i) (0-based), elem[i] = first element reaching it
function orbit_reps(P::Vector{Vector{Int32}}, n::Int)
    rep = Vector{Int32}(undef, n); elem = Vector{UInt8}(undef, n)
    for i in 1:n
        r = typemax(Int32); e = 0
        for (g, p) in enumerate(P)
            p[i] < r && (r = p[i]; e = g)
        end
        rep[i] = r; elem[i] = e
    end
    return rep, elem
end

"""
    FoldedTable

Folded, bit-packed pair table. `T[i,j]` (1-based `i`, `j`) equals the 0-based entry `angij(S; inv)[i,j]` up to the
near-ties described in `Rotij`'s `Fold.jl` header. `kind` is `:k` (angij2k, inv=true) or `:a` (angij2a).
"""
struct FoldedTable
    kind::Symbol
    n::Int
    bits::Int
    nr::Int                    # stored rows
    nc::Int                    # stored columns
    rrow::Vector{Int32}        # compact row (0-based) of the row representative of i
    rg::Vector{UInt8}          # group element (1-based) with s_g(i) = representative
    ccol::Vector{Int32}        # compact column of the column representative of j  (:a only)
    ch::Vector{UInt8}          # right element (1-based) with t_h(c) = j            (:a only)
    PL::Matrix{Int32}          # n × |G_left|,  s_g(j) (0-based)
    PR::Matrix{Int32}          # n × |G_right|, t_h(j) (0-based)
    PLi::Matrix{Int32}         # inverse permutations of PL
    data::Vector{UInt64}       # nr*nc entries of `bits` bits
    keep::Any                  # memory map owner (when read with mmap)
end

nbits(n::Int) = max(1, 64 - leading_zeros(UInt64(n - 1)))

@inline function getbits(w::AbstractVector{UInt64}, e::Int, bits::Int)
    pos = e * bits
    wi = (pos >>> 6) + 1; off = pos & 63
    mask = (UInt64(1) << bits) - UInt64(1)
    v = w[wi] >>> off
    off + bits > 64 && (v |= w[wi+1] << (64 - off))
    return Int(v & mask)
end

function pack(M::AbstractMatrix{<:Integer}, bits::Int)
    nr, nc = size(M)
    w = zeros(UInt64, (nr * nc * bits + 63) >>> 6 + 1)
    e = 0
    for r in 1:nr, c in 1:nc          # entry index e = (r-1)*nc + (c-1)
        v = UInt64(M[r, c]); pos = e * bits
        wi = (pos >>> 6) + 1; off = pos & 63
        w[wi] |= v << off
        off + bits > 64 && (w[wi+1] |= v >>> (64 - off))
        e += 1
    end
    return w
end

function Base.getindex(T::FoldedTable, i::Integer, j::Integer)
    gi = T.rg[i]; r = Int(T.rrow[i])
    if T.kind === :k
        c = Int(T.PL[j, gi]) + 1                       # P[i,j] = P[rep, s_g(j)]
        return getbits(T.data, r * T.nc + c - 1, T.bits)
    else
        c = Int(T.ccol[j])
        v = getbits(T.data, r * T.nc + c, T.bits)      # Q[rep, column rep]
        v = Int(T.PR[v+1, T.ch[j]])                    # Q[rep, j] = t_h(Q[rep, c])
        return Int(T.PLi[v+1, gi])                     # Q[i, j]   = s_g^-1(Q[rep, j])
    end
end

Base.size(T::FoldedTable) = (T.n, T.n)
Base.size(T::FoldedTable, d::Integer) = d <= 2 ? T.n : 1
Base.Matrix(T::FoldedTable) = Int32[T[i, j] for i in 1:T.n, j in 1:T.n]

"""bytes of the table with its index arrays (what is read at run time), and the size of the full 16-bit table."""
storage_bytes(T::FoldedTable) = sizeof(T.data) + sizeof(T.rrow) + sizeof(T.rg) + sizeof(T.ccol) + sizeof(T.ch) + sizeof(T.PL) + sizeof(T.PR)
storage_bytes_full(T::FoldedTable) = 2 * T.n^2

"""
    fold_angij(S; inv=true, sym=symmetries(S)) -> FoldedTable

Builds only the orbit representatives with exact searches (|G| times fewer than `angij`; `|G|` = 6 for
`inv=true`, 12 for `inv=false` on the HEALPix x psi sequences) and packs them.
"""
function fold_angij(S::RotSamples; inv::Bool=true, sym=symmetries(S))
    n = length(S)
    left, right = sym
    isempty(left) && (left = [Int32.(0:n-1)])
    isempty(right) && (right = [Int32.(0:n-1)])
    bits = nbits(n)
    rrep, rg = orbit_reps(left, n)                       # s_g(i) = rep, minimum over the group
    PL = reduce(hcat, left); PLi = reduce(hcat, inverse_perms(left)); PR = reduce(hcat, right)
    rows = sort!(unique(Int.(rrep)))                     # representative rows (0-based)
    ridx = Dict(r => k - 1 for (k, r) in enumerate(rows))
    rrow = Int32[ridx[Int(r)] for r in rrep]
    if inv
        cols = collect(0:n-1)
        ccol = Int32.(0:n-1); ch = ones(UInt8, n)
    else
        Ri = inverse_perms(right)                        # j = t_h(c)  =>  c = t_h^-1(j)
        crep = Vector{Int32}(undef, n); ch = Vector{UInt8}(undef, n)
        for j in 1:n
            r = typemax(Int32); e = 0
            for (h, p) in enumerate(Ri)
                p[j] < r && (r = p[j]; e = h)
            end
            crep[j] = r; ch[j] = e
        end
        cols = sort!(unique(Int.(crep)))
        cidx = Dict(c => k - 1 for (k, c) in enumerate(cols))
        ccol = Int32[cidx[Int(c)] for c in crep]
    end
    nr, nc = length(rows), length(cols)
    D = Matrix{IDX}(undef, nr, nc)
    Threads.@threads for a in 1:nr
        ri = getmat(S.R, rows[a] + 1)
        inv && (ri = transpose9(ri))
        best = Best(); hint = 0
        for b in 1:nc
            k, _ = nearest(S, mul9(ri, getmat(S.R, cols[b] + 1)); hint=hint, best=best)
            hint = k
            D[a, b] = k - 1
        end
    end
    return FoldedTable(inv ? :k : :a, n, bits, nr, nc, rrow, rg, ccol, ch, PL, PR, PLi, pack(D, bits), nothing)
end

# ---------------------------------------------------------------- file: header (8-byte words) + padded sections
const FOLD_MAGIC = 0x52464c4431          # "RFLD1"

pad8(x) = (x + 7) & ~7

"""`write_folded(path, T)`: raw little-endian file, sections 8-byte aligned so it can be memory-mapped (`read_folded`)."""
function write_folded(path::AbstractString, T::FoldedTable)
    open(path, "w") do io
        hdr = Int64[FOLD_MAGIC, T.kind === :k ? 0 : 1, T.n, T.bits, T.nr, T.nc, size(T.PL, 2), size(T.PR, 2), length(T.data)]
        write(io, hdr)
        for a in (T.rrow, T.rg, T.ccol, T.ch, T.PL, T.PR, T.data)
            write(io, a)
            write(io, zeros(UInt8, pad8(sizeof(a)) - sizeof(a)))
        end
    end
    return path
end

"""`read_folded(path; mmap=true)`: the table, with its arrays memory-mapped (only touched pages are read)."""
function read_folded(path::AbstractString; mmap::Bool=true)
    bytes = mmap ? Mmap.mmap(path, Vector{UInt8}) : read(path)
    hdr = unsafe_wrap(Array, Ptr{Int64}(pointer(bytes)), 9)
    hdr[1] == FOLD_MAGIC || error("$path: not a folded table")
    kind = hdr[2] == 0 ? :k : :a
    n, bits, nr, nc, ngl, ngr, nw = Int.(hdr[3:9])
    off = 72
    function take(::Type{T}, dims...) where {T}
        len = prod(dims)
        a = unsafe_wrap(Array, Ptr{T}(pointer(bytes) + off), dims)
        off += pad8(len * sizeof(T))
        return a
    end
    rrow = take(Int32, n); rg = take(UInt8, n); ccol = take(Int32, n); ch = take(UInt8, n)
    PL = take(Int32, n, ngl); PR = take(Int32, n, ngr); data = take(UInt64, nw)
    PLi = reduce(hcat, inverse_perms([PL[:, g] for g in 1:ngl]))
    return FoldedTable(kind, n, bits, nr, nc, rrow, rg, ccol, ch, PL, PR, PLi, data, bytes)
end
