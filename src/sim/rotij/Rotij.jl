"""
    Rotij

Rotation lookup tables for a list of Euler angles (`rot.eul`, ZXZ, rad), the Julia version of the
C tools in `../src`. Everything is built on demand from the sample list in seconds:

- `angijk`   nearest sample of every (alpha, cos beta, gamma) bin            (`angijk.dat.gz`)
- `angij`    nearest sample of R_i^T R_j (`inv=true`, A2-A1: `angij2k`) or R_i R_j (`inv=false`,
             A1+A2: `angij2a`)                                                  (`angij2k/2a.dat.gz`)
             Optional O(1) approximation from the `angijk` table: `angij(S; A=angijk(S))`, `angij_lookup`.
- `fold_angij` symmetry-folded, bit-packed, memory-mappable `angij` tables (`Fold.jl`; prototype)
- `rotm`     inverse rotation matrices, 9 doubles per sample                    (`rotm.dat.gz`)
- `chkmin`   nearest-neighbour rotation angle of every sample                   (`rot.eul.chkmin.nk3`)

Nearest sample = exact (up to exact-distance ties) via a ball tree on the rotation matrices; the
distance is the rotation angle acos((tr(R1' R2)-1)/2). No external dependencies (standard library: Printf, Mmap); use `julia -t auto`.
Index tables hold 0-based sample indices, as in the `.dat.gz` files of the C pipeline.
"""
module Rotij

using Printf, Mmap

export RotSamples, load_eul, nearest, angijk, angij, bin_lookup, angij_lookup, rotm, chkmin, write_tables, build_all,
       symmetries, fold_angij, FoldedTable, write_folded, read_folded, storage_bytes, storage_bytes_full

const LEAF = 8

# ---------------------------------------------------------------- rotations
const Mat9 = NTuple{9,Float64}      # 3x3 matrix, row-major (same layout as the C code)

"""R = Rz(a) Rx(b) Rz(c); `Euler2Rot` of the C code with (a,b,c) = (e1,e2,e3)."""
function euler2rot(a::Float64, b::Float64, c::Float64)::Mat9
    ca, sa = cos(a), sin(a)
    cb, sb = cos(b), sin(b)
    cc, sc = cos(c), sin(c)
    return (ca*cc - sa*cb*sc, -ca*sc - sa*cb*cc,  sa*sb,
            sa*cc + ca*cb*sc, -sa*sc + ca*cb*cc, -ca*sb,
            sb*sc,             sb*cc,             cb)
end

euler2rot(a::Real, b::Real, c::Real) = euler2rot(Float64(a), Float64(b), Float64(c))

transpose9(m::Mat9)::Mat9 = (m[1], m[4], m[7], m[2], m[5], m[8], m[3], m[6], m[9])

function mul9(a::Mat9, b::Mat9)::Mat9
    return (a[1]*b[1]+a[2]*b[4]+a[3]*b[7], a[1]*b[2]+a[2]*b[5]+a[3]*b[8], a[1]*b[3]+a[2]*b[6]+a[3]*b[9],
            a[4]*b[1]+a[5]*b[4]+a[6]*b[7], a[4]*b[2]+a[5]*b[5]+a[6]*b[8], a[4]*b[3]+a[5]*b[6]+a[6]*b[9],
            a[7]*b[1]+a[8]*b[4]+a[9]*b[7], a[7]*b[2]+a[8]*b[5]+a[9]*b[8], a[7]*b[3]+a[8]*b[6]+a[9]*b[9])
end

# score = M.N (Frobenius); larger = closer; angle = acos((score-1)/2)
@inline function score(R::Matrix{Float64}, i::Int, q::Mat9)
    s = 0.0
    @inbounds for k in 1:9
        s += R[k, i] * q[k]
    end
    return s
end
@inline score2angle(s::Float64) = acos(clamp((s - 1.0) / 2.0, -1.0, 1.0))
@inline getmat(R::Matrix{Float64}, i::Int)::Mat9 =
    @inbounds (R[1,i], R[2,i], R[3,i], R[4,i], R[5,i], R[6,i], R[7,i], R[8,i], R[9,i])

# ---------------------------------------------------------------- ball tree
struct BallTree
    center::Vector{Int}
    lo::Vector{Int}
    hi::Vector{Int}
    left::Vector{Int}        # 0 = leaf
    right::Vector{Int}
    rad::Vector{Float64}     # largest rotation angle from the centre to a sample below
    rcos::Vector{Float64}
    rsin::Vector{Float64}
    idx::Vector{Int}         # sample rows, permuted so that each node owns idx[lo:hi]
end

function build_tree(R::Matrix{Float64})
    n = size(R, 2)
    idx = collect(1:n)
    t = BallTree(Int[], Int[], Int[], Int[], Int[], Float64[], Float64[], Float64[], idx)
    build!(t, R, 1, n)
    return t
end

function build!(t::BallTree, R::Matrix{Float64}, lo::Int, hi::Int)
    idx = t.idx
    id = length(t.center) + 1
    for v in (t.center, t.lo, t.hi, t.left, t.right); push!(v, 0); end
    for v in (t.rad, t.rcos, t.rsin); push!(v, 0.0); end
    t.lo[id] = lo; t.hi[id] = hi
    # a = farthest from the first member, b = farthest from a
    a = idx[lo]; smin = Inf
    m0 = getmat(R, idx[lo])
    for i in lo:hi
        s = score(R, idx[i], m0)
        if s < smin; smin = s; a = idx[i]; end
    end
    ma = getmat(R, a)
    b = a; smin = Inf
    for i in lo:hi
        s = score(R, idx[i], ma)
        if s < smin; smin = s; b = idx[i]; end
    end
    t.center[id] = a
    rmax = 0.0
    for i in lo:hi
        rmax = max(rmax, score2angle(score(R, idx[i], ma)))
    end
    t.rad[id] = rmax; t.rcos[id] = cos(rmax); t.rsin[id] = sin(rmax)
    hi - lo + 1 <= LEAF && return id
    mb = getmat(R, b)
    mid = lo - 1                       # partition: nearer to a first, then nearer to b
    for i in lo:hi
        if score(R, idx[i], ma) >= score(R, idx[i], mb)
            mid += 1
            idx[i], idx[mid] = idx[mid], idx[i]
        end
    end
    (mid < lo || mid == hi) && (mid = (lo + hi - 1) ÷ 2)   # duplicates: split by position
    l = build!(t, R, lo, mid)
    r = build!(t, R, mid + 1, hi)
    t.left[id] = l; t.right[id] = r
    return id
end

mutable struct Best          # state of one nearest-sample search
    s::Float64               # score of the best sample so far
    i::Int
    ang::Float64
    c::Float64
    sn::Float64
end
Best() = Best(-Inf, 0, Inf, 0.0, 0.0)

@inline function setbest!(b::Best, i::Int, s::Float64)
    c = clamp((s - 1.0) / 2.0, -1.0, 1.0)
    b.s = s; b.i = i; b.c = c; b.sn = sqrt(1.0 - c*c); b.ang = acos(c)
    return nothing
end

function visit!(b::Best, t::BallTree, R::Matrix{Float64}, id::Int, s::Float64, q::Mat9, skip::Int)
    @inbounds begin
        ctr = t.center[id]
        if s > b.s && ctr != skip
            setbest!(b, ctr, s)
        end
        rad = t.rad[id]
        if b.ang + rad < pi                       # else nothing can be pruned yet
            c = (s - 1.0) / 2.0
            thr = b.c * t.rcos[id] - b.sn * t.rsin[id]    # cos(best + rad)
            c <= thr - 1e-12 && return nothing    # angle(query,centre) >= best+rad: nothing below can win
        end
        l = t.left[id]
        if l == 0
            for k in t.lo[id]:t.hi[id]
                j = t.idx[k]
                j == skip && continue
                sj = score(R, j, q)
                sj > b.s && setbest!(b, j, sj)
            end
            return nothing
        end
        r = t.right[id]
        sl = score(R, t.center[l], q)
        sr = score(R, t.center[r], q)
        if sl >= sr
            visit!(b, t, R, l, sl, q, skip); visit!(b, t, R, r, sr, q, skip)
        else
            visit!(b, t, R, r, sr, q, skip); visit!(b, t, R, l, sl, q, skip)
        end
    end
    return nothing
end

# ---------------------------------------------------------------- samples
"""
    RotSamples(eul)    # eul: 3×n matrix of Euler angles (ZXZ, rad)

The sample list with its rotation matrices and ball tree (built in O(n log n)).
"""
struct RotSamples
    eul::Matrix{Float64}     # 3×n
    R::Matrix{Float64}       # 9×n, row-major matrices of Rz(e1) Rx(e2) Rz(e3)
    tree::BallTree
end

function RotSamples(eul::AbstractMatrix{<:Real})
    size(eul, 1) == 3 || throw(ArgumentError("eul must be 3×n"))
    e = Matrix{Float64}(eul)
    n = size(e, 2)
    R = Matrix{Float64}(undef, 9, n)
    for i in 1:n
        R[:, i] .= euler2rot(e[1,i], e[2,i], e[3,i])
    end
    return RotSamples(e, R, build_tree(R))
end

Base.length(S::RotSamples) = size(S.eul, 2)

"""Read `rot.eul` (whitespace separated Euler angles, one sample per line)."""
function load_eul(file::AbstractString)
    vals = Float64[]
    for line in eachline(file)
        f = split(line)
        isempty(f) && continue
        length(f) >= 3 || error("$file: bad line: $line")
        append!(vals, parse.(Float64, f[1:3]))
    end
    return RotSamples(reshape(vals, 3, :))
end

"""
    nearest(S, q::NTuple{9,Float64}; hint=0, skip=0) -> (row, angle)

Nearest sample (1-based row) to the rotation matrix `q` (row-major), and the rotation angle (rad).
`hint` = a row to start from (any row; speeds up queries close to the previous one), `skip` = row
to exclude.
"""
function nearest(S::RotSamples, q::Mat9; hint::Int=0, skip::Int=0, best::Best=Best())
    best.s = -Inf; best.i = 0; best.ang = Inf
    if hint > 0 && hint != skip
        setbest!(best, hint, score(S.R, hint, q))
    end
    root = S.tree.center[1]
    visit!(best, S.tree, S.R, 1, score(S.R, root, q), q, skip)
    return best.i, best.ang
end

nearest(S::RotSamples, a::Real, b::Real, c::Real; kw...) =
    nearest(S, euler2rot(Float64(a), Float64(b), Float64(c)); kw...)

# ---------------------------------------------------------------- tables
const IDX = Int32

"""
    angijk(S; nbin=120) -> Array{Int32,3}

`A[ai+1,aj+1,ak+1]` = 0-based nearest sample to the bin centre alpha = -pi + ai*2pi/nbin,
cos(beta) = 1 - (aj+0.5)*2/nbin, gamma = -pi + ak*2pi/nbin (same bins as `rotij_idps3.x rot.eul nbin`).
"""
function angijk(S::RotSamples; nbin::Int=120)
    A = Array{IDX,3}(undef, nbin, nbin, nbin)
    nd = nbin
    Threads.@threads for ai in 0:nd-1
        best = Best(); hint = 0
        a0 = -pi + ai * (2pi / nd)
        for aj in 0:nd-1
            a1 = acos(1.0 - aj * (2.0 / nd) - 1.0 / nd)
            for ak in 0:nd-1
                a2 = -pi + ak * (2pi / nd)
                j, _ = nearest(S, euler2rot(a0, a1, a2); hint=hint, best=best)
                hint = j
                A[ai+1, aj+1, ak+1] = j - 1
            end
        end
    end
    return A
end

"""
    bin_lookup(A, q::NTuple{9,Float64}) -> Int

Approximate nearest sample (0-based) of the rotation matrix `q` (row-major, Rz Rx Rz) by looking up the
bin table `A = angijk(S; nbin)`: Euler angles of `q`, bins `b1=(alpha+pi)nbin/2pi`, `b2=(1-cos beta)nbin/2`,
`b3=(gamma+pi)nbin/2pi`, `round` for alpha and gamma, `floor` (clamped) for beta because the bin centres in
beta are half a bin offset. At beta ~ 0 or pi only alpha+gamma (alpha-gamma) is defined; gamma = 0 is used.
The result is the nearest sample of the bin centre, so it can be a neighbour of the exact nearest sample
(4608 samples, random pairs: same sample in 89%, mean excess angle 0.09 deg, 99% below 1.8 deg, worst 8 deg).
"""
function bin_lookup(A::AbstractArray{<:Integer,3}, q::Mat9)
    nd = size(A, 1)
    if sqrt(q[3]^2 + q[6]^2) < 1e-9
        a = atan(q[4], q[1]); g = 0.0
    else
        a = atan(q[3], -q[6]); g = atan(q[7], q[8])
    end
    b2 = (1 - clamp(q[9], -1.0, 1.0)) * nd / 2
    i1 = mod(round(Int, (a + pi) * nd / 2pi), nd) + 1
    i3 = mod(round(Int, (g + pi) * nd / 2pi), nd) + 1
    i2 = clamp(floor(Int, b2), 0, nd - 1) + 1
    return Int(A[i1, i2, i3])
end

"""`angij_lookup(S, A, i, j; inv=true)`: approximate `angij(S; inv)[i,j]` (0-based) from the bin table `A` (see `bin_lookup`)."""
function angij_lookup(S::RotSamples, A::AbstractArray{<:Integer,3}, i::Int, j::Int; inv::Bool=true)
    ri = getmat(S.R, i)
    inv && (ri = transpose9(ri))
    return bin_lookup(A, mul9(ri, getmat(S.R, j)))
end

"""
    angij(S; inv=true, A=nothing) -> Matrix{Int32}

`M[i+1,j+1]` = 0-based nearest sample to `R_i^T R_j` (`inv=true`: relative rotation A2-A1,
`angij2k`) or to `R_i R_j` (`inv=false`: A1+A2, `angij2a`). Exact by default (n^2 searches, ~23 s each
at 4608 samples with 8 threads). With a bin table `A = angijk(S)` the entries are the O(1) approximation of
`bin_lookup` instead (n^2 lookups, well under a second at 4608; the table itself is not needed at run time).
"""
function angij(S::RotSamples; inv::Bool=true, A=nothing)
    n = length(S)
    M = Matrix{IDX}(undef, n, n)
    if A !== nothing
        Threads.@threads for i in 1:n
            for j in 1:n
                M[i, j] = angij_lookup(S, A, i, j; inv=inv)
            end
        end
        return M
    end
    Threads.@threads for i in 1:n
        best = Best(); hint = 0
        ri = getmat(S.R, i)
        inv && (ri = transpose9(ri))
        for j in S.tree.idx        # tree order = neighbours first; left-multiplying by R_i is an isometry,
            k, _ = nearest(S, mul9(ri, getmat(S.R, j)); hint=hint, best=best)   # so the previous result is a good hint
            hint = k
            M[i, j] = k - 1
        end
    end
    return M
end

"""`rotm(S)` -> 9×n matrix, column i = row-major `Euler2Rot(-e3,-e2,-e1)` (the inverse rotation), as `build_rotm`."""
function rotm(S::RotSamples)
    n = length(S)
    out = Matrix{Float64}(undef, 9, n)
    for i in 1:n
        out[:, i] .= euler2rot(-S.eul[3,i], -S.eul[2,i], -S.eul[1,i])
    end
    return out
end

"""`chkmin(S)` -> (j, angle): 1-based nearest other sample and rotation angle (rad) of every sample."""
function chkmin(S::RotSamples)
    n = length(S)
    j = Vector{Int}(undef, n); d = Vector{Float64}(undef, n)
    Threads.@threads for i in 1:n
        j[i], d[i] = nearest(S, getmat(S.R, i); skip=i)
    end
    return j, d
end

# ---------------------------------------------------------------- files
function write_raw(path::AbstractString, A)
    if endswith(path, ".gz")
        open(pipeline(`gzip -n -c`; stdout=path), "w") do io
            write(io, A)
        end
    else
        open(io -> write(io, A), path, "w")
    end
    return path
end

function index_eltype(n::Int, index_type)
    index_type === nothing || return index_type
    n <= 32768 || error("n=$n does not fit the 16-bit index of the .dat files; pass index_type=Int32")
    return Int16
end

"""
    write_tables(S, outdir; nbin=120, index_type=nothing, pair=:exact, tables=(:rot,:rotm,:chkmin,:angijk,:angij2k,:angij2a))

Writes the files of the C pipeline into `outdir`: `rot.dat.gz` (raw doubles), `rotm.dat.gz`,
`rot.eul.chkmin.nk3` (text, sorted by angle), `angijk.dat.gz`, `angij2k.dat.gz`, `angij2a.dat.gz`
(raw 16-bit 0-based indices, `index_type=Int32` for more than 32768 samples). `pair=:lookup` fills `angij2k`/`angij2a`
from the `angijk` table (approximate, seconds; see `bin_lookup`) instead of the exact searches.
"""
function write_tables(S::RotSamples, outdir::AbstractString; nbin::Int=120, index_type=nothing, pair::Symbol=:exact,
                      tables=(:rot, :rotm, :chkmin, :angijk, :angij2k, :angij2a))
    mkpath(outdir)
    n = length(S)
    T = index_eltype(n, index_type)
    p(f) = joinpath(outdir, f)
    :rot in tables && write_raw(p("rot.dat.gz"), S.eul)
    :rotm in tables && write_raw(p("rotm.dat.gz"), rotm(S))
    if :chkmin in tables
        j, d = chkmin(S)
        lines = [@sprintf("%8d%8d%12.6f", i - 1, j[i] - 1, d[i]) for i in 1:n]
        key(l) = parse(Float64, split(l)[3])
        sort!(lines; by=l -> (key(l), l))      # like `sort -nk3` (ties: whole line)
        open(p("rot.eul.chkmin.nk3"), "w") do io
            foreach(l -> println(io, l), lines)
        end
    end
    pair in (:exact, :lookup) || throw(ArgumentError("pair must be :exact or :lookup"))
    A = (:angijk in tables || (pair === :lookup && (:angij2k in tables || :angij2a in tables))) ? angijk(S; nbin=nbin) : nothing
    if :angijk in tables
        write_raw(p("angijk.dat.gz"), T.(permutedims(A, (3, 2, 1))))   # ai slowest, ak fastest
    end
    if :angij2k in tables
        write_raw(p("angij2k.dat.gz"), T.(permutedims(angij(S; inv=true, A=pair === :lookup ? A : nothing))))   # i slowest, j fastest
    end
    if :angij2a in tables
        write_raw(p("angij2a.dat.gz"), T.(permutedims(angij(S; inv=false, A=pair === :lookup ? A : nothing))))
    end
    return outdir
end

"""`build_all(eulfile, outdir; kw...)`: load `eulfile` and write all tables (see `write_tables`)."""
build_all(eulfile::AbstractString, outdir::AbstractString; kw...) =
    write_tables(load_eul(eulfile), outdir; kw...)

include("Fold.jl")

end # module
