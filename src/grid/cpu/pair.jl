# pair.jl - low-energy poses of the protein pair and their pair-wise energies.
#
#   select_poses : the SelErn step of fmapgrevst -6 (grid cells below a cutoff -> "ang ijk score" lines, fmap.log format)
#   read_poses   : reader of such a file (ZDread)
#   pair_energies: fmapdd / fmapdd.atm (direct atom-atom energies of the listed poses; per-atom sums for the atm mode)
#   residue_sums, bfactor_pdb, pose_centers_pdb : resern.py, data2bf.py, fmappdb.py -cpdb
#
# Conventions of the C code: pose = rotation (Euler ZXZ, rot * xyz) of the ligand and a shift of ijk*dx (grid index,
# periodic with period l*dx); the receptor stays at the origin. Energies use the .vdw atoms as read (no radius or charge
# scaling, as fmapdd), soft core rlow = 1 A.

# ---------------------------------------------------------------- pose files

"Header (l, dx, receptor and ligand file) and the poses of an fmap.log-style file."
struct Poses
    l::Int
    dx::Float64
    rec::String
    lig::String
    rcen::NTuple{3,Float64}
    lcen::NTuple{3,Float64}
    ang::Vector{NTuple{3,Float64}}
    ijk::Vector{NTuple{3,Int}}
    score::Vector{Float64}
end

function write_pose_header(io::IO, l::Integer, dx::Real, rec::AbstractString, lig::AbstractString)
    @printf(io, "%d\t%f\n", l, dx)
    @printf(io, "%f\t%f\t%f\n", 0.0, 0.0, 0.0)
    @printf(io, "%s\t%8.3f%8.3f%8.3f\n", rec, 0.0, 0.0, 0.0)
    @printf(io, "%s\t%8.3f%8.3f%8.3f\n", lig, 0.0, 0.0, 0.0)
end

function read_poses(file::AbstractString)
    lines = readlines(file)
    length(lines) >= 4 || error("$file: not a pose file (header missing)")
    h1 = split(lines[1]); h3 = split(lines[3]); h4 = split(lines[4])
    l = parse(Int, h1[1]); dx = parse(Float64, h1[2])
    cen(h) = (parse(Float64, h[2]), parse(Float64, h[3]), parse(Float64, h[4]))
    ang = NTuple{3,Float64}[]; ijk = NTuple{3,Int}[]; score = Float64[]
    for ln in @view lines[5:end]
        p = split(ln)
        length(p) >= 7 || continue
        push!(ang, (parse(Float64, p[1]), parse(Float64, p[2]), parse(Float64, p[3])))
        push!(ijk, (round(Int, parse(Float64, p[4])), round(Int, parse(Float64, p[5])), round(Int, parse(Float64, p[6]))))
        push!(score, parse(Float64, p[7]))
    end
    return Poses(l, dx, String(h3[1]), String(h4[1]), cen(h3), cen(h4), ang, ijk, score)
end

"""
    select_poses(io, ctx, ang, ecut)

Print the cells of the energy grid of one orientation that are not excluded and lie below `ecut` (kcal/mol),
in the order and format of the C SelErn: `ang1 ang2 ang3 i j k score`. The first array dimension of the Julia grids
is the C index k, the last is i. Methods for the CPU and GPU contexts.
"""
function select_poses end

function select_poses(io::IO, ctx::SoftCtx, ang, ecut::Real)
    l = ctx.l; comb = ctx.combined; vol = ctx.vol
    n = 0
    @inbounds for c in 1:l, b in 1:l, a in 1:l
        if !vol[a, b, c] && comb[a, b, c] < ecut
            @printf(io, "%10.6f%10.6f%10.6f%8d%8d%8d%10.3f\n", ang[1], ang[2], ang[3], c - 1, b - 1, a - 1, comb[a, b, c])
            n += 1
        end
    end
    return n
end

# ---------------------------------------------------------------- pair energies

"Atoms as structure of arrays."
struct AtomSoA
    x::Vector{Float64}; y::Vector{Float64}; z::Vector{Float64}
    q::Vector{Float64}; r::Vector{Float64}; A::Vector{Float64}; B::Vector{Float64}
    info::Vector{String}
end
AtomSoA(a::Vector{Atom}) = AtomSoA([t.x for t in a], [t.y for t in a], [t.z for t in a], [t.q for t in a],
                                   [t.r for t in a], [t.Asq for t in a], [t.Bsq for t in a], [t.info for t in a])
Base.length(a::AtomSoA) = length(a.x)

"Parameters of the pair energy (SOLV of the C code): inverse Debye length, scales, cutoffs, soft core, period."
struct PairSys
    kap::Float64
    es::Float64        # escl * 332 / sdie
    vscl::Float64
    rEup2::Float64
    rup2::Float64
    rlow2::Float64
    L::Float64         # periodic box l*dx
end

"Per-atom sums of the three terms (vol, ele, vdw), unscaled, for one thread."
mutable struct AtomAcc
    rec::Matrix{Float64}   # 3 x nrec
    lig::Matrix{Float64}
end
AtomAcc(nrec, nlig) = AtomAcc(zeros(3, nrec), zeros(3, nlig))

"Wrapped shift (grid cells) of an index, as fmappdb.py: the ligand centre is at wrap(ijk)*dx."
wrap_ijk(i::Integer, l::Integer) = i <= l ÷ 2 ? i : i - l

"""
    pose_terms(rec, lig, R, off, ps, acc)

Excluded-volume contacts (count), electrostatic and van der Waals sums (unscaled) of one pose: ligand rotated by `R`
(rows of the rotation matrix), shifted by `off` (A). If `acc` is given the pair values are added to both atoms.
"""
function pose_terms(rec::AtomSoA, lig::AtomSoA, R, off::NTuple{3,Float64}, ps::PairSys, acc::Union{Nothing,AtomAcc})
    nr = length(rec); nl = length(lig)
    L = ps.L; hL = 0.5 * L
    cut2 = max(ps.rEup2, ps.rup2)
    vol = 0.0; ele = 0.0; vdw = 0.0
    @inbounds for j in 1:nl
        lx = R[1][1] * lig.x[j] + R[1][2] * lig.y[j] + R[1][3] * lig.z[j] + off[1]
        ly = R[2][1] * lig.x[j] + R[2][2] * lig.y[j] + R[2][3] * lig.z[j] + off[2]
        lz = R[3][1] * lig.x[j] + R[3][2] * lig.y[j] + R[3][3] * lig.z[j] + off[3]
        qj = lig.q[j]; rj = lig.r[j]; Aj = lig.A[j]; Bj = lig.B[j]
        sv = 0.0; se = 0.0; sw = 0.0
        for i in 1:nr
            dx = rec.x[i] - lx; dx = dx > hL ? dx - L : (dx < -hL ? dx + L : dx)
            dy = rec.y[i] - ly; dy = dy > hL ? dy - L : (dy < -hL ? dy + L : dy)
            dz = rec.z[i] - lz; dz = dz > hL ? dz - L : (dz < -hL ? dz + L : dz)
            d2 = dx * dx + dy * dy + dz * dz
            d2 > cut2 && continue
            rs = rec.r[i] + rj
            v = d2 > rs * rs ? 0.0 : 1.0
            e = 0.0
            if d2 <= ps.rEup2
                dd = sqrt(d2 > ps.rlow2 ? d2 : ps.rlow2)
                e = rec.q[i] * qj * exp(-1.0 * dd * ps.kap) / dd
            end
            w = 0.0
            d2m = d2 > ps.rlow2 ? d2 : ps.rlow2
            if d2m <= ps.rup2
                d6 = d2m * d2m * d2m
                w = rec.A[i] * Aj / (d6 * d6) - rec.B[i] * Bj / d6
            end
            sv += v; se += e; sw += w
            if acc !== nothing
                acc.rec[1, i] += v; acc.rec[2, i] += e; acc.rec[3, i] += w
                acc.lig[1, j] += v; acc.lig[2, j] += e; acc.lig[3, j] += w
            end
        end
        vol += sv; ele += se; vdw += sw
    end
    return vol, ele, vdw
end

"Result of `pair_energies`: per pose (total, vol, ele, vdw), and the per-atom sums when `atm = true`."
struct PairResult
    ern::Matrix{Float64}            # 4 x npose: total, vol, ele*es, vdw*vscl
    rec::Union{Nothing,Matrix{Float64}}   # 4 x nrec, summed over poses: vol, ele, vdw (scaled), total
    lig::Union{Nothing,Matrix{Float64}}
end

function pair_system(params::Dict{String,Float64}, l::Integer, dx::Real)
    es = get(params, "escl", 1.0) * 332.0 / params["sdie"]
    rc = params["rcut"]
    return PairSys(params["kappa"], es, params["vscl"], rc * rc, rc * rc, 1.0, l * dx)
end

"""
    pair_energies(rec, lig, poses, ps; atm = false) -> PairResult

fmapdd (and fmapdd.atm with `atm = true`) for all poses of `poses`, threaded over the poses.
"""
function pair_energies(rec::Vector{Atom}, lig::Vector{Atom}, poses::Poses, ps::PairSys; atm::Bool = false)
    R_ = AtomSoA(rec); L_ = AtomSoA(lig)
    np = length(poses.ang)
    ern = zeros(4, np)
    nchunk = max(1, min(np, Threads.nthreads()))
    chunks = [c for c in Iterators.partition(1:np, cld(np, nchunk))]
    accs = atm ? [AtomAcc(length(R_), length(L_)) for _ in chunks] : nothing
    tasks = map(enumerate(chunks)) do (ci, rng)
        Threads.@spawn for p in rng
            a = poses.ang[p]
            R = euler_to_rotmat(a[1], a[2], a[3])
            ijk = poses.ijk[p]
            off = (wrap_ijk(ijk[1], poses.l) * poses.dx, wrap_ijk(ijk[2], poses.l) * poses.dx, wrap_ijk(ijk[3], poses.l) * poses.dx)
            v, e, w = pose_terms(R_, L_, R, off, ps, atm ? accs[ci] : nothing)
            ern[2, p] = v; ern[3, p] = e * ps.es; ern[4, p] = w * ps.vscl
            ern[1, p] = ern[3, p] + ern[4, p]
        end
    end
    foreach(wait, tasks)
    atm || return PairResult(ern, nothing, nothing)
    function fin(sel)
        s = sum(getfield(a, sel) for a in accs)          # 3 x n: vol, ele, vdw
        out = zeros(4, size(s, 2))
        out[1, :] = s[1, :]; out[2, :] = s[2, :] .* ps.es; out[3, :] = s[3, :] .* ps.vscl
        out[4, :] = out[2, :] .+ out[3, :]
        return out
    end
    return PairResult(ern, fin(:rec), fin(:lig))
end

# ---------------------------------------------------------------- C-format output

"The pose lines of fmapdd: `ang ijk score total vol ele vdw`."
function write_pair_poses(io::IO, poses::Poses, res::PairResult)
    for p in eachindex(poses.ang)
        a = poses.ang[p]; k = poses.ijk[p]
        @printf(io, "%10.6f%10.6f%10.6f%8.0f%8.0f%8.0f%10.3f%16.8e%16.8e%16.8e%16.8e\n",
                a[1], a[2], a[3], k[1], k[2], k[3], poses.score[p], res.ern[1, p], res.ern[2, p], res.ern[3, p], res.ern[4, p])
    end
end

volchk(v) = v > 0.001 ? 1.0 : 0.0

"The ATOM lines of fmapdd.atm for one molecule (values summed over the nv poses; B column = -total/nv, clipped)."
function write_atom_lines(io::IO, atoms::Vector{Atom}, ern::Matrix{Float64}, nv::Integer)
    for (i, a) in enumerate(atoms)
        bf = clamp(-ern[4, i] / nv, -9.99, 9.99)
        @printf(io, "%30s%8.3f%8.3f%8.3f%6.2f%6.2f%16.8e%16.8e%16.8e%16.8e\n",
                a.info, a.x, a.y, a.z, volchk(ern[1, i]), bf, ern[1, i], ern[2, i], ern[3, i], ern[4, i])
    end
end

# ---------------------------------------------------------------- residues (resern.py, data2bf.py, fmappdb.py)

"Residue sums of per-atom values (consecutive atoms with the same residue number and name)."
struct ResSum
    key::String         # resname, chain, resnum, insertion code (info[18:27], what data2bf matches on)
    resname::String
    resnum::Int
    vol::Float64
    ele::Float64
    vdw::Float64
    tot::Float64
    ca::Int             # index of the CA-like atom (first atom if none)
end

function residue_sums(atoms::Vector{Atom}, ern::Matrix{Float64})
    out = ResSum[]
    isempty(atoms) && return out
    favs = (" CA ", " C3 ", " C3H")
    cur = nothing; vol = ele = vdw = tot = 0.0; ca = 0; first_i = 1
    flush!() = cur === nothing || push!(out, ResSum(cur[1], cur[2], cur[3], vol, ele, vdw, tot, ca == 0 ? first_i : ca))
    for (i, a) in enumerate(atoms)
        s = rpad(a.info, 30)
        key = (s[18:27], s[18:20], something(tryparse(Int, strip(s[23:26])), 0), s[23:26] * s[18:20])
        if cur === nothing || key[4] != cur[4]
            flush!()
            cur = key; vol = ele = vdw = tot = 0.0; ca = 0; first_i = i
        end
        vol += ern[1, i]; ele += ern[2, i]; vdw += ern[3, i]; tot += ern[4, i]
        ca == 0 && s[13:16] in favs && (ca = i)
    end
    flush!()
    return out
end

"Per-residue interaction energy per pose, averaged over the two partners (resdA.txt): (key, value) pairs."
function residue_energy(rec::Vector{ResSum}, lig::Vector{ResSum}, nconf::Integer)
    length(rec) == length(lig) || error("receptor and ligand have different residue lists")
    return [(r.key, (r.tot + l.tot) / 2.0 / nconf) for (r, l) in zip(rec, lig)]
end

"B-factor PDB (data2bf.py): values scaled so that the largest |value| is 10, written into the B column of the .vdw atoms."
function bfactor_pdb(io::IO, vdwfile::AbstractString, data::Vector{<:Tuple})
    d = Dict{String,Float64}(k => v for (k, v) in data)
    vmax = maximum(abs, values(d))
    vmax == 0 && (vmax = 1.0)
    for line in eachline(vdwfile)
        startswith(line, "ATOM") || continue
        key = line[18:27]
        if haskey(d, key)
            @printf(io, "%s%6.2f%6.2f\n", line[1:54], 1.0, d[key] / vmax * 10.0)
        else
            @printf(io, "%s%6.2f%6.2f\n", line[1:54], 0.0, 0.0)
        end
    end
end

"Ligand centres of the poses as pseudo atoms (fmappdb.py -cpdb); the centre is at wrap(ijk)*dx from the receptor."
function pose_centers_pdb(io::IO, poses::Poses; first_n::Integer = typemax(Int))
    for p in 1:min(length(poses.ijk), first_n)
        k = poses.ijk[p]
        i = p - 1
        @printf(io, "ATOM  %5d  N   ALA  %4d    %8.3f%8.3f%8.3f\n", i % 100000, i % 10000,
                wrap_ijk(k[1], poses.l) * poses.dx, wrap_ijk(k[2], poses.l) * poses.dx, wrap_ijk(k[3], poses.l) * poses.dx)
    end
end
