# PD6D trajectory handling (pd6d.c): per-protein translation + Euler angles.
const MAXLENLINE = 255

# Emulates sscanf("%16d%16lf%16lf...") : each field skips whitespace and reads at most 16 chars
function _scan_fields(s::AbstractString, nf::Int)
    vals = Float64[]
    i = 1
    n = ncodeunits(s)
    while length(vals) < nf
        while i <= n && isspace(s[i]); i += 1; end
        i > n && break
        j = min(i + 15, n)
        m = match(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?", SubString(s, i, j))
        m === nothing && break
        push!(vals, parse(Float64, m.match))
        i += ncodeunits(m.match)
    end
    return vals
end

# Returns (id, tr, xr)
function pd6d_readline(line::AbstractString, txt::Bool)
    tr = zeros(3); xr = zeros(3); id = 0
    if txt
        v = _scan_fields(line, 7)
        length(v) >= 1 && (id = Int(v[1]))
        for k in 1:3
            length(v) >= 1 + k && (tr[k] = v[1+k])
            length(v) >= 4 + k && (xr[k] = v[4+k])
        end
    else
        # PDB-like: id at cols 7-11, six %8f fields from col 31
        id = something(tryparse(Int, strip(line[7:min(11, end)])), 0)
        for k in 1:6
            a = 31 + 8 * (k - 1)
            a > ncodeunits(line) && break
            f = something(tryparse(Float64, strip(line[a:min(a + 7, end)])), 0.0)
            k <= 3 ? (tr[k] = f) : (xr[k-3] = f)
        end
    end
    return id, tr, xr
end

_is_record(line, txt) = txt ? !(startswith(line, "MODEL") || startswith(line, "ENDMDL") || startswith(line, "#")) :
                              (startswith(line, "ATOM") || startswith(line, "HETATM"))

# Read the first model: returns vectors of translations and Euler angles
function read_pd6d(filename::AbstractString, txt::Bool)
    isfile(filename) || exit(1)
    trs = Vector{Float64}[]; xrs = Vector{Float64}[]
    preid = 0
    for line in eachline(filename)
        _is_record(line, txt) || continue
        id, tr, xr = pd6d_readline(line, txt)
        id > preid || break     # first model only (PD6DCountAtoms)
        preid = id
        push!(trs, tr); push!(xrs, xr)
    end
    return trs, xrs
end

# Rotation matrix (row-major 3x3 as in C) from Euler angles
function euler2rot(psi, theta, phi)
    r11 = cos(psi) * cos(phi) - sin(psi) * cos(theta) * sin(phi)
    r21 = sin(psi) * cos(phi) + cos(psi) * cos(theta) * sin(phi)
    r31 = sin(theta) * sin(phi)
    r12 = -cos(psi) * sin(phi) - sin(psi) * cos(theta) * cos(phi)
    r22 = -sin(psi) * sin(phi) + cos(psi) * cos(theta) * cos(phi)
    r32 = sin(theta) * cos(phi)
    r13 = sin(psi) * sin(theta)
    r23 = -cos(psi) * sin(theta)
    r33 = cos(theta)
    return (r11, r12, r13, r21, r22, r23, r31, r32, r33)
end

# Replicate `original` once per transformation (main.c: process_trajectory)
function process_trajectory(original::Structure, traj_file::AbstractString, is_txt::Bool)
    trs, xrs = read_pd6d(traj_file, is_txt)
    nprot = length(trs)
    @printf("Successfully read %d proteins from trajectory file\n", nprot)
    n0 = length(original)
    out = Vector{Atom}(undef, n0 * nprot)
    Threads.@threads for t in 1:nprot
        tr = trs[t]; xr = xrs[t]
        r = euler2rot(xr[1], xr[2], xr[3])
        off = (t - 1) * n0
        @inbounds for i in 1:n0
            a = original[i]
            out[off+i] = Atom(a.element,
                tr[1] + a.x * r[1] + a.y * r[2] + a.z * r[3],
                tr[2] + a.x * r[4] + a.y * r[5] + a.z * r[6],
                tr[3] + a.x * r[7] + a.y * r[8] + a.z * r[9],
                a.vol)
        end
    end
    @printf("Trajectory processing complete - created structure with %d atoms\n", length(out))
    return out
end
