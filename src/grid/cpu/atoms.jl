# atoms.jl - records shared by the grid generator (ext/common.h ATOM/SOLV, tools/fmap.py).

"One .vdw atom: position (Angstrom, or grid units after scaling), charge, radius, sqrt(A), sqrt(B)."
struct Atom
    info::String
    x::Float64
    y::Float64
    z::Float64
    q::Float64
    r::Float64
    Asq::Float64
    Bsq::Float64
end

"Solvent / cutoff parameters (SOLV)."
struct Solv
    Cut::Float64
    VLow::Float64
    kappa::Float64
    sdie::Float64
    kBT::Float64
    escl::Float64
    vscl::Float64
end

"Cell format of the generated grid, as the C build's -DTERN_BYTES: 2 = [elec,vdw] per cell, 1 = pre-summed."
function tern_bytes_from_env()
    tb = parse(Int, get(ENV, "TERN_BYTES", "2"))
    tb in (1, 2) || error("TERN_BYTES must be 1 or 2, got $tb")
    return tb
end

"Directory with lattice.txt and the rotation tables (sspNF<nrot>/)."
function data_dir()
    d = get(ENV, "FMAPB2_DATA_PATH", "")
    isempty(d) || return d
    return normpath(joinpath(@__DIR__, "..", "..", "..", "data"))
end

"""
    read_parameters(file) -> Dict

The numbers fmap.py needs from parms.txt (`key : value [unit]` lines).
"""
function read_parameters(filename::AbstractString)
    isfile(filename) || (println(stderr, "Error: Parameter file $filename not found"); exit(1))
    needed = ("kbt", "qscl", "sdie", "kappa", "rscl", "escl", "vscl", "rcut", "spacing", "blen")
    params = Dict{String,Float64}()
    for line in eachline(filename)
        line = strip(line)
        (occursin(':', line) && !startswith(line, '#')) || continue
        key, value = split(line, ':'; limit = 2)
        key = strip(key)
        key in needed || continue
        toks = split(strip(value))
        isempty(toks) && continue
        v = tryparse(Float64, toks[1])
        v === nothing || (params[key] = v)
    end
    return params
end

"Read a .vdw (extended PQR) file: x y z q r at fixed columns, sqrt(A) sqrt(B) from column 103."
function read_vdw(filename::AbstractString)
    isfile(filename) || (println(stderr, "Error: File $filename not found"); exit(1))
    atoms = Atom[]
    for line in eachline(filename)
        (startswith(line, "ATOM") || startswith(line, "HETATM")) || continue
        try
            info = line[1:min(30, lastindex(line))]
            x = parse(Float64, line[31:38]); y = parse(Float64, line[39:46]); z = parse(Float64, line[47:54])
            q = parse(Float64, line[55:62]); r = parse(Float64, line[63:70])
            s = 70 + 8 + 16 + 16               # python: line[start:start+16] after 70+8+16+16 chars
            Asq = parse(Float64, line[s+1:s+16]); Bsq = parse(Float64, line[s+17:s+32])
            push!(atoms, Atom(info, x, y, z, q, r, Asq, Bsq))
        catch
            println(stderr, "Warning: Could not parse line: ", strip(line))
        end
    end
    return atoms
end

function read_angles(filename::AbstractString)
    isfile(filename) || (println(stderr, "Error: File $filename not found"); exit(1))
    angles = NTuple{3,Float64}[]
    for line in eachline(filename)
        p = split(line)
        length(p) >= 3 && push!(angles, (parse(Float64, p[1]), parse(Float64, p[2]), parse(Float64, p[3])))
    end
    return angles
end

"ZXZ Euler angles -> rotation matrix (tools/fmap.py euler_to_rotmat)."
function euler_to_rotmat(psi, theta, phi)
    cp, sp = cos(psi), sin(psi)
    ct, st = cos(theta), sin(theta)
    cf, sf = cos(phi), sin(phi)
    return ((cp * cf - sp * ct * sf, -cp * sf - sp * ct * cf, sp * st),
            (sp * cf + cp * ct * sf, -sp * sf + cp * ct * cf, -cp * st),
            (st * sf, st * cf, ct))
end

function rotate_atoms(atoms::Vector{Atom}, R)
    return [Atom(a.info,
                 R[1][1] * a.x + R[1][2] * a.y + R[1][3] * a.z,
                 R[2][1] * a.x + R[2][2] * a.y + R[2][3] * a.z,
                 R[3][1] * a.x + R[3][2] * a.y + R[3][3] * a.z,
                 a.q, a.r, a.Asq, a.Bsq) for a in atoms]
end

scale_coordinates(atoms::Vector{Atom}, dx) =
    [Atom(a.info, a.x / dx, a.y / dx, a.z / dx, a.q, a.r, a.Asq, a.Bsq) for a in atoms]

"libm pow(): what python's `**` / math.pow call (Julia's `^` can differ in the last bit)."
libm_pow(x::Float64, y::Float64) = ccall(:pow, Float64, (Float64, Float64), x, y)
