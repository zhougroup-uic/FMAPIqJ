# pre.jl - fmappre: PQR -> parms.txt + sub{A,B}.vdw (tools/fmappre.py).

const VDW_DATA = Dict(
    "H" => (1.78179743628, 0.0200), "O" => (2.85087589805, 0.2000), "C" => (3.56359487256, 0.1500),
    "N" => (3.11814551349, 0.1600), "S" => (3.56359487256, 0.2000), "P" => (3.74177461619, 0.2),
    "IM" => (4.40103966761, 0.1), "Li" => (2.02590368505, 0.0183), "IP" => (3.32839761097, 0.00277),
    "Na" => (3.32839761097, 0.00277), "K" => (4.73601758563, 0.000328), "Rb" => (5.26699322165, 0.00017),
    "Cs" => (6.04920229617, 8.06e-05), "MG" => (1.412252648, 0.8947), "C0" => (3.05239718809, 0.459789),
    "Zn" => (1.95997717991, 0.0125), "F" => (3.11814551349, 0.061), "Cl" => (3.47094140587, 0.265),
    "Br" => (3.95559030854, 0.32), "I" => (4.18722397526, 0.4), "IB" => (8.9089871814, 0.1))

const ELEM_WEIGHTS = Dict('C' => 12.01, 'N' => 14.01, 'O' => 16.00, 'P' => 30.97, 'S' => 32.06, 'H' => 1.01)

function get_kappa(ionic_strength, sdie, temp)
    N_A = 6.022045000e+23; e_c = 4.803242384e-10; k_B = 1.380662000e-16
    return sqrt(ionic_strength * 1.0e-16 * (8.0 * pi * N_A * e_c * e_c) / (1000.0 * sdie * k_B * temp))
end

get_kbt(temp) = (N_A = 6.022045000e+23; k_B = 3.297623030e-24; N_A * k_B * temp / 1000.0)

"Malmberg & Maryott (1956) dielectric constant of water, temp in Kelvin."
function get_water_die(temp)
    t = temp - 273.15
    return 87.740 - 0.40008 * t + 9.398e-4 * t * t - 1.410e-6 * t * t * t
end

is_atom_line(line) = startswith(line, "ATOM") || startswith(line, "HETATM")
get_xyz(line) = (parse(Float64, line[31:38]), parse(Float64, line[39:46]), parse(Float64, line[47:54]))

function get_maxrad(fname)
    rad = 0.0
    for line in eachline(fname)
        is_atom_line(line) || continue
        x, y, z = get_xyz(line)
        rad = max(rad, x * x + y * y + z * z)
    end
    return sqrt(rad)
end

function get_mass(fname)
    mass = 0.0
    for line in eachline(fname)
        is_atom_line(line) || continue
        elem = line[13] == 'H' ? 'H' : line[14]
        haskey(ELEM_WEIGHTS, elem) && (mass += ELEM_WEIGHTS[elem])
    end
    return mass
end

get_charge(fname) = sum((parse(Float64, line[55:62]) for line in eachline(fname) if is_atom_line(line)); init = 0.0)

function get_max_atom_radius(vdw_fname)
    rad = 0.0
    for line in eachline(vdw_fname)
        startswith(line, "ATOM") && (rad = max(rad, parse(Float64, line[63:70])))
    end
    return rad
end

calculate_vscl(mass) = round(0.18 * (80 + 0.27 * mass / 1000.0) / (80 + mass / 1000.0); digits = 3)

"Write the centred .vdw file (LJ parameters from the atom name); returns nothing."
function vdw_assign_center(pdb_fname, vdw_fname)
    lines = String[]; coords = NTuple{3,Float64}[]
    for line in eachline(pdb_fname)
        if is_atom_line(line)
            push!(lines, line); push!(coords, get_xyz(line))
        end
    end
    n = length(coords)
    cen = (sum(c[1] for c in coords) / n, sum(c[2] for c in coords) / n, sum(c[3] for c in coords) / n)
    open(vdw_fname, "w") do io
        for line in lines
            # Keep the atom name as is (fmappre.py 384a910: "keep H?\d\d as is"): the old code overwrote the 2nd
            # character of H names (HG11 -> HH11), which made those H unknown to sim2iq_pre.
            info = line[1:30]
            key = line[13] == 'H' ? "H" : string(info[14])
            haskey(VDW_DATA, key) || error("no VDW parameters for atom type '$key' in line: $line")
            sig, eps = VDW_DATA[key]
            sig6 = libm_pow(sig, 6.0)
            sqA = sqrt(4.0 * eps * sig6 * sig6)
            sqB = sqrt(4.0 * eps * sig6)
            x, y, z = get_xyz(line)
            write(io, info, @sprintf("%8.3f%8.3f%8.3f", x - cen[1], y - cen[2], z - cen[3]), line[55:62],
                  @sprintf("%8.4f%8s%16.11f%16.11f%16.8e%16.8e\n", sig / 2.0, key, sig, eps, sqA, sqB))
        end
    end
end

"""
    fmappre(pqr_a, ion, temp_c, escl, vscl; pqr_b = nothing, out = stdout)

parms.txt content on `out`; also writes subA.vdw (and subB.vdw) into the current directory.
`escl` and `vscl` as given on the command line ("-" for the defaults / auto-vscl).
"""
function fmappre(pqr_a::AbstractString, ion::Float64, temp_c::Float64, escl_arg::AbstractString, vscl_arg::AbstractString;
                 pqr_b::Union{Nothing,AbstractString} = nothing, out::IO = stdout)
    temp_k = temp_c + 273.15
    escl = escl_arg == "-" ? 1.0 : parse(Float64, escl_arg)
    charge_a = get_charge(pqr_a); mass_a = get_mass(pqr_a)
    charge_b = mass_b = 0.0
    mass = mass_a
    if pqr_b !== nothing
        charge_b = get_charge(pqr_b); mass_b = get_mass(pqr_b)
        mass = sqrt(mass_a * mass_b)
    end
    vscl = vscl_arg == "-" ? calculate_vscl(mass) : parse(Float64, vscl_arg)

    sdie = get_water_die(temp_k)
    kappa = get_kappa(ion, sdie, temp_k)
    kbt = get_kbt(temp_k)
    qscl = 1.0 + 0.025 * libm_pow(ion, -0.4)

    p(k, fmt, args...) = write(out, k, " : ", Printf.format(Printf.Format(fmt), args...))
    row(name, fmt, args...) = write(out, rpad(name, 12), " : ", Printf.format(Printf.Format(fmt), args...))
    row("Temperature", "%16.8E K\n", temp_k)
    row("kbt", "%16.8E\n", kbt)
    row("ion", "%16.8E M\n", ion)
    row("qscl", "%16.8E\n", qscl)
    row("sdie", "%16.8E\n", sdie)
    row("kappa", "%16.8E\n", kappa)
    row("lambda", "%16.8E angstrom\n", 1.0 / kappa)
    row("rscl", "%16.3f\n", 1.08)
    row("escl", "%16.3f\n", escl)
    row("vscl", "%16.3f\n", vscl)
    row("ChargeA", "%16.0f\n", charge_a)
    row("MassA", "%16.3f Da\n", mass_a)
    vdw_assign_center(pqr_a, "subA.vdw")
    max_rad_a = get_maxrad("subA.vdw")
    row("MaxRA", "%16.3f angstrom\n", max_rad_a)
    local max_r, atom_r
    if pqr_b !== nothing
        row("ChargeB", "%16.0f\n", charge_b)
        row("MassB", "%16.3f Da\n", mass_b)
        vdw_assign_center(pqr_b, "subB.vdw")
        max_rad_b = get_maxrad("subB.vdw")
        row("MaxRB", "%16.3f angstrom\n", max_rad_b)
        max_r = (max_rad_a + max_rad_b) / 2.0
        atom_r = max(get_max_atom_radius("subA.vdw"), get_max_atom_radius("subB.vdw"))
    else
        max_r = max_rad_a
        atom_r = get_max_atom_radius("subA.vdw")
    end
    row("mass", "%16.3f Da\n", mass)
    row("MaxR", "%16.3f angstrom\n", max_r)
    rcut = trunc(Int, 1.0 / kappa * 3.0)
    rcut < 36 && (rcut = 36)
    row("rcut", "%16.3f angstrom\n", Float64(rcut))
    spacing = 0.6
    blen = 2 * (trunc(Int, (2 * (max_r + atom_r) + rcut) / spacing) + 1)
    row("spacing", "%16.1f\n", spacing)
    row("blen", "%16.0f\n", Float64(blen))
    return nothing
end
