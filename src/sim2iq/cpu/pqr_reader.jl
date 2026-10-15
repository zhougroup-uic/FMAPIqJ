struct Atom
    element::String
    x::Float64
    y::Float64
    z::Float64
    vol::Float64
end

const Structure = Vector{Atom}

const MAX_ELEMENT_LENGTH = 8

# Element string -> element type (first letter, case-insensitive)
function get_element_type(element::AbstractString)
    s = lstrip(element)
    isempty(s) && return ELEMENT_UNKNOWN
    c = uppercase(s[1])
    c == 'H' && return ELEMENT_H
    c == 'C' && return ELEMENT_C
    c == 'N' && return ELEMENT_N
    c == 'O' && return ELEMENT_O
    c == 'P' && return ELEMENT_P
    c == 'S' && return ELEMENT_S
    return ELEMENT_UNKNOWN
end

# Emulates C strtof on a short fixed-width field, returning a Float32 (value as Float64)
function strtof(s::AbstractString)
    m = match(r"^\s*[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?", s)
    m === nothing && return 0.0
    return Float64(parse(Float32, strip(m.match)))
end

# Read atomic structure from a PQR file (fixed columns, as in pqr_reader.c)
function read_pqr_file(filename::AbstractString)
    isfile(filename) || (println(stderr, "Error opening file: ", filename); exit(1))
    structure = Atom[]
    open(filename, "r") do io
        while !eof(io)
            line = readline(io; keep = true)
            (startswith(line, "ATOM") || startswith(line, "HETATM")) || continue
            n = ncodeunits(line)
            element = ""
            if n >= 16
                element = String(strip(line[13:16]))
                length(element) >= MAX_ELEMENT_LENGTH && (element = element[1:MAX_ELEMENT_LENGTH-1])
            end
            if n >= 54
                x = strtof(line[31:38])
                y = strtof(line[39:46])
                z = strtof(line[47:54])
            else
                print(stderr, "Warning: Line too short for coordinates: ", line)
                continue
            end
            radius = n >= 70 ? strtof(line[63:70]) : 0.0
            cm = get_cromer_mann_params(get_element_type(element))
            vol = if radius > 0.0 && cm !== nothing
                (4.0 / 3.0) * π * (radius * cm.scl)^3
            elseif cm !== nothing
                (4.0 / 3.0) * π * (cm.vdw_radius * cm.scl)^3
            else
                (4.0 / 3.0) * π * 1.7 * 1.7 * 1.7
            end
            push!(structure, Atom(element, x, y, z, vol))
        end
    end
    @printf("Read %d atoms from %s\n", length(structure), filename)
    return structure
end
