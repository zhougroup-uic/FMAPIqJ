sphere_radius_from_volume(volume) = cbrt(3.0 * volume / (4π))

# Map all atoms onto `grid` (N×N×N Float32, periodic) following grid_mapping.c.
# Each atom is spread over a small cube around it, restricted to r² <= cutoff²,
# then normalised according to rdf_params.norm_mode.
function map_atoms_to_grid!(grid::Array{Float32,3}, grid_size::Int, grid_spacing::Float64,
                            structure::Structure, cutoff_offset::Float64, p::RdfParams)
    fill!(grid, 0.0f0)
    N = grid_size
    mode = p.norm_mode
    maxpts = 0
    idx = Vector{Int}(undef, 0)
    r2v = Vector{Float64}(undef, 0)
    con = Vector{Float64}(undef, 0)
    for atom in structure
        etype = get_element_type(atom.element)
        cm = get_cromer_mann_params(etype)
        # Unknown elements: C would dereference NULL for volume/shell modes; skip them here
        if cm === nothing
            mode == NORM_ELECTRON_COUNT || mode == NORM_DEFAULT || continue
        end
        cutoff = if mode == NORM_ELECTRON_COUNT || mode == NORM_DEFAULT
            3.0
        elseif mode == NORM_VOLUME
            2.0 * cm.vdw_radius
        else
            cutoff_offset + cm.vdw_radius
        end
        small = 2 * ceil(Int, cutoff / grid_spacing) + 1
        if small^3 > maxpts
            maxpts = small^3
            resize!(idx, maxpts); resize!(r2v, maxpts); resize!(con, maxpts)
        end
        ax = floor(Int, atom.x / grid_spacing)
        ay = floor(Int, atom.y / grid_spacing)
        az = floor(Int, atom.z / grid_spacing)
        ox = ax - small ÷ 2; oy = ay - small ÷ 2; oz = az - small ÷ 2
        relx = atom.x / grid_spacing - ox
        rely = atom.y / grid_spacing - oy
        relz = atom.z / grid_spacing - oz
        cut2 = cutoff * cutoff
        n = 0
        @inbounds for x in 0:small-1
            dx = (x - relx) * grid_spacing
            fx = mod(ox + x, N)
            for y in 0:small-1
                dy = (y - rely) * grid_spacing
                fy = mod(oy + y, N)
                for z in 0:small-1
                    dz = (z - relz) * grid_spacing
                    r2 = dx * dx + dy * dy + dz * dz
                    if r2 <= cut2
                        n += 1
                        fz = mod(oz + z, N)
                        idx[n] = 1 + fx + fy * N + fz * N * N
                        r2v[n] = r2
                    end
                end
            end
        end
        n == 0 && continue
        total = 0.0
        @inbounds for j in 1:n
            c = if mode == NORM_VOLUME
                simple_gauss_rdf(r2v[j], atom.vol, p.rho0)
            elseif mode == NORM_NONE
                1.0
            else
                multigauss_rdf(r2v[j], cm, p.global_b)
            end
            con[j] = c
            total += c
        end
        total > 0.0 || continue
        nf = if mode == NORM_ELECTRON_COUNT
            (cm !== nothing && cm.electrons > 0) ? cm.electrons / total : 1.0 / total
        elseif mode == NORM_VOLUME
            atom.vol * p.rho0 / total
        elseif mode == NORM_DEFAULT
            1.0 / total
        else
            1.0
        end
        @inbounds for j in 1:n
            c = con[j] * nf
            c > 0.0 && (grid[idx[j]] += Float32(c))
        end
    end
    return grid
end
