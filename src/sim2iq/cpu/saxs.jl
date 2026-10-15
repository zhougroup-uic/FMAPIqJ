# Add source grid values to target grid with scaling
function add2grid!(source::Array{Float32,3}, target::Array{Float32,3}, scale::Float64)
    @inbounds Threads.@threads for i in eachindex(source, target)
        target[i] = Float32(Float64(target[i]) + Float64(source[i]) * scale)
    end
end

# Add a uniform value to target wherever source > 0.5
function addU2grid!(source::Array{Float32,3}, target::Array{Float32,3}, grid_spacing::Float64, scale::Float64)
    dV = grid_spacing^3
    cnt = 0
    @inbounds for i in eachindex(source, target)
        if source[i] > 0.5f0
            target[i] = Float32(Float64(target[i]) + scale * dV)
            cnt += 1
        end
    end
    @printf("Volume: %.12e\n", cnt * dV)
end

# B-factor from displacement
u2B(u) = 8.0 * π^2 * u * u

# Multi-step calculation of SAXS data:
#   1 vacuum density, 2 + excluded volume, 3 + outer shell, 4 + inner shell
function calculate_saxs_data_multi_step(structure::Structure, grid_size::Int, grid_spacing::Float64,
                                        steps::Int, cutoff_offset::Float64, se_weight::Float64,
                                        ou_weight::Float64, in_weight::Float64, b_factor::Float64)
    t0 = time()
    el() = time() - t0
    N = grid_size
    vacuum = zeros(Float32, N, N, N)
    other = steps > 1 ? zeros(Float32, N, N, N) : nothing
    work = steps > 1 ? zeros(Float32, N, N, N) : nothing

    @printf("[%.3f s] Mapping atoms to grid...\n", el())
    p = RdfParams(global_b = b_factor, rho0 = 0.334, norm_mode = NORM_ELECTRON_COUNT)
    map_atoms_to_grid!(vacuum, N, grid_spacing, structure, 0.0, p)

    if steps > 1
        @printf("[%.3f s] Mapping solvent excluded volume...\n", el())
        p.norm_mode = NORM_VOLUME
        map_atoms_to_grid!(work, N, grid_spacing, structure, 0.0, p)
        add2grid!(work, other, se_weight)
    end
    if steps > 2
        @printf("[%.3f s] Mapping outer shell...\n", el())
        p.norm_mode = NORM_NONE
        map_atoms_to_grid!(work, N, grid_spacing, structure, cutoff_offset, p)
        addU2grid!(work, other, grid_spacing, ou_weight)
    end
    if steps > 3
        @printf("[%.3f s] Mapping inner shell...\n", el())
        p.norm_mode = NORM_NONE
        map_atoms_to_grid!(work, N, grid_spacing, structure, 0.0, p)
        addU2grid!(work, other, grid_spacing, in_weight)
    end
    work = nothing

    @printf("[%.3f s] Calculating FFTs...\n", el())
    qv = create_q_vectors(N, grid_spacing)
    fft_vac = ComplexF32.(vacuum)
    vacuum = nothing
    calculate_fft!(fft_vac)
    fft_other = nothing
    if steps > 1
        fft_other = ComplexF32.(other)
        other = nothing
        calculate_fft!(fft_other)
    end

    @printf("[%.3f s] Binning intensity...\n", el())
    bins = create_bin_data(qv)
    accumulate_intensity!(bins, fft_vac, fft_other, qv, b_factor)
    result = create_final_data(bins)
    @printf("[%.3f s] SAXS calculation complete.\n", el())
    return result
end
