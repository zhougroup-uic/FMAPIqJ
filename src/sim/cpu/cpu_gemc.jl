# cpu_gemc.jl - sequential CPU GEMC engine, a line-by-line port of src/cpu/simulation_gemc.c
# (single-threaded build): same RNG draw order, same slot-compact bookkeeping, so a run with the
# same seed and arguments follows the C engine's trajectory.
#
# Slots are 1-based here; [1,n_particles] are active. Removal shifts later slots down
# (order-preserving), insertion appends.

mutable struct GemcBox
    max_capacity::Int
    n_particles::Int
    box_len::NTuple{3,Float64}
    temperature::Float64

    particles::Vector{Particle}        # max_capacity slots; types of unused slots are pre-assigned
    particle_energies::Vector{Float64}
    energy_matrix::Matrix{Float64}     # [i,j] pair energy for slot order min(i,j) < max(i,j)
    total_energy::Float64

    ctxs::Vector{FmapCtx}              # n_types*(n_types+1)/2 pair contexts
    n_types::Int
    n_type_a_active::Int
    nbin::Int

    max_translation_param::Float64
    max_rotation_param::Float64

    n_moves::Int
    n_accepted::Int
    n_transfer_attempts::Int
    n_transfer_accepted::Int

    target_ratio::Float64
    js::Float64
    rs::Float64
    ntry::Int
    naccept::Int
    nrtry::Int
    naccept_rot::Int
    js_max::Float64
    rs_max::Float64

    rng::LCG
    trial_energies::Vector{Float64}
end

# ---- type-aware energy ----

@inline function get_ctx(box::GemcBox, t1::Integer, t2::Integer)
    idx = ctx_index(t1, t2, box.n_types)
    if idx < 0 || idx >= length(box.ctxs)
        error("get_ctx index out of range (t1=$t1 t2=$t2 n_types=$(box.n_types) idx=$idx n_ctxs=$(length(box.ctxs)))")
    end
    return @inbounds box.ctxs[idx+1]
end

@inline function particle_energy(c::FmapCtx, box_len::NTuple{3,Float64}, p1::Particle, p2::Particle)
    dx = apply_pbc(p1.x - p2.x, box_len[1])
    dy = apply_pbc(p1.y - p2.y, box_len[2])
    dz = apply_pbc(p1.z - p2.z, box_len[3])
    return fmap_pair_energy(c, dx, dy, dz, p1.rot_idx, p2.rot_idx)
end

@inline function pair_energy_idx(box::GemcBox, i::Int, j::Int)
    pa = box.particles[i < j ? i : j]
    pb = box.particles[i < j ? j : i]
    return particle_energy(get_ctx(box, pa.type, pb.type), box.box_len, pa, pb)
end

# ---- particle moves (particle_fmap.c) ----

function random_particle(box::GemcBox, type::Integer, box_len::NTuple{3,Float64}, rng::LCG, c::FmapCtx)
    nbin = c.nbin
    x = lcg_random!(rng) * box_len[1]; b1 = lcg_random!(rng) * nbin
    y = lcg_random!(rng) * box_len[2]; b2 = lcg_random!(rng) * nbin
    z = lcg_random!(rng) * box_len[3]; b3 = lcg_random!(rng) * nbin
    return Particle(x, y, z, b1, b2, b3, bins_to_rot_idx(c, b1, b2, b3), Int32(type))
end

function trial_move(p::Particle, max_trans::Float64, max_rot::Float64,
                    box_len::NTuple{3,Float64}, rng::LCG, c::FmapCtx)
    x = p.x; y = p.y; z = p.z; b1 = p.b1; b2 = p.b2; b3 = p.b3
    # x += (r-0.5)*max_trans and b += ((r-0.5)*max_rot)*3.0 as fused multiply-adds (gcc -march=native)
    x = muladd(lcg_random!(rng) - 0.5, max_trans, x); b1 = muladd((lcg_random!(rng) - 0.5) * max_rot, ROT_DEFAULT, b1)
    y = muladd(lcg_random!(rng) - 0.5, max_trans, y); b2 = muladd((lcg_random!(rng) - 0.5) * max_rot, ROT_DEFAULT, b2)
    z = muladd(lcg_random!(rng) - 0.5, max_trans, z); b3 = muladd((lcg_random!(rng) - 0.5) * max_rot, ROT_DEFAULT, b3)
    x = rem(x, box_len[1]); x < 0 && (x += box_len[1])
    y = rem(y, box_len[2]); y < 0 && (y += box_len[2])
    z = rem(z, box_len[3]); z < 0 && (z += box_len[3])
    nb = c.nbin
    b1 = wrap_bin(b1, nb); b2 = wrap_bin(b2, nb); b3 = wrap_bin(b3, nb)
    return Particle(x, y, z, b1, b2, b3, bins_to_rot_idx(c, b1, b2, b3), p.type)
end

# ---- energy matrix ----

function calculate_all_energies!(box::GemcBox)
    fill!(box.particle_energies, 0.0)
    fill!(box.energy_matrix, 0.0)
    n = box.n_particles
    for i in 1:n, j in i+1:n
        e = pair_energy_idx(box, i, j)
        box.energy_matrix[i, j] = e
        box.energy_matrix[j, i] = e
        box.particle_energies[i] += e
        box.particle_energies[j] += e
    end
end

function calculate_initial_energy!(box::GemcBox)
    t = 0.0
    for i in 1:box.n_particles, j in i+1:box.n_particles
        t += box.energy_matrix[i, j]
    end
    box.total_energy = t
end

# ---- active particle selection ----

function get_random_active(box::GemcBox)
    box.n_particles == 0 && return 0
    target = unsafe_trunc(Int, lcg_random!(box.rng) * box.n_particles)
    target >= box.n_particles && (target = box.n_particles - 1)
    return target + 1                       # 1-based slot
end

find_empty_slot(box::GemcBox) = box.n_particles < box.max_capacity ? box.n_particles + 1 : 0

# Remove active slot idx: later slots shift down by one (particles, energies, matrix rows and
# columns); the removed particle is parked at slot n.
function remove_slot!(box::GemcBox, idx::Int)
    n = box.n_particles
    M = box.energy_matrix
    pp = box.particles[idx]
    for j in idx:n-1
        box.particles[j] = box.particles[j+1]
        box.particle_energies[j] = box.particle_energies[j+1]
    end
    box.particles[n] = pp
    box.particle_energies[n] = 0.0
    # rows shift up, then every row's columns shift left
    for r in idx:n-1, c in 1:size(M, 2)
        M[r, c] = M[r+1, c]
    end
    for c in 1:size(M, 2); M[n, c] = 0.0; end
    for r in 1:n
        for c in idx:n-1
            M[r, c] = M[r, c+1]
        end
        M[r, n] = 0.0
    end
    box.n_particles = n - 1
end

# ---- initialization ----

function place_particles_random!(box::GemcBox, target::Int)
    placed = box.n_particles
    i = box.n_particles                     # 0-based slot, as in C
    while i < box.max_capacity && placed < target
        ok = false
        for _ in 1:1000
            t = box.particles[i+1].type
            c0 = get_ctx(box, t, t)
            box.particles[i+1] = random_particle(box, t, box.box_len, box.rng, c0)
            ins = 0.0
            for j in 1:i
                e = pair_energy_idx(box, i + 1, j)
                ins += e
                box.energy_matrix[i+1, j] = e
                box.energy_matrix[j, i+1] = e
            end
            if ins < 999.0
                box.particle_energies[i+1] = ins
                for j in 1:i
                    box.particle_energies[j] += box.energy_matrix[i+1, j]
                end
                box.particles[i+1].type == 0 && (box.n_type_a_active += 1)
                placed += 1
                box.n_particles = placed
                ok = true
                break
            end
        end
        if !ok
            @printf("Warning: failed to place particle at slot %d (placed %d/%d)\n", i, placed, target)
            break
        end
        half = target ÷ 2 > 0 ? target ÷ 2 : 1
        (placed % half == 0 || placed == target) && @printf("Placed %d/%d\n", placed, target)
        i += 1
    end
    return 1
end

# ---- displacement MC ----

function calc_delta!(box::GemcBox, pid::Int, trial::Particle)
    new_e = 0.0
    for j in 1:box.n_particles
        box.trial_energies[j] = 0.0
        j == pid && continue
        pj = box.particles[j]
        c = get_ctx(box, trial.type, pj.type)
        e = pid < j ? particle_energy(c, box.box_len, trial, pj) : particle_energy(c, box.box_len, pj, trial)
        box.trial_energies[j] = e
        new_e += e
    end
    return new_e - box.particle_energies[pid]
end

function accept_displacement!(box::GemcBox, pid::Int, trial::Particle, delta::Float64)
    box.particles[pid] = trial
    new_e = 0.0
    for j in 1:box.n_particles
        j == pid && continue
        ne = box.trial_energies[j]
        oe = box.energy_matrix[pid, j]
        box.energy_matrix[pid, j] = ne
        box.energy_matrix[j, pid] = ne
        box.particle_energies[j] += (ne - oe)
        new_e += ne
    end
    box.particle_energies[pid] = new_e
    box.total_energy += delta
end

function displacement_step!(box::GemcBox)
    pid = get_random_active(box)
    pid < 1 && return
    is_trans = lcg_random!(box.rng) < 0.5
    t = box.particles[pid].type
    c = get_ctx(box, t, t)
    trial = trial_move(box.particles[pid], is_trans ? box.js : 0.0, is_trans ? 0.0 : box.rs,
                       box.box_len, box.rng, c)
    delta = calc_delta!(box, pid, trial)
    accept = delta <= 0.0 || lcg_random!(box.rng) < exp(-delta / box.temperature)
    if accept
        accept_displacement!(box, pid, trial, delta)
        box.n_accepted += 1
    end
    box.n_moves += 1
    if is_trans
        box.ntry += 1
        accept && (box.naccept += 1)
    else
        box.nrtry += 1
        accept && (box.naccept_rot += 1)
    end
end

"Per-move adaptive step sizes (sim_gemc_adjust_step_sizes)."
function adjust_step_sizes!(box::GemcBox)
    if box.ntry > 0
        ratio = box.naccept / box.ntry
        if ratio < box.target_ratio
            box.js > box.max_translation_param && (box.js *= 0.99)
        else
            box.js < box.js_max && (box.js *= 1.01)
        end
        box.ntry = 0; box.naccept = 0
    end
    if box.nrtry > 0
        ratio = box.naccept_rot / box.nrtry
        if ratio < box.target_ratio
            box.rs > box.max_rotation_param && (box.rs *= 0.99)
        else
            box.rs < box.rs_max && (box.rs *= 1.01)
        end
        box.nrtry = 0; box.naccept_rot = 0
    end
end

# ---- GEMC transfer ----

function gemc_transfer!(from::GemcBox, to::GemcBox)
    (from.n_particles == 0 || to.n_particles >= to.max_capacity) && return
    idx_from = get_random_active(from)
    idx_from < 1 && return
    idx_to = find_empty_slot(to)
    idx_to < 1 && return

    energy_old = from.particle_energies[idx_from]
    ptype = from.particles[idx_from].type

    c = get_ctx(to, ptype, ptype)
    to.particles[idx_to] = random_particle(to, ptype, to.box_len, to.rng, c)

    energy_new = 0.0
    for j in 1:idx_to-1
        energy_new += pair_energy_idx(to, idx_to, j)
    end

    vol_from = from.box_len[1] * from.box_len[2] * from.box_len[3]
    vol_to = to.box_len[1] * to.box_len[2] * to.box_len[3]
    beta = 1.0 / from.temperature
    ln_acc = muladd(-beta, energy_new - energy_old,
                    log(vol_to / vol_from) + log(Float64(from.n_particles) / Float64(to.n_particles + 1)))
    accept = ln_acc >= 0.0 || lcg_random!(from.rng) < exp(ln_acc)

    if accept
        for j in 1:from.n_particles
            j == idx_from && continue
            from.particle_energies[j] -= from.energy_matrix[idx_from, j]
        end
        from.total_energy -= energy_old
        remove_slot!(from, idx_from)
        if ptype == 0
            from.n_type_a_active -= 1; to.n_type_a_active += 1
        end

        to.particle_energies[idx_to] = 0.0
        for j in 1:to.max_capacity
            if j >= idx_to
                to.energy_matrix[idx_to, j] = 0.0
                to.energy_matrix[j, idx_to] = 0.0
                continue
            end
            e = pair_energy_idx(to, idx_to, j)
            to.energy_matrix[idx_to, j] = e
            to.energy_matrix[j, idx_to] = e
            to.particle_energies[idx_to] += e
            to.particle_energies[j] += e
        end
        to.total_energy += to.particle_energies[idx_to]
        to.n_particles += 1

        from.n_transfer_accepted += 1
        to.n_transfer_accepted += 1
    end
    from.n_transfer_attempts += 1
    to.n_transfer_attempts += 1
end

# ---- NVT same-box insertion (replaces the inter-box transfer for a single box) ----
#
# Same bookkeeping as gemc_transfer!, but source and target are the same box: a random active
# particle is removed and re-inserted at a uniformly random position/orientation (appended at the
# highest slot, like a transfer into a box). N and V are unchanged, so the acceptance is
# exp(-beta*(E_insert - E_removal)). Returns true if accepted.
function nvt_insert!(box::GemcBox)
    n = box.n_particles
    n == 0 && return false
    box.n_transfer_attempts += 1               # counted like a transfer in the log
    idx = get_random_active(box)
    energy_old = box.particle_energies[idx]
    ptype = box.particles[idx].type

    c = get_ctx(box, ptype, ptype)
    trial = random_particle(box, ptype, box.box_len, box.rng, c)

    # Trial sits at the last slot (n), so every other particle precedes it in slot order.
    energy_new = 0.0
    for j in 1:n
        j == idx && continue
        pj = box.particles[j]
        energy_new += particle_energy(get_ctx(box, pj.type, ptype), box.box_len, pj, trial)
    end

    ln_acc = -(energy_new - energy_old) / box.temperature
    accept = ln_acc >= 0.0 || lcg_random!(box.rng) < exp(ln_acc)
    accept || return false

    for j in 1:n
        j == idx && continue
        box.particle_energies[j] -= box.energy_matrix[idx, j]
    end
    box.total_energy -= energy_old
    remove_slot!(box, idx)                      # n -> n-1, later slots shift down

    idx_to = box.n_particles + 1                # == n, the freed top slot
    box.particles[idx_to] = trial
    box.particle_energies[idx_to] = 0.0
    for j in 1:box.max_capacity
        if j >= idx_to
            box.energy_matrix[idx_to, j] = 0.0
            box.energy_matrix[j, idx_to] = 0.0
            continue
        end
        e = pair_energy_idx(box, idx_to, j)
        box.energy_matrix[idx_to, j] = e
        box.energy_matrix[j, idx_to] = e
        box.particle_energies[idx_to] += e
        box.particle_energies[j] += e
    end
    box.total_energy += box.particle_energies[idx_to]
    box.n_particles += 1
    box.n_transfer_accepted += 1
    return true
end

# Insert one particle into an empty slot from an implicit zero-energy reservoir.
function gemc_reservoir_insert!(box::GemcBox)
    idx_to = find_empty_slot(box)
    idx_to < 1 && return false
    ptype = box.particles[idx_to].type
    c = get_ctx(box, ptype, ptype)
    box.particles[idx_to] = random_particle(box, ptype, box.box_len, box.rng, c)

    energy_new = 0.0
    for j in 1:idx_to-1
        energy_new += pair_energy_idx(box, idx_to, j)
    end
    beta = 1.0 / box.temperature
    accept = energy_new <= 0.0 || lcg_random!(box.rng) < exp(-beta * energy_new)
    accept || return false

    box.particle_energies[idx_to] = 0.0
    for j in 1:box.max_capacity
        if j >= idx_to
            box.energy_matrix[idx_to, j] = 0.0
            box.energy_matrix[j, idx_to] = 0.0
            continue
        end
        e = pair_energy_idx(box, idx_to, j)
        box.energy_matrix[idx_to, j] = e
        box.energy_matrix[j, idx_to] = e
        box.particle_energies[idx_to] += e
        box.particle_energies[j] += e
    end
    box.total_energy += box.particle_energies[idx_to]
    box.n_particles += 1
    ptype == 0 && (box.n_type_a_active += 1)
    return true
end

# ---- construction / configuration I/O ----

"""
    create_box(n_particles, box_length, temperature, max_trans_param, max_rot_param, seed,
               ctxs, n_types, zratio, target_ratio)

`sim_gemc_create_box`. `ctxs` are the n_types*(n_types+1)/2 pair contexts (already carrying
their elec/vdw scales).
"""
function create_box(n_particles::Int, box_length::Float64, temperature::Float64,
                    max_trans_param::Float64, max_rot_param::Float64, seed::Integer,
                    ctxs::Vector{FmapCtx}, n_types::Int, zratio::Float64, target_ratio::Float64)
    mt = max_trans_param * 1.62
    nbin = ctxs[1].nbin
    return GemcBox(n_particles, 0, (box_length, box_length, box_length * zratio), temperature,
                   fill(Particle(), n_particles), zeros(n_particles), zeros(n_particles, n_particles), 0.0,
                   ctxs, n_types, 0, nbin,
                   mt, max_rot_param,
                   0, 0, 0, 0,
                   target_ratio, mt, max_rot_param, 0, 0, 0, 0,
                   box_length,                        # js_max: caps the shift at box_len/2
                   Float64(nbin) / ROT_DEFAULT,       # rs_max: caps the bin offset at nbin/2
                   LCG(seed), zeros(n_particles))
end

"Assign the types of slots [first, last] (1-based) from cumulative per-type counts; the rest get last_t."
function assign_types!(box::GemcBox, first::Int, counts::AbstractVector{<:Integer}, n_counts::Int)
    pos = first - 1                          # 0-based next slot
    for t in 0:n_counts-1
        pos < box.max_capacity || break
        stop = min(pos + counts[t+1], box.max_capacity)
        while pos < stop
            box.particles[pos+1] = with_type(box.particles[pos+1], t); pos += 1
        end
    end
    last_t = n_counts < box.n_types ? n_counts : box.n_types - 1
    while pos < box.max_capacity
        box.particles[pos+1] = with_type(box.particles[pos+1], last_t); pos += 1
    end
end

function initialize_random!(box::GemcBox, target_count::Int, counts_per_box::AbstractVector{<:Integer}, n_counts::Int)
    target_count = min(target_count, box.max_capacity)
    assign_types!(box, 1, counts_per_box, n_counts)
    place_particles_random!(box, target_count)
    calculate_initial_energy!(box)
    return true
end

const _HDR_RE = r"^# \d+ particles, energy=\S+, nA=\d+, js=(\S+), rs=(\S+)"

"Parse a configuration line `id x y z e1 e2 e3 [type]`; nothing if fewer than 6 numbers follow the id."
function parse_particle_line(line::AbstractString)
    tok = split(line)
    length(tok) >= 7 && tryparse(Int, tok[1]) !== nothing || return nothing
    v = Vector{Float64}(undef, 6)
    for k in 1:6
        x = tryparse(Float64, tok[k+1]); x === nothing && return nothing
        v[k] = x
    end
    type = length(tok) >= 8 ? something(tryparse(Int, tok[8]), 0) : 0
    return v, type
end

function load_configuration!(box::GemcBox, filename::AbstractString, target_count::Int)
    isfile(filename) || (println("Error: cannot open $filename"); return false)
    c0 = box.ctxs[1]
    loaded = 0
    println("Loading $filename...")
    for line in eachline(filename)
        loaded < box.max_capacity || break
        if startswith(line, '#')
            m = match(_HDR_RE, line)
            if m !== nothing
                js = tryparse(Float64, m.captures[1]); rs = tryparse(Float64, m.captures[2])
                if js !== nothing && rs !== nothing
                    box.js = js; box.rs = rs
                end
            end
            continue
        end
        r = parse_particle_line(line)
        r === nothing && continue
        v, type = r
        b1, b2, b3 = euler_to_bins(v[4], v[5], v[6], c0.nbin)
        box.particles[loaded+1] = Particle(v[1], v[2], v[3], b1, b2, b3, bins_to_rot_idx(c0, b1, b2, b3), Int32(type))
        type == 0 && (box.n_type_a_active += 1)
        loaded += 1
    end
    box.n_particles = loaded
    println("Loaded $loaded particles.")
    calculate_all_energies!(box)
    if target_count > loaded
        @printf("Adding %d particles randomly to reach %d...\n", target_count - loaded, target_count)
        place_particles_random!(box, target_count)
    end
    calculate_initial_energy!(box)
    return true
end

get_type_count(box::GemcBox, type::Integer) = count(i -> box.particles[i].type == type, 1:box.n_particles)

function topup!(box::GemcBox, target_count::Int, topup_counts::AbstractVector{<:Integer}, n_counts::Int)
    target_count = min(target_count, box.max_capacity)
    target_count <= box.n_particles && return true
    start = box.n_particles                   # first empty slot, 0-based
    for t in 0:n_counts-1
        cnt = topup_counts[t+1]
        i = start
        while i < box.max_capacity && cnt > 0
            box.particles[i+1] = with_type(box.particles[i+1], t); cnt -= 1; i += 1
        end
    end
    last_t = n_counts < box.n_types ? n_counts : box.n_types - 1
    for i in start:box.max_capacity-1
        box.particles[i+1] = with_type(box.particles[i+1], last_t)
    end
    place_particles_random!(box, target_count)
    calculate_initial_energy!(box)
    return true
end

function particle_line(id::Integer, p::Particle, nbin::Integer)
    e1, e2, e3 = bins_to_euler_fma(p.b1, p.b2, p.b3, nbin)
    return @sprintf("%16d%16.10f%16.10f%16.10f%16.10f%16.10f%16.10f%8d\n",
                    id, p.x, p.y, p.z, e1, e2, e3, p.type)
end

function save_configuration(box::GemcBox, filename::AbstractString)
    open(filename, "w") do f
        write(f, @sprintf("# %d particles, energy=%.6f, nA=%d, js=%.10g, rs=%.10g\n",
                          box.n_particles, box.total_energy, box.n_type_a_active, box.js, box.rs))
        for i in 1:box.n_particles
            write(f, particle_line(i, box.particles[i], box.nbin))
        end
    end
    println("Saved $filename")
    return true
end

run_steps!(box::GemcBox, n::Integer) = (for _ in 1:n; displacement_step!(box); end; nothing)
rand_unit!(box::GemcBox) = lcg_random!(box.rng)

function particle_coords(box::GemcBox, slot::Int)      # 1-based slot, 6 coords (pos + euler)
    p = box.particles[slot]
    e = bins_to_euler_fma(p.b1, p.b2, p.b3, box.nbin)
    return (p.x, p.y, p.z, e[1], e[2], e[3])
end
