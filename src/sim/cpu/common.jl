# common.jl - RNG, particle record, orientation-bin conversions (particle_fmap.c / fmap_energy.c).

const TERN_SCL = 8.0          # tErn_SCL (tern.h)
const TERN_MAX = Int8(127)    # tErn_MAX: excluded-volume sentinel in channel 0 / the single byte
const ROT_DEFAULT = 3.0       # rotation displacement in bin units (particle_fmap.c rot_default)
const MAX_N_CTXS = 15
const TWO_PI = 2.0 * pi

"Grid cell format from the environment, as the C builds' -DTERN_BYTES: 2 = [elec,vdw], 1 = pre-summed."
function tern_bytes_from_env()
    tb = parse(Int, get(ENV, "TERN_BYTES", "2"))
    tb in (1, 2) || error("TERN_BYTES must be 1 or 2, got $tb")
    return tb
end

# ---- LCG (glibc constants, 31-bit state), identical to lcg_random / lcg_next_dev ----

mutable struct LCG
    state::UInt64
end
LCG(seed::Integer) = LCG(UInt64(seed % UInt32))

@inline lcg_step(s::UInt64) = (s * UInt64(1103515245) + UInt64(12345)) & UInt64(0x7fffffff)
@inline lcg_unit(s::UInt64) = Float64(s) / Float64(0x7fffffff)

@inline function lcg_random!(l::LCG)
    l.state = lcg_step(l.state)
    return lcg_unit(l.state)
end

# ---- particle ----

"One particle: position, continuous orientation bins, precomputed rotation index, type."
struct Particle
    x::Float64; y::Float64; z::Float64
    b1::Float64; b2::Float64; b3::Float64
    rot_idx::Int32
    type::Int32
end
Particle() = Particle(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, Int32(0), Int32(0))

@inline with_type(p::Particle, t::Integer) =
    Particle(p.x, p.y, p.z, p.b1, p.b2, p.b3, p.rot_idx, Int32(t))

# ---- euler <-> bins (nbin bins per rotation period, from the grid header) ----

function euler_to_bins(e1::Float64, e2::Float64, e3::Float64, nbin::Integer)
    a = rem(e1 + pi, TWO_PI); a < 0 && (a += TWO_PI)
    b1 = (a * nbin) / TWO_PI
    beta = max(0.0, min(pi, e2))
    b2 = ((1.0 - cos(beta)) * nbin) / 2.0
    g = rem(e3 + pi, TWO_PI); g < 0 && (g += TWO_PI)
    b3 = (g * nbin) / TWO_PI
    return b1, b2, b3
end

function bins_to_euler(b1::Float64, b2::Float64, b3::Float64, nbin::Integer)
    e1 = -pi + (b1 / nbin) * TWO_PI
    e2 = acos(max(-1.0, min(1.0, 1.0 - b2 / nbin * 2.0)))
    e3 = -pi + (b3 / nbin) * TWO_PI
    return e1, e2, e3
end

# bins -> euler as the CPU C build computes it (-M_PI + x*2pi fused).
function bins_to_euler_fma(b1::Float64, b2::Float64, b3::Float64, nbin::Integer)
    e1 = muladd(b1 / nbin, TWO_PI, -pi)
    e2 = acos(max(-1.0, min(1.0, 1.0 - b2 / nbin * 2.0)))
    e3 = muladd(b3 / nbin, TWO_PI, -pi)
    return e1, e2, e3
end

@inline function wrap_bin(b::Float64, nbin::Integer)
    b = rem(b, Float64(nbin))
    return b < 0 ? b + nbin : b
end

# C: (int)round(x) (half away from zero), then (idx<0) ? idx+n : (idx>=n) ? 0 : idx
@inline c_round_int(x::Float64) = unsafe_trunc(Int64, round(x, RoundNearestTiesAway))
@inline wrap_idx(i::Int64, n::Int64) = i < 0 ? i + n : (i >= n ? Int64(0) : i)

"Sorted type pair (t1,t2) -> context index (get_ctx / ctx_index_dev); 0-based in and out."
@inline function ctx_index(t1::Integer, t2::Integer, n_types::Integer)
    i = min(t1, t2); j = max(t1, t2)
    return (i * (2 * n_types - 1 - i)) ÷ 2 + j
end

"Parse comma-separated integers (parse_type_totals_v); an empty string gives an empty list."
function parse_type_totals(s::AbstractString, max_n::Integer = MAX_N_CTXS)
    out = Int[]
    for tok in split(s, ',')
        isempty(tok) && continue
        length(out) >= max_n && break
        push!(out, c_atoi(tok))
    end
    return out
end

"C atoi: optional blanks/sign, leading digits, anything else ignored."
function c_atoi(s::AbstractString)
    m = match(r"^\s*([+-]?\d+)", s)
    return m === nothing ? 0 : parse(Int, m.captures[1])
end

"C atof for the plain decimal / exponent forms used on the command lines."
function c_atof(s::AbstractString)
    m = match(r"^\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)", s)
    return m === nothing ? 0.0 : parse(Float64, m.captures[1])
end

const _tern_warned = Ref(false)
"Warn once when a scale is requested that the 1-byte cell format cannot apply (tern_check_scales)."
function tern_check_scales(tern::Integer, es::Real, vs::Real)
    if tern == 1 && (es != 1.0 || vs != 1.0) && !_tern_warned[]
        _tern_warned[] = true
        println(stderr, "Warning: elec_scale=$es vdw_scale=$vs requested but NOT applied - this run uses the 1-byte tErn format ",
                "(TERN_BYTES=1): elec+vdw are pre-summed in the grid, so there is no separate channel to scale; the lookup runs as ",
                "if both were 1.0. kT (TSIM/TSIMS) still varies correctly. Use TERN_BYTES=2 (and a 2-channel grid) to use them. ",
                "(Printed once per process.)")
    end
    return nothing
end
