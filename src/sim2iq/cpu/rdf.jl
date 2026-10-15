# Normalization modes for RDF density mapping
@enum NormalizationMode NORM_DEFAULT = 0 NORM_ELECTRON_COUNT = 1 NORM_VOLUME = 2 NORM_NONE = 3

mutable struct RdfParams
    global_b::Float64
    vol::Float64
    rho0::Float64
    norm_mode::NormalizationMode
end
RdfParams(; global_b = 0.0, vol = 0.0, rho0 = 0.0, norm_mode = NORM_DEFAULT) =
    RdfParams(global_b, vol, rho0, norm_mode)

# Multi-Gaussian RDF from Cromer-Mann parameters
@inline function multigauss_rdf(r2::Float64, cm::CromerMann4Gauss, B::Float64)
    ff = 0.0
    @inbounds for i in 1:4
        B_term = cm.b[i] + B
        prefactor = (4π / B_term)^1.5 * cm.a[i]
        ff += prefactor * exp(-4π^2 * r2 / B_term)
    end
    return ff
end

# Simple Gaussian RDF
@inline function simple_gauss_rdf(r2::Float64, V::Float64, rho0::Float64)
    B_term = 4π * V^(2.0 / 3.0)
    prefactor = 8 * rho0 * π^1.5 * V / B_term^1.5
    return prefactor * exp(-4π^2 * r2 / B_term)
end
