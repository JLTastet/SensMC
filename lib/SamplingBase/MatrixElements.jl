# TODO: modularise

using .Particles


@doc raw"""
    MatrixElementSampler

Abstract supertype of all matrix element samplers.

!!! note

    No matrix elements are currently supported for the scalar portal.

    The utility functions present in this file can nonetheless be used as a base to implement your own matrix elements.


!!! tip "Tutorial"

    Here is a short tutorial on how to implement a new matrix element:

    ```julia
    # Define a struct that is a subtype of `MatrixElementSampler`
    struct MySampler <: MatrixElementSampler
        ps :: NBodySampler{3, Float64}
        sup_weight :: Float64
        parent :: Field
        children :: NTuple{3, Field}
    end

    # Define some utility functions (get parent mass, pretty-print sampler, create `PointParticle` objects)
    parent_mass(sampler::MySampler) = mass(sampler.parent)

    function Base.show(io::IO, sp::MySampler)
        print(io, "MySampler($(sp.parent.id) ->")
        for ch in sp.children
            print(io, " $(ch.id)")
        end
        print(io, ")")
    end

    _make_particles(sp::MySampler, Pi::NTuple{3,Vec4{Float64}}) = _generic_make_particles(sp, Pi)

    # The actual matrix element, defined in terms of the parent and children momenta
    my_matrix_element(P0, P1, P2, P3) = (P1+P2)⋅(P1+P2)

    # Re-use the generic function to sample an event uniformly in phase space and compute its weight
    _sample_weighted(sp::MySampler, P0::Vec4) = _sample_weighted(my_matrix_element, sp.ps, P0)

    # Re-use the generic function to perform rejection sampling
    sample(sp::MySampler, X0::Particle) = _generic_weighted_sample(sp, X0)

    # Define a convenience constructor
    function MySampler(parent::Field, children...)
        ps = NBodySampler(mass(parent), mass.(children)...) # Initialize the phase-space sampler
        sup_weight = _burn_in(my_matrix_element, ps, 10000, 5.) # Estimate an upper bound on the matrix element weight
        MySampler(ps, sup_weight, parent, children)
    end

    # Instantiate the sampler
    sp = MySampler(Field(25), Field(22), Field(22), Field(22))

    # Sample one event from it
    decay_products = sample(sp, PointParticle(Field(25)))

    # Mutating version of `sample`
    function sample!(particles::Vector{<: PointParticle}, sampler::MySampler, X0::PointParticle)
        resize!(particles, 3)
        particles .= sample(sampler, X0)
        particles
    end

    particles = PointParticle[]

    sample!(particles, sp, PointParticle(Field(25)))
    ```
"""
abstract type MatrixElementSampler end

parent_mass(::MatrixElementSampler) :: Float64 = error("Not implemented")


#####################################################
### Utility functions for matrix element sampling ###
#####################################################


function _burn_in(
    me::ME,
    ps::NBodySampler,
    samples::Int,
    nsigmas::Float64
) where {ME}

    max_weight = -Inf
    P0 = Vec4(ps.m0, 0, 0, 0)
    for k in 1:samples
        w, _ = _sample_weighted(me, ps, P0)
        max_weight = max(w, max_weight)
    end

    # Correct for bias
    max_weight /= (1 - 1/samples)

    # Add safety margin
    max_weight * (1 + nsigmas/samples)
end

function _burn_in(
    me::ME,
    ps₁::NBodySampler,
    ps₂::NBodySampler,
    samples::Int,
    nsigmas::Float64
) where {ME}

    max_weight = -Inf
    P0 = Vec4(ps₁.m0, 0, 0, 0)
    for k in 1:samples
        w, _ = _sample_weighted(me, ps₁, ps₂, P0)
        max_weight = max(w, max_weight)
    end

    # Correct for bias
    max_weight /= (1 - 1/samples)

    # Add safety margin
    max_weight * (1 + nsigmas/samples)
end

function _sample_weighted(
    me::ME,
    ps::NBodySampler{N,Float64},
    PH::Vec4
) where {ME, N}

    Pi = sample(ps, PH)
    weight = me(PH, Pi...)
    weight, Pi
end

function _sample_weighted(
    me::ME,
    ps₁::NBodySampler{M,Float64},
    ps₂::NBodySampler{N,Float64},
    PH::Vec4
) where {ME, M, N}

    Pi₁ = sample(ps₁, PH)
    Pi₂ = sample(ps₂, Pi₁[M])
    weight = me(PH, Pi₁..., Pi₂...)
    weight, (Base.front(Pi₁)..., Pi₂...)
end

function _generic_make_particles(sampler::MatrixElementSampler, Pi::NTuple{N,Vec4{Float64}}) where {N}
    NTuple{N,PointParticle{Float64}}(
        PointParticle(f, P) for (f, P) in zip(sampler.children, Pi))
end

function _generic_weighted_sample(sampler::MatrixElementSampler, XH::Particle)
    if field(XH) ≠ sampler.parent
        throw(ArgumentError("Parent particle does not match"))
    end
    PH = P_(XH)
    while true
        w, Pi = _sample_weighted(sampler, PH)
        if w > rand() * sampler.sup_weight
            return _make_particles(sampler, Pi)
        end
    end
end


#######################
### Matrix elements ###
#######################


## Uniform sampler

struct UniformSampler{N} <: MatrixElementSampler
    phase_space_sampler :: NBodySampler{N, Float64}
    parent :: Field
    children :: NTuple{N, Field}
end

parent_mass(sampler::UniformSampler) = sampler.phase_space_sampler.m0

function Base.show(io::IO, sp::UniformSampler)
    print(io, "UniformSampler($(sp.parent.id) ->")
    for ch in sp.children
        print(io, " $(ch.id)")
    end
    print(io, ")")
end

"""
    UniformSampler(parent, children...)

A sampler that samples particles uniformly in phase space.
"""
function UniformSampler(parent, children...; kwargs...)
    ps_sampler = NBodySampler(mass(parent), mass.(children)...; kwargs...)
    UniformSampler{length(ps_sampler.mi)}(ps_sampler, parent, children)
end

# Special treatment for "1-body decays" (e.g. K0 -> K_L0)
function sample(sampler::UniformSampler{1}, XH::Particle)
    (PointParticle{Float64}(sampler.children[1], P_(XH)),)
end

function sample(sampler::UniformSampler{N}, XH::Particle) where {N}
    Pi = sample(sampler.phase_space_sampler, P_(XH))
    NTuple{N,PointParticle{Float64}}(
        PointParticle(f, P) for (f, P) in zip(sampler.children, Pi))
end

function sample!(particles::Vector{<: PointParticle}, sampler::UniformSampler{N}, XH::PointParticle) where {N}
    resize!(particles, N)
    particles .= sample(sampler, XH)
    particles
end


## Narrow-width approximation for two subsequent decays

# Samples multiple successive decays within the narrow-width approximation
struct NWASampler{Samplers} <: MatrixElementSampler
    samplers :: Samplers
end

"""
    NWASampler(samplers::MatrixElementSampler...)

Samples multiple subsequent decays in the narrow-width approximation.

The last decay product in the sampler N becomes the parent in the sampler (N+1).
"""
function NWASampler(samplers::MatrixElementSampler...)
    for i in length(samplers)-1
        @assert samplers[i+1].parent == samplers[i].children[end]
    end
    NWASampler(samplers)
end

function Base.show(io::IO, sp::NWASampler)
    print(io, "NWASampler(")
    for (i, sub) in enumerate(sp.samplers)
        print(io, (i > 1) ? " -> " : "", sub)
    end
    print(io, ")")
end

function sample!(particles::Vector{<: PointParticle}, sampler::NWASampler, XH::PointParticle)
    resize!(particles, 0)
    X :: PointParticle = XH
    for sp in sampler.samplers
        Xs = sample(sp, X)
        X = Xs[end]
        for i in 1:length(Xs)-1
            push!(particles, Xs[i])
        end
    end
    push!(particles, X)
    particles
end
