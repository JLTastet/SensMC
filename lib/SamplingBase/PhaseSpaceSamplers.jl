import StatsBase.sample

kallen_triangle(a, b, c) = a^2 + b^2 + c^2 - 2*a*b - 2*b*c - 2*c*a
const λ_ = kallen_triangle

function kallen_positive(a, b, c)
    λ = kallen_triangle(a, b, c)
    # λ could be negative due to underflow. Ensure it is always positive.
    λ >= zero(λ) ? λ : zero(λ)
end
const λ⁺_ = kallen_positive

function _sample_2body_cm(m0, m1, m2)

    pAbs = √λ⁺_(m0^2, m1^2, m2^2) / 2m0
    p1 = pAbs * rand(SpatialVector{Float64})

    E1 = √(m1^2 + pAbs^2)
    E2 = √(m2^2 + pAbs^2)

    P1 = Vec4(E1, +p1)
    P2 = Vec4(E2, -p1)

    P1, P2
end

function sample_2body(m0, m1, m2, P0::Vec4)

    P1_CM, P2_CM = _sample_2body_cm(m0, m1, m2)

    β = Vec3(P0) / P0.t
    P1 = boost(P1_CM, -β)
    P2 = boost(P2_CM, -β)

    P1, P2
end

function sample_2body(m0::T, m1::T, m2::T, p0::Vec3{T}=zero(Vec3{T})) where {T <: AbstractFloat}

    E0 = √(m0^2 + p0⋅p0)
    P0 = Vec4(E0, p0)

    sample_2body(m0, m1, m2, P0)
end

function _sample_weighted_intermediate_masses(m0::T, mi::SVector{N,T}) where {N, T <: AbstractFloat}

    # Validate arguments
    if N < 2
        error("Need at least two decay products.")
    end

    # Generate a list of random intermediate masses
    Δm = m0 - sum(mi)
    r = MVector{N,T}(undef)
    r[1] = zero(T)
    r[N] = one(T)
    r_ = rand(MVector{N-2,T})
    sort!(r_)
    r[2:N-1] .= r_
    Σmi = cumsum(mi) .+ r .* Δm

    # Evaluate the Jacobian
    J = one(T)
    for i in 2:N
        J *= √λ⁺_(Σmi[i]^2, Σmi[i-1]^2, mi[i]^2) / 2Σmi[i]
    end

    Σmi, J
end

function _sample_intermediate_masses(m0::T, mi::SVector{N,T}, supJ::T) where {N, T <: AbstractFloat}

    while true
        Σmi, J = _sample_weighted_intermediate_masses(m0, mi)
        if J > supJ*rand()
            return Σmi
        end
    end
end

function _sample_nbody(m0::T, mi::SVector{N,T}, supJ::T, P0::Vec4) where {N, T <: AbstractFloat}

    Σmi = _sample_intermediate_masses(m0, mi, supJ)
    Pi  = MVector{N,Vec4{T}}(undef)
    ΣPi = MVector{N,Vec4{T}}(undef)
    ΣPi[N] = P0
    for i in N:-1:2
        ΣPi[i-1], Pi[i] = sample_2body(Σmi[i], Σmi[i-1], mi[i], ΣPi[i])
    end
    Pi[1] = ΣPi[1]
    Pi
end

struct NBodySampler{N, T <: AbstractFloat}
    m0 :: T
    mi :: SVector{N,T}
    supJ :: T
end

function NBodySampler(
    m0::T, masses::T...;
    burn_in_samples::Int=1000,
    nsigmas::Float64=5.,
) where {T <: AbstractFloat}

    # Convert arguments
    N = length(masses)
    mi = SVector(masses)

    # Use burn-in samples to estimate the maximum of J
    maxJ = -Inf
    if N > 2
        for k in 1:burn_in_samples
            _, J = _sample_weighted_intermediate_masses(m0, mi)
            maxJ = max(maxJ, J)
        end
    end

    # Correct the maximum for bias due to the finite number of samples
    maxJ /= (1 - 1/burn_in_samples)

    # Add a small safety margin to make sure that we have an actual upper bound for J
    supJ = maxJ * (1 + nsigmas/burn_in_samples)

    # Create sampler
    NBodySampler{N,T}(m0, mi, supJ)
end

function sample(sampler::NBodySampler{N,T}, P0::Vec4{T}) where {N, T <: AbstractFloat}
    Pi = _sample_nbody(sampler.m0, sampler.mi, sampler.supJ, P0)
    NTuple{N,Vec4{T}}(Pi[i] for i in 1:N)
end

function sample(sampler::NBodySampler{N,T}, p0::Vec3{T}=zero(Vec3{T})) where {N, T <: AbstractFloat}
    P0 = Vec4(√(sampler.m0^2+p0⋅p0), p0)
    sample(sampler, P0)
end
