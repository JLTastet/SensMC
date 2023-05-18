module Particles


# Imports
import Base: show, ==
import LorentzVectors: boost
using LorentzVectors
using PyCall: pyimport
using Memoize: @memoize
const pdg_tables = pyimport("particletools.tables")


# Exports
export Field, Particle, PointParticle
export pdata
export mass, M_
export four_momentum, P_
export energy, E_
export momentum, p_
export velocity, β_
export field
export boost
export show, ==


# Global constants
const pdata = pdg_tables.PYTHIAParticleData()


# This function is memoized to minimize the overhead of calling Python
@memoize mass(id::Int) = pdata.mass(id)


struct Field
    id :: Int
    mass :: Float64
end

Field(id::Int) = Field(id, mass(id))

mass(f::Field) = f.mass

function ==(f1::Field, f2::Field)
    f1.id == f2.id
end


abstract type Particle end

field(X::Particle) = error("Not implemented!")

mass(X::Particle) = mass(X.field)
const M_ = mass

four_momentum(X::Particle) = error("Not implemented!")
const P_ = four_momentum

energy(X::Particle) = P_(X).t
const E_ = energy

momentum(X::Particle) = Vec3(P_(X))
const p_ = momentum

velocity(X::Particle) = p_(X) / E_(X)
const β_ = velocity


"""
Represents a particle with the spin information averaged out.
"""
struct PointParticle{T <: Real} <: Particle
    field :: Field
    p :: LorentzVector{T}
end

function PointParticle(f::Field, p::Vec3=zero(Vec3))
    m = mass(f)
    PointParticle(f, LorentzVector(√(m^2+p⋅p), p))
end

function show(io::IO, X::PointParticle)
    print(io, "Particle(")
    print(io, field(X).id)
    print(io, ")")
end

field(X::PointParticle) = X.field

four_momentum(X::PointParticle) = X.p

boost(X::PointParticle, β) = PointParticle(field(X), boost(P_(X), β))


end # module Particles
