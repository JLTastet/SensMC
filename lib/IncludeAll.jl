using Distributed
@everywhere using Pkg
@everywhere Pkg.activate("$(@__DIR__)/..")

@everywhere using Memoize
@everywhere using LorentzVectors
@everywhere import LorentzVectors: boost
@everywhere using LinearAlgebra
@everywhere using StaticArrays
@everywhere using SharedArrays
@everywhere using Distributions
@everywhere using StatsBase
@everywhere using Missings
@everywhere using PyCall

@everywhere begin

    include("$(@__DIR__)/SetPaths.jl")

    include("$(@__DIR__)/Util/Units.jl")

    include("$(@__DIR__)/SamplingBase/Particles.jl")
    include("$(@__DIR__)/SamplingBase/PhaseSpaceSamplers.jl")
    include("$(@__DIR__)/SamplingBase/MatrixElements.jl")

end # @everywhere

nothing # Do not return anything when including this file.
