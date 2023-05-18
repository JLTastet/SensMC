module Histograms


export Histogram
export +,-,*,/
export fill!, copy
export midpoints, counts
export underflow, overflow
export project


"""
# Histogram

A histogram type inspired from ROOT's `TH1`

`Histogram{T<:Real, N, Edge}`

Constructors:
* `Histogram(edges...)`
* `Histogram{T<:Real, N, Edge}(edges, counts)`
* `Histogram{T<:Real, N, Edge}(edges)`
"""
struct Histogram{T <: Real, N, Edge}
    edges  :: NTuple{N, Edge}
    counts :: Array{T, N}

    # Keep default constructor
    function Histogram{T,N,Edge}(edges::NTuple{N,Edge}, counts::Array{T,N}) where {N, T <: Real, Edge <: AbstractArray{T,1}}
        new{T,N,Edge}(edges, counts)
    end

    # Define new inner constuctor
    function Histogram{T,N,Edge}(edges::NTuple{N,Edge}) where {N, T <: Real, Edge <: AbstractArray{T,1}}
        # N + 1 edges → N + 2 bins (including underflow & overflow)
        ct_dims = map(edge_vec -> size(edge_vec,1) + 1, edges)
        counts = zeros(T, ct_dims)
        new{T,N,Edge}(edges, counts)
    end
end

"""
Outer constructor for convenience
"""
Histogram(edges::Edge...) where {T <: Real, Edge <: AbstractArray{T,1}} = Histogram{T,length(edges),Edge}(edges)

import Base: +,-,*,/

for op in (:+,:-,:*,:/)
    @eval begin
        function ($op)(lh::Histogram{T,N,E}, rh::Histogram{T,N,E}) where {T,N,E}
            @assert lh.edges == rh.edges
            Histogram{T,N,E}(lh.edges, broadcast($op, lh.counts, rh.counts))
        end

        function ($op)(lh::Histogram{T,N,E}, rh::Real) where {T,N,E}
            Histogram{T,N,E}(lh.edges, broadcast($op, lh.counts, rh))
        end

        function ($op)(lh::Real, rh::Histogram{T,N,E}) where {T,N,E}
            Histogram{T,N,E}(rh.edges, broadcast($op, lh, rh.counts))
        end
    end
end

@doc """
#### Derive common arithmetic operations

Adapted from https://discourse.julialang.org/t/concise-way-to-create-numeric-type/4900/6

New methods for binary operators:
* `+(h1, h2)`
* `-(h1, h2)`
* `*(h1, h2)`
* `/(h1, h2)`
""" Base.:+;

"""
#### Define functions to fill the histogram

Helper function `find_bin_idx(ar, el)`

Return the index in `ar` of the bin to which `el` belongs.

Special values:
* `0` → underflow
* `size(ar,1)` → overflow
"""
find_bin_idx(ar, el) = searchsortedfirst(ar, el, lt=(<=)) - 1

find_bin_idx(ar::StepRangeLen, el) = Int(floor((el - ar.ref.hi) / ar.step.hi)) + ar.offset

import Base: fill!
"""
Function `fill!`

Methods:
* `fill!(hist, x, w)`
* `fill!(hist, x)`
* `fill!(hist, array, weights)`
* `fill!(hist, array)`
"""
function fill!(hist::Histogram{T,N,E}, x::NTuple{N,U}, w::V) where {T,U,V,N,E}
    bin_idx = ntuple(
        i -> clamp( find_bin_idx(hist.edges[i], x[i]) + 1, 1, size(hist.counts, i) ),
        Val(N))
    hist.counts[bin_idx...] += w
    bin_idx
end

fill!(hist::Histogram{T,N,E}, x::NTuple{N,U}) where {T,U,N,E} = fill!(hist, x, one(T))

function fill!(hist::Histogram{T,N,E}, array::AbstractArray{NTuple{N,T}}, weights::AbstractArray{T}) where {T,N,E}
    for (x,w) in zip(array, weights)
        fill!(hist, x, w);
    end
end

fill!(hist::Histogram{T,N,E}, array::AbstractArray{NTuple{N,T}}) where {T,N,E} = fill!(hist, array, one(T):one(T))

import Base: copy
"""
#### Misc. functions

Add method `copy(hist)`
"""
copy(hist::Histogram{T,N,E}) where {T,N,E} =
    Histogram{T,N,E}(Tuple(copy(ed) for ed in hist.edges), copy(hist.counts))

"""
Returns the midpoints of each bin. This is useful for plotting.
"""
midpoints(hist::Histogram) = [(edges[1:end-1] .+ edges[2:end]) ./ 2 for edges in hist.edges]

"""
Returns the counts for each bin (excluding underflow and overflow). This is useful for plotting.
"""
counts(hist::Histogram{T,N,E}) where {T,N,E} = hist.counts[[2:size(hist.counts, d)-1 for d in 1:N]...]

"""
Return respectively the counts for the underflow and overflow bins (1D only for now).
"""
underflow(hist::Histogram{T,1,E}) where {T,E} = hist.counts[1]
overflow( hist::Histogram{T,1,E}) where {T,E} = hist.counts[end]

"""
#### Sums bins along the given ``D`` dimensions, resulting in a ``(N-D)``-dimensional histogram

Function `project`

Methods:
* `project(hist, dim)`
* `project(hist, dims...)`
"""
function project(hist::Histogram{T,N,E}, dim::Int) where {T,N,E}
    new_dims = collect(d for d in 1:N if d ≠ dim)
    projection = Histogram{T,N-1,E}(hist.edges[new_dims])
    projection.counts .= dropdims(sum(hist.counts, dims=dim), dims=dim)
    projection
end

"""
Project along multiple dimensions using recursion
"""
function project(hist::Histogram{T,N,E}, dims::Int...) where {T,N,E}
    d1 = dims[1]
    projection_1 = project(hist, d1)
    new_dims = tuple(collect(d > d1 ? d-1 : d for d in dims if d ≠ d1)...)
    project(projection_1, new_dims...)
end


end # module Histograms