import MDBM
import MDBM: Axis, MDBM_Problem
import DataFrames
import CSV
import JSON
using Roots
using OrderedCollections
using DataStructures
import Base: pop!
using Multisets
using Logging

###################
### Sensitivity ###
###################

## Compute the number of events, for fixed parameters

"""
    compute_yield(simulation, geometry, Nsamples, acceptance) -> yield

Computes the number of accepted events, averaged over `Nsamples` samples.

`acceptance` takes the same values as `mode` in `event_in_acceptance`.
"""
function compute_yield(sim::Simulation, geom::Geometry, Nsamples::Int,
                       acceptance::Symbol)

    evt = EventRecord()
    weights = zeros(Nsamples)
    accepted = zeros(Nsamples)
    for k in 1:Nsamples
        success = false
        while !success
            success = sample_event!(evt, sim)
        end
        accepted[k] = event_in_acceptance(evt, geom, sim.fip.id, acceptance)
        weights[k] = evt.weight
    end
    weights ./= evt.counter
    sum(weights .* accepted)
end

"""
    compute_scalar_portal_yield(
        mS, θ, α, NPOT or L_int,
        meson_spectrums_source, production_processes,
        geometry, Nsamples,
        acceptance=:two_tracks_zero_charge, kwargs...
    ) -> yield

Helper function for computing the number of accepted events for a dark scalar.

The arguments are forwarded to `make_scalar_portal_simulation` and `compute_yield`.
See their respective documentations for more information.
"""
function compute_scalar_portal_yield(mS::Float64, θ::Float64, α::Float64,
                                     NPOT_or_Lint::Float64, meson_spectrums_source::Symbol,
                                     production_processes::Vector{String},
                                     geom::Geometry, Nsamples::Int;
                                     acceptance::Symbol=:two_tracks_zero_charge, kwargs...)

    L_min, L_max = covering_shell(geom)
    sim = make_scalar_portal_simulation(mS, θ, α, NPOT_or_Lint, meson_spectrums_source,
                                        production_processes, L_max/m;
                                        L_min_meters=L_min/m, kwargs...)
    compute_yield(sim, geom, Nsamples, acceptance)
end

"""
    compute_critical_yield(CL=0.95, rtol=1e-3, xmax=25.) -> critical_yield

Computes the expected signal yield above which a non-observation (in a zero-background setting)
leads to exclusion of the signal hypothesis at the confidence level `CL` or higher.

`rtol` is the relative tolerance for the bisection method, and `xmax` the upper bound of the initial bracketing interval.
"""
function compute_critical_yield(; CL=0.95, rtol=1e-3, xmax=25.)
    find_zero(x -> cdf(Poisson(x), 0) - (1-CL), (1.,xmax), Bisection(), xrtol=rtol)
end

## 1d bisection method, for a fixed mass

"""
    compute_scalar_portal_sensitivity(
        mS, couplings_ratio::NTuple{2,Float64}, scale_bracket::NTuple{2,Float64},
        NPOT or L_int, meson_spectrums_source, production_processes, geometry,
        Nsamples=10000, CL=0.95, rtol=1e-2, verbose=false,
        logger=ConsoleLogger(Warn), kwargs...
    ) -> coupling_scale

Computes the sensitivity to the scalar portal for a fixed mass, by bisecting the critical mixing angle.

Its unique parameters are:
- `couplings_ratio`: the ratio of the mixing angle and quartic coupling, i.e. (θ : α);
- `scale_bracket`: a bracket in the signal "scale" (which multiplies the normalized `couplings_ratio`), which should contain the critical scale; for mixing-only, the bracket corresponds to (θ\\_min, θ\\_max);
- `Nsamples`: the number of Monte-Carlo samples to generate; more samples lead to a smaller statistical uncertainty;
- `verbose`: whether to print the trace of the bisection (for debugging);
- `logger`: a custom logger can be passed to override the default loglevel (e.g. pass `ConsoleLogger(Info)` for more detailed logs).

The remaining parameters are the same as for `compute_scalar_portal_yield` and `compute_critical_yield`. See their respective documentations for details.

!!! note

    This function uses the non-squared mixing angle, i.e. θ instead of θ².

!!! warning "Limitation"

    Setting α ≠ 0 is currently not fully supported, so for the moment `coupling_ratio` should be set to (1., 0.).
"""
function compute_scalar_portal_sensitivity(
    mS::Float64, couplings_ratio::NTuple{2,Float64}, scale_bracket::NTuple{2,Float64},
    NPOT_or_Lint::Float64, meson_spectrums_source::Symbol,
    production_processes::Vector{String}, geom::Geometry;
    Nsamples::Int=10000, CL=0.95, rtol=1e-2, verbose=false,
    logger=ConsoleLogger(Warn), kwargs...)

    couplings_ratio = [couplings_ratio...] / sum(couplings_ratio) # Make sure the ratio sums to unity
    @assert sum(couplings_ratio) ≈ 1.
    critical_yield = compute_critical_yield(CL=CL, rtol=rtol)

    function f(μ)
        compute_scalar_portal_yield(
            mS, μ*couplings_ratio[1], μ*couplings_ratio[2], NPOT_or_Lint,
            meson_spectrums_source, production_processes, geom, Nsamples;
            kwargs...
        ) - critical_yield
    end

    with_logger(logger) do
        if verbose
            println("Lower yield = ", f(scale_bracket[1]) + critical_yield)
            println("Upper yield = ", f(scale_bracket[2]) + critical_yield)
        end
        try
            find_zero(f, scale_bracket, Bisection(); xrtol=rtol, verbose=verbose)
        catch
            missing
        end
    end
end

## 2d bisection method

"""
    sigmoid(x, slope)

A sigmoid function, that is zero at `x == 0` and ±1 at ±Inf.

This function does not affect the `x == 0` isocontour, but is helpful to compress large values and make the 2d bisection better-behaved, especially when combined with a logarithm.
"""
sigmoid(x, k) = 2 / (1 + exp(-k*x)) - 1

"""
    find_scalar_portal_sensitivity_mdbm(
        log10_mS_range, log10_scale_range, couplings_ratio::NTuple{2,Float64},
        NPOT or L_int, meson_spectrums_source, production_processes, geometry;
        iterations=3, Nsamples=10000, CL=0.95,
        logger=ConsoleLogger(Warn), kwargs...
    ) -> (mS_list, scale_list, debug)

Computes the full sensitivity curve for the scalar portal, using the multi-dimensional bisection method (MDBM) in 2 dimensions.

Its unique parameters are:
- `log10_mS_range`: a coarse range of log10(masses) that will be evaluated eagerly, before starting to refine the grid; it should ideally contain the entire curve;
- `log10_scale_range`: a coarse range of log10(couplings scale) (log10(θ) for mixing-only) that will be evaluated eagerly, similar to above;
- `couplings_ratio`: the ratio of the mixing angle and quartic coupling, i.e. (θ : α);
- `iterations`: the number of times the grid will be refined when performing the 2d bisection;
- `Nsamples`: the number of Monte-Carlo samples to generate; more samples lead to a smaller statistical uncertainty;
- `CL`: the desired confidence level for the expected exclusion limit, assuming nearly-zero background;
- `logger`: a custom logger can be passed to override the default loglevel (e.g. pass `ConsoleLogger(Info)` for more detailed logs).

The remaining parameters are the same as for `compute_scalar_portal_yield`. See its documentation for details.

!!! note

    This function uses the non-squared mixing angle, i.e. θ instead of θ².

!!! warning "Limitation"

    Setting α ≠ 0 is currently not fully supported, so for the moment `coupling_ratio` should be set to (1., 0.).
"""
function find_scalar_portal_sensitivity_mdbm(
    log10_mS_range::AbstractRange{Float64}, log10_scale_range::AbstractRange{Float64},
    couplings_ratio::NTuple{2,Float64}, NPOT_or_Lint::Float64, meson_spectrums_source::Symbol,
    production_processes::Vector{String}, geom::Geometry;
    iterations::Int=3, Nsamples::Int=10000, CL::Float64=0.95,
    logger=ConsoleLogger(Warn), kwargs...)

    log10_mS_axis = Axis(log10_mS_range)
    log10_scale_axis = Axis(log10_scale_range)
    critical_yield = compute_critical_yield(CL=CL, rtol=1e-4)
    couplings_ratio = [couplings_ratio...] / sum(couplings_ratio) # Make sure the ratio sums to unity
    function f(log10_mS, log10_μ)
        with_logger(logger) do
            yield = try
                compute_scalar_portal_yield(
                    10.0^log10_mS, 10.0^log10_μ*couplings_ratio[1], 10.0^log10_μ*couplings_ratio[2],
                    NPOT_or_Lint, meson_spectrums_source, production_processes, geom, Nsamples; kwargs...)
            catch e
                if isa(e, PhaseSpaceError)
                    0.0
                else
                    @show 10.0^log10_mS
                    @show 10.0^log10_μ
                    throw(e)
                end
            end
            sigmoid(log(yield/critical_yield), 1.)
        end
    end
    mdbm = MDBM_Problem(f, [log10_mS_axis, log10_scale_axis])
    MDBM.solve!(mdbm, iterations)
    log10_mS_eval, log10_μ_eval = MDBM.getevaluatedpoints(mdbm)
    log10_mS_sol, log10_μ_sol = MDBM.getinterpolatedsolution(mdbm)
    edge_connection = MDBM.connect(mdbm)
    debug = (mass_evaluated=10.0 .^ log10_mS_eval, signal_scale_evaluated=10.0 .^ log10_μ_eval,
             mdbm=mdbm, edge_connection=edge_connection)
    10.0 .^ log10_mS_sol, 10.0 .^ log10_μ_sol, debug
end

"""
    save_sensitivity_mdbm(path, name, M_sol, μ_sol, debug, column_names=("M", "scale"))

Saves the full output of the MDBM algorithm for further processing.

Arguments:
- `path`: the directory where to save the files;
- `name`: the base name that will be included in every file name;
- `M_sol`: the list of masses produced by MDBM;
- `μ_sol`: the list of coupling scales produced by MDBM;
- `debug`: additional information (all evaluated points, the edge connection, the MDBM object) that may be useful to save;
- `column_names`: a tuple specifying how the "mass" and "scale" columns should be named in the CSV files.
"""
function save_sensitivity_mdbm(
    path::AbstractString, name::AbstractString,
    M_sol::AbstractVector{<:AbstractFloat}, μ_sol::AbstractVector{<:AbstractFloat}, debug;
    column_names::NTuple{2,String}=("M", "scale"))

    df_sol = DataFrames.DataFrame([M_sol, μ_sol], ["$(name)_sol" for name in column_names])
    df_eval = DataFrames.DataFrame([debug.mass_evaluated, debug.signal_scale_evaluated],
                                   ["$(name)_evaluated" for name in column_names])
    df_connect = DataFrames.DataFrame(first=first.(debug.edge_connection), second=last.(debug.edge_connection))
    CSV.write(joinpath(path, "$(name).csv"), df_sol)
    CSV.write(joinpath(path, "$(name)_grid.csv"), df_eval)
    CSV.write(joinpath(path, "$(name)_connection.csv"), df_connect, writeheader=false)
    open_loops, closed_loops = find_loops(debug.edge_connection)
    d = OrderedDict("open_loops" => open_loops, "closed_loops" => closed_loops)
    open(joinpath(path, "$(name)_loops.json"), "w+") do f
        JSON.print(f, d, 4)
    end
    if length(open_loops) + length(closed_loops) == 1
        perm = vcat(open_loops..., closed_loops...)
        df_sorted = df_sol[perm,:]
        CSV.write(joinpath(path, "$(name)_sorted.csv"), df_sorted)
    end
    nothing
end

"""
    find_and_save_scalar_portal_sensitivity_mdbm(path, name, args..., kwargs...)

Sequentially runs `find_scalar_portal_sensitivity_mdbm` and `save_sensitivity_mdbm`, forwarding arguments as needed.

!!! tip

    Because MDBM can take a long time to run, this function can be useful to avoid forgetting to save its output.
"""
function find_and_save_scalar_portal_sensitivity_mdbm(
    path::AbstractString, name::AbstractString, args...;
    column_names::NTuple{2,String}=("mS", "signal_scale"), kwargs...)

    mS_sol, μ_sol, debug = find_scalar_portal_sensitivity_mdbm(args...; kwargs...)
    save_sensitivity_mdbm(path, name, mS_sol, μ_sol, debug; column_names=column_names)
end

"""
    pop!(m::Multiset)

Randomly remove and element from a `Multiset` and return it.
"""
function pop!(m::Multiset{T}) where {T}
    v = rand(m)
    push!(m, v, -1)
    v
end

"""
    find_loops(edge_connection) -> (open_loops_list, closed_loops_list)

Form sets of open and closed loops from the edge connection produced by MDBM.

!!! info

    Due to either physical features or noise, the sensitivity curve may form disconnected islands, some of which may be open.

    This function essentially sorts the indices of the curve into a set of islands, that may then be plotted separately (for instance).
"""
function find_loops(connection::Vector{NTuple{2,Int}})
    # We play dominoes with indices to link the points to each other...
    multiplicities = Multiset(vcat(first.(connection), last.(connection)))
    @assert sum(values(multiplicities)) % 2 == 0
    edges = OrderedSet(connection)
    # Select points with a non-trivial number of edges
    nodes = Multiset([x for x in multiplicities if multiplicities.data[x] != 2])
    # First, find open loops
    open_loops = MutableLinkedList{Int}[]
    while !isempty(nodes)
        ll = MutableLinkedList{Int}(pop!(nodes))
        complete = false
        push!(open_loops, ll)
        while !complete
            edge = first(e for e in edges if last(ll) in e)
            new_index = only(x for x in edge if x != last(ll))
            push!(ll, new_index)
            pop!(edges, edge)
            if new_index in nodes
                push!(nodes, new_index, -1)
                complete = true
            end
        end
    end
    # Second, find closed loops
    closed_loops = MutableLinkedList{Int}[]
    while length(edges) > 0
        ll = MutableLinkedList{Int}(first(edges)...)
        popfirst!(edges)
        push!(closed_loops, ll)
        closed = last(ll) == first(ll)
        while !closed
            edge = only(e for e in edges if last(ll) in e)
            new_index = only(x for x in edge if x != last(ll))
            push!(ll, new_index)
            pop!(edges, edge)
            closed = last(ll) == first(ll)
        end
    end
    # Convert linked lists to arrays
    open_loops_arr = Vector{Int}[]
    closed_loops_arr = Vector{Int}[]
    for loop in open_loops
        push!(open_loops_arr, [x for x in loop])
    end
    for loop in closed_loops
        push!(closed_loops_arr, [x for x in loop])
    end
    open_loops_arr, closed_loops_arr
end
