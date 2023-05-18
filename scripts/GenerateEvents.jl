using LinearAlgebra
using Distributions
using StatsBase
import Base: empty!, convert, showerror, show
using OrderedCollections
using PyCall
const uproot = pyimport("uproot")
using DelimitedFiles, Distributions

const scalar_portal = pyimport("scalar_portal")
const particle_data = pyimport("scalar_portal.data.particles")

const pdg_tables = pyimport("particletools.tables")
const pythia_pdg = pdg_tables.PYTHIAParticleData() # Table from particletools, misses data on anti-particles

const SCALAR_ID = 9900025;
const FIP_IDs = [SCALAR_ID];

# Define the picobarn as the base cross-section unit. Not to be mixed with GeV-based natural units.
const pb = 1.0
const mb = 1000 * pb
const fb = pb / 1000
const ab = fb / 1000

#################
### Spectrums ###
#################

## FairShip spectrums

mutable struct FairShipSpectrums
    tree_iterator :: PyObject
    counter :: Int
    index :: Int
    id :: Vector{Float32}
    E  :: Vector{Float32}
    px :: Vector{Float32}
    py :: Vector{Float32}
    pz :: Vector{Float32}
    M  :: Vector{Float32}
    filepath :: String
    step_size :: String
end

function _load_new_chunk!(sp::FairShipSpectrums)
    chunk = try
        sp.tree_iterator.__next__()
    catch ex
        if ex isa PyCall.PyError && ex.T == py"StopIteration"
            @warn "Reached the end of the meson list, looping back. This may violate the i.i.d. assumption." maxlog=1
            sp.tree_iterator = uproot.iterate("$(sp.filepath):pythia6;1", ["id", "px", "py", "pz", "E", "M"],
                                           library="np", step_size=sp.step_size)
            sp.tree_iterator.__next__()
        else
            throw(ex)
        end
    end
    sp.index = 1
    sp.id = chunk["id"] :: Vector{Float32}
    sp.E  = chunk["E" ] :: Vector{Float32}
    sp.px = chunk["px"] :: Vector{Float32}
    sp.py = chunk["py"] :: Vector{Float32}
    sp.pz = chunk["pz"] :: Vector{Float32}
    sp.M  = chunk["M" ] :: Vector{Float32}
end

"""
    FairShipSpectrums(filepath)

Samples heavy mesons one-by-one from one of the ROOT files used by FairShip.

Uses `uproot` to iterate over the file, and loops back when reaching the end.
"""
function FairShipSpectrums(filepath; step_size="1 MB")
    tree_iterator = uproot.iterate("$filepath:pythia6;1", ["id", "px", "py", "pz", "E", "M"],
                                   library="np", step_size=step_size)
    sp = FairShipSpectrums(tree_iterator,0, 1,
                           Float32[], Float32[], Float32[],
                           Float32[], Float32[], Float32[],
                           filepath, step_size)
    _load_new_chunk!(sp)
    sp
end

"""
    sample(sampler::FairShipSpectrums)

Sample one single meson from `FairShipSpectrums`. Returns a `PointParticle`.
"""
function sample(sp::FairShipSpectrums)
    momentum = Vec4(sp.E[sp.index], sp.px[sp.index], sp.py[sp.index], sp.pz[sp.index])
    field = Field(sp.id[sp.index], sp.M[sp.index])
    particle = PointParticle{Float64}(field, momentum)
    sp.counter += 1
    if sp.index < length(sp.id)
        sp.index += 1
    else
        _load_new_chunk!(sp)
    end
    @assert sp.index <= length(sp.id)
    particle
end

## FORESEE spectrums

struct DiscretizedFORESEESpectrum
    field :: Field
    midpoints :: Vector{NTuple{2,Float64}}
    weights :: Weights{Float64, Float64, Vector{Float64}}
    table_path :: String
end

function show(io::IO, sp::DiscretizedFORESEESpectrum)
    print(io, "DiscretizedFORESEESpectrum(field=")
    print(io, sp.field.id)
    print(io, ", table_path=")
    print(io, sp.table_path)
    print(io, ")")
end

"""
    DiscretizedFORESEESpectrum(tables_dir, pdg_code::Int; L_int=1.)

Sample from one of the (θ, |p|) tables used in FORESEE [arXiv:2105.07077], without performing any interpolation.

!!! warning

    Due to the discretization involved, this is only a rough approximation of the true spectrum.

    A more precise sampler may be implemented in the future.

If `L_int == 1.`, the weights correspond to the cross-section in picobarns.
If `L_int` is an integrated luminosity in picobarns, the weights correspond to physical counts.
"""
function DiscretizedFORESEESpectrum(tables_dir::AbstractString, pdg_code::Int; L_int::Float64=1.)
    table_path = joinpath(tables_dir, "Pythia8_14TeV_$(pdg_code).txt")
    if !isfile(table_path)
        throw(ArgumentError("File Pythia8_14TeV_$(pdg_code).txt not found in directory $tables_dir for particle code $pdg_code."))
    end
    mat = readdlm(table_path; comments=true, comment_char='#')
    θ = 10. .^ mat[:,1]
    p = 10. .^ mat[:,2] * GeV
    w = weights(L_int * mat[:,3])
    @assert all(w .>= 0)
    nonzero = w .> 0 # Drop zero weights, which will never be sampled anyway
    w = w[nonzero]
    perm = sortperm(w, rev=true) # Put the largest weights first for faster sampling
    DiscretizedFORESEESpectrum(
        Field(pdg_code),
        collect(zip(θ[nonzero][perm], p[nonzero][perm])),
        w[perm],
        table_path)
end

"""
    sample(sampler::DiscretizedFORESEESpectrum)

Sample a single meson from the corresponding `DiscretizedFORESEESpectrum`. Returns a `PointParticle`.
"""
function sample(sp::DiscretizedFORESEESpectrum)
    φ = rand(Uniform(0, 2π))
    # θ, p = sample(sp.midpoints, sp.weights)
    idx = sample(1:size(sp.weights,1), sp.weights) # Faster, somehow
    θ, p = sp.midpoints[idx]
    px = p * sin(θ) * cos(φ)
    py = p * sin(θ) * sin(φ)
    pz = p * cos(θ)
    PointParticle(sp.field, Vec3(px, py, pz))
end

## Create weighted lists of meson spectrums, for the different meson species

const fairship_pdg_ids = Dict(
    :charm  => [-411, 411, -421, 421, -431, 431, -4122, 4122, -4132, 4132, -4232, 4232, -4332, 4332],
    :beauty => [-511, 511, -521, 521, -531, 531, -5122, 5122, -5132, 5132, -5232, 5232, -5332, 5332],
)

"""
    list_heavy_meson_spectrums_fairship(NPOT, flavor, [meson_spectrums_root]) \
        -> ((weight, probabilities::Vector, spectrums::Vector), meson_ids::Vector)

Loads the FairShip spectrums for either `flavor==:charm` or `:beauty` mesons.

The `FairShipSpectrums` object takes care of sampling from the various meson species,
therefore a single sampler is returned, with unit probability.

!!! note

    This function requires the following external resources:
    - Charm: Cascade-parp16-MSTP82-1-MSEL4-978Bpot.root (MD5 = 402535074194f9c4bec79fe6333c95af)
    - Beauty: Cascade-run0-19-parp16-MSTP82-1-MSEL5-5338Bpot.root (MD5 = 29140ca106955a75a113c0ff569117f8)
"""
function list_heavy_meson_spectrums_fairship(NPOT::Float64, flavor::Symbol;
                                             meson_spectrums_root=missing)
    @info "Some heavy hadrons do not have any decay channels implemented and are currently ignored." maxlog=1
    χqq, cascade_enhancement, data_file = if flavor == :charm
        1.7e-3, 2.3, "Cascade-parp16-MSTP82-1-MSEL4-978Bpot.root"
    elseif flavor == :beauty
        1.6e-7, 1.7, "Cascade-run0-19-parp16-MSTP82-1-MSEL5-5338Bpot.root"
    else
        error("`flavor` must be :charm or :beauty")
    end
    N_mesons = NPOT * 2 * χqq * cascade_enhancement
    if ismissing(meson_spectrums_root)
        meson_spectrums_root = ENV["MESON_SPECTRUMS"]
    end
    full_path = joinpath(meson_spectrums_root, data_file)
    spectrums = FairShipSpectrums(full_path)
    (weight=N_mesons, probabilities=weights(@SVector [1.]), spectrums=@SVector [spectrums]), fairship_pdg_ids[flavor]
end

"""
    list_heavy_meson_spectrums_foresee(L_int, flavor::Int, [meson_spectrums_root]) \
        -> ((weight, probabilities::Vector, spectrums::Vector), meson_ids::Vector)

Loads the relevant spectrums from the FORESEE tables for the selected `flavor ∈ [:charm, :beauty]`.

Returns a list of spectrums (one per meson species), with probabilities proportional to the number of mesons.

!!! note

    This function requires the following external resources:
    - Charm:
      - Pythia8\\_14TeV\\_-411.txt (MD5 = 996db931fb07fdfad3ebe757f23ef93a)
      - Pythia8\\_14TeV\\_-421.txt (MD5 = 111a1a69d762a675ba5137c262d21a45)
      - Pythia8\\_14TeV\\_-431.txt (MD5 = ff41456fe7b236caceeb89a6dc6e62a3)
      - Pythia8\\_14TeV\\_411.txt (MD5 = e70035238c7c86a5edfd8b5403222536)
      - Pythia8\\_14TeV\\_421.txt (MD5 = e7a577ad88943569818521aea5772789)
      - Pythia8\\_14TeV\\_431.txt (MD5 = a53df94e45e183048c030397d556b1e6)
    - Beauty:
       - Pythia8\\_14TeV\\_-511.txt (MD5 = 08d4eaa9b9319cd47c42450f8f880d78)
       - Pythia8\\_14TeV\\_-521.txt (MD5 = 3eefe4ef0d56de9885e1280b5f58e8eb)
       - Pythia8\\_14TeV\\_-531.txt (MD5 = ec4f516bf172aa89c0b7c4d7bec1b144)
       - Pythia8\\_14TeV\\_511.txt (MD5 = 728a1d20942c3221a2d20aeb38776edb)
       - Pythia8\\_14TeV\\_521.txt (MD5 = 67319e3f9ae28d3c66ef86b726049f7b)
       - Pythia8\\_14TeV\\_531.txt (MD5 = f54f06330c0a3779eebb74798175a4d9)

    At the time of writing the documentation, these files can be found at the
    [FORESEE GitHub repository](https://github.com/KlingFelix/FORESEE/tree/main/files/hadrons/14TeV/Pythia8).
"""
function list_heavy_meson_spectrums_foresee(L_int::Float64, flavor::Symbol; meson_spectrums_root=missing)
    meson_pdg_codes = if flavor == :charm
        [-411, 411, -421, 421, -431, 431]
    elseif flavor == :beauty
        [-511, 511, -521, 521, -531, 531]
    else
        error("`flavor` must be :charm or :beauty")
    end
    if ismissing(meson_spectrums_root)
        meson_spectrums_root = ENV["MESON_SPECTRUMS"]
    end
    spectrums = [DiscretizedFORESEESpectrum(meson_spectrums_root, pdg_code; L_int=L_int) for pdg_code in meson_pdg_codes]
    spectrum_weights = [sp.weights.sum for sp in spectrums]
    (
        (
            weight=sum(spectrum_weights),
            probabilities=weights(spectrum_weights / sum(spectrum_weights)),
            spectrums=spectrums
        ),
        meson_pdg_codes
    )
end

"""
    list_heavy_meson_spectrums(which, NPOT or L_int, [meson_spectrums_root]) \
        -> ((weight, probabilities::Vector, spectrums::Vector), meson_ids::Vector)

Loads the relevant spectrums, using external data files from FairShip or FORESEE.

Possible values: `which=:fairship_charm`, `:fairship_beauty`, `:foresee_charm` or `:foresee_beauty`.

See the documentation of `list_heavy_meson_spectrums_fairship` and `list_heavy_meson_spectrums_foresee` fore details.
"""
function list_heavy_meson_spectrums(which::Symbol, NPOT_or_Lint::Float64; meson_spectrums_root=missing)
    if which == :fairship_charm
        list_heavy_meson_spectrums_fairship(NPOT_or_Lint, :charm; meson_spectrums_root=meson_spectrums_root)
    elseif which == :fairship_beauty
        list_heavy_meson_spectrums_fairship(NPOT_or_Lint, :beauty; meson_spectrums_root=meson_spectrums_root)
    elseif which == :foresee_charm
        list_heavy_meson_spectrums_foresee(NPOT_or_Lint, :charm, meson_spectrums_root=meson_spectrums_root)
    elseif which == :foresee_beauty
        list_heavy_meson_spectrums_foresee(NPOT_or_Lint, :beauty, meson_spectrums_root=meson_spectrums_root)
    else
        error("`which` must be :fairship_charm, :fairship_beauty, :foresee_charm or :foresee_beauty")
    end
end


################################
### FIP production and decay ###
################################

## Utility functions and types

# Shorthands for various complicated types
const weights_t = typeof(weights(zeros(1)))
const weighted_samplers_t = @NamedTuple{weight::Float64,probabilities::weights_t,samplers::Vector{MatrixElementSampler}}

"""
    PhaseSpaceError(msg)

Warn that the FIP cannot be produced due to no phase space being available.
"""
struct PhaseSpaceError <: Exception
    msg::String
end

Base.showerror(io::IO, e::PhaseSpaceError) = print(io, e.msg)

"""
    get_pdg_id(name)

Returns the particle's PDG code, given its name.
"""
@memoize function get_pdg_id(particle::String)
    particle_data.get_pdg_id(particle)
end

"""
    get_mass(name or PDG code)

Returns the particle's mass, given its plain-text name or its PDG code.
"""
@memoize function get_mass(particle::Union{Int,String})
    particle_data.get_mass(particle)
end

## Dark scalar production channels

"""
    list_open_scalar_production_channels(M, θ, α, production_processes, meson_pdg_ids) \
        -> Dict(meson_id => production_channels::weighted_samplers_t)

Generate the samplers for the production processes of a dark scalar produced in heavy meson decays.

Parameters:
- `M`: the dark scalar mass,
- `θ`: its mixing angle with the SM Higgs,
- `α`: its quartic coupling,
- `production_processes`: a list of `String` representing the production processes to enable (see the documentation of the Python package `scalar_portal` for details),
- `meson_pdg_ids`: the PDG codes of the parent mesons that will be decayed to dark scalars.

!!! warning

    For 3-body decays, hadronic matrix elements are currently *not* implemented. Instead, sampling is performed uniformly in phase space.
"""
function list_open_scalar_production_channels(M::Float64, θ::Float64, α::Float64,
                                              production_processes::Vector{String}, meson_pdg_ids::Vector{Int};
                                              burn_in_samples::Int=10000, nsigmas::Float64=5., mode=:phase_space)
    @assert mode == :phase_space # Other modes not implemented for the scalar
    # List all scalar production channels using the `scalar_portal` packages

    model = scalar_portal.Model()
    for proc in production_processes
        model.production.enable(proc)
    end
    res = model.compute_branching_ratios(M, theta=θ, alpha=α)

    # Define scalar with mass M
    S = Field(SCALAR_ID, M)

    # Just a dict declaration, with the explicit type specified for performance reasons
    production_channels = Dict{Int,weighted_samplers_t}()
    # Fill the dictionary with a list of branching fractions and MC samplers, for each parent meson
    for parent_id in meson_pdg_ids
        branching_ratios = Float64[]
        samplers = MatrixElementSampler[]
        for (key, ch) in res.production._channels
            # For now, let’s assume channels are defined for mesons with code > 0
            @assert convert(Int, get_pdg_id(ch._parent)) > 0
            sgn = sign(parent_id)
            parent = Field(sgn * get_pdg_id(ch._parent), get_mass(ch._parent))
            children = [Field(sgn * get_pdg_id(c), get_mass(c)) for c in ch._other_children]
            push!(children, S)
            if parent.id == parent_id
                # Skip kinematically closed channels
                if mass(parent) <= sum(mass.(children))
                    continue
                end
                br = res.production.branching_ratios[key]
                @info "Production matrix elements not implemented for the scalar portal." maxlog=1
                sampler = UniformSampler(parent, children...; burn_in_samples=burn_in_samples, nsigmas=nsigmas)
                push!(branching_ratios, br)
                push!(samplers, sampler)
            end
        end

        # Importance sampling
        weight = sum(branching_ratios)
        if weight == 0.
            @info "No scalar production channel found in $parent_id decays."
            continue
        end
        probabilities = branching_ratios / sum(branching_ratios)
        @assert weight > 0
        @assert sum(probabilities) ≈ 1.
        @assert weight * probabilities ≈ branching_ratios
        perm = sortperm(probabilities, rev=true)

        production_channels[parent_id] = (weight=weight, probabilities=weights(probabilities[perm]), samplers=samplers[perm])
    end

    if isempty(production_channels)
        throw(PhaseSpaceError("no phase space available for the scalar production."))
    end

    production_channels
end

## Dark scalar decay channels

"""
    list_open_scalar_decay_channels(M, θ, α) -> (total_width, decay_channels::weighted_samplers_t)

Generate the samplers for the decays of a dark scalar.

Parameters:
- `M`: the dark scalar mass,
- `θ`: its mixing angle with the SM Higgs,
- `α`: its quartic coupling.

Returns the total width of the dark scalar in GeV in addition to the decay channels.
"""
function list_open_scalar_decay_channels(M::Float64, θ::Float64, α::Float64)
    # List the visible scalar decay channels using the scalar_portal module

    model = scalar_portal.Model()
    if M < 2.0GeV
        model.decay.enable("LightScalar")
    else
        model.decay.enable("HeavyScalar_LightCharmedHadrons")
    end
    res = model.compute_branching_ratios(M, theta=θ, alpha=α)
    total_width = res.decay.total_width

    # Define scalar with mass M
    S = Field(SCALAR_ID, M)

    # For decays to quarks, replace the kinematic mass with the one of the lightest meson with the same quantum numbers
    override_quark_masses = Dict(
        "s"    => get_mass("K"),
        "sbar" => get_mass("K"),
        "c"    => get_mass("D"),
        "cbar" => get_mass("D"),
        "b"    => get_mass("B"),
        "bbar" => get_mass("B"),
    )
    function get_kinematic_mass(particle::String)
        get(override_quark_masses, particle, get_mass(particle))
    end

    # Create MC samplers for each scalar decay channel
    # Each of them is weighted by its branching fraction
    branching_ratios = Float64[]
    scalar_decay_samplers = MatrixElementSampler[]

    for (key, ch) in res.decay._channels
        # Skip the channel if it is kinematically closed
        BR = res.decay.branching_ratios[key]
        if BR == 0
            continue
        elseif key == "S -> mesons..."
            @info "Multi-meson decay channel not simulated."
            continue
        end

        sampler = UniformSampler(S, (Field(get_pdg_id(c), get_kinematic_mass(c)) for c in ch._children)...)
        push!(branching_ratios     , BR     )
        push!(scalar_decay_samplers, sampler)
    end

    if sum(branching_ratios) == 0.
        throw(PhaseSpaceError("no phase space available for the scalar decay."))
    end

    # Importance sampling
    weight = sum(branching_ratios)
    probabilities = branching_ratios / sum(branching_ratios)
    @assert sum(probabilities) ≈ 1.
    @assert weight * probabilities ≈ branching_ratios
    perm = sortperm(probabilities, rev=true)

    total_width, (weight=weight, probabilities=weights(probabilities[perm]), samplers=scalar_decay_samplers[perm])
end

const sm_decays_t = Vector{Tuple{Float64, Vector{Int64}}}

# FIXME: workaround for https://github.com/afedynitch/particletools/issues/7
function _remove_incorrectly_conjugated_pdg_ids(channel::eltype(sm_decays_t), valid_pdg_ids::Vector{Int64})
    br, children = channel
    correct_children = [(child in valid_pdg_ids) ? child : abs(child) for child in children]
    @assert all(child in valid_pdg_ids for child in correct_children)
    (br, correct_children)
end

"""
    make_stable_list(min_metastable_lifetime_second))

Lists the particles whose lifetime exceeds the specified one, and which are thus considered metastable.
"""
@memoize function make_stable_list(min_metastable_lifetime_seconds::Float64)
    pdg_tables.make_stable_list(min_metastable_lifetime_seconds)
end

"""
    get_decay_channels(pdg_code) -> [(BR, [children PDG codes...]), ...]

Returns the decay channels of the specified SM particle, as a list containing, for each entry,
the branching ratio and the PDG codes of the decay products.
"""
@memoize function get_decay_channels(particle::Int)
    pythia_pdg.decay_channels(particle)
end

"""
    list_SM_decay_channels(min_metastable_lifetime_seconds=1e-8, verbose=true) \
        -> Dict(pdg_code => sm_decay_channels::weighted_samplers_t)

Returns the SM decay channels of the SM particles whose lifetime exceeds `min_metastable_lifetime_seconds`.

If `verbose=true`, prints the PDG codes of the particles considered as metastable.
"""
function list_SM_decay_channels(; min_metastable_lifetime_seconds=1e-8, verbose=true)
    # Decay channels for other unstable particles
    # Dictionary
    # Parent PDG code => [(BR, [children...]), ...]
    # Branching ratios taken from the PYTHIA database through the `particletools` package.
    metastable_particles = Set(make_stable_list(min_metastable_lifetime_seconds))

    sm_decays = Dict{Int, sm_decays_t}()
    particles_to_decay = Set([+13, -13, +15, -15])
    valid_pdg_ids = [id for (id, data) in pythia_pdg.iteritems()
                        if (0 < abs(id) < 1000000) || (abs(id) > 9000000)] # Remove SUSY et al.

    function decay_children!(decays::sm_decays_t)
        for decay in decays
            br, children = decay
            for child in children
                push!(particles_to_decay, child)
            end
        end
    end

    # Decay the D mesons # FIXME: automatically select which mesons should be decayed
    push!(particles_to_decay, +411, -411, +421, -421)

    remaining_particles = setdiff(particles_to_decay, metastable_particles, keys(sm_decays))
    while !isempty(remaining_particles)
        particle = first(remaining_particles)
        # List the decay channels of the selected particle
        decays = [_remove_incorrectly_conjugated_pdg_ids(channel, valid_pdg_ids)
                  for channel in get_decay_channels(particle)]
        # Make sure that its decay products will be decayed, too
        sm_decays[particle] = decays
        decay_children!(decays)
        remaining_particles = setdiff(particles_to_decay, metastable_particles, keys(sm_decays))
    end

    if verbose
        metastable_particles = setdiff(particles_to_decay, keys(sm_decays)) |> collect
        @info "Particles considered as metastable: $metastable_particles"
    end

    # Just a dict declaration, with the explicit type specified for performance reasons
    sm_channels = Dict{Int,weighted_samplers_t}()

    for (parent_pdg_id, decays) in sm_decays
        if isempty(decays)
            continue
        end

        branching_ratios = Float64[]
        samplers = MatrixElementSampler[]
        for (br, children_pdg_ids) in decays
            sampler = UniformSampler(Field(parent_pdg_id), Field.(children_pdg_ids)...)
            push!(branching_ratios, br)
            push!(samplers, sampler)
        end

        br_sum = sum(branching_ratios)
        if br_sum <= 0.99 # Check that we don't ignore too many channels...
            @warn "Branching ratios for particle $parent_pdg_id only sum up to $br_sum."
        end
        @assert br_sum <= 1+eps(typeof(br_sum)) # ... and that we don't invent any.

        # Importance sampling
        weight = br_sum
        probabilities = branching_ratios / br_sum
        @assert sum(probabilities) ≈ 1.
        @assert weight * probabilities ≈ branching_ratios
        perm = sortperm(probabilities, rev=true)

        sm_channels[parent_pdg_id] = (weight=weight, probabilities=weights(probabilities[perm]), samplers=samplers[perm]) # Named tuple ≈ anonymous struct
    end

    sm_channels
end

##################
### Simulation ###
##################

## Instantiate the simulation

# Uniform importance distribution
struct UniformBias
    L_min :: Float64
    L_max :: Float64
end

struct Simulation{Bias <: Union{Missing,UniformBias}, WeightedMesonSpectrums}
    fip :: Field
    total_fip_width :: Float64
    bias :: Bias
    meson_spectrums :: WeightedMesonSpectrums
    fip_prod_channels  :: Dict{Int,weighted_samplers_t}
    fip_decay_channels :: weighted_samplers_t
    sm_decay_channels  :: Dict{Int,weighted_samplers_t}
    unstable_pdg_ids :: Vector{Int}
    metadata :: OrderedDict{String,Any}
    target_atomic_weight :: Float64 # Should be 96 for the default molybdenum target
end

"""
    Simulation(...)

An object representing a generic simulation of a FIP produced through the decay of heavy particles,
and decaying in a displaced detector.

Parameters:
- `fip_id`: the PDG code of the FIP,
- `M_fip`: the FIP mass in GeV,
- `Γ_fip`: the total width of the FIP in GeV,
- `meson_spectrums`: weighted samplers from which to sample mesons of various species,
- `fip_prod_channels`: a dictionary containing, for each meson species, weighted samplers for the FIP production,
- `fip_decay_channels`: weighted samplers to decay the FIP,
- `metadata`: a dictionary containing some metadata to associate with the simulation,
- `importance_sampling=true`: whether to enable importance sampling of the FIP decay vertex,
- `L_min_meters=0.`: if using importance sampling, the minimum distance from the target, at which to sample the FIP decay vertex,
- `L_max_meters=Inf`: if using importance sampling, the maximum distance from the target, at which to sample the FIP decay vertex,
- `target_atomic_weight=96.`: (for beam-dump experiments only) if using the FairShip spectrums with a target composed of something else than molybdenum, the atomic weight of the target element, used to rescale the elastic scattering cross-section.
"""
function Simulation(fip_id::Int, M_fip::Float64, Γ_fip::Float64, meson_spectrums,
                    fip_prod_channels::Dict{Int,weighted_samplers_t},
                    fip_decay_channels::weighted_samplers_t,
                    metadata::OrderedDict{String,Any};
                    L_min_meters::Float64=0., L_max_meters::Float64=Inf,
                    importance_sampling::Bool=true, target_atomic_weight::Float64=96.)

    fip = Field(fip_id, M_fip)

    bias = if importance_sampling == true
        L_min = L_min_meters * m
        L_max = L_max_meters * m
        UniformBias(L_min, L_max)
    else
        @assert L_min_meters == 0
        @assert L_max_meters == Inf
        missing
    end

    sm_decay_channels = list_SM_decay_channels()
    unstable_pdg_ids = collect(keys(sm_decay_channels))

    Simulation(fip, Γ_fip, bias, meson_spectrums,
               fip_prod_channels, fip_decay_channels,
               sm_decay_channels, unstable_pdg_ids,
               metadata, target_atomic_weight)
end

"""
    make_scalar_portal_simulation(
        MS, θ, α, NPOT or L_int,
        meson_spectrums_source, production_processes,
        L_max_meters, L_min_meters=0., ...
    ) -> Simulation

A helper function to create a `Simulation` of the dark scalar. See `Simulation` for more details.

Main parameters:
- `MS`: the scalar mass,
- `θ`: its mixing angle to the SM Higgs,
- `α`: its quartic coupling,
- `NPOT` or `L_int`: specify the total number of protons on target (POT) for beam-dump experiments, or the integrated luminosity in inverse picobarns for collider experiments,
- `meson_spectrums_source`: specify from which external source to sample the parent mesons (see `list_heavy_meson_spectrums` for allowed values),
- `production_processes`: the list of production processes to enable (see `list_open_scalar_production_channels` for details),
- `L_max_meters`: see `Simulation` for details; must be set to `Inf` if importance sampling is disabled,
- `L_min_meters=0.`, `importance_sampling=true`, `target_atomic_weight=96.`: see `Simulation` for details,
- `meson_spectrums_root=missing`: where to search for external files; if `missing`, query the environment variable `MESON_SPECTRUMS`.
"""
function make_scalar_portal_simulation(
    MS::Float64, θ::Float64, α::Float64,
    NPOT_or_Lint::Float64, meson_spectrums_source::Symbol, production_processes::Vector{String},
    L_max_meters::Float64;
    L_min_meters::Float64=0.,
    meson_spectrums_root=missing,
    importance_sampling=true,
    target_atomic_weight=96.)

    SUPPORTED_MESON_SPECTRUM_SOURCES = [:fairship_beauty, :foresee_beauty]
    if !(meson_spectrums_source in SUPPORTED_MESON_SPECTRUM_SOURCES)
        throw(ArgumentError("Unsupported meson spectrum source (supported: $SUPPORTED_MESON_SPECTRUM_SOURCES)."))
    end

    meson_spectrums, meson_pdg_ids =
        list_heavy_meson_spectrums(meson_spectrums_source, NPOT_or_Lint;
                                   meson_spectrums_root=meson_spectrums_root)

    if α != 0
        @error "The simulation has not been validated when more than one scalar is produced."
        # To fully support the quartic coupling, `sample_event!` should be modified to decay all FIPs (at short lifetimes),
        # or to rescale the event weights by a factor of 2 when using importance sampling.
    end

    scalar_prod_channels = list_open_scalar_production_channels(MS, θ, α, production_processes, meson_pdg_ids)

    Γtot, scalar_decay_channels = list_open_scalar_decay_channels(MS, θ, α)

    intensity_tag = if meson_spectrums_source == :fairship_beauty # Beam dump setup
        "protons_on_target"
    elseif meson_spectrums_source == :foresee_beauty # Collider setup
        @assert target_atomic_weight == 96. # No target => no correction needed
        "L_int"
    else
        @assert false
    end
    metadata = OrderedDict{String,Any}(
        "scalar_mass" => MS,
        "mixing_angles" => θ,
        "quartic_coupling" => α,
        "total_scalar_width" => Γtot,
        intensity_tag => NPOT_or_Lint,
        "fip_L_min_meters" => L_min_meters,
        "fip_L_max_meters" => L_max_meters,
    )

    Simulation(SCALAR_ID, MS, Γtot, meson_spectrums,
               scalar_prod_channels, scalar_decay_channels, metadata;
               L_min_meters=L_min_meters, L_max_meters=L_max_meters,
               importance_sampling=importance_sampling,
               target_atomic_weight=target_atomic_weight)
end

## Sample the FIP decay vertex

"""
    generate_decay_vertex(PN, sim::Simulation{Missing}) -> (weight, vertex)

Sample the FIP decay vertex along its momentum, without importance sampling.

Returns a unit weight and the vertex.
"""
function generate_decay_vertex(PN::Vec4, sim::Simulation{Missing})
    # Compute the FIP velocity and boost factor from its momentum
    β = Vec3(PN) / PN.t
    γ = 1/√(1-β⋅β)
    # Generate a decay vertex according to the exponential distribution
    τ = rand(Exponential(1/sim.total_fip_width))
    # Convert the proper time to the decay vertex in the lab frame
    vertex = β*γ*τ

    1., vertex
end

"""
    generate_decay_vertex(PN, sim::Simulation{UniformBias}) -> (weight, vertex)

Sample the FIP decay vertex along its momentum, uniformly within a specified interval of distance to the target,
using importance sampling.

Returns the importance weight and the vertex.
"""
function generate_decay_vertex(PN::Vec4, sim::Simulation{UniformBias})
    # Compute the FIP velocity and boost factor from its momentum
    β = Vec3(PN) / PN.t
    γ = 1/√(1-β⋅β)
    # Generate a decay vertex uniformly between L_min and L_max
    L = rand(Uniform(sim.bias.L_min, sim.bias.L_max))
    vertex = L * normalize(β)
    # Importance sampling weight = true distribution / importance distribution
    L_exp = γ * norm(β) / sim.total_fip_width
    ΔL = sim.bias.L_max-sim.bias.L_min
    weight = ΔL / L_exp * exp(- L / L_exp)

    weight, vertex
end

## Event record

const particle_t = PointParticle{Float64}

mutable struct EventRecord
    weight :: Float64
    buffer :: Vector{particle_t}
    record :: Vector{particle_t}
    parent_index :: Vector{Int}
    is_live :: Vector{Bool}
    is_final :: Vector{Bool}
    vertex :: Union{Missing,Vec3{Float64}}
    counter :: Int
end

"""
    EventRecord()

Instantiate a new event record, capable of holding all initial, intermediate and final particles and their decay graph,
as well as the event weight, decay vertex, and generated events counter.
"""
EventRecord() = EventRecord(1., particle_t[], particle_t[], Int[], Bool[], Bool[], missing, 0)

"""
    empty!(er::EventRecord)

Empty the event record before generating a new event.

This does *not* reset its counter.
"""
function empty!(er::EventRecord)
    er.weight = 1.
    empty!(er.buffer)
    empty!(er.record)
    empty!(er.parent_index)
    empty!(er.is_live)
    empty!(er.is_final)
    vertex = missing
    # counter is *not* reset!
end

"""
    reset_counter!(event_record)

Reset the number of generated events to zero.
"""
function reset_counter!(er::EventRecord)
    er.counter = 0
end

"""
    sample_event!(event_record, simulation) -> Bool

Run the simulation and sample a new event, that will populate the event record.

This is the function doing most of the work. See the relevant appendix in the SensCalc paper for a high-level description.

The event record is automatically emptied before generating a new event, and its counter is incremented by 1.

Returns `true` if an event was successfully generated, `false` otherwise. In the latter case, do *not* modify the counter, and try generating a new event.
Failed generation can occur when the sampling probabilities do not sum up to one and/or when some channels are missing; it is normal behaviour.

!!! tip

    As a general rule, once the desired number of events has been generated, their weights should be divided by the value of the `counter` stored in the `EventRecord`.
    The sum of weights will then correspond to the number of events generated, while `sum(weights .* acceptance)` will correspond to the number of accepted events.
"""
function sample_event!(evt::EventRecord, sim::Simulation) :: Bool
    # NOTE: remember to divide the final weight by the number of events

    # Make room for the new event, and reset the event weight to unity
    empty!(evt)

    # Increment the counter
    evt.counter += 1

    # Sample a heavy meson
    evt.weight *= sim.meson_spectrums.weight
    meson_spectrum = sample(sim.meson_spectrums.spectrums, sim.meson_spectrums.probabilities)
    meson = sample(meson_spectrum)
    push!(evt.record, meson)
    push!(evt.is_live, true)
    push!(evt.parent_index, 0)

    # Reweight the heavy meson yield to take the target composition into account (only works for elastic scattering)
    # This assumes the use of the FairShip spectrums, computed for a molybdenum target
    # When using a collider setup and keeping the default atomic weight, this will be a no-op
    evt.weight *= (sim.target_atomic_weight / 96.)^0.29

    # Decay the heavy meson to a FIP
    if !(meson.field.id in keys(sim.fip_prod_channels))
        evt.weight = 0.
        return false
    end
    fip_prod_channels = sim.fip_prod_channels[meson.field.id]
    evt.weight *= fip_prod_channels.weight
    prod_sampler = sample(fip_prod_channels.samplers, fip_prod_channels.probabilities)
    sample!(evt.buffer, prod_sampler, meson)
    evt.is_live[1] = false # The meson is now decayed
    append!(evt.record, evt.buffer) # Append the decay products to the event record
    for i in 1:length(evt.buffer)
        push!(evt.is_live, true) # All decay products are live
        push!(evt.parent_index, 1) # Set the parent of the decay products
    end

    # Find the FIP
    fip_index = findfirst(p -> p.field == sim.fip, evt.record)
    fip = evt.record[fip_index]

    # Generate a decay vertex for the FIP
    # The distance is constrained to be within L_min and L_max
    # (the efficiency must be zero outside this range)
    # Importance sampling is used to compensate for the biased distribution
    decay_weight, evt.vertex = generate_decay_vertex(P_(fip), sim)
    evt.weight *= decay_weight

    # Mark all current particles as final
    # Only the FIP decay products may be further decayed
    resize!(evt.is_final, length(evt.record))
    fill!(evt.is_final, true)

    # Decay the FIP
    evt.weight *= sim.fip_decay_channels.weight
    decay_sampler = sample(sim.fip_decay_channels.samplers, sim.fip_decay_channels.probabilities)
    sample!(evt.buffer, decay_sampler, fip)
    evt.is_live[fip_index] = false
    append!(evt.record, evt.buffer)
    for i in 1:length(evt.buffer)
        push!(evt.is_live, true)
        push!(evt.parent_index, fip_index)
        # A particle is unstable if it has SM decays
        push!(evt.is_final, !(evt.buffer[i].field.id in sim.unstable_pdg_ids))
    end

    # Evaluates to true if any particle is not final
    function exist_unstable_particles()
        any_unstable_particles = false
        for i in 1:length(evt.record)
            if !evt.is_final[i]
                any_unstable_particles = true
                break
            end
        end
        any_unstable_particles
    end
    # Recursively decay all unstable SM particles
    while exist_unstable_particles()
        for i in 1:length(evt.record)
            if !evt.is_final[i]
                particle = evt.record[i]
                sm_channels = sim.sm_decay_channels[particle.field.id]
                evt.weight *= sm_channels.weight
                sm_decay_sampler = sample(sm_channels.samplers, sm_channels.probabilities)
                sample!(evt.buffer, sm_decay_sampler, particle)
                evt.is_live[i] = false
                evt.is_final[i] = true # Decayed particles are considered final since they cannot decay further
                append!(evt.record, evt.buffer)
                for j in 1:length(evt.buffer)
                    push!(evt.is_live, true)
                    push!(evt.parent_index, i)
                    push!(evt.is_final, !(evt.buffer[j].field.id in sim.unstable_pdg_ids))
                end
            end
        end
    end
    @assert length(evt.record) == length(evt.parent_index) == length(evt.is_live) == length(evt.is_final)
    @assert all(evt.is_final)
    @assert !ismissing(evt.vertex)
    @assert evt.weight >= 0.
    empty!(evt.buffer)

    true
end

#################
### Detectors ###
#################

abstract type Geometry end

## The frustum geometry used by SHiP and SHADOWS

"""
    FrustumGeometry(...)

A (possibly-degenerate) pyramidal frustum.

This shape encompasses both the SHiP and SHADOWS geometries.

The following parameters must be specified, both at the `start` and `end` of the decay vessel, plus at the far end of the detector (`enddet`):
- `z`: the position of the plane along the beam axis,
- `δx`: the detector width,
- `δy`: the detector height,
- `x0`: the horizontal shift at the corresponding z,
- `y0`: the vertical shift at the corresponding z.

If the values for the detector end are not specified, assume that it has the
same dimensions as the end of the decay vessel and is collinear to the beam.
"""
struct FrustumGeometry <: Geometry
    z_start :: Float64
    z_end   :: Float64
    z_enddet :: Float64
    δx_start :: Float64
    δx_end   :: Float64
    δx_enddet :: Float64
    δy_start :: Float64
    δy_end   :: Float64
    δy_enddet :: Float64
    x0_start :: Float64
    x0_end   :: Float64
    x0_enddet :: Float64
    y0_start :: Float64
    y0_end   :: Float64
    y0_enddet :: Float64
end

function FrustumGeometry(
    z_start::Float64, z_end::Float64, z_enddet::Float64,
    δx_start::Float64, δx_end::Float64, δy_start::Float64, δy_end::Float64,
    x0_start::Float64, x0_end::Float64, y0_start::Float64, y0_end::Float64)

    FrustumGeometry(
        z_start, z_end, z_enddet,
        δx_start, δx_end, δx_end,
        δy_start, δy_end, δy_end,
        x0_start, x0_end, x0_end,
        y0_start, y0_end, y0_end,
    )
end

"""
    CenteredFrustumGeometry(...)

A pyramidal frustum centered around the beam axis. See `FrustumGeometry` for details.
"""
function CenteredFrustumGeometry(
    z_start::Float64, z_end::Float64, z_enddet::Float64,
    δx_start::Float64, δx_end::Float64, δy_start::Float64, δy_end::Float64)

    FrustumGeometry(
        z_start, z_end, z_enddet,
        δx_start, δx_end, δy_start, δy_end,
        0., 0., 0., 0.,
    )
end

const SHiP_geometry = CenteredFrustumGeometry(
    110m-6087cm, 110m-6087cm+5056cm, 110m,
    150cm, 500cm, 430cm, 1100cm)

@assert SHiP_geometry.z_end - SHiP_geometry.z_start ≈ 5056cm
@assert SHiP_geometry.z_enddet - SHiP_geometry.z_start ≈ 6087cm

const SHiP_geometry_ECN3 = CenteredFrustumGeometry(
    38m, 38m+50m, 38m+50m+15m,
    1.2m, 4m, 3.9m, 8m)

const SHADOWS_geometry_EoI = FrustumGeometry(
    10m, 30m, 40m,
    2.5m, 2.5m, 2.5m, 2.5m,
    225cm, 225cm, 0cm, 0cm)

const SHADOWS_geometry_LoI = FrustumGeometry(
    14m, 34m, 46m,
    2.5m, 2.5m, 2.5m, 2.5m,
    225cm, 225cm, 0cm, 0cm)

# SHADOWS but neglecting the length of the detector, for testing
const SHADOWS_geometry_EoI_short = FrustumGeometry(
    10m, 30m, 30m,
    2.5m, 2.5m, 2.5m, 2.5m,
    225cm, 225cm, 0cm, 0cm)

const SHADOWS_geometry_LoI_short = FrustumGeometry(
    14m, 34m, 34m,
    2.5m, 2.5m, 2.5m, 2.5m,
    225cm, 225cm, 0cm, 0cm)

"""
    ProjectiveGeometry(...)

A projective geometry (for testing).

This implements a `FrustumGeometry` spanning a fixed solid angle as seen from the target.
"""
function ProjectiveGeometry(
    z_start::Float64, z_end::Float64, z_enddet::Float64,
    δx_enddet::Float64, δy_enddet::Float64,
    x0_enddet::Float64, y0_enddet::Float64)

    R_start = z_start / z_enddet
    R_end = z_end / z_enddet

    FrustumGeometry(
        z_start, z_end, z_enddet,
        R_start*δx_enddet, R_end*δx_enddet, δx_enddet,
        R_start*δy_enddet, R_end*δy_enddet, δy_enddet,
        R_start*x0_enddet, R_end*x0_enddet, x0_enddet,
        R_start*y0_enddet, R_end*y0_enddet, y0_enddet,
    )
end

"""
    vertex_in_acceptance(event_record, geom::FrustumGeometry) -> Bool

Test whether the decay vertex is within the specified pyramidal frustum.
"""
function vertex_in_acceptance(evt::EventRecord, geom::FrustumGeometry)
    # s ∈ [0,1], used for linear interpolation
    s = (evt.vertex.z - geom.z_start) / (geom.z_end - geom.z_start)
    δx_at_z = geom.δx_start + s * (geom.δx_end - geom.δx_start)
    δy_at_z = geom.δy_start + s * (geom.δy_end - geom.δy_start)
    x0_at_z = geom.x0_start + s * (geom.x0_end - geom.x0_start)
    y0_at_z = geom.y0_start + s * (geom.y0_end - geom.y0_start)
    (evt.vertex.z > geom.z_start) && (evt.vertex.z < geom.z_end) &&
        (abs(evt.vertex.x-x0_at_z) < δx_at_z/2) && (abs(evt.vertex.y-y0_at_z) < δy_at_z/2)
end

"""
    particle_crosses_detector(particle, origin, geom::FrustumGeometry) -> Bool

Test whether the `particle`, crossing through `origin`, passes through the entire detector with momentum along the beam axis.
"""
function particle_crosses_detector(particle::PointParticle, origin::Vec3, geom::FrustumGeometry)
    p = p_(particle)
    x_end = origin.x + (geom.z_end-origin.z) * p.x/p.z
    y_end = origin.y + (geom.z_end-origin.z) * p.y/p.z
    enters_detector = (abs(x_end-geom.x0_end) < geom.δx_end/2) && (abs(y_end-geom.y0_end) < geom.δy_end/2)
    x_enddet = origin.x + (geom.z_enddet-origin.z) * p.x/p.z
    y_enddet = origin.y + (geom.z_enddet-origin.z) * p.y/p.z
    exits_detector = (abs(x_enddet-geom.x0_enddet) < geom.δx_enddet/2) && (abs(y_enddet-geom.y0_enddet) < geom.δy_enddet/2)
    (p.z > 0) && enters_detector && exits_detector
end

"""
    covering_shell(geom::FrustumGeometry) -> (L_min, L_max)

A shell that completely covers the decay vessel, used for importance sampling.

Returns both the small and large radii of the shell.
"""
function covering_shell(geom::FrustumGeometry)
    @assert geom.z_start >= 0
    δx_min_start = max(abs(geom.x0_start) - geom.δx_start/2, 0)
    δy_min_start = max(abs(geom.y0_start) - geom.δy_start/2, 0)
    # Distance to the closest surface, edge or corner
    L_low = √(geom.z_start^2 + δx_min_start^2 + δy_min_start^2)
    δx_max_end = abs(geom.x0_end) + geom.δx_end/2
    δy_max_end = abs(geom.y0_end) + geom.δy_end/2
    # Distance to the furthest surface, edge or corner
    L_high = √(geom.z_end^2 + δx_max_end^2 + δy_max_end^2)
    L_low, L_high
end

## The geometry used by the MATHUSLA experiment

# The y axis is taken to be upward
"""
    MATHUSLAGeometry(z_start, z_end, y_bottom, y_top, y_topdet, x_width)

Approximates the geometry of the MATHUSLA experiment.

The exact dimensions can be specified using the following parameters:
- `z_start`: the distance from the target to the start of the detector, as projected along the beamline,
- `z_end`: the distance from the target to the end of the detector, as projected along the beamline,
- `y_bottom`: the distance between the target and the bottom plane of the decay volume, along the vertical axis,
- `y_top`: the distance between the target and the top plane of the decay volume, along the vertical axis,
- `y_topdet`: the distance between the target and the top plane of the tracker, along the vertical axis,
- `x_width`: the width of the detector, along the horizontal axis perpendicular to the beamline. The detector is assumed to be horizontally centered with respect to the beamline.
"""
struct MATHUSLAGeometry <: Geometry
    z_start :: Float64
    z_end :: Float64
    y_bottom :: Float64
    y_top :: Float64
    y_topdet :: Float64
    x_width :: Float64
end

const MATHUSLA50_geometry_LoI = MATHUSLAGeometry(
    100m, 100m+50m,
    100m, 120m, 124m,
    50m)

const MATHUSLA100_geometry_LoI = MATHUSLAGeometry(
    100m, 100m+100m,
    100m, 120m, 124m,
    100m)

const MATHUSLA200_geometry_LoI = MATHUSLAGeometry(
    100m, 100m+200m,
    100m, 120m, 124m,
    200m)

const MATHUSLA100_geometry_homepage = MATHUSLAGeometry(
    70m, 70m+100m,
    60m, 85m, 89m,
    100m)

"""
    vertex_in_acceptance(event_record, geom::MATHUSLAGeometry) -> Bool

Test whether the decay vertex is within the MATHUSLA decay volume.
"""
function vertex_in_acceptance(evt::EventRecord, geom::MATHUSLAGeometry)
    (
        (geom.z_start < evt.vertex.z < geom.z_end)
        && (geom.y_bottom < evt.vertex.y < geom.y_top)
        && (abs(evt.vertex.x) < geom.x_width/2)
    )
end

"""
    particle_crosses_detector(particle, origin, geom::MATHUSLAGeometry) -> Bool

Test whether the `particle`, crossing through `origin`, passes through the entire MATHUSLA tracker with an upward momentum.
"""
function particle_crosses_detector(particle::PointParticle, origin::Vec3, geom::MATHUSLAGeometry)
    p = p_(particle)
    # Extrapolate the particle trajectory to the top of the decay vessel / beginning of the detector
    z_top = origin.z + (p.z/p.y) * (geom.y_top - origin.y)
    x_top = origin.x + (p.x/p.y) * (geom.y_top - origin.y)
    enters_detector = (geom.z_start < z_top < geom.z_end) && (abs(x_top) < geom.x_width/2)
    # Extrapolate the particle trajectory to the top of the detector plane
    z_topdet = origin.z + (p.z/p.y) * (geom.y_topdet - origin.y)
    x_topdet = origin.x + (p.x/p.y) * (geom.y_topdet - origin.y)
    exits_detector = (geom.z_start < z_topdet < geom.z_end) && (abs(x_topdet) < geom.x_width/2)
    # Require the particle to fully cross the detector in the upward direction (the y axis is up)
    (p.y > 0) && enters_detector && exits_detector
end

"""
    covering_shell(geom::MATHUSLAGeometry) -> (L_min, L_max)

A shell that completely covers the MATHUSLA decay volume, used for importance sampling.

Returns both the small and large radii of the shell.
"""
function covering_shell(geom::MATHUSLAGeometry)
    @assert geom.z_start > 0
    @assert geom.y_bottom > 0
    # Assumes the detector to be centered around the x axis
    # Distance to the closest lower edge
    L_low = √(geom.z_start^2 + geom.y_bottom^2)
    # Distance to the furthest upper corner
    L_high = √(geom.z_end^2 + geom.y_top^2 + (geom.x_width/2)^2)
    L_low, L_high
end

"""
    decay_product_crosses_detector(particle, event_record, geometry) -> Bool

Tests whether the `particle`, originating from the `event_record`'s decay vertex, fully crosses the detector.
"""
function decay_product_crosses_detector(particle::PointParticle, evt::EventRecord, geom::Geometry)
    particle_crosses_detector(particle, evt.vertex, geom)
end

##################
### Acceptance ###
##################

## Utility functions

"""
    get_charge(pdg_code) -> charge

Returns the particle's charge, given its PDG code.
"""
@memoize function get_charge(pdg_code::Int)
    if pdg_code in FIP_IDs
        0.0
    else
        pythia_pdg.charge(pdg_code)
    end
end

"""
    is_charged(pdg_code) -> Bool

Returns whether a particle is charged, given its PDG code.
"""
function is_charged(pdg_code::Int)
    abs(get_charge(pdg_code)) > 0.
end

"""
    is_jet(pdg_code) -> Bool

Returns whether the particle is a parton than may produce a jet.

!!! warning

    Neither parton shower nor hadronization are currently implemented.
"""
@memoize function is_jet(pdg_code::Int)
    res = abs(pdg_code) in [1,2,3,4,21]
    if res == true
        @info "Neither parton shower nor hadronization is implemented; jets are treated as partons." maxlog=1
    end
    res
end

## Generic acceptance function, used to dispatch to specific implementations

"""
    event_in_acceptance(event_record, geometry, fip_pdg_code, mode::Symbol) -> Bool

Given an `event_record`, detector `geometry` and the `fip_pdg_code`,
determines whether the event passes the acceptance criterion specified by `mode`.

This is a generic function that dispatches to specific implementations depending on the `mode`.

The following values are supported for `mode`:
- `:two_tracks_zero_charge`: two opposite charged tracks, or two jets, or two neutral tracks made of photons and/or K_L0;
- `:two_opposite_charge_tracks`: two opposite charged tracks, or two jets;
- `:two_charged_tracks_anysign`: two charged tracks with any signs;
- `:fip_only`: just checks if the FIP's trajectory goes through the detector.

In all cases, the decay vertex must be located inside the decay volume, and the relevant tracks must fully cross the detector.
"""
function event_in_acceptance(evt::EventRecord, geom::Geometry, fip_id::Int, mode::Symbol)
    if mode == :two_tracks_zero_charge
        acceptance_two_tracks_zero_total_charge_including_jets(evt, geom, fip_id)
    elseif mode == :two_opposite_charge_tracks
        acceptance_vertex_plus_two_opposite_charge_tracks_or_jets(evt, geom, fip_id)
    elseif mode == :two_charged_tracks_anysign
        acceptance_vertex_plus_two_charged_tracks_any_signs(evt, geom, fip_id)
    elseif mode == :fip_only
        acceptance_vertex_plus_intersects_detector(evt, geom, fip_id)
    else
        throw(ArgumentError("Unsupported `mode`, choose from [:fip_only, :two_charged_tracks_anysign, :two_opposite_charge_tracks, :two_tracks_zero_charge]."))
    end
end

## Implementations of the specific acceptance criteria

function acceptance_vertex_plus_two_charged_tracks_any_signs(evt::EventRecord, geom::Geometry, fip_id::Int)
    if !vertex_in_acceptance(evt, geom)
        return false
    end
    N_charged_tracks_in_acceptance = 0
    fip_index = only(i for i in 1:length(evt.record) if evt.record[i].field.id == fip_id)
    for i in fip_index+1:length(evt.record)
        particle = evt.record[i]
        if evt.is_live[i] && is_charged(particle.field.id) && decay_product_crosses_detector(particle, evt, geom)
            N_charged_tracks_in_acceptance += 1
        end
    end
    if N_charged_tracks_in_acceptance < 2
        return false
    end
    true
end

function acceptance_vertex_plus_two_opposite_charge_tracks_or_jets(evt::EventRecord, geom::Geometry, fip_id::Int)
    if !vertex_in_acceptance(evt, geom)
        return false
    end
    N_positively_charged_tracks_in_acceptance = 0
    N_negatively_charged_tracks_in_acceptance = 0
    N_jets_in_acceptance = 0
    fip_index = only(i for i in 1:length(evt.record) if evt.record[i].field.id == fip_id)
    for i in fip_index+1:length(evt.record)
        particle = evt.record[i]
        if evt.is_live[i] && decay_product_crosses_detector(particle, evt, geom)
            if is_jet(particle.field.id)
                N_jets_in_acceptance += 1
            elseif is_charged(particle.field.id)
                charge = get_charge(particle.field.id)
                @assert charge != 0.
                if charge > 0
                    N_positively_charged_tracks_in_acceptance += 1
                else
                    N_negatively_charged_tracks_in_acceptance += 1
                end
            end
        end
    end
    (
        (N_jets_in_acceptance >= 2) ||
        ((N_positively_charged_tracks_in_acceptance >= 1) && (N_negatively_charged_tracks_in_acceptance >= 1))
    )
end

function acceptance_vertex_plus_intersects_detector(evt::EventRecord, geom::Geometry, fip_id::Int)
    fip = only(p for p in evt.record if p.field.id == fip_id)
    vertex_in_acceptance(evt, geom) && particle_crosses_detector(fip, zero(Vec3), geom)
end

function acceptance_two_tracks_zero_total_charge_including_jets(evt::EventRecord, geom::Geometry, fip_id::Int)
    if !vertex_in_acceptance(evt, geom)
        return false
    end
    N_neutral_tracks_in_acceptance = 0
    N_positively_charged_tracks_in_acceptance = 0
    N_negatively_charged_tracks_in_acceptance = 0
    N_jets_in_acceptance = 0
    fip_index = only(i for i in 1:length(evt.record) if evt.record[i].field.id == fip_id)
    for i in fip_index+1:length(evt.record)
        particle = evt.record[i]
        if evt.is_live[i] && decay_product_crosses_detector(particle, evt, geom)
            if is_jet(particle.field.id)
                N_jets_in_acceptance += 1
            elseif particle.field.id in [22, 130]
                N_neutral_tracks_in_acceptance += 1
            elseif is_charged(particle.field.id)
                charge = get_charge(particle.field.id)
                @assert charge != 0.
                if charge > 0
                    N_positively_charged_tracks_in_acceptance += 1
                else
                    N_negatively_charged_tracks_in_acceptance += 1
                end
            end
        end
    end
    (
        (N_jets_in_acceptance >= 2) ||
        (N_neutral_tracks_in_acceptance >= 2) ||
        ((N_positively_charged_tracks_in_acceptance >= 1) && (N_negatively_charged_tracks_in_acceptance >= 1))
    )
end
