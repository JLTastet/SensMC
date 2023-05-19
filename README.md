# SensMC

A customizable Monte-Carlo event generator written in Julia, used to validate SensCalc.

In its current version, it can simulate a dark scalar mixed with the Standard Model Higgs at the SHiP, SHADOWS and MATHUSLA experiments.

## Installation

### Cloning the repository

You will need to clone both this repository and its submodule `scalar_portal`:
```
$ git clone --recurse-submodules https://github.com/JLTastet/SensMC.git
$ cd SensMC/
```
⚠️ The `--recurse-submodules` option is necessary in order for git to also clone the `scalar_portal` package.

### Setting up the environment

Then, install the Python dependencies, optionally within a virtual environment (recommended):
```
$ pyenv virtualenv 3.11.2 senscalc
$ pyenv activate senscalc
$ pip install -r requirements.txt
```

Install Julia if needed (tested with version `1.8.5`), and instantiate the project:
```
$ julia --project=. -q  # Start julia using the project found in the current directory
(SensMC) pkg> instantiate
```
This will install all the necessary dependencies and precompile them.

⚠️ **Important:** for the code to work, you must activate the `SensMC` Julia environment whenever you start `julia`. This can be done either by passing the `--project=.` option on the command line, or at the REPL prompt by pressing `]` to go into "pkg" mode and then entering `activate .`. If everything worked, the prompt in "pkg" mode should change to `(SensMC) pkg>`.

In case you encounter a `TaskFailedException`, `LoadError: PyError` or `ModuleNotFoundError` when loading code, it most likely means that something is wrong with how the environment is set up. Please refer to the "Troubleshooting" section below.

### External resources

In order to sample heavy mesons produced in the beam dump or at the LHC, this software depends on some external resources.

#### Cascade files for the beam dump facility

Those files are currently not publicly accessible, but they may be requested from the SHiP collaboration.
The files are the following, and should be placed in the folder `data/FairShip`:
* Charmed mesons: `Cascade-parp16-MSTP82-1-MSEL4-978Bpot.root`
  (MD5 = 402535074194f9c4bec79fe6333c95af)
* Beauty mesons: `Cascade-run0-19-parp16-MSTP82-1-MSEL5-5338Bpot.root`
  (MD5 = 29140ca106955a75a113c0ff569117f8)

#### Heavy meson spectrums from FORESEE for the LHC (included!)

Those files can be found in the [FORESEE GitHub repository](https://github.com/KlingFelix/FORESEE/tree/main/files/hadrons/14TeV/Pythia8).
For convenience, and thanks to the permissive license of FORESEE, we already include these files in `data/FORESEE_spectrums`, but please remember to cite the relevant [references](https://github.com/KlingFelix/FORESEE/tree/main#references) if you use them.

## Quick start guide

First, load the core library using the `IncludeAll.jl` file:
```julia
julia> include("lib/IncludeAll.jl");
```

The scripts implementing the FIP model, the sensitivity computation and the event display are found in the `scripts` subdirectory.

### `scripts/GenerateEvents.jl`

This script implements a large part of the actual simulation (minus the phase space sampling and the branching ratios calculations).

You will almost always want to load it:
```julia
julia> include("scripts/GenerateEvents.jl");
```

To start generating events, instantiate a `Simulation` object with the right parameters and create an `EventRecord`:
```julia
help?> make_scalar_portal_simulation
julia> sim = make_scalar_portal_simulation(
           4.0GeV, 1e-6, 0., 3*ab^-1, :foresee_beauty, ["B -> S pi", "B -> S K?"], 200.;
           meson_spectrums_root="data/FORESEE_spectrums", L_min_meters=90.);
julia> evr = EventRecord();
julia> sample_event!(evr, sim)
```
If `sample_event!` returns `false`, try sampling a new event until it returns `true`.
The final state particles can then be extracted as follows:
```julia
julia> final_state_particles = [evr.record[i] for i in 1:length(evr.record) if evr.parent_index[i] > 1 && evr.is_live[i] && evr.is_final[i]]
```
while `evr.weight` holds the event weight, and `evr.counter` the number of events generated so far.

#### Note: importance sampling

SensMC is a weight-based Monte-Carlo generator that heavily relies on importance sampling.
As such, the expectation value of the weight of a generated event (including "failed" events that return `false` and have zero weight) is equal to the number of physical events (_within_ the domain of the importance distribution).
The number of events can therefore be estimated as the sum of event weights divided by the number of generated events (as recorded by `evr.counter`).

Disabling (temporarily) importance sampling for the FIP decay vertex (but not for the branching ratios), we can obtain the total number of dark scalars produced:
```julia
sim = make_scalar_portal_simulation(
    4.0GeV, 1e-6, 0., 3*ab^-1, :foresee_beauty, ["B -> S pi", "B -> S K?"], Inf; # L_max must be set to infinity when the decay vertex importance sampling is disabled
    meson_spectrums_root="data/FORESEE_spectrums", importance_sampling=false);

evr = EventRecord()
N_events = 100000

w = zeros(N_events)
for k in 1:length(w)
   while !(sample_event!(evr, sim)) end # Skip failed events
   w[k] = evr.weight
end

@show evr.counter # This will usually be larger than N_events
w ./= evr.counter
print("Total number of dark scalars produced: ", sum(w))
```

Similarly, the physical number of events that pass some acceptance criterion can be estimated as the weighted average of the "accepted" boolean variable.

Because we will always require the FIP to decay within the decay volume, the acceptance is uniformly zero for events that decay outside of it.
It is therefore wasteful to generate those events in the first place, and we can use importance sampling for the position of the decay vertex in order to only generate events which have a chance of being inside the decay volume (with an appropriate reweighting). 
This is done by specifying in the laboratory frame a distance interval (L_min, L_max) from the target, in which the decay vertex must be located. This interval must fully cover the decay volume (see `covering_shell` for more information).

For simplicity, let’s just require here that the FIP decays within the decay volume, with a trajectory that intersects the detector:
```julia
sim = make_scalar_portal_simulation(
    4.0GeV, 1e-6, 0., 3*ab^-1, :foresee_beauty, ["B -> S pi", "B -> S K?"], 200.; # L_max = 200 m (mandatory parameter)
    meson_spectrums_root="data/FORESEE_spectrums", L_min_meters=90.); # L_min = 90 m (zero by default)
geometry = MATHUSLA100_geometry_homepage # The geometry at https://mathusla-experiment.web.cern.ch/ (as of 2023-05-18)

evr = EventRecord()
w = zeros(N_events)
accepted = zeros(Bool, N_events) # Store whether each even is accepted or rejected
for k in 1:length(w)
   while !(sample_event!(evr, sim)) end # Skip failed events
   w[k] = evr.weight
   fip = only(p for p in evr.record if p.field.id == 9900025) # Match the PDG code of the dark scalar
   accepted[k] = vertex_in_acceptance(evr, geometry) && particle_crosses_detector(fip, zero(Vec3), geometry)
end

w ./= evr.counter
@show sum(w) # This won’t match the total number of events, because we only generate those events which have a chance of being accepted
print("Total number of accepted dark scalars: ", sum(w .* accepted))
```

Here, we have essentially reimplemented the function `compute_yield` from `ScanSensitivity.jl`.
This script contains various functions that abstract away the above technical details and compute the event counts or the sensitivity.

### `scripts/ScanSensitivity.jl`

This script contains various functions for computing the number of events passing some acceptance criterion or the sensitivity of the selected experiment.

For instance, the number of events passing the simplified acceptance criterion that we used above can be easily computed using:
```julia
julia> include("scripts/ScanSensitivity.jl");
julia> compute_scalar_portal_yield(
           4.0GeV, 1e-6, 0., 3*ab^-1, :foresee_beauty,
           ["B -> S pi", "B -> S K?"], MATHUSLA100_geometry_homepage, 1000000
           ; meson_spectrums_root="data/FORESEE_spectrums", acceptance=:fip_only)
```

This script also contains functions for computing the sensitivity at a fixed mass or to compute the full 2d sensitivity curve, using in both cases bisection methods.
The reader is referred to the Jupyter notebooks in the `notebooks` subdirectory for usage examples, as well as to the documentation of `compute_scalar_portal_sensitivity` and `find_scalar_portal_sensitivity_mdbm`.

### `scripts/EventDisplay.jl`

As the name indicates, this script contains objects and functions used to display generated events, along with an outline of the experiment, in an interactive 3d visualization.
This is useful for verifying the implementation of the experiment geometry and of the acceptance criterion.

In order to sample and display events proportionally to their physical frequency, one can use an `UnweightedEventDisplay` object.
Let’s use it to display events from the above simulation, along with the acceptance of the individual decay products of the FIP.

```julia
julia> include("scripts/EventDisplay.jl");
julia> evd = UnweightedEventDisplay(sim, MATHUSLA100_geometry_homepage; z_max_meters=200., colorscheme=color_by_acceptance)
julia> sample_event!(evd) # This could be looped until the event satisfies the acceptance criterion
```

See the documentation of `EventDisplay` for more details and parameters.

## Documentation

This code makes extensive use of docstrings. You can consult them directly from the REPL by pressing the `?` key at the beginning of a line:
```
help?> Simulation
```

## Troubleshooting

### Environment problems

**Typical symptoms:** `TaskFailedException`, `LoadError: PyError`, `ModuleNotFoundError`, ...

If you are using a Python virtual environment (especially if using something different from `pyenv`) or if you have multiple versions of Python installed on your system, it is possible that Julia is not calling the correct one.

First, don't forget to enable your Python virtual environment _before_ starting `julia`, if you are using one.
Then, check that you are also activating the Julia environment, either by passing the `--project=.` option on the command line when you launch `julia`, or at the REPL prompt by pressing `]` to go into "pkg" mode and then entering `activate .`. If everything worked, the prompt in "pkg" mode should change to `(SensMC) pkg>`.

You can finally check if `PyCall` is configured to use the desired Python version by checking whether the outputs of `using PyCall; PyCall.python` and `Sys.which("python")` agree (note that this is a necessary but not sufficient condition for things to work).
If they differ, rebuild `PyCall` against the correct Python version as follows:
```julia
julia> ENV["PYTHON"] = Sys.which("python")
(SensMC) pkg> build PyCall
```
Then restart `julia` to apply the changes.

If you are using an ARM-based Mac, both Julia and Python should be built for the same CPU architecture.
