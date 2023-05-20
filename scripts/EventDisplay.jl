import Makie, GLMakie
import Base: display, convert

#####################
### Event display ###
#####################

## Utility functions and types

"""
    rotate_coordinates(point)

Maps the coordinates from the system where the beam axis z is horizontal and y is vertical
to a system where z is up, which is more appropriate for 3d plotting.

**Note:** although the rotation has determinant +1, it flips the sign of the x axis.
"""
function rotate_coordinates(p::Union{Makie.Point3f0,Makie.Vec3f0})
    typeof(p)(-p[1], p[3], p[2])
end

convert(::Type{Makie.Point3f0}, p::Vec3) = Makie.Point3f0(p.x, p.y, p.z)
convert(::Type{Makie.Vec3f0}, p::Vec3) = Makie.Vec3f0(p.x, p.y, p.z)

## Plot the detector

"""
    plot!(scene, geometry::FrustumGeometry)

Plot a `FrustumGeometry` in the event display.
"""
function plot!(scene::Makie.LScene, g::FrustumGeometry; kwargs...)
    vtx_start = [Vec3(g.x0_start-g.δx_start/2, g.y0_start-g.δy_start/2, g.z_start),
                 Vec3(g.x0_start-g.δx_start/2, g.y0_start+g.δy_start/2, g.z_start),
                 Vec3(g.x0_start+g.δx_start/2, g.y0_start+g.δy_start/2, g.z_start),
                 Vec3(g.x0_start+g.δx_start/2, g.y0_start-g.δy_start/2, g.z_start)]
    vtx_end = [Vec3(g.x0_end-g.δx_end/2, g.y0_end-g.δy_end/2, g.z_end),
               Vec3(g.x0_end-g.δx_end/2, g.y0_end+g.δy_end/2, g.z_end),
               Vec3(g.x0_end+g.δx_end/2, g.y0_end+g.δy_end/2, g.z_end),
               Vec3(g.x0_end+g.δx_end/2, g.y0_end-g.δy_end/2, g.z_end)]
    vtx_enddet = [Vec3(g.x0_enddet-g.δx_enddet/2, g.y0_enddet-g.δy_enddet/2, g.z_enddet),
                  Vec3(g.x0_enddet-g.δx_enddet/2, g.y0_enddet+g.δy_enddet/2, g.z_enddet),
                  Vec3(g.x0_enddet+g.δx_enddet/2, g.y0_enddet+g.δy_enddet/2, g.z_enddet),
                  Vec3(g.x0_enddet+g.δx_enddet/2, g.y0_enddet-g.δy_enddet/2, g.z_enddet)]
    segments = Vec3{Float32}[]
    for i in 1:length(vtx_start)
        push!(segments, vtx_start[i], vtx_start[mod1(i+1, end)])
    end
    for i in 1:length(vtx_end)
        push!(segments, vtx_end[i], vtx_end[mod1(i+1, end)])
    end
    for i in 1:length(vtx_enddet)
        push!(segments, vtx_enddet[i], vtx_enddet[mod1(i+1, end)])
    end
    @assert length(vtx_start) == length(vtx_end) == length(vtx_enddet)
    for i in 1:length(vtx_start)
        push!(segments, vtx_start[i], vtx_end[i], vtx_end[i], vtx_enddet[i])
    end
    segments /= m
    mk_segments = [rotate_coordinates(convert(Makie.Point3f0, p)) for p in segments]
    Makie.linesegments!(scene, mk_segments, linewidth=2, color=:white)
    center = mean([vtx_start; vtx_enddet]) / m
    cam = Makie.cameracontrols(scene.scene)
    cam.lookat[] = convert(Makie.Vec3f0, center) |> rotate_coordinates
    cam.eyeposition[] = Makie.Vec3f0(-30., 5., center.z+20.) |> rotate_coordinates
end

"""
    plot!(scene, geometry::MATHUSLAGeometry)

Plot a `MATHUSLAGeometry` in the event display.
"""
function plot!(scene::Makie.LScene, g::MATHUSLAGeometry; kwargs...)
    vtx_bottom = [Vec3(-g.x_width/2, g.y_bottom, g.z_start),
                  Vec3(-g.x_width/2, g.y_bottom, g.z_end  ),
                  Vec3(+g.x_width/2, g.y_bottom, g.z_end  ),
                  Vec3(+g.x_width/2, g.y_bottom, g.z_start)]
    vtx_top = [vtx + Vec3(0., g.y_top - g.y_bottom, 0.) for vtx in vtx_bottom]
    vtx_topdet = [vtx + Vec3(0., g.y_topdet - g.y_bottom, 0.) for vtx in vtx_bottom]
    segments = Vec3{Float32}[]
    for i in 1:length(vtx_bottom)
        push!(segments, vtx_bottom[i], vtx_bottom[mod1(i+1, end)])
    end
    for i in 1:length(vtx_top)
        push!(segments, vtx_top[i], vtx_top[mod1(i+1, end)])
    end
    for i in 1:length(vtx_topdet)
        push!(segments, vtx_topdet[i], vtx_topdet[mod1(i+1, end)])
    end
    @assert length(vtx_bottom) == length(vtx_top) == length(vtx_topdet)
    for i in 1:length(vtx_bottom)
        push!(segments, vtx_bottom[i], vtx_top[i], vtx_top[i], vtx_topdet[i])
    end
    segments /= m
    mk_segments = [rotate_coordinates(convert(Makie.Point3f0, p)) for p in segments]
    Makie.linesegments!(scene, mk_segments, linewidth=2, color=:white)
    center = mean([vtx_bottom; vtx_topdet]) / m
    cam = Makie.cameracontrols(scene.scene)
    cam.lookat[] = convert(Makie.Vec3f0, center) |> rotate_coordinates
    cam.eyeposition[] = Makie.Vec3f0(-50., center.y+30., center.z-25.) |> rotate_coordinates
end

## Event display

struct EventDisplay{Figure,LScene,Axis,CameraControls,Color,Geom <: Geometry}
    sim :: Simulation
    evt :: EventRecord
    geom :: Geom
    fig :: Figure
    lscene :: LScene
    ax :: Axis
    cameracontrols :: CameraControls
    z_max :: Float64
    vertices :: Makie.Observable{Vector{Makie.Point3f0}}
    segments :: Makie.Observable{Vector{Makie.Point3f0}}
    colors :: Makie.Observable{Vector{Color}}
    clear :: Bool
    colorscheme :: Function
end

"""
    uniform_color_scheme(color::Symbol) -> function

A uniform color scheme, used to plot all particle trajectories using the same color.
"""
function uniform_color_scheme(color)
    (particle::PointParticle, ::EventDisplay) -> color
end

"""
    EventDisplay(
        simulation, geometry,
        resolution=(1600,800), clear=true,
        colorscheme=uniform_color_scheme(:white), z_max_meters=150.,
        show_decay_vertex=false, show_trajectories=true
    )

A 3d event display, used to show an event along with an outline of the detector.

Arguments:
- `simulation`: the `Simulation` object used to simulate the FIP;
- `geometry`: the detector `Geometry`, used for displaying the detector and computing the acceptance;
- `resolution`: the resolution (in pixels) of the 3d viewer;
- `clear`: whether to clear the previous event when displaying a new one;
- `colorscheme`: a function used to decide the color of each particle's trajectory, using e.g. its properties
   (currently supported values are `uniform_color_scheme(color)`, `color_by_charge` and `color_by_acceptance`);
- `z_max_meters`: up to which z coordinate to show the particle trajectories;
- `show_decay_vertex`: whether to highlight the FIP decay vertex;
- `show_trajectories`: whether to display the particle trajectories.

!!! tip

    The event display can be useful for validating the detector geometry, or the implementation of an acceptance criterion.
"""
function EventDisplay(sim::Simulation, geom::Geometry; resolution=(1600,800), clear=true,
                      colorscheme=uniform_color_scheme(:white), z_max_meters=150.,
                      show_decay_vertex=false, show_trajectories=true)
    Makie.set_theme!(backgroundcolor = :black)
    fig = Makie.Figure(resolution=resolution)
    lscene = fig[1,1] = Makie.LScene(fig)
    plot!(lscene, geom)
    cam = Makie.cameracontrols(lscene.scene)
    ax = lscene.scene[Makie.OldAxis]
    ax[:ticks][:textcolor] = :white
    segments = Makie.Observable(zeros(Makie.Point3f0,2))
    vertices = Makie.Observable([zero(Makie.Point3f0)])
    colors = Makie.Observable([:white])
    if show_decay_vertex
        Makie.scatter!(lscene, vertices, color=:white, markersize=100, overdraw=true)
    end
    if show_trajectories
        Makie.linesegments!(lscene, segments, linewidth=2, color=colors, overdraw=true)
    end
    EventDisplay(sim, EventRecord(), geom, fig, lscene, ax, Makie.cameracontrols(lscene.scene),
                 z_max_meters * m, vertices, segments, colors, clear, colorscheme)
end

display(ed::EventDisplay) = display(ed.fig)

## Display an event

"""
    plot!(event_display)

Plot in the `EventDisplay` the event currently held in its `evt::EventRecord` field.
"""
function plot!(ed::EventDisplay)
    fip_index = only(i for i in 1:length(ed.evt.record) if ed.evt.record[i].field == ed.sim.fip)
    fip_segment = [rotate_coordinates(convert(Makie.Point3f0, p)) for p in [zero(Vec3), ed.evt.vertex / m]]
    push!(ed.vertices.val, rotate_coordinates(convert(Makie.Point3f0, ed.evt.vertex / m)))
    append!(ed.segments.val, fip_segment)
    push!(ed.colors.val, ed.colorscheme(ed.evt.record[fip_index], ed))
    if ed.evt.vertex.z >= ed.z_max
        @warn "The FIP decayed too far, and the event may not display properly. Enable importance sampling and set L_max_meters < z_max_meters in Simulation to avoid this issue." maxlog=1
    end
    for i in fip_index+1:length(ed.evt.record)
        if ed.evt.is_live[i]
            p = P_(ed.evt.record[i])
            segment = [rotate_coordinates(convert(Makie.Point3f0, p))
                       for p in [ed.evt.vertex, ed.evt.vertex + Vec3(p)/abs(p.z) * abs(ed.z_max - ed.evt.vertex.z)] / m]
            append!(ed.segments.val, segment)
            push!(ed.colors.val, ed.colorscheme(ed.evt.record[i], ed))
        end
    end
    notify(ed.vertices)
    notify(ed.segments)
    notify(ed.colors)
end

"""
    sample_event!(event_display::EventDisplay)

Randomly sample a new event from the `Simulation` and plot it in the `EventDisplay`.

!!! warning

    The event weight is discarded in this process, therefore the frequency of plotted events will not match their physical frequency.

    In order to plot events with the right physical frequency, use an `UnweightedEventDisplay` instead.
"""
function sample_event!(ed::EventDisplay)
    success = false
    while !success
        success = sample_event!(ed.evt, ed.sim)
    end
    if ed.clear
        empty!(ed.vertices.val)
        empty!(ed.segments.val)
        empty!(ed.colors.val)
    end
    plot!(ed)
end

"""
    color_by_charge(particle, event_display)

A color scheme that colors a particle's track according to its charge:
- positively charged: red,
- negatively charged: cyan,
- neutral but visible: yellow,
- invisible (neutrinos & FIPs): grey.
"""
function color_by_charge(particle::PointParticle, ::EventDisplay)
    id = particle.field.id
    charge = get_charge(id)
    if abs(id) in vcat([12,14,16], FIP_IDs)
        :grey
    elseif charge > 0
        :red
    elseif charge < 0
        :cyan
    else
        :yellow
    end
end

"""
    color_by_acceptance(particle, event_display)

A color scheme that colors a particle's track according to whether it is accepted or not.

Legend:
- Accepted:
  - Charged particles -> green
  - Jets & (meta-)stable neutral particles (γ, K_L0) -> yellow
  - Other neutral particles -> white
- Rejected:
  - Charged particles & jets & (meta-)stable neutral particles (γ, K_L0) -> red
  - Other neutral particles -> grey
- FIPs -> white
"""
function color_by_acceptance(particle::PointParticle, ed::EventDisplay)
    if !vertex_in_acceptance(ed.evt, ed.geom)
        :orange
    else
        id = particle.field.id
        charge = get_charge(id)
        if id in FIP_IDs
            :white
        elseif decay_product_crosses_detector(particle, ed.evt, ed.geom)
            if is_jet(id) || (id in [22, 130])
                :yellow
            elseif charge != 0
                :chartreuse
            else
                :white
            end
        else
            if (charge != 0) || is_jet(id) || (id in [22, 130])
                :red
            else
                :grey
            end
        end
    end
end

## Unweight the events using rejection sampling before displaying them

struct UnweightedEventDisplay
    event_display :: EventDisplay
    max_weight :: Float64
    # Allows computing the final weights using the number of events requested, instead of generated
    success_rate :: Float64
end

"""
    UnweightedEventDisplay(simulation, geometry, kwargs...)

A wrapper for `EventDisplay` that takes event weights into account and performs rejection sampling when sampling events,
in order to plot them with a frequency proportional to their true physical frequency.

All arguments are the same as `EventDisplay` (except for the addition of the number of `burn_in_samples` used by rejection sampling).
"""
function UnweightedEventDisplay(sim::Simulation, geom::Geometry;
                                burn_in_samples::Int=1000, kwargs...)
    ed = EventDisplay(sim, geom; kwargs...)
    max_weight = -Inf
    N_generated = 0
    for k in 1:burn_in_samples
        success = false
        while !success
            success = sample_event!(ed.evt, ed.sim)
            N_generated += 1
        end
        max_weight = max(max_weight, ed.evt.weight)
    end
    success_rate = burn_in_samples / N_generated
    @assert success_rate <= 1
    UnweightedEventDisplay(ed, max_weight, success_rate)
end

display(uw::UnweightedEventDisplay) = display(uw.event_display)

"""
    sample_event!(event_display::UnweightedEventDisplay)

Randomly sample a new event from the `Simulation` and plot it in the event display.

This version takes event weights into account in order to plot them with the correct frequency.
"""
function sample_event!(uw::UnweightedEventDisplay)
    ed = uw.event_display
    # Use rejection sampling
    success = false
    while !success
        success = sample_event!(ed.evt, ed.sim) && ed.evt.weight > rand() * uw.max_weight
    end
    if ed.clear
        empty!(ed.vertices.val)
        empty!(ed.segments.val)
        empty!(ed.colors.val)
    end
    plot!(ed)
end
