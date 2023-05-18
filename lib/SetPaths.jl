using PyCall

begin
    local fullpaths = [
        # Add path to each module here
        (@__DIR__),
        joinpath((@__DIR__), "Simulation"),
    ]

    for path in fullpaths
        if !(path in pyimport("sys")."path")
            pushfirst!(PyVector(pyimport("sys")."path"), path)
        end
    end
end;
