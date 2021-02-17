using DataFrames
using CSV
using ArgParse



function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--datafile"
            help = "Path to ngsim_point.csv"
        "--vehiclenum", "-v"
            help = "Vehicle UniqueID number to focus on"
            arg_type = Int
            default = 3000
        "--radius"
            help = "radius around focus vehicle to include"
            arg_type = Float64
            default = 100.0
        "--trim"
            help = "Number of timesteps to remove from begin and end of focal dataset"
            default = 50
            arg_type = Int
        "--long"
            help = "Whether or not to tidy the dataframe"
            action= :store_true
        "vars"
            help = "Which data-containing variables to include"
            default = ["Local_X", "Local_Y"]
        "--output"
            help = "Path to write output dataset"
            default = "./output_dataset.csv"
    end

    return parse_args(s)
end


function main()
    parsed_args = parse_commandline()
    println("Parsed args:")
    for (arg,val) in parsed_args
        println("  $arg  =>  $val")
    end
    println("Reading dataset...")
    dataset = DataFrame(CSV.File(parsed_args["datafile"]))
    unique!(dataset, ["timestep", "Unique_Veh_ID"])
    dataset.timestep = round.(Int, dataset.timestep) # stored as floats
    focaldf = filter(:Unique_Veh_ID => x -> x == parsed_args["vehiclenum"], dataset)
    maxtime = maximum(focaldf[!, :timestep])
    mintime = minimum(focaldf[!, :timestep])

    # Select rows
    outdf = filter!(:timestep => x -> mintime <= x <= maxtime, dataset)
    rename!(focaldf, :Global_X => :Focal_X, :Global_Y => :Focal_Y)
    outdf = leftjoin(outdf, focaldf[:, ["timestep", "Focal_X", "Focal_Y"]], on="timestep")
    outdf.dist_to_focal = sqrt.((outdf.Global_X - outdf.Focal_X).^2 + (outdf.Global_Y - outdf.Focal_Y).^2)
    filter!(:dist_to_focal => x -> x <= parsed_args["radius"], outdf)
    # Select columsn
    vars_to_include = copy(parsed_args["vars"])
    pushfirst!(vars_to_include, "timestep")
    pushfirst!(vars_to_include, "Unique_Veh_ID")
    select!(outdf, vars_to_include)
    if parsed_args["long"] == true
        outdf = stack(outdf, parsed_args["vars"], ["timestep", "Unique_Veh_ID"];
                    variable_name="channel", value_name="value")
        rename!(outdf, :timestep => :sample_id, :Unique_Veh_ID => :observation_id)
    end
    CSV.write(parsed_args["output"], outdf)
end

main()
