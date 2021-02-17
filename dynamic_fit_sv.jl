using DataFrames
using CSV
using ArgParse
using MultiChannelFactorAnalysis


function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--datafile"
            help = "Path to single_vehicle.csv"
        "--name", "-n"
            help = "Prefix for output files"
            default = "output"
        "--maxmissing"
            help = "radius around focus vehicle to include"
            arg_type = Float64
            default = 0.1
        "--window"
            help = "How many timesteps of history to use"
            default = 50
            arg_type = Int
        "--step"
            help="How many timesteps to step ahead by"
            default=1
            arg_type=Int
        "--tmin"
            help="Which timestep to start at"
            default = -1
            arg_type=Int
        "--tmax"
            help="Which timestep to stop at"
            default = -1
            arg_type=Int
        "--adaptivefac"
            help="Whether to adaptively identify factor structure"
            action= :store_true
        "--nshared"
            help = "Number of common factors"
            arg_type=Int
            default = 1
        "--outfolder"
            help = "Path to write outputs"
            default = "."
        "nspecific"
            nargs = '*'
            arg_type = Int
            help = "Numbers of channel-specific factors"
            default = [1, 1]
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

    tmin = parsed_args["tmin"] == -1 ? minimum(dataset.sample_id) : parsed_args["tmin"]
    tmax = parsed_args["tmin"] == -1 ? maximum(dataset.sample_id) : parsed_args["tmax"]
    windowsize = parsed_args["window"]
    if tmax - tmin < windowsize
        error("$tmax less than $windowsize from $tmin")
    end

    tstart = tmin + windowsize
    params_longs = []
    factors_longs = []
    fitted_longs = []
    for t in tstart:windowsize:tmax
        print("Fitting $t...")
        windowdf = dataset[t - windowsize .< dataset.sample_id .<= t, :]
        windowdata = data_fromlong(windowdf)
        sample_ids = windowdata.sample_ids
        obs_ids = windowdata.obs_ids
        if parsed_args["adaptivefac"]
            error("Adaptive fac not implemented.")
        else
            #nshared = parse(Int, parsed_args[""])
            #nspecific = [parse(Int, p) for p in parsed_args["nspecific"]]
            factorlayout = MultiChannelFactorLayout(parsed_args["nshared"], parsed_args["nspecific"])
        end
        windowfit = fit(windowdata, factorlayout; maxiter=20)
        windowparams = windowfit.finalparams
        windowfactors = extract_factors(windowdata, windowparams)
        windowfitted = predict(windowfactors, windowparams)

        windowparams_long = to_long(windowparams; obs_ids)
        windowfactors_long = to_long(windowfactors; sample_ids)
        windowfitted_long = to_long(MultiChannelFactorAnalysis.unstack(windowfitted, windowdata.channellayout);
            obs_ids=obs_ids, sample_ids=sample_ids)

        windowparams_long[!, :windowtime] .= t
        windowfactors_long[!, :windowtime] .= t
        windowfitted_long[!, :windowtime] .= t
        push!(params_longs, windowparams_long)
        push!(factors_longs, windowfactors_long)
        push!(fitted_longs, windowfitted_long)
    end
    ## combine
    all_params = vcat(params_longs...)
    all_factors = vcat(factors_longs...)
    all_fitted = vcat(fitted_longs...)

    ## output
    CSV.write("$(parsed_args["outfolder"])/$(parsed_args["name"])_params.csv", all_params)
    CSV.write("$(parsed_args["outfolder"])/$(parsed_args["name"])_factors.csv", all_factors)
    CSV.write("$(parsed_args["outfolder"])/$(parsed_args["name"])_fitted.csv", all_fitted)
end

main()
