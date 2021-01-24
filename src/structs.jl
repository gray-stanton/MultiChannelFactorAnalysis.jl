struct MultiChannelData
    values :: Array{Array{Array{Float64, 1}}}
    nchannels :: Int
    nobs_per_channel :: Array{Float64, 1}
    nsamples :: Int
    if nchannels <= 0 || nsamples <= 0 || any([p <= 0 for p in nobs_per_channel])
        throw(ArgumentError("Must have non-zero channels/samples/observations"))
    elseif size(values)[1] != nsamples
        throw(ArgumentError("Values don't match number of samples"))
    elseif any([size(samp)[1] != nchannels for samp in values])
        throw(ArgumentError("Values don't match number of channels"))
    elseif any([any([size(c)[1] != nchan for c, nchan in zip(samp, nobs_per_channel)]) for samp in values])
        throw(ArgumentError("Values don't match number of observations per channel"))
    else
        MultiChannelData(values, nchannels, nobs_per_channel, nsamples) = new(values, nchannels, nobs_per_channel, nsamples)
    end
end

struct MultiChannelFactorLayout
    nchannels :: Int
    nchannelshared :: Int
    nchannelspecific :: Array{Int, 1}
end

 
