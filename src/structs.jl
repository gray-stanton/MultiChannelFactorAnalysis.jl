struct MultiChannelLayout
    nobs_per_channel :: Array{Int, 1}
    nchannels :: Int
    nobs :: Int
    # Inner constructor to validate fields
    MultiChannelLayout(nobs_per_channel) =
        any([p <= 0 for p in nobs_per_channel]) ? throw(ArgumentError("Must have non-empty channels")) :
             new(nobs_per_channel, size(nobs_per_channel)[1], sum(nobs_per_channel))
end



struct MultiChannelData
    values :: Array{Array{Array{Float64, 1}}}
    channellayout :: MultiChannelLayout
    nsamples :: Int
    function check_conds(values, channellayout, nsamples)
        if nsamples <= 0
            throw(ArgumentError("Must have non-zero samples"))
        elseif size(values)[1] != nsamples
            throw(ArgumentError("Values don't match number of samples"))
        elseif any([size(samp)[1] != channellayout.nchannels for samp in values])
            throw(ArgumentError("Values don't match number of channels"))
        elseif any([any([size(c)[1] != nchan for (c, nchan) in zip(samp, channellayout.nobs_per_channel)]) for samp in values])
            throw(ArgumentError("Values don't match number of observations per channel"))
        else
            new(values, channellayout, nsamples)
        end
    end
    MultiChannelData(values, channellayout, nsamples) = check_conds(values, channellayout, nsamples)
end

struct MultiChannelFactorLayout
    nchannelshared :: Int # Number of channel-shared factors
    nspecific_per_channel :: Array{Int, 1}  # Number of channel-specific factors for each chan.
    nchannels :: Int # Number of channels
    nchannelspecific :: Int # Total channel-specific factors
    nfactors :: Int # Total number of factors
    MultiChannelFactorLayout(nchannelshared, nspecific_per_channel) = new(
        nchannelshared,
        nspecific_per_channel,
        size(nspecific_per_channel)[1],
        sum(nspecific_per_channel),
        nchannelshared + sum(nspecific_per_channel),
        )
end

struct MultiChannelFactors
    factorlayout :: MultiChannelFactorLayout
    fcommon :: Matrix{Float64}
    funique :: Matrix{Float64}
    funique_by_channel :: Array{Matrix{Float64}, 1}
    factors :: Matrix{Float64}
    function check_conds(factorlayout, fcommon, funique)
        funique_by_channel = extract_horizontal_blocks(funique, factorlayout.nspecific_per_channel; get_view=false)
        factors = vcat(fcommon, funique)
        if size(fcommon)[1] != factorlayout.nchannelshared
            throw(ArgumentError("Number of laid-out common factors does not match supplied"))
        elseif size(funique)[1] != factorlayout.nchannelspecific
            throw(ArgumentError("Number of laid-out unique factors does not match supplied"))
        elseif size(fcommon)[2] != size(funique)[2]
            throw(ArgumentError("Length of common and unique factors do not match"))
        else
            new(
            factorlayout,
            fcommon,
            funique,
            funique_by_channel,
            factors
            )
        end
    end
    MultiChannelFactors(factorlayout, fcommon, funique) = check_conds(factorlayout, fcommon, funique)
end

struct MCFMParams
    factorlayout :: MultiChannelFactorLayout
    channellayout :: MultiChannelLayout
    H :: Matrix{Float64}
    G :: Matrix{Float64}
    Σ :: Matrix{Float64}
    H_by_channel :: Array{Matrix{Float64},1}
    G_by_channel :: Array{Matrix{Float64},1}
    Σ_by_channel :: Array{Matrix{Float64},1}
    loading :: Matrix{Float64}
    function check_conds(factorlayout, channellayout, H, G, Σ)
        #TODO: Validate shapes of H/G/Sig by factorlayout
        H_by_channel = extract_horizontal_blocks(H, channellayout.nobs_per_channel; get_view=false)
        G_by_channel = extract_diagonal_blocks(G, channellayout.nobs_per_channel, factorlayout.nspecific_per_channel; get_view=false)
        Σ_by_channel = extract_diagonal_blocks(Σ, channellayout.nobs_per_channel, channellayout.nobs_per_channel; get_view=false)
        loading = hcat(H, G)
        new(factorlayout, channellayout, H, G, Σ, H_by_channel, G_by_channel, Σ_by_channel, loading)
    end
    MCFMParams(factorlayout, channellayout, H, G, Σ) = check_conds(factorlayout, channellayout, H, G, Σ)
end

struct MCFMHistory
    niter :: Int
    maxiter :: Int
    tol :: Float64
    parampath :: Array{MCFMParams, 1}
end


struct MCFMFit
    finalparams :: MCFMParams
    history :: MCFMHistory
end
