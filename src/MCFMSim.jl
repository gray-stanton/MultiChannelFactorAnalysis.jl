
function simulate(nsamples :: Int, params :: MCFMParams, factor_func, error_func)
    if nsamples <= 0
        throw(ArgumentError("Number of samples must be positive"))
    end
    factorlayout = params.factorlayout
    channellayout = params.channellayout
    factors = factor_func(nsamples, factorlayout) :: MultiChannelFactors
    #TODO : errors should take Sigma from params
    errors  = error_func(nsamples, channellayout) :: MultiChannelData
    exactvalues  = params.loading * factors.factors
    exactdata  = unstack(exactvalues, channellayout)
    data = exactdata + errors
    return data
end


function periodic_factors(nsamples, factorlayout::MultiChannelFactorLayout)
    freqs = 2*π .* (1:factorlayout.nfactors)
    factors = transpose(hcat([sin.((1:nsamples)/nsamples * f) for f in freqs ]...))
    fcommon = factors[1:factorlayout.nchannelshared, :]
    funique = factors[(factorlayout.nchannelshared+1):end, :]
    out = MultiChannelFactors(factorlayout, fcommon, funique)
    return out
end

function classical_factors(nsamples, factorlayout::MultiChannelFactorLayout)
    # Independent N(0, 1) for all samples is classical
    dist = Normal(0, 1)
    factors = rand(dist, (factorlayout.nfactors, nsamples))
    fcommon = factors[1:factorlayout.nchannelshared, :]
    funique = factors[(factorlayout.nchannelshared+1):end, :]
    out = MultiChannelFactors(factorlayout, fcommon, funique)
    return out
end

function indep_gauss_errors(nsamples, channellayout::MultiChannelLayout; sd=0.1)
    dist = Normal(0, sd)
    errors = unstack(rand(dist, (channellayout.nobs, nsamples)), channellayout)
    return errors
end



function simChannel(H, G, fc, fu, edist::Distribution)
    if length(edist) == 1
        # Univariate dist, sample iid
        errs = rand(edist, size(H)[1])
    elseif length(edist) == size(H)[1]
        # Multivariate dist, sample once
        errs = rand(edist, 1)
        errs = errs[:, 1]
    else
        throw(ArgumentError("Distribution dim invalid"))
    end
    x = H * fc + G * fu +  errs
    return(x)
end

function simFactor(ncommon, nuniques)
    dist = Normal(0, 1) # Classical FA: factors iid N(0, 1)
    fc = rand(dist, ncommon)
    fus = [rand(dist, n) for n in nuniques]
    return fc, fus
end


function simACData(N, Hs, Gs, edists; ρ=0.9)
    if length(Hs) != length(Gs) || length(Hs) != length(edists)
        throw(ArgumentError("Length mismatch"))
    end
    nc = size(Hs[1])[2]
    nus = [size(G)[2] for G in Gs ]
    dat = Array{Array{Array{Float64, 1}, 1}, 1}(undef, N)
    for n in 1:N
        fc, fus = simFactor(nc, nus) # Almost certain we want a new factor draw for each sample
        xs = Array{Array{Float64, 1}, 1}(undef, length(Hs))
        for i in 1:length(Hs)
            x = simChannel(Hs[i], Gs[i], fc, fus[i], edists[i])
            xs[i] = x
        end
        dat[n] = xs
    end
    return(dat)
end

function simData(N, Hs, Gs, edists; factor_func = simFactor)
    if length(Hs) != length(Gs) || length(Hs) != length(edists)
        throw(ArgumentError("Length mismatch"))
    end
    nc = size(Hs[1])[2]
    nus = [size(G)[2] for G in Gs ]
    dat = Array{Array{Array{Float64, 1}, 1}, 1}(undef, N)
    for n in 1:N
        fc, fus = simFactor(nc, nus) # Almost certain we want a new factor draw for each sample
        xs = Array{Array{Float64, 1}, 1}(undef, length(Hs))
        for i in 1:length(Hs)
            x = simChannel(Hs[i], Gs[i], fc, fus[i], edists[i])
            xs[i] = x
        end
        dat[n] = xs
    end
    return(dat)
end

function randomLoadingNorm(L, p; target_tr=1)
    top = rand(Normal(0, 1), p, p)
    Q1 = LowerTriangular(top)
    Q2 = rand(Normal(0, 1), (L-p), p)
    Q = vcat(Q1, Q2)
    trace = tr(Q * transpose(Q))
    Q = sqrt(target_tr/trace) * Q
    diag_signs = sign.(diag(Q))
    Q= Q * diagm(diag_signs)
    return Q
end

function randomLoadingUnif(L, p)
    dist = DiscreteUniform(-2, 2)
    Q = rand(dist, (L, p))
    return(Q)
end

function randomLoadings(Ls, nc, nus, loadingfunc)
    nchan = length(Ls)
    Hs = [loadingfunc(L, nc) for L in Ls]
    Gs = [loadingfunc(L, nu) for (L, nu) in zip(Ls, nus)]
    return Hs, Gs
end

function LyuouLoadings1(Ls, nc, nus)
    N = sum(Ls)
    Hs = [ transpose(hcat([[0.8^f * (-1)^(i+f) for f in 1:nc] for i in 1:L]...)) for L in Ls]
    Gs = [ones(L, nu) for (L, nu) in zip(Ls, nus)]
    Gs = [(length(Ls) - i + 1) .* Gs[i] for i in 1:length(Gs)]
    Gs[1] = Gs[1] .+ [i/(2*Ls[1]) for i in 1:Ls[1]]
    Σs = [diagm(repeat([1], L)) for L in Ls ]
    return Hs, Gs, Σs
end


function powerStructLoadings(Ls, nc, nus;commonpow=0.3, uniquepow=0.3, sigmapow=0.4)
    Hs = extract_blocks(randomLoadingNorm(sum(Ls), nc), Ls, nc; diag=false)
    Gs = [randomLoadingNorm(L, nu) for (L, nu) in zip(Ls, nus)]
    Σs = [diagm(repeat([4/(3*L)], L)) for L in Ls ]

    return Hs, Gs, Σs
end
