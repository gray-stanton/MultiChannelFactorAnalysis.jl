

function simChannel(H, G, fc, fu, edist::Distribution)
    if length(edist) == 1
        # Univariate dist, sample iid
        errs = rand(edist, size(H)[1])
    elseif length(edist) == size(H)[1]
        # Multivariate dist, sample once
        errs = rand(edist, 1)
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


function simData(N, Hs, Gs, edists)
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
