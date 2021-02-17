
function chooseFactorStructure(X :: Matrix, channel_layout :: MultiChannelLayout)
    n = size(X)[1]
    T = size(X)[2]
    XtX = 1/(n-1)*transpose(X) * X
    S   = 1/(n-1)*X * transpose(X)
    overall_evs = reverse(eigen(XtX.values, dims=1))
    X_by_channel = extract_horizontal_blocks(X, channel_layout.nobs_per_channel)
    channel_evs = [reverse(eigen(transpose(Xc) * Xc).values) for Xc in X_by_channel]
    D = blockdiag(extract_diagonal_blocks(S,   channel_layout.nobs_per_channel, channel_layout.nobs_per_channel))
    U = S - D

    off_diag_evs = reverse(eigen(XtX.values, dims=1)[T-n+1:T])
end






function Hstep(S :: Symmetric, params :: MCFMParams)
    U = params.factorlayout.nchannelspecific # total num of unique factors
    L = params.channellayout.nobs # total number of obs, across all channels.
    G = params.G
    Σ = params.Σ
    p = params.factorlayout.nchannelshared

    E = G * transpose(G) + Σ # eqn 24
    Eisqrt = sqrt(inv(E)) # inefficient
    Esqrt  = sqrt(E)

    Swhite = Eisqrt * S * Eisqrt # eqn 25
    Swhite_eigen = svd(Swhite)
    W = Swhite_eigen.U

    #TODO: Notation in paper is ambigious, double check defn of D.
    ds = max.(Swhite_eigen.S[1:p]  .- 1, zeros(p))
    Dsqrt = vcat(sqrt(diagm(ds)) , zeros((L-p, p)))

    B = Esqrt * W * Dsqrt # before eqn 11
    Q = LTOrthog(B, p)


    Hnew = B * Q
    Hnew = Array{Float64, 2}(Hnew)
    return Hnew
end


function _Gstep_channel(S, H, Σ, pj)
    Lj = size(H)[1]
    P = S - H * transpose(H) # eqn 44
    Σsqrt = sqrt(Σ)
    Σisqrt = sqrt(inv(Σ))
    Pwhite = Σisqrt * P * Σisqrt
    Pwhite_eigen = svd(Pwhite)
    W = Pwhite_eigen.U

    ds = max.(Pwhite_eigen.S[1:pj] .-1, zeros(pj))
    Dsqrt = vcat(sqrt(diagm(ds)) , zeros((Lj-pj, pj))) #TODO: Still questionable.

    B = Σsqrt * W * Dsqrt
    Q = LTOrthog(B, pj)
    Gnew = B * Q
    return Gnew
end

function Gstep(S :: Symmetric, params :: MCFMParams)
    Sblocks = extract_diagonal_blocks(S, params.channellayout.nobs_per_channel,params.channellayout.nobs_per_channel)
    L = params.channellayout.nobs
    Gblocks = Array{Float64, 2}[]
    for i in 1:params.channellayout.nchannels
        if params.factorlayout.nspecific_per_channel[i] == 0
            # If no channel specific factors, don't run Gstep
            Gblock = zeros(params.channellayout.nobs_per_channel[i], 0)
        else
            Gblock = _Gstep_channel(Sblocks[i],
                params.H_by_channel[i],
                params.Σ_by_channel[i],
                params.factorlayout.nspecific_per_channel[i])
            end
        push!(Gblocks, Gblock)
    end
    return blockdiag(Gblocks)
end


function Σstep(S :: Symmetric, params :: MCFMParams)
    H = params.H
    G = params.G
    Est = S - H * transpose(H) - G * transpose(G)
    Σnew = Matrix(Diagonal(Est)) # extract only the diagonal elems.
    return Σnew
end


function fit(data :: MultiChannelData, factorlayout :: MultiChannelFactorLayout; tol=1e-4, maxiter=1e6, center=true)
    # Data is assumed to be in a ragged 3d array structure: sample x channel x observation
    X = stack(data)
    if center
        X = X .- mean(X, dims=2)
    end
    # Initialize parameters
    H =  zeros(data.channellayout.nobs, factorlayout.nchannelshared)
    G =  zeros(data.channellayout.nobs, factorlayout.nchannelspecific)
    Σ =  Matrix(I(data.channellayout.nobs))
    params = MCFMParams(factorlayout, data.channellayout, H, G, Σ)
    fitpath = MCFMHistory(1, maxiter, tol, [params])

    # Compute sample cov
    S = Symmetric(1/(data.nsamples-1) * X * transpose(X))

    iter = 2
    converged = false
    while iter < maxiter && !converged
        H = Hstep(S, params)
        params = MCFMParams(factorlayout, data.channellayout, H, G, Σ)
        G = Gstep(S, params)
        params = MCFMParams(factorlayout, data.channellayout, H, G, Σ)
        Σ = Σstep(S, params)
        params = MCFMParams(factorlayout, data.channellayout, H, G, Σ)
        fitpath = MCFMHistory(iter, maxiter, tol, vcat(fitpath.parampath..., [params]))
        if _checkConverged(fitpath, tol)
            _converged = true
            fitpath
        end
        iter = iter + 1
    end
    finalfit = MCFMFit(params, fitpath)
    return finalfit
end

#
# function fitMFA(data, Ls, p, pjs; tol=1e-4, maxiter=1e6)
#
#     X, Xobs, Xchan = stack_and_view(data)
#     # Remove column means
#     #X = X .- mean(X; dims=2) # centering is wrong.
#     # Initialize parameter matrices
#     N = length(data)
#     L = sum(Ls)
#     U = sum(pjs)
#
#     H = zeros(L, p)
#     H_old = zeros(L, p)
#     G = zeros(L, U)
#     G_old = zeros(L, U)
#     Σ = Matrix(I(L))
#     Σ_old = Matrix(I(L))
#
#
#     # Compute sample cov
#     S = Symmetric(1/(N-1) * X * transpose(X))
#
#     #loop vars
#     k = 0
#     converged = false
#     while k < maxiter && !converged
#         H = Hstep(S, G, Σ, p)
#         Hblocks = extract_blocks(H, Ls, [p]; diag=false)
#         Sblocks = extract_blocks(S, Ls, Ls; diag=true)
#         Σblocks = extract_blocks(Σ, Ls, Ls; diag=true)
#         G = Gstep(Sblocks, Hblocks, Σblocks, pjs)
#         Σ = Σstep(S, H, G)
#         if _checkConverged(H_old, G_old, Σ_old, H, G, Σ; tol=tol)
#             converged = true
#             break
#         end
#         k = k + 1
#         H_old = H
#         G_old = G
#         Σ_old = Σ
#     end
#     # Construct MFA object
#     if !converged
#         @warn "Max iteration {maxiter} reached, model has not converged"
#     end
#     return H, G, Σ
# end

function _checkConverged(fitpath :: MCFMHistory, tol)
    iter = fitpath.niter
    H_old = fitpath.parampath[iter-1].H
    G_old = fitpath.parampath[iter-1].G
    Σ_old = fitpath.parampath[iter-1].Σ
    H_new = fitpath.parampath[iter].H
    G_new = fitpath.parampath[iter].G
    Σ_new = fitpath.parampath[iter].Σ
    Hdiffsize = norm(H_new - H_old)
    Gdiffsize = norm(G_new - G_old)
    Σdiffsize = norm(Σ_new - Σ_old)
    Hconv = Hdiffsize/length(H_new) <= tol
    Gconv = Gdiffsize/length(G_new) <= tol
    Σconv = Σdiffsize/size(Σ_old)[1] <= tol # diagonal
    if Hconv && Gconv && Σconv
        return true
    else
        return false
    end
end
#
# function _checkConverged(H_old, G_old, Σ_old, H_new, G_new, Σ_new; tol=1e-6)
#
#     Hdiffsize = norm(H_new - H_old)
#     Gdiffsize = norm(G_new - G_old)
#     Σdiffsize = norm(Σ_new - Σ_old)
#     Hconv = Hdiffsize/length(H_new) <= tol
#     Gconv = Gdiffsize/length(G_new) <= tol
#     Σconv = Σdiffsize/size(Σ_old)[1] <= tol # diagonal
#     if Hconv && Gconv && Σconv
#         return true
#     else
#         return false
#     end
# end


function project(data::MultiChannelData, factors::MultiChannelFactors)
    # Project each factor onto space spanned by common + specific factors
    X = stack(data)
    Xblks = extract_horizontal_blocks(X, data.channellayout.nobs_per_channel)
    Fblks = [hcat(transpose(factors.fcommon), transpose(fu)) for fu in funique_by_channel]
    Xfitteds = Matrix{Float64}(undef, data.channellayout.nchannels)
    for c in 1:data.channellayout.nchannels
        Ft = transpose(F)
        proj = Fblk[c] * pinv(Ft * F) * Ft
        Xblkfit = proj * Xblks[c]
        Xfitteds[c] = Xblkfit
    end
    Xfitted = vcat(Xfitted...)
    out = unstack(Xfitted, data.channellayout)
    return out
end

function extract_factors(data :: MultiChannelData, params ::MCFMParams; center=true)
    H = params.H
    G = params.G
    Σ = params.Σ
    H_by_channel = params.H_by_channel
    G_by_channel = params.G_by_channel
    Σ_by_channel = params.Σ_by_channel
    X = stack(data)
    if center
        X = X .- mean(X, dims=2)
    end
    R = H * transpose(H) + G * transpose(G) + Σ
    Rinv = inv(R)
    fcommon = transpose(H) * inv(R) * X
    funique = zeros(params.factorlayout.nchannelspecific, data.nsamples)
    Xblks = extract_horizontal_blocks(X, data.channellayout.nobs_per_channel)
    j = 0
    for c in 1:data.channellayout.nchannels
        if params.factorlayout.nspecific_per_channel[c] == 0
            continue
        else
            Ri = H_by_channel[c] * transpose(H_by_channel[c]) + G_by_channel[c] * transpose(G_by_channel[c]) + Σ_by_channel[c]
            fu = transpose(G_by_channel[c]) * inv(Ri) * Xblks[c]
            funique[(j+1):(j+size(fu)[1]), :] = fu
            j += size(fu)[1]
        end
    end
    factors = MultiChannelFactors(params.factorlayout, fcommon, funique)
    return factors
end


# function extract_factors(H, G, Σ, X, Ls, pjs)
#     R = H * transpose(H) + G * transpose(G) + Σ
#     #Gblks = extract_blocks(G, Ls; diag=true)
#     #Σblks = extract_blocks(G, Ls; diag=true)
#     #Xblks = extract_blocks(X, Ls; diag=false)
#     Rinv = inv(R)
#     fcommon = transpose(H) * inv(R) * X
#     Hblks = extract_blocks(H, Ls, 1; diag=false)
#     Gblks = extract_blocks(G, Ls, pjs; diag=true)
#     Σblks = extract_blocks(Σ, Ls, Ls; diag=true)
#     Xblks = extract_blocks(X, Ls, 1; diag=false)
#     funique = zeros(sum(pjs), size(X)[2])
#     j=0
#     for (Hi, Gi, Σi, Xi) in zip(Hblks, Gblks, Σblks, Xblks)
#         Ri = Hi * transpose(Hi) + Gi * transpose(Gi) + Σi
#         fu = transpose(Gi) * inv(Ri) * Xi
#         funique[(j+1):(j+size(fu)[1]), :] = fu
#         j += size(fu)[1]
#     end
#     fcommon = transpose(fcommon)
#     funique = transpose(funique)
#     return fcommon, funique
# end

function predict(factors :: MultiChannelFactors, params :: MCFMParams)
    preds = params.loading * factors.factors
    return preds
end
#
# function predict(H, G, Σ, X, Ls)
#     fcom, funi = extract_factors(H, G, Σ, X, Ls)
#     preds = H * (fcom) + G * transpose(funi)
#     return(preds)
# end
