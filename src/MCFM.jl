

function Hstep(S :: Symmetric, G, Σ, p)
    U = size(G)[2] # total num of unique factors
    L = size(G)[1] # total number of obs, across all channels.

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

function Gstep(Sblocks, Hblocks, Σblocks, pjs)
    L = sum([size(Hblock)[2] for Hblock in Hblocks])
    u = sum(pjs)
    Gblocks = Array{Float64, 2}[]
    for i in 1:length(pjs)
        Gblock = _Gstep_channel(Sblocks[i], Hblocks[i], Σblocks[i], pjs[i])
        push!(Gblocks, Gblock)
    end
    return blockdiag(Gblocks)
end


function Σstep(S, H, G)
    Est = S - H * transpose(H) - G * transpose(G)
    Σnew = Matrix(Diagonal(Est)) # extract only the diagonal elems.
    return Σnew
end



function fitMFA(data, Ls, p, pjs; tol=1e-6, maxiter=1e6)
    # Data is assumed to be in a ragged 3d array structure: sample x channel x observation
    X, Xobs, Xchan = stack_and_view(data)
    # Remove column means
    #X = X .- mean(X; dims=2) # centering is wrong.
    # Initialize parameter matrices
    N = length(data)
    L = sum(Ls)
    U = sum(pjs)

    H = zeros(L, p)
    H_old = zeros(L, p)
    G = zeros(L, U)
    G_old = zeros(L, U)
    Σ = Matrix(I(L))
    Σ_old = Matrix(I(L))


    # Compute sample cov
    S = Symmetric(1/(N-1) * X * transpose(X))

    #loop vars
    k = 0
    converged = false
    while k < maxiter && !converged
        H = Hstep(S, G, Σ, p)
        Hblocks = extract_blocks(H, Ls, [p]; diag=false)
        Sblocks = extract_blocks(S, Ls, Ls; diag=true)
        Σblocks = extract_blocks(Σ, Ls, Ls; diag=true)
        G = Gstep(Sblocks, Hblocks, Σblocks, pjs)
        Σ = Σstep(S, H, G)
        if _checkConverged(H_old, G_old, Σ_old, H, G, Σ; tol=tol)
            converged = true
            break
        end
        k = k + 1
        H_old = H
        G_old = G
        Σ_old = Σ
    end
    # Construct MFA object
    if !converged
        @warn "Max iteration {maxiter} reached, model has not converged"
    end
    return H, G, Σ
end

function _checkConverged(H_old, G_old, Σ_old, H_new, G_new, Σ_new; tol=1e-6)
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

function extract_factors(H, G, Σ, X, Ls, pjs)
    R = H * transpose(H) + G * transpose(G) + Σ
    #Gblks = extract_blocks(G, Ls; diag=true)
    #Σblks = extract_blocks(G, Ls; diag=true)
    #Xblks = extract_blocks(X, Ls; diag=false)
    Rinv = inv(R)
    fcommon = transpose(H) * inv(R) * X
    Hblks = extract_blocks(H, Ls, 1; diag=false)
    Gblks = extract_blocks(G, Ls, pjs; diag=true)
    Σblks = extract_blocks(Σ, Ls, Ls; diag=true)
    Xblks = extract_blocks(X, Ls, 1; diag=false)
    funique = zeros(sum(pjs), size(X)[2])
    j=0
    for (Hi, Gi, Σi, Xi) in zip(Hblks, Gblks, Σblks, Xblks)
        Ri = Hi * transpose(Hi) + Gi * transpose(Gi) + Σi
        fu = transpose(Gi) * inv(Ri) * Xi
        funique[(j+1):(j+size(fu)[1]), :] = fu
        j += size(fu)[1]
    end
    fcommon = transpose(fcommon)
    funique = transpose(funique)
    return fcommon, funique
end

function predict(H, G, Σ, X, Ls)
    fcom, funi = extract_factors(H, G, Σ, X, Ls)
    preds = H * transpose(fcom) + G * transpose(funi)
    return(preds)
end
