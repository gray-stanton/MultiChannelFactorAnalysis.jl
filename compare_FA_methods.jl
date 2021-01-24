using CSV
using MultiChannelFactorAnalysis
using DataFrames
using Statistics
using Distributions
using LinearAlgebra
#using Arpack
using ArnoldiMethod

function fitlargeFA(data, Ls, p, pjs)
    X, Xobs, Xchan = MultiChannelFactorAnalysis.stack_and_view(data)
    k = p+sum(pjs)
    K = maximum(size(X))
    decomp, history = partialschur(transpose(X) * X; nev=k, which=LM())
    ls, evecs = partialeigen(decomp)
    evecs = reverse(evecs, dims=2)
    Fhat = sqrt(size(X)[2]) * evecs
    Chat = X * Fhat / size(X)[2]
    H = Chat[:, 1:p]
    G = Chat[:, (p+1):end]
    #Gs = MultiChannelFactorAnalysis.extract_blocks(Chat[:, (p+1):end] , Ls, pjs;diag=true)
    #H = MultiChannelFactorAnalysis.flip_signs(H * MultiChannelFactorAnalysis.LTOrthog(H, p))
    #Gs = [MultiChannelFactorAnalysis.flip_signs(G * MultiChannelFactorAnalysis.LTOrthog(G, size(G)[2])) for G in Gs]
    #G = MultiChannelFactorAnalysis.blockdiag(Gs)

    return H, G
end

function fitIGFM(data, Ls, p, pjs, delta)
    H, G = fitlargeFA(data, Ls, p , pjs)
    N = sum(Ls)
    T = length(data)
    m = length(Ls)
    #delta * log(1/mest * 1/log(p)* time )*(1/sqrt(time) * sqrt(log(p)) + 1/sqrt(p))
    τ = delta * log(1/m *1/log(N)  * T ) * (1/sqrt(T) * sqrt(log(N)) + 1/sqrt(N))
    Gthresh = G
    Gthresh[abs.(Gthresh) .< τ] .= 0
    rbounds = vcat([0], cumsum(Ls))
    cbounds = vcat([0], cumsum(pjs))

    #block_sums = [[ sum(abs.(Gthresh[rstart:rstop, cstart:cstop])) for  (cstart, cstop) in zip(cbounds .+ 1, cbounds[2:end])]
    #        for (rstart, rstop) in zip(rbounds .+ 1, rbounds[2:end])]

    #permutes = [argmax(block_sums[i, :][1]) for i in 1:size(block_sums)[1]]
    #perm_Gthresh = blockdiag()
    return H, Gthresh
end
Ls = [120, 120, 120]
p = 1
pjs = [1, 1, 1]
N=450

function loading_trial(N, Ls, p, pjs; factor_func=MultiChannelFactorAnalysis.simFactor)
    Hs, Gs, Sigs = MultiChannelFactorAnalysis.powerStructLoadings(Ls, p, pjs)
    H = vcat(Hs...)
    G = MultiChannelFactorAnalysis.blockdiag(Gs)
    edists = [ MvNormal(Sig) for (L, Sig) in zip(Ls, Sigs)]
    data = MultiChannelFactorAnalysis.simData(N, Hs, Gs, edists)
    H1, G1, Sig1 = MultiChannelFactorAnalysis.fitMFA(data, Ls, p, pjs; maxiter=20)
    #G1s = MultiChannelFactorAnalysis.extract_blocks(G1, Ls, pjs; diag=true)
    #H1 = MultiChannelFactorAnalysis.flip_signs(H1)
    #G1s = [MultiChannelFactorAnalysis.flip_signs(G * MultiChannelFactorAnalysis.LTOrthog(G, size(G)[2])) for G in G1s]
    #G1 = MultiChannelFactorAnalysis.blockdiag(G1s)
    H2, G2 = fitlargeFA(data, Ls, p, pjs)
    R = H * transpose(H) + G * transpose(G)
    R1 = H1 * transpose(H1) + G1 * transpose(G1)
    R2 = H2 * transpose(H2) + G2 * transpose(G2)
    return norm(R - R1)^2/norm(R)^2, norm(R - R2)^2/norm(R)^2
end

function get_trial_results(times, d)
    Ls = [d, floor(Int32,d/2), floor(Int32, d/4)]
    p = 2
    pjs = repeat([1], 3)
    log1s = []
    log2s = []
    for N in 50:50:450
        print(N)
        r1s = []
        r2s = []
        for i in 1:times
            r1, r2 = loading_trial(N, Ls, p, pjs)
            push!(r1s, log(r1))
            push!(r2s, log(r2))
        end
        push!(log1s, mean(r1s))
        push!(log2s, mean(r2s))
    end
    return (log1s, log2s)
end


ds = [10, 20, 40, 60, 100]
fixed_p_loss3 = []
fixed_p_loss4 = []
for d in ds
    print(d)
    v1, v2 = get_trial_results(times, d)
    push!(fixed_p_loss3, v1)
    push!(fixed_p_loss4, v2)
end

df1 = DataFrame(MCFMlosses = vcat(fixed_p_loss3...),
                IGFMlosses = vcat(fixed_p_loss4...),
                d    =  repeat(ds, inner=[9]),
                N          = repeat(collect(50:50:450), outer=[5])
)

CSV.write("/home/gray/research/ngsim/FactorComparison_pfixed.csv", df1)
3
pnratios = [1/15, 2/15, 3/15, 4/15, 5/15, 10/15, 15/15]
times = 10
fixed_p_loss1 = []
fixed_p_loss2 = []
for rat in pnratios
    print(rat)
    s1, s2 = get_trial_results2(times, rat)
    push!(fixed_p_loss1, s1)
    push!(fixed_p_loss1, s2)
end


df1 = DataFrame(MCFMlosses = vcat(fixed_p_loss1[1], fixed_p_loss1[3], fixed_p_loss1[5], fixed_p_loss1[7], fixed_p_loss1[9]),
                IGFMlosses = vcat(fixed_p_loss1[2], fixed_p_loss1[4], fixed_p_loss1[6],fixed_p_loss1[8], fixed_p_loss1[10]),
                pnratio    = repeat([1/15, 2/15, 3/15, 4/15, 5/15], inner=[9]),
                N          = repeat(collect(50:50:450), outer=[5])
)

CSV.write("/home/gray/research/ngsim/FactorComparison_pn_grow.csv", df1)


function get_trial_results2(times, pnratio)
    p = 1
    pjs = repeat([1], 3)
    log1s = []
    log2s = []
    for N in 50:50:450
        print(N)
        d = floor(Int32, N * pnratio)
        Ls = repeat([d], 3)
        r1s = []
        r2s = []
        for i in 1:times
            r1, r2 = loading_trial2(N, Ls, p, pjs)
            push!(r1s, log(r1))
            push!(r2s, log(r2))
        end
        push!(log1s, mean(r1s))
        push!(log2s, mean(r2s))
    end
    return (log1s, log2s)
end


function get_trial_results3(times, d)
    p = 1
    pjs = repeat([1], 3)
    log1s = []
    log2s = []
    for N in 50:50:450
        print(N)
        Ls = repeat([d], 3)
        r1s = []
        r2s = []
        for i in 1:times
            r1, r2 = loading_trial2(N, Ls, p, pjs)
            push!(r1s, log(r1))
            push!(r2s, log(r2))
        end
        push!(log1s, mean(r1s))
        push!(log2s, mean(r2s))
    end
    return (log1s, log2s)
end


function loading_trial2(N, Ls, p, pjs; factor_func=MultiChannelFactorAnalysis.simFactor)
    Hs, Gs, Sigs = MultiChannelFactorAnalysis.LyuouLoadings1(Ls, p, pjs)
    H = vcat(Hs...)
    G = MultiChannelFactorAnalysis.blockdiag(Gs)
    edists = [ MvNormal(Sig) for (L, Sig) in zip(Ls, Sigs)]
    data = MultiChannelFactorAnalysis.simData(N, Hs, Gs, edists)
    H1, G1, Sig1 = MultiChannelFactorAnalysis.fitMFA(data, Ls, p, pjs; maxiter=15)
    #G1s = MultiChannelFactorAnalysis.extract_blocks(G1, Ls, pjs; diag=true)
    #H1 = MultiChannelFactorAnalysis.flip_signs(H1)
    #G1s = [MultiChannelFactorAnalysis.flip_signs(G * MultiChannelFactorAnalysis.LTOrthog(G, size(G)[2])) for G in G1s]
    #G1 = MultiChannelFactorAnalysis.blockdiag(G1s)
    H2, G2 = fitIGFM(data, Ls, p, pjs, 1)
    R = H * transpose(H) + G * transpose(G)
    R1 = H1 * transpose(H1) + G1 * transpose(G1)
    R2 = H2 * transpose(H2) + G2 * transpose(G2)
    return norm(R - R1)^2/norm(R)^2, norm(R - R2)^2/norm(R)^2
end
