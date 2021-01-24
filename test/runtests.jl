using MultiChannelFactorAnalysis: simFactor, randomLoadingNorm, randomLoadings, simData, MultiChannelData
using Test
using CSV

const datadir = joinpath(dirname(@__FILE__), "..", "data")

@testset "MCFMSim" begin
    fc, fus = simFactor(3, [4, 5, 6])
    @test length(fc) == 3
    @test length(fus)== 3
    @test length(fus[1]) == 4
    Ls = [5, 3, 4]
    nc = 3
    nus = [2, 2, 3]
    Hs, Gs = randomLoadings(Ls, nc, nus)
    @test length(Hs) == length(Ls)
    @test size(Hs[1])[2] == nc
    @test sum([size(H)[1] for H in Hs]) == sum(Ls)
end

@testset "MCFM" begin
    Ls = [20, 15, 10]
    N  = 100
    p = 2
    ps = [4, 3, 2]
    lf = (l, p) ->  randomLoadingNorm(l, p; target_tr=0.3)
    Hs, Gs = randomLoadings(Ls, p, ps, lf)
    edists = vcat([repeat([Normal(0, sqrt(0.4 / L))], L) for L in Ls]...)



@testset "MCFMStructs" begin
    c = 3
    n = 10
    nobs_per_chan = [2, 2, 1]
    vals = [[zeros(p) for p in nobs_per_chan] for i in 1:n]
    @test MultiChannelData(vals, c, nobs_per_chan, n).values = vals
    @test_throws ArgumentError MultiChannelData(vals, 0, nobs_per_chan, n)
    @test_throws ArgumentError(vals, c-1, nobs_per_chan, n)
    @test_throws ArgumentError(vals, c, nobs_per_chan .+ 1, n )
    @test_throws ArgumentError(vals, c, nobs_per_chan, n+1)
end
