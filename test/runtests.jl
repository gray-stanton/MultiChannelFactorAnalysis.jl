using MultiChannelFactorAnalysis
using Test
using Statistics
using LinearAlgebra
#using CSV

const datadir = joinpath(dirname(@__FILE__), "..", "data")

# @testset "MCFMSim" begin
#     fc, fus = simFactor(3, [4, 5, 6])
#     @test length(fc) == 3
#     @test length(fus)== 3
#     @test length(fus[1]) == 4
#     Ls = [5, 3, 4]
#     nc = 3
#     nus = [2, 2, 3]
#     Hs, Gs = randomLoadings(Ls, nc, nus)
#     @test length(Hs) == length(Ls)
#     @test size(Hs[1])[2] == nc
#     @test sum([size(H)[1] for H in Hs]) == sum(Ls)
# end

# @testset "MCFM" begin
#     Ls = [20, 15, 10]
#     N  = 100
#     p = 2
#     ps = [4, 3, 2]
#     lf = (l, p) ->  randomLoadingNorm(l, p; target_tr=0.3)
#     Hs, Gs = randomLoadings(Ls, p, ps, lf)
#     edists = vcat([repeat([Normal(0, sqrt(0.4 / L))], L) for L in Ls]...)
# end

@testset "All tests" begin

@testset "simulation" begin
    factorlayout = MultiChannelFactorLayout(1, [1, 1])
    channellayout = MultiChannelLayout([5, 5])
    nsamples = 100
    @test periodic_factors(nsamples, factorlayout).factors[1, :] == sin.((1:nsamples)/nsamples * 2 * pi)
    gauss_fac = classical_factors(nsamples, factorlayout)
    @test abs(mean(gauss_fac.factors[1, :])) <= 0.2
    @test size(gauss_fac.factors) == (3, 100)
    @test std(gauss_fac.factors[1, :]) <= 1.2
    @test std(gauss_fac.factors[1, :]) >= 0.8
    errors = indep_gauss_errors(nsamples, channellayout)
    @test size(errors.values)[1] == nsamples

    #Simulation
    H = reshape([1; 0.5; -0.5; 0.2; -0.2; -1; 0.2; 2.0; 1.0; 0.2], (10, 1))
    G = [0.5 0; 0.2 0; 0.1 0; -0.1 0; 0.1 0; 0 0.5; 0 0.3; 0 0.2; 0 -0.1; 0 -1]
    Sig = diagm(repeat([0.1], 10))
    params = MCFMParams(factorlayout, channellayout, H, G, Sig)
    simdata = simulate(nsamples, params, classical_factors, indep_gauss_errors)
    @test abs(mean(stack(simdata)[1, :])) <= 0.25
    @test abs(std(stack(simdata)[1, :]) - sqrt(1.1)) <= 0.35
end

@testset "data" begin
    c = 2
    n = 2
    nobs_per_chan = [2, 2]
    vals = [[j.* ones(p) .+ i for (j, p) in enumerate(nobs_per_chan)] for i in 1:n]
    channellayout = MultiChannelLayout(nobs_per_chan)
    data = MultiChannelData(vals, channellayout, n)
    @test stack(data) == [2.0 3.0; 2.0 3.0; 3.0 4.0; 3.0 4.0]
    @test unstack(stack(data), channellayout).values == data.values
    @test (data + data).values[1][1][1] == 4.0
end


@testset "layout" begin
    # Channel Layout
    clay = MultiChannelLayout([5, 3, 1])
    @test clay.nobs == 9
    @test clay.nchannels == 3
    @test_throws ArgumentError MultiChannelLayout([-2, 0, 1])
    # Factor Layout
    flay = MultiChannelFactorLayout(3, [2, 1, 1])
    @test flay.nchannels == 3
    @test flay.nchannelspecific == 4
    @test flay.nfactors == 7
end
@testset "utils" begin
    m = [1 2 3; 4 5 6; 7 8 9]
    b1 = [[1 2 3], [4 5 6; 7 8 9]]
    b2 = [[1 2; 4 5], reshape([9], 1, 1)]
    @test extract_horizontal_blocks(m, [1, 2]) == b1
    @test extract_diagonal_blocks(m, [2, 1], [2, 1]) == b2
    m[1, 1] = 300
    @test b1[1][1] == 1
    @test extract_horizontal_blocks(m, [1, 2]; get_view=true)[1][1] == 300
end



@testset "MCFMStructs" begin
    c = 3
    n = 10
    nobs_per_chan = [2, 2, 1]
    vals = [[zeros(p) for p in nobs_per_chan] for i in 1:n]
    channellayout = MultiChannelLayout(nobs_per_chan)
    @test MultiChannelData(vals, channellayout, n).values == vals
    @test_throws ArgumentError MultiChannelData(vals[1:(end-1), :], channellayout, n)
    @test_throws ArgumentError MultiChannelData(vals, MultiChannelLayout([2, 1, 1]), n)
    @test_throws ArgumentError MultiChannelData(vals, channellayout, 0 )
    @test_throws ArgumentError MultiChannelData(vals, channellayout, n+1)
end
end
