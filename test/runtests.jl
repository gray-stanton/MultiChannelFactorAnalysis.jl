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


        #from long, no missing
        df = DataFrame(sample_id = [1,  1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
                channel = ["a", "a", "b", "b", "a", "a", "b", "b", "a", "a", "b", "b"],
                observation_id = [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2 ],
                value = [0.5, 0.3, 0.2, 0.1, 0.3, 0.4, 0.6, 0.2, 0.3, 0.5, 0.2, 0.1])
        data = data_fromlong(df)
        @test data.values[1][1][1] == 0.5
        @test data.values[2][2][2] == 0.2

        #from long, some missing to trim
        df2 = DataFrame(sample_id = [1, 1, 1, 1, 2, 2, 3, 3],
                channel = ["a", "a", "b", "b", "a", "b", "a", "b"],
                observation_id = [1, 2, 1, 2, 1, 1, 1, 1],
                value = [0.5, 0.3, 0.2, 0.1, 0.3, 0.4, 0.6, 0.3])
        @test dat2.channellayout.nobs== 6

        # from long, some missing to interpolate
        df3 = DataFrame(sample_id = [1,  1, 1, 1, 2, 2, 2, 2, 3, 3],
                channel = ["a", "a", "b", "b", "a", "a", "b", "b", "a", "b"],
                observation_id = [1, 2, 1, 2, 1, 2, 1, 2, 1, 1 ],
                value = [0.5, 0.3, 0.2, 0.1, 0.3, 0.4, 0.6, 0.2, 0.3, 0.2])
        dat3 = data_fromlong(df3; max_missingpcent =0.5)
        @test df3.values[3][1][2] == 0.5
        @test dat3.nobs == 12



    end

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

    @testset "factortest" begin
        M1 = [3; 2; 1; 2; 3; 4; 2; 1.0; 3.0; 2.0] * transpose([3; 2; 1; 2; 3; 4; 2; 1.0; 3.0; 2.0])
        M2 = M +  [5.0; -3.0; .2; 0.6; 0.3; 4.1; 3.2; -2.6; 4.0; 5.0] * transpose([5.0; -3.0; .2; 0.6; 0.3; 4.1; 3.2; -2.6; 4.0; 5.0])
        M3b = randn(100, 30)
        M3  = M3b * transpose(M3b)
        @test eigenvalue_ratio_test(reverse(eigvals(M1))) == 1
        @test eigenvalue_ratio_test(reverse(eigvals(M2))) == 2
        @test eigenvalue_ratio_test(reverse(eigvals(M3)); threshold=5) = 0


    @testset "fitting" begin
        factorlayout = MultiChannelFactorLayout(1, [1, 1])
        channellayout = MultiChannelLayout([5, 5])
        nsamples = 1000
        H = reshape([1; 0.5; -0.5; 0.2; -0.2; -1; 0.2; 2.0; 1.0; 0.2], (10, 1))
        G = [0.5 0; 0.2 0; 0.1 0; -0.1 0; 0.1 0; 0 0.5; 0 0.3; 0 0.2; 0 -0.1; 0 -1]
        Sig = diagm(repeat([0.1], 10))
        params = MCFMParams(factorlayout, channellayout, H, G, Sig)
        simdata = simulate(nsamples, params, classical_factors, indep_gauss_errors)
        fitted = fit(simdata,  factorlayout; maxiter=1e2)
        fitH = fitted.finalparams.H
        fitG = fitted.finalparams.G
        fitSig = fitted.finalparams.Σ
        @test norm(transpose(fitH) * fitH - transpose(H) * H)^2/norm(transpose(H) * H)^2 <= 1
        @test norm(transpose(fitG) * fitG - transpose(G) * G)^2/norm(transpose(G) * G)^2 <= 1
        #@test norm(fitSig - Sig)^2/norm(Sig)^2 <= 1


        # 0 channel-specific factors in some channels
        factorlayout = MultiChannelFactorLayout(1, [2, 0])
        channellayout = MultiChannelLayout([5, 5])
        nsamples = 1000
        H = reshape([1; 0.5; -0.5; 0.2; -0.2; -1; 0.2; 2.0; 1.0; 0.2], (10, 1))
        G = [0.5 0; 0.2 -0.1; 0.1 0.1; -0.1 2.0; 0.1 -2.0; 0 0; 0 0; 0 0; 0.0 0.0; 0.0 0.0]
        Sig = diagm(repeat([0.1], 10))
        params = MCFMParams(factorlayout, channellayout, H, G, Sig)
        simdata = simulate(nsamples, params, classical_factors, indep_gauss_errors)
        fitted = fit(simdata,  factorlayout; maxiter=1e2)
        fitH = fitted.finalparams.H
        fitG = fitted.finalparams.G
        fitSig = fitted.finalparams.Σ
        @test norm(transpose(fitH) * fitH - transpose(H) * H)^2/norm(transpose(H) * H)^2 <= 1
        @test norm(transpose(fitG) * fitG - transpose(G) * G)^2/norm(transpose(G) * G)^2 <= 1


        # Common factors only
        factorlayout = MultiChannelFactorLayout(2, [0, 0])
        channellayout = MultiChannelLayout([5, 5])
        nsamples = 1000
        H = [1 0.0; 0.5 0.2; -0.5 -0.1; 0.2 -0.3; -0.2 0.4; -1 -0.2; 0.2 0.6; 2.0 -0.2; 1.0 0.4; 0.2 -0.2]
        G = zeros(10, 0)
        Sig = diagm(repeat([0.1], 10))
        params = MCFMParams(factorlayout, channellayout, H, G, Sig)
        simdata = simulate(nsamples, params, classical_factors, indep_gauss_errors)
        fitted = fit(simdata,  factorlayout; maxiter=1e2)
        fitH = fitted.finalparams.H
        fitG = fitted.finalparams.G
        fitSig = fitted.finalparams.Σ
        facs = extract_factors(simdata, fitted.finalparams)
        #@test norm(fitSig - Sig)^2/norm(Sig)^2 <= 1
    end


    @testset "extraction" begin
        factorlayout = MultiChannelFactorLayout(1, [1, 1])
        channellayout = MultiChannelLayout([5, 5])
        nsamples = 100
        H = reshape([1; 0.5; -0.5; 0.2; -0.2; -1; 0.2; 2.0; 1.0; 0.2], (10, 1))
        G = [0.5 0; 0.2 0; 0.1 0; -0.1 0; 0.1 0; 0 0.5; 0 0.3; 0 0.2; 0 -0.1; 0 -1]
        Sig = diagm(repeat([0.1], 10))
        params = MCFMParams(factorlayout, channellayout, H, G, Sig)
        simdata = simulate(nsamples, params, periodic_factors, indep_gauss_errors)
        true_fac = periodic_factors(nsamples, factorlayout )
        fitted_fac = extract_factors(simdata, params)
        @test norm(true_fac.factors - fitted_fac.factors)^2/norm(true_fac.factors)^2 <= 1

        errorless_preds   = predict(true_fac, params)
        preds = predict(fitted_fac, params)
        @test norm(errorless_preds - stack(simdata))^2/norm(stack(simdata)) <= 1
        @test norm(preds - stack(simdata))^2/norm(stack(simdata)) <= 1
    end


    @testset "writeout" begin
        factorlayout = MultiChannelFactorLayout(1, [1, 1])
        channellayout = MultiChannelLayout([5, 5])
        nsamples = 100
        H = reshape([1; 0.5; -0.5; 0.2; -0.2; -1; 0.2; 2.0; 1.0; 0.2], (10, 1))
        G = [0.5 0; 0.2 0; 0.1 0; -0.1 0; 0.1 0; 0 0.5; 0 0.3; 0 0.2; 0 -0.1; 0 -1]
        Sig = diagm(repeat([0.1], 10))
        params = MCFMParams(factorlayout, channellayout, H, G, Sig)
        param_long = to_long(params)
        @test size(parm_long) == (20, 4)
        fac = periodic_factors(nsamples, factorlayout )
        @test size(to_long(fac)) == (100, 3)

        df = DataFrame(sample_id = [1,  1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
                channel = ["a", "a", "b", "b", "a", "a", "b", "b", "a", "a", "b", "b"],
                observation_id = [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2 ],
                value = [0.5, 0.3, 0.2, 0.1, 0.3, 0.4, 0.6, 0.2, 0.3, 0.5, 0.2, 0.1])
        data = data_fromlong(df)
end
