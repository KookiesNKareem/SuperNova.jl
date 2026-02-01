# Factor Models Tests

using Test
using QuantNova
using Random
using LinearAlgebra
using Statistics: mean

Random.seed!(42)

@testset "Factor Models" begin

    @testset "CAPM Regression" begin
        n = 252
        market = 0.0003 .+ 0.01 .* randn(n)

        # Generate returns with known beta
        true_beta = 1.2
        true_alpha = 0.0002  # Daily alpha
        noise = 0.005 .* randn(n)
        returns = true_alpha .+ true_beta .* market .+ noise

        result = capm_regression(returns, market)

        @test result.beta ≈ true_beta atol=0.3
        @test result.alpha ≈ true_alpha * 252 atol=0.1  # Annualized
        @test 0 <= result.r_squared <= 1
    end

    @testset "Factor Regression" begin
        n = 252
        k = 3
        factors = randn(n, k)

        true_betas = [0.8, 0.5, -0.3]
        true_alpha = 0.0001
        returns = true_alpha .+ factors * true_betas .+ 0.003 .* randn(n)

        result = factor_regression(returns, factors)

        @test length(result.betas) == k
        @test result.betas[1] ≈ true_betas[1] atol=0.2
        @test result.r_squared > 0.5
        @test result.adj_r_squared <= result.r_squared
    end

    @testset "Fama-French Regression" begin
        n = 252
        mkt = 0.0003 .+ 0.01 .* randn(n)
        smb = 0.0001 .+ 0.005 .* randn(n)
        hml = 0.00005 .+ 0.005 .* randn(n)
        mom = 0.0002 .+ 0.006 .* randn(n)

        # Growth stock: positive mkt beta, negative hml (growth not value)
        returns = 0.0001 .+ 1.1 .* mkt .- 0.3 .* smb .- 0.4 .* hml .+ 0.003 .* randn(n)

        # 3-factor
        ff3 = fama_french_regression(returns, mkt, smb, hml)
        @test ff3 isa FamaFrenchResult
        @test ff3.market_beta ≈ 1.1 atol=0.3
        @test ff3.hml_beta < 0  # Growth tilt
        @test ff3.mom_beta == 0.0  # Not included

        # 4-factor
        ff4 = fama_french_regression(returns, mkt, smb, hml; mom=mom)
        @test ff4.mom_beta != 0.0
    end

    @testset "Factor Construction" begin
        n = 100
        k = 5
        returns = randn(n, k)

        # Market factor
        mkt = construct_market_factor(returns)
        @test length(mkt) == n

        # Long-short factor
        signal = [1.0, 0.5, 0.0, -0.5, -1.0]  # Rank signal
        ls_factor = construct_long_short_factor(returns, signal; quantile=0.4)
        @test length(ls_factor) == n
    end

    @testset "Return Attribution" begin
        n = 252
        factors = randn(n, 2)
        betas = [0.8, 0.4]
        alpha = 0.05  # 5% annual alpha
        returns = (alpha/252) .+ factors * betas .+ 0.002 .* randn(n)

        attr = return_attribution(returns, factors, betas, alpha;
                                   factor_names=["MKT", "SMB"])

        @test attr.total_return ≈ sum(returns) atol=0.01
        @test haskey(attr.factor_contributions, "MKT")
        @test haskey(attr.factor_contributions, "SMB")
    end

    @testset "Rolling Beta/Alpha" begin
        n = 252
        factor = 0.01 .* randn(n)
        returns = 0.0001 .+ 1.2 .* factor .+ 0.003 .* randn(n)

        betas = rolling_beta(returns, factor; window=60)
        alphas = rolling_alpha(returns, factor; window=60)

        @test length(betas) == n
        @test length(alphas) == n
        @test all(isnan.(betas[1:59]))  # Not enough data
        @test !isnan(betas[60])

        # Rolling beta should be near true beta
        avg_beta = mean(filter(!isnan, betas))
        @test avg_beta ≈ 1.2 atol=0.3
    end

    @testset "Style Analysis" begin
        n = 252
        k = 4
        style_returns = randn(n, k)

        # Manager is 60% style 1, 40% style 2
        true_weights = [0.6, 0.4, 0.0, 0.0]
        returns = style_returns * true_weights .+ 0.001 .* randn(n)

        result = style_analysis(returns, style_returns)

        @test length(result.weights) == k
        @test sum(result.weights) ≈ 1.0 atol=0.01
        @test all(result.weights .>= -0.01)  # Non-negative (within tolerance)
        @test result.r_squared > 0.8
        @test result.tracking_error >= 0
    end

    @testset "Tracking Error & Information Ratio" begin
        n = 252
        benchmark = 0.0003 .+ 0.01 .* randn(n)
        returns = benchmark .+ 0.0001 .+ 0.002 .* randn(n)  # Outperformer

        te = tracking_error(returns, benchmark)
        ir = information_ratio(returns, benchmark)

        @test te > 0
        @test te ≈ 0.002 * sqrt(252) atol=0.02
        @test ir isa Float64

        # Perfect tracking = 0 TE
        te_perfect = tracking_error(benchmark, benchmark)
        @test te_perfect ≈ 0.0 atol=1e-10
    end

    @testset "Capture Ratios" begin
        n = 252
        benchmark = 0.01 .* randn(n)

        # Good manager: captures upside, avoids downside
        good_returns = benchmark .* 0.5  # 50% of benchmark moves
        good_returns[benchmark .> 0] .= benchmark[benchmark .> 0] .* 1.2  # 120% up capture
        good_returns[benchmark .< 0] .= benchmark[benchmark .< 0] .* 0.7  # 70% down capture

        up_cap = up_capture_ratio(good_returns, benchmark)
        down_cap = down_capture_ratio(good_returns, benchmark)
        cap_ratio = capture_ratio(good_returns, benchmark)

        @test up_cap ≈ 1.2 atol=0.1
        @test down_cap ≈ 0.7 atol=0.1
        @test cap_ratio > 1.0  # Good manager
    end

end
