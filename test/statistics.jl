# Statistical Testing Tests

using Test
using QuantNova
using Random

Random.seed!(42)

@testset "Statistical Testing" begin

    @testset "Sharpe Ratio" begin
        # Generate returns with known Sharpe
        n = 252
        daily_ret = 0.0004  # ~10% annual
        daily_vol = 0.01   # ~16% annual
        returns = daily_ret .+ daily_vol .* randn(n)

        sr = sharpe_ratio(returns)
        @test sr isa Float64
        @test -2 < sr < 4  # Reasonable range

        # Zero returns should have ~0 Sharpe
        zero_returns = 0.01 .* randn(n)
        @test abs(sharpe_ratio(zero_returns)) < 1.0
    end

    @testset "Sharpe Confidence Interval" begin
        returns = 0.0004 .+ 0.01 .* randn(500)

        ci = sharpe_confidence_interval(returns; confidence=0.95)

        @test ci.sharpe isa Float64
        @test ci.lower < ci.sharpe < ci.upper
        @test ci.std_error > 0

        # 99% CI should be wider than 95%
        ci_99 = sharpe_confidence_interval(returns; confidence=0.99)
        @test (ci_99.upper - ci_99.lower) > (ci.upper - ci.lower)
    end

    @testset "Sharpe T-stat and P-value" begin
        # Strong positive returns
        good_returns = 0.001 .+ 0.01 .* randn(252)
        t = sharpe_t_stat(good_returns)
        p = sharpe_pvalue(good_returns; alternative=:greater)

        @test t > 0
        @test 0 <= p <= 1

        # Flat returns should have p-value near 0.5
        flat_returns = 0.01 .* randn(252)
        p_flat = sharpe_pvalue(flat_returns; alternative=:greater)
        @test 0.2 < p_flat < 0.8
    end

    @testset "Probabilistic Sharpe Ratio" begin
        returns = 0.0003 .+ 0.01 .* randn(252)

        # PSR vs 0 benchmark
        psr = probabilistic_sharpe_ratio(returns, 0.0)
        @test 0 <= psr <= 1

        # PSR should be lower vs higher benchmark
        psr_high = probabilistic_sharpe_ratio(returns, 1.0)
        @test psr_high < psr
    end

    @testset "Deflated Sharpe Ratio" begin
        returns = 0.0004 .+ 0.01 .* randn(252)

        # DSR with 1 trial = PSR vs 0
        dsr_1 = deflated_sharpe_ratio(returns, 1)
        psr_0 = probabilistic_sharpe_ratio(returns, 0.0)
        @test abs(dsr_1 - psr_0) < 0.1

        # DSR with many trials should be lower (more skeptical)
        dsr_100 = deflated_sharpe_ratio(returns, 100)
        @test dsr_100 < dsr_1
    end

    @testset "Compare Sharpe Ratios" begin
        # Strategy A better than B
        returns_a = 0.0006 .+ 0.01 .* randn(252)
        returns_b = 0.0002 .+ 0.01 .* randn(252)

        result = compare_sharpe_ratios(returns_a, returns_b)

        @test result.z_stat isa Float64
        @test 0 <= result.pvalue <= 1
        @test result.sharpe_diff isa Float64

        # Bootstrap method
        result_boot = compare_sharpe_ratios(returns_a, returns_b; method=:bootstrap)
        @test 0 <= result_boot.pvalue <= 1
    end

    @testset "Minimum Backtest Length" begin
        # More trials = longer backtest needed to prove significance
        # Use high SR (3.5) to exceed expected max under null for both trial counts
        min_len_100 = minimum_backtest_length(3.5, 100)
        min_len_10 = minimum_backtest_length(3.5, 10)
        @test min_len_100 > min_len_10

        # Should return positive integer
        @test minimum_backtest_length(3.0, 10) > 0
        @test minimum_backtest_length(4.0, 50) > 0
    end

    @testset "Overfitting Detection" begin
        # In-sample >> out-sample suggests overfitting
        p_overfit = probability_of_backtest_overfitting(2.0, 0.5, 100)
        @test 0 <= p_overfit <= 1

        # Good out-sample = less likely overfit
        p_good = probability_of_backtest_overfitting(1.5, 1.2, 10)
        @test p_good < p_overfit
    end

    @testset "CPCV PBO" begin
        returns = 0.0003 .+ 0.01 .* randn(252)

        pbo = combinatorial_purged_cv_pbo(returns, 8)
        @test 0 <= pbo <= 1
    end

    @testset "Information Coefficient" begin
        predictions = randn(100)
        outcomes = 0.3 .* predictions .+ 0.7 .* randn(100)  # Correlated

        ic = information_coefficient(predictions, outcomes)
        @test -1 <= ic <= 1
        @test ic > 0  # Should be positive

        # Perfect correlation
        ic_perfect = information_coefficient(predictions, predictions)
        @test ic_perfect ≈ 1.0
    end

    @testset "Hit Rate" begin
        predictions = randn(100)
        outcomes = 0.5 .* predictions .+ 0.5 .* randn(100)

        hr = hit_rate(predictions, outcomes)
        @test 0 <= hr <= 1
        @test hr > 0.4  # Should be better than random

        # Test significance
        sig = hit_rate_significance(hr, 100)
        @test sig.z_stat isa Float64
        @test 0 <= sig.pvalue <= 1
    end

end
