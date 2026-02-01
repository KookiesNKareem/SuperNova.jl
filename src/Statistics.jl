module Statistics

using Statistics: mean, std, var, cov, cor
using LinearAlgebra: I, norm
using Distributions: Normal, TDist, cdf, quantile

# =============================================================================
# Sharpe Ratio Statistics
# =============================================================================

"""
    sharpe_ratio(returns; rf=0.0, periods_per_year=252) -> Float64

Compute annualized Sharpe ratio.
"""
function sharpe_ratio(returns::Vector{Float64}; rf::Float64=0.0, periods_per_year::Int=252)
    rf_per_period = rf / periods_per_year
    excess = returns .- rf_per_period
    mean(excess) / std(excess) * sqrt(periods_per_year)
end

"""
    sharpe_std_error(returns, sharpe; periods_per_year=252) -> Float64

Standard error of Sharpe ratio estimate. Lo (2002) formula.
SE(SR) ≈ sqrt((1 + 0.5*SR²) / n)
"""
function sharpe_std_error(returns::Vector{Float64}, sharpe::Float64; periods_per_year::Int=252)
    n = length(returns)
    # Annualized SE
    sqrt((1 + 0.5 * sharpe^2) / n) * sqrt(periods_per_year)
end

"""
    sharpe_confidence_interval(returns; rf=0.0, confidence=0.95, periods_per_year=252)

Confidence interval for Sharpe ratio.

Returns (sharpe, lower, upper, std_error)
"""
function sharpe_confidence_interval(returns::Vector{Float64};
                                     rf::Float64=0.0,
                                     confidence::Float64=0.95,
                                     periods_per_year::Int=252)
    sr = sharpe_ratio(returns; rf, periods_per_year)
    se = sharpe_std_error(returns, sr; periods_per_year)

    z = quantile(Normal(), 1 - (1 - confidence) / 2)
    lower = sr - z * se
    upper = sr + z * se

    (sharpe=sr, lower=lower, upper=upper, std_error=se)
end

"""
    sharpe_t_stat(returns; rf=0.0, benchmark_sharpe=0.0) -> Float64

T-statistic for Sharpe ratio vs benchmark (default: testing if SR > 0).
"""
function sharpe_t_stat(returns::Vector{Float64}; rf::Float64=0.0, benchmark_sharpe::Float64=0.0)
    sr = sharpe_ratio(returns; rf)
    se = sharpe_std_error(returns, sr)
    (sr - benchmark_sharpe) / se
end

"""
    sharpe_pvalue(returns; rf=0.0, benchmark_sharpe=0.0, alternative=:greater) -> Float64

P-value for Sharpe ratio hypothesis test.
- :greater - test if SR > benchmark (one-tailed)
- :two_sided - test if SR ≠ benchmark
"""
function sharpe_pvalue(returns::Vector{Float64};
                       rf::Float64=0.0,
                       benchmark_sharpe::Float64=0.0,
                       alternative::Symbol=:greater)
    t = sharpe_t_stat(returns; rf, benchmark_sharpe)
    n = length(returns)
    dist = TDist(n - 1)

    if alternative == :greater
        1 - cdf(dist, t)
    elseif alternative == :two_sided
        2 * (1 - cdf(dist, abs(t)))
    else
        cdf(dist, t)  # :less
    end
end

# =============================================================================
# Probabilistic Sharpe Ratio (Bailey & López de Prado)
# =============================================================================

"""
    probabilistic_sharpe_ratio(returns, benchmark_sharpe; rf=0.0) -> Float64

Probability that true Sharpe exceeds benchmark, accounting for skewness/kurtosis.
Bailey & López de Prado (2012).
"""
function probabilistic_sharpe_ratio(returns::Vector{Float64},
                                     benchmark_sharpe::Float64;
                                     rf::Float64=0.0)
    n = length(returns)
    sr = sharpe_ratio(returns; rf)

    # Compute skewness and kurtosis
    μ = mean(returns)
    σ = std(returns)
    z = (returns .- μ) ./ σ
    skew = mean(z.^3)
    kurt = mean(z.^4) - 3  # Excess kurtosis

    # Adjusted standard error (Lo 2002 with skew/kurt correction)
    se_adj = sqrt((1 + 0.5*sr^2 - skew*sr + (kurt/4)*sr^2) / (n - 1))

    # PSR = probability that SR > benchmark
    z_score = (sr - benchmark_sharpe) / se_adj
    cdf(Normal(), z_score)
end

"""
    deflated_sharpe_ratio(returns, n_trials; rf=0.0, expected_max_sr=nothing) -> Float64

Deflated Sharpe Ratio - adjusts for multiple testing (strategy selection bias).
Bailey & López de Prado (2014).

`n_trials` = number of strategies/parameters tested before selecting this one.
"""
function deflated_sharpe_ratio(returns::Vector{Float64},
                                n_trials::Int;
                                rf::Float64=0.0,
                                expected_max_sr::Union{Nothing,Float64}=nothing)
    n = length(returns)
    sr = sharpe_ratio(returns; rf)

    # Expected maximum SR under null (all strategies have SR=0)
    if expected_max_sr === nothing
        if n_trials <= 1
            expected_max_sr = 0.0  # No selection bias with single trial
        else
            γ = 0.5772156649  # Euler-Mascheroni constant
            expected_max_sr = sqrt(2 * log(n_trials)) - (γ + log(π)) / (2 * sqrt(2 * log(n_trials)))
        end
    end

    # PSR vs expected max
    probabilistic_sharpe_ratio(returns, expected_max_sr; rf)
end

# =============================================================================
# Strategy Comparison Tests
# =============================================================================

"""
    compare_sharpe_ratios(returns_a, returns_b; rf=0.0, method=:jobson_korkie)

Test if strategy A has significantly higher Sharpe than strategy B.

Methods:
- :jobson_korkie - Jobson-Korkie (1981) test
- :bootstrap - Bootstrap test (more robust)

Returns (z_stat, pvalue, sharpe_diff)
"""
function compare_sharpe_ratios(returns_a::Vector{Float64},
                                returns_b::Vector{Float64};
                                rf::Float64=0.0,
                                method::Symbol=:jobson_korkie)
    if method == :jobson_korkie
        _jk_test(returns_a, returns_b; rf)
    else
        _bootstrap_sharpe_test(returns_a, returns_b; rf)
    end
end

# Jobson-Korkie (1981) test for comparing Sharpe ratios
function _jk_test(returns_a::Vector{Float64}, returns_b::Vector{Float64}; rf::Float64=0.0)
    n = length(returns_a)
    @assert length(returns_b) == n "Returns must have same length"

    rf_per_period = rf / 252
    μa = mean(returns_a) - rf_per_period
    μb = mean(returns_b) - rf_per_period
    σa = std(returns_a)
    σb = std(returns_b)
    ρ = cor(returns_a, returns_b)

    sr_a = μa / σa
    sr_b = μb / σb

    # Jobson-Korkie variance of difference
    θ = (1/n) * (2 * (1 - ρ) + 0.5 * (sr_a^2 + sr_b^2 - 2*sr_a*sr_b*ρ^2))

    z = (sr_a - sr_b) / sqrt(θ)
    pval = 1 - cdf(Normal(), z)

    (z_stat=z, pvalue=pval, sharpe_diff=(sr_a - sr_b) * sqrt(252))
end

# Bootstrap test for Sharpe ratio comparison
function _bootstrap_sharpe_test(returns_a::Vector{Float64}, returns_b::Vector{Float64};
                                 rf::Float64=0.0, n_bootstrap::Int=10000)
    n = length(returns_a)
    sr_a = sharpe_ratio(returns_a; rf)
    sr_b = sharpe_ratio(returns_b; rf)
    observed_diff = sr_a - sr_b

    # Bootstrap under null (no difference)
    combined = [returns_a; returns_b]
    diffs = Float64[]

    for _ in 1:n_bootstrap
        idx = rand(1:2n, 2n)
        boot_a = combined[idx[1:n]]
        boot_b = combined[idx[n+1:end]]
        push!(diffs, sharpe_ratio(boot_a; rf) - sharpe_ratio(boot_b; rf))
    end

    pval = mean(diffs .>= observed_diff)
    z = (observed_diff - mean(diffs)) / std(diffs)

    (z_stat=z, pvalue=pval, sharpe_diff=observed_diff)
end

# =============================================================================
# Overfitting Detection
# =============================================================================

"""
    minimum_backtest_length(sharpe_target, n_trials; confidence=0.95) -> Int

Minimum backtest length needed to achieve target Sharpe with given confidence,
accounting for multiple testing. Bailey et al. (2015).

More trials = higher expected max SR under null = need more data to prove significance.
"""
function minimum_backtest_length(sharpe_target::Float64, n_trials::Int; confidence::Float64=0.95)
    z = quantile(Normal(), confidence)

    # Expected max SR under null (selection bias from trying n_trials strategies)
    γ = 0.5772156649
    if n_trials <= 1
        e_max_sr = 0.0
    else
        e_max_sr = sqrt(2 * log(n_trials)) - (γ + log(π)) / (2 * sqrt(2 * log(n_trials)))
    end

    # Need observed SR to exceed (expected max + margin)
    # The required standard error: SE < (sr_target - e_max_sr) / z
    # SE(SR) ≈ sqrt((1 + 0.5*sr^2) / n), approximately sqrt(1/n) for small SR
    # For significance: sr_target > e_max_sr + z * sqrt((1 + 0.5*sr^2)/n)
    # Solving for n: n > (1 + 0.5*sr^2) * z^2 / (sr_target - e_max_sr)^2

    margin = sharpe_target - e_max_sr
    if margin <= 0
        return 10000  # Target SR too low to ever prove significance
    end

    # Bailey et al. formula: n ≥ (1 + 0.5*SR²) * z² / (SR - E[maxSR])²
    min_n = (1 + 0.5 * sharpe_target^2) * z^2 / margin^2
    max(1, ceil(Int, min_n))
end

"""
    probability_of_backtest_overfitting(in_sample_sr, out_sample_sr, n_trials) -> Float64

Estimate probability that backtest is overfit.
High if in-sample >> out-sample and many trials.
"""
function probability_of_backtest_overfitting(in_sample_sr::Float64,
                                              out_sample_sr::Float64,
                                              n_trials::Int)
    # Expected SR degradation from overfitting
    γ = 0.5772156649
    expected_inflation = sqrt(2 * log(n_trials)) - (γ + log(π)) / (2 * sqrt(2 * log(n_trials)))

    # Ratio of expected to observed degradation
    observed_degradation = in_sample_sr - out_sample_sr

    if in_sample_sr <= 0
        return 1.0
    end

    # Simple heuristic: P(overfit) based on how much SR degraded vs expected
    clamp(observed_degradation / (expected_inflation + 0.1), 0.0, 1.0)
end

"""
    combinatorial_purged_cv_pbo(returns, n_paths; train_frac=0.5) -> Float64

Probability of Backtest Overfitting via Combinatorial Purged Cross-Validation.
Bailey et al. (2017). Simplified implementation.

Returns probability that strategy is overfit.
"""
function combinatorial_purged_cv_pbo(returns::Vector{Float64},
                                      n_paths::Int=16;
                                      train_frac::Float64=0.5)
    n = length(returns)
    train_size = floor(Int, n * train_frac)

    # Generate multiple train/test splits
    in_sample_srs = Float64[]
    out_sample_srs = Float64[]

    for _ in 1:n_paths
        # Random split (simplified - full CPCV uses combinatorial)
        perm = randperm(n)
        train_idx = perm[1:train_size]
        test_idx = perm[train_size+1:end]

        train_returns = returns[sort(train_idx)]
        test_returns = returns[sort(test_idx)]

        push!(in_sample_srs, sharpe_ratio(train_returns))
        push!(out_sample_srs, sharpe_ratio(test_returns))
    end

    # PBO = fraction where rank changes (in-sample best ≠ out-sample best)
    # Simplified: fraction where out-sample SR < 0 when in-sample > 0
    overfit_count = sum((in_sample_srs .> 0) .& (out_sample_srs .< 0))
    positive_in_sample = sum(in_sample_srs .> 0)

    positive_in_sample > 0 ? overfit_count / positive_in_sample : 1.0
end

using Random: randperm

# =============================================================================
# Information Coefficient & Hit Rate
# =============================================================================

"""
    information_coefficient(predictions, outcomes) -> Float64

Correlation between predictions and outcomes. Key alpha metric.
"""
function information_coefficient(predictions::Vector{Float64}, outcomes::Vector{Float64})
    cor(predictions, outcomes)
end

"""
    hit_rate(predictions, outcomes) -> Float64

Fraction of correct directional predictions.
"""
function hit_rate(predictions::Vector{Float64}, outcomes::Vector{Float64})
    correct = sum(sign.(predictions) .== sign.(outcomes))
    correct / length(predictions)
end

"""
    hit_rate_significance(hit_rate, n_predictions; benchmark=0.5) -> (z_stat, pvalue)

Test if hit rate is significantly above benchmark (default 50%).
"""
function hit_rate_significance(hr::Float64, n::Int; benchmark::Float64=0.5)
    se = sqrt(benchmark * (1 - benchmark) / n)
    z = (hr - benchmark) / se
    pval = 1 - cdf(Normal(), z)
    (z_stat=z, pvalue=pval)
end

# =============================================================================
# Exports
# =============================================================================

export sharpe_ratio, sharpe_std_error, sharpe_confidence_interval
export sharpe_t_stat, sharpe_pvalue
export probabilistic_sharpe_ratio, deflated_sharpe_ratio
export compare_sharpe_ratios
export minimum_backtest_length, probability_of_backtest_overfitting
export combinatorial_purged_cv_pbo
export information_coefficient, hit_rate, hit_rate_significance

end
