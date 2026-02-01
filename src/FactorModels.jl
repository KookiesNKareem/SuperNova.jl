module FactorModels

using Statistics: mean, std, var, cov, cor
using LinearAlgebra: I, pinv, diag, norm, dot

# =============================================================================
# Factor Regression
# =============================================================================

"""
    RegressionResult

Results from factor regression.
"""
struct RegressionResult
    alpha::Float64           # Annualized alpha
    betas::Vector{Float64}   # Factor loadings
    r_squared::Float64       # R² (explained variance)
    adj_r_squared::Float64   # Adjusted R²
    alpha_tstat::Float64     # T-stat for alpha
    alpha_pvalue::Float64    # P-value for alpha
    residual_vol::Float64    # Idiosyncratic volatility (annualized)
    factor_names::Vector{String}
end

"""
    factor_regression(returns, factors; factor_names=nothing, rf=0.0)

Regress strategy returns on factor returns.

returns = α + Σ(βᵢ * factorᵢ) + ε

Returns RegressionResult with alpha, betas, and statistics.
"""
function factor_regression(returns::Vector{Float64},
                           factors::Matrix{Float64};
                           factor_names::Union{Nothing,Vector{String}}=nothing,
                           rf::Float64=0.0)
    n, k = size(factors)
    @assert length(returns) == n "Returns and factors must have same length"

    # Excess returns
    rf_per_period = rf / 252
    y = returns .- rf_per_period
    X = hcat(ones(n), factors)  # Add intercept

    # OLS: β = (X'X)⁻¹X'y
    XtX_inv = pinv(X' * X)
    coeffs = XtX_inv * X' * y

    alpha_daily = coeffs[1]
    betas = coeffs[2:end]

    # Fitted values and residuals
    y_hat = X * coeffs
    residuals = y - y_hat

    # R² and adjusted R²
    ss_res = sum(residuals.^2)
    ss_tot = sum((y .- mean(y)).^2)
    r2 = 1 - ss_res / ss_tot
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)

    # Standard errors
    mse = ss_res / (n - k - 1)
    se = sqrt.(diag(XtX_inv) .* mse)
    alpha_se = se[1]

    # T-stat and p-value for alpha
    alpha_tstat = alpha_daily / alpha_se
    alpha_pvalue = 2 * (1 - _tcdf(abs(alpha_tstat), n - k - 1))

    # Annualize
    alpha_annual = alpha_daily * 252
    residual_vol = std(residuals) * sqrt(252)

    names = factor_names !== nothing ? factor_names : ["F$i" for i in 1:k]

    RegressionResult(alpha_annual, betas, r2, adj_r2, alpha_tstat, alpha_pvalue,
                     residual_vol, names)
end

# Simple t-distribution CDF approximation
function _tcdf(t, df)
    # Normal approximation for large df
    if df > 30
        return _normcdf(t)
    end
    # Rough approximation using beta function relationship
    x = df / (df + t^2)
    0.5 + sign(t) * 0.5 * (1 - x^(df/2))
end

# Normal CDF approximation (Abramowitz & Stegun)
function _normcdf(x)
    t = 1.0 / (1.0 + 0.2316419 * abs(x))
    d = 0.3989422804014327  # 1/sqrt(2π)
    p = d * exp(-x^2 / 2) * t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))))
    x >= 0 ? 1 - p : p
end

# =============================================================================
# CAPM (Single Factor)
# =============================================================================

"""
    capm_regression(returns, market_returns; rf=0.0)

Capital Asset Pricing Model regression.

Returns (alpha, beta, r_squared, alpha_tstat, alpha_pvalue)
"""
function capm_regression(returns::Vector{Float64},
                         market_returns::Vector{Float64};
                         rf::Float64=0.0)
    factors = reshape(market_returns, :, 1)
    result = factor_regression(returns, factors; factor_names=["Market"], rf)

    (alpha=result.alpha,
     beta=result.betas[1],
     r_squared=result.r_squared,
     alpha_tstat=result.alpha_tstat,
     alpha_pvalue=result.alpha_pvalue,
     residual_vol=result.residual_vol)
end

# =============================================================================
# Fama-French Factors
# =============================================================================

"""
    FamaFrenchResult

Results from Fama-French factor analysis.
"""
struct FamaFrenchResult
    alpha::Float64           # Annualized alpha
    market_beta::Float64     # MKT-RF loading
    smb_beta::Float64        # Size factor (Small Minus Big)
    hml_beta::Float64        # Value factor (High Minus Low)
    mom_beta::Float64        # Momentum factor (optional, 0 if 3-factor)
    r_squared::Float64
    alpha_tstat::Float64
    alpha_pvalue::Float64
    residual_vol::Float64
end

"""
    fama_french_regression(returns, mkt, smb, hml; mom=nothing, rf=0.0)

Fama-French 3-factor (or 4-factor with momentum) regression.

Factors should be factor returns (not cumulative).
"""
function fama_french_regression(returns::Vector{Float64},
                                 mkt::Vector{Float64},
                                 smb::Vector{Float64},
                                 hml::Vector{Float64};
                                 mom::Union{Nothing,Vector{Float64}}=nothing,
                                 rf::Float64=0.0)
    if mom !== nothing
        factors = hcat(mkt, smb, hml, mom)
        names = ["MKT-RF", "SMB", "HML", "MOM"]
    else
        factors = hcat(mkt, smb, hml)
        names = ["MKT-RF", "SMB", "HML"]
    end

    result = factor_regression(returns, factors; factor_names=names, rf)

    FamaFrenchResult(
        result.alpha,
        result.betas[1],
        result.betas[2],
        result.betas[3],
        mom !== nothing ? result.betas[4] : 0.0,
        result.r_squared,
        result.alpha_tstat,
        result.alpha_pvalue,
        result.residual_vol
    )
end

# =============================================================================
# Factor Construction (from returns)
# =============================================================================

"""
    construct_market_factor(returns_matrix, weights=nothing) -> Vector{Float64}

Construct market factor from asset returns matrix (n_periods × n_assets).
Default: equal-weighted market return.
"""
function construct_market_factor(returns::Matrix{Float64};
                                  weights::Union{Nothing,Vector{Float64}}=nothing)
    n, k = size(returns)
    if weights === nothing
        weights = fill(1.0/k, k)
    end
    returns * weights
end

"""
    construct_long_short_factor(returns_matrix, signal; quantile=0.3) -> Vector{Float64}

Construct long-short factor: long top quantile, short bottom quantile.
"""
function construct_long_short_factor(returns::Matrix{Float64},
                                      signal::Vector{Float64};
                                      quantile::Float64=0.3)
    n, k = size(returns)
    @assert length(signal) == k "Signal must have length = number of assets"

    n_long = max(1, floor(Int, k * quantile))
    n_short = n_long

    sorted_idx = sortperm(signal, rev=true)
    long_idx = sorted_idx[1:n_long]
    short_idx = sorted_idx[end-n_short+1:end]

    # Equal-weight within legs
    factor_returns = Vector{Float64}(undef, n)
    for t in 1:n
        long_ret = mean(returns[t, long_idx])
        short_ret = mean(returns[t, short_idx])
        factor_returns[t] = long_ret - short_ret
    end

    factor_returns
end

# =============================================================================
# Return Attribution
# =============================================================================

"""
    AttributionResult

Factor-based return attribution.
"""
struct AttributionResult
    total_return::Float64
    factor_contributions::Dict{String,Float64}
    alpha_contribution::Float64
    residual_contribution::Float64
end

"""
    return_attribution(returns, factors, betas, alpha; factor_names=nothing)

Decompose total return into factor contributions + alpha + residual.
"""
function return_attribution(returns::Vector{Float64},
                            factors::Matrix{Float64},
                            betas::Vector{Float64},
                            alpha::Float64;
                            factor_names::Union{Nothing,Vector{String}}=nothing)
    n, k = size(factors)
    names = factor_names !== nothing ? factor_names : ["F$i" for i in 1:k]

    total = sum(returns)

    # Factor contributions
    contributions = Dict{String,Float64}()
    for i in 1:k
        contributions[names[i]] = betas[i] * sum(factors[:, i])
    end

    # Alpha contribution (daily alpha × n periods)
    alpha_contrib = (alpha / 252) * n

    # Residual
    factor_total = sum(values(contributions))
    residual = total - factor_total - alpha_contrib

    AttributionResult(total, contributions, alpha_contrib, residual)
end

"""
    rolling_beta(returns, factor; window=60) -> Vector{Float64}

Compute rolling beta to a factor.
"""
function rolling_beta(returns::Vector{Float64}, factor::Vector{Float64}; window::Int=60)
    n = length(returns)
    betas = fill(NaN, n)

    for t in window:n
        y = returns[t-window+1:t]
        x = factor[t-window+1:t]
        # β = Cov(y,x) / Var(x)
        betas[t] = cov(y, x) / var(x)
    end

    betas
end

"""
    rolling_alpha(returns, factor; window=60, rf=0.0) -> Vector{Float64}

Compute rolling alpha vs a factor (annualized).
"""
function rolling_alpha(returns::Vector{Float64}, factor::Vector{Float64};
                       window::Int=60, rf::Float64=0.0)
    n = length(returns)
    alphas = fill(NaN, n)
    rf_daily = rf / 252

    for t in window:n
        y = returns[t-window+1:t] .- rf_daily
        x = factor[t-window+1:t]
        β = cov(y, x) / var(x)
        α = mean(y) - β * mean(x)
        alphas[t] = α * 252  # Annualize
    end

    alphas
end

# =============================================================================
# Style Analysis (Sharpe 1992)
# =============================================================================

"""
    StyleAnalysisResult

Results from returns-based style analysis.
"""
struct StyleAnalysisResult
    weights::Vector{Float64}      # Style weights (sum to 1)
    style_names::Vector{String}
    r_squared::Float64
    tracking_error::Float64       # Annualized
end

"""
    style_analysis(returns, style_returns; style_names=nothing)

Returns-based style analysis (Sharpe 1992).
Constrained regression: weights ≥ 0, sum to 1.

Finds style portfolio that best replicates manager returns.
"""
function style_analysis(returns::Vector{Float64},
                        style_returns::Matrix{Float64};
                        style_names::Union{Nothing,Vector{String}}=nothing,
                        max_iter::Int=1000,
                        tol::Float64=1e-6)
    n, k = size(style_returns)
    names = style_names !== nothing ? style_names : ["Style$i" for i in 1:k]

    # Constrained least squares via iterative projection
    # min ||y - Xw||² s.t. w ≥ 0, sum(w) = 1

    w = fill(1.0/k, k)  # Initial equal weights

    for _ in 1:max_iter
        # Gradient step
        residual = returns - style_returns * w
        grad = -2 * style_returns' * residual

        # Simple gradient descent with projection
        step = 0.001
        w_new = w - step * grad

        # Project to simplex (non-negative, sum to 1)
        w_new = _project_simplex(w_new)

        if norm(w_new - w) < tol
            w = w_new
            break
        end
        w = w_new
    end

    # Compute fit statistics
    fitted = style_returns * w
    residual = returns - fitted
    r2 = 1 - var(residual) / var(returns)
    te = std(residual) * sqrt(252)

    StyleAnalysisResult(w, names, r2, te)
end

# Project vector onto probability simplex
function _project_simplex(v::Vector{Float64})
    n = length(v)
    u = sort(v, rev=true)
    cssv = cumsum(u)
    rho = findlast(i -> u[i] + (1 - cssv[i]) / i > 0, 1:n)
    rho = rho === nothing ? 1 : rho
    theta = (cssv[rho] - 1) / rho
    max.(v .- theta, 0)
end

# =============================================================================
# Information Ratio & Tracking Error
# =============================================================================

"""
    tracking_error(returns, benchmark_returns) -> Float64

Annualized tracking error (volatility of active returns).
"""
function tracking_error(returns::Vector{Float64}, benchmark::Vector{Float64})
    active = returns - benchmark
    std(active) * sqrt(252)
end

"""
    information_ratio(returns, benchmark_returns) -> Float64

Annualized information ratio = active return / tracking error.
"""
function information_ratio(returns::Vector{Float64}, benchmark::Vector{Float64})
    active = returns - benchmark
    mean(active) * 252 / (std(active) * sqrt(252))
end

"""
    up_capture_ratio(returns, benchmark_returns) -> Float64

Up capture = mean(portfolio | benchmark > 0) / mean(benchmark | benchmark > 0)
"""
function up_capture_ratio(returns::Vector{Float64}, benchmark::Vector{Float64})
    up_mask = benchmark .> 0
    sum(up_mask) == 0 && return NaN
    mean(returns[up_mask]) / mean(benchmark[up_mask])
end

"""
    down_capture_ratio(returns, benchmark_returns) -> Float64

Down capture = mean(portfolio | benchmark < 0) / mean(benchmark | benchmark < 0)
Lower is better (lose less when market falls).
"""
function down_capture_ratio(returns::Vector{Float64}, benchmark::Vector{Float64})
    down_mask = benchmark .< 0
    sum(down_mask) == 0 && return NaN
    mean(returns[down_mask]) / mean(benchmark[down_mask])
end

"""
    capture_ratio(returns, benchmark_returns) -> Float64

Capture ratio = up_capture / down_capture.
> 1 means manager adds value (captures more upside, less downside).
"""
function capture_ratio(returns::Vector{Float64}, benchmark::Vector{Float64})
    up_capture_ratio(returns, benchmark) / down_capture_ratio(returns, benchmark)
end

# =============================================================================
# Exports
# =============================================================================

export RegressionResult, factor_regression
export capm_regression
export FamaFrenchResult, fama_french_regression
export construct_market_factor, construct_long_short_factor
export AttributionResult, return_attribution
export rolling_beta, rolling_alpha
export StyleAnalysisResult, style_analysis
export tracking_error, information_ratio
export up_capture_ratio, down_capture_ratio, capture_ratio

end
