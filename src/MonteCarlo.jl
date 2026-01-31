module MonteCarlo

using Random
using Statistics: mean, std
using ..Core: ADBackend
using ..AD: gradient, current_backend, ForwardDiffBackend

# ============================================================================
# Path Dynamics
# ============================================================================

abstract type AbstractDynamics end

"""
    GBMDynamics(r, σ)

Geometric Brownian Motion: dS = r*S*dt + σ*S*dW
"""
struct GBMDynamics{T1,T2} <: AbstractDynamics
    r::T1      # risk-free rate
    sigma::T2  # volatility
end

"""
    HestonDynamics(r, v0, κ, θ, ξ, ρ)

Heston stochastic volatility model.
"""
struct HestonDynamics{T} <: AbstractDynamics
    r::T      # risk-free rate
    v0::T     # initial variance
    kappa::T  # mean reversion speed
    theta::T  # long-term variance
    xi::T     # vol of vol
    rho::T    # correlation
end

# ============================================================================
# Payoffs
# ============================================================================

abstract type AbstractPayoff end

struct EuropeanCall <: AbstractPayoff
    K::Float64
end

struct EuropeanPut <: AbstractPayoff
    K::Float64
end

struct AsianCall <: AbstractPayoff
    K::Float64
end

struct AsianPut <: AbstractPayoff
    K::Float64
end

struct UpAndOutCall <: AbstractPayoff
    K::Float64
    barrier::Float64
end

struct DownAndOutPut <: AbstractPayoff
    K::Float64
    barrier::Float64
end

# Payoff evaluation
payoff(p::EuropeanCall, path) = max(path[end] - p.K, 0.0)
payoff(p::EuropeanPut, path) = max(p.K - path[end], 0.0)
payoff(p::AsianCall, path) = max(mean(path) - p.K, 0.0)
payoff(p::AsianPut, path) = max(p.K - mean(path), 0.0)

function payoff(p::UpAndOutCall, path)
    maximum(path) >= p.barrier ? 0.0 : max(path[end] - p.K, 0.0)
end

function payoff(p::DownAndOutPut, path)
    minimum(path) <= p.barrier ? 0.0 : max(p.K - path[end], 0.0)
end

# ============================================================================
# Path Generation
# ============================================================================

"""
    simulate_gbm(S0, T, nsteps, dynamics; rng=Random.default_rng())

Generate a single GBM path.
"""
function simulate_gbm(S0, T, nsteps, dynamics::GBMDynamics; rng=Random.default_rng())
    dt = T / nsteps
    path = Vector{typeof(S0 * one(dynamics.r))}(undef, nsteps + 1)
    path[1] = S0

    sqrt_dt = sqrt(dt)
    drift = (dynamics.r - 0.5 * dynamics.sigma^2) * dt

    for i in 1:nsteps
        Z = randn(rng)
        path[i+1] = path[i] * exp(drift + dynamics.sigma * sqrt_dt * Z)
    end

    return path
end

"""
    simulate_gbm_antithetic(S0, T, nsteps, dynamics; rng=Random.default_rng())

Generate two antithetic GBM paths for variance reduction.
"""
function simulate_gbm_antithetic(S0, T, nsteps, dynamics::GBMDynamics; rng=Random.default_rng())
    dt = T / nsteps
    path1 = Vector{typeof(S0 * one(dynamics.r))}(undef, nsteps + 1)
    path2 = Vector{typeof(S0 * one(dynamics.r))}(undef, nsteps + 1)
    path1[1] = S0
    path2[1] = S0

    sqrt_dt = sqrt(dt)
    drift = (dynamics.r - 0.5 * dynamics.sigma^2) * dt

    for i in 1:nsteps
        Z = randn(rng)
        path1[i+1] = path1[i] * exp(drift + dynamics.sigma * sqrt_dt * Z)
        path2[i+1] = path2[i] * exp(drift + dynamics.sigma * sqrt_dt * (-Z))
    end

    return path1, path2
end

"""
    simulate_heston(S0, T, nsteps, dynamics; rng=Random.default_rng())

Generate a single Heston path using full truncation scheme.
"""
function simulate_heston(S0, T, nsteps, dynamics::HestonDynamics; rng=Random.default_rng())
    dt = T / nsteps
    sqrt_dt = sqrt(dt)

    path = Vector{typeof(S0 * one(dynamics.r))}(undef, nsteps + 1)
    path[1] = S0
    v = dynamics.v0

    for i in 1:nsteps
        Z1 = randn(rng)
        Z2 = dynamics.rho * Z1 + sqrt(1 - dynamics.rho^2) * randn(rng)

        v_plus = max(v, 0.0)
        sqrt_v = sqrt(v_plus)

        path[i+1] = path[i] * exp((dynamics.r - 0.5 * v_plus) * dt + sqrt_v * sqrt_dt * Z1)
        v = v + dynamics.kappa * (dynamics.theta - v_plus) * dt + dynamics.xi * sqrt_v * sqrt_dt * Z2
    end

    return path
end

# ============================================================================
# Monte Carlo Pricing
# ============================================================================

"""
    MCResult

Result of Monte Carlo simulation.
"""
struct MCResult{T}
    price::T
    stderr::T
    npaths::Int
    ci_lower::T  # 95% CI
    ci_upper::T
end

"""
    mc_price(S0, T, payoff, dynamics; npaths=10000, nsteps=252, antithetic=true, rng=nothing)

Price a derivative using Monte Carlo simulation.

# Arguments
- `S0` - Initial spot price
- `T` - Time to maturity
- `payoff` - Payoff structure
- `dynamics` - Price dynamics (GBM or Heston)
- `npaths` - Number of simulation paths
- `nsteps` - Time steps per path
- `antithetic` - Use antithetic variates for variance reduction
- `rng` - Random number generator (optional)

# Returns
MCResult with price, standard error, and confidence interval.
"""
function mc_price(S0, T, pf::AbstractPayoff, dynamics::GBMDynamics;
                  npaths::Int=10000, nsteps::Int=252, antithetic::Bool=true,
                  rng=nothing)

    rng = isnothing(rng) ? Random.default_rng() : rng
    r = dynamics.r
    df = exp(-r * T)

    # Use eltype that supports AD (Dual numbers)
    RT = typeof(S0 * one(r))
    payoffs = Vector{RT}(undef, antithetic ? npaths ÷ 2 : npaths)

    if antithetic
        for i in 1:(npaths ÷ 2)
            path1, path2 = simulate_gbm_antithetic(S0, T, nsteps, dynamics; rng=rng)
            payoffs[i] = 0.5 * (payoff(pf, path1) + payoff(pf, path2))
        end
    else
        for i in 1:npaths
            path = simulate_gbm(S0, T, nsteps, dynamics; rng=rng)
            payoffs[i] = payoff(pf, path)
        end
    end

    price_est = df * mean(payoffs)
    stderr = df * std(payoffs) / sqrt(length(payoffs))

    return MCResult(price_est, stderr, npaths, price_est - 1.96*stderr, price_est + 1.96*stderr)
end

function mc_price(S0, T, pf::AbstractPayoff, dynamics::HestonDynamics;
                  npaths::Int=10000, nsteps::Int=252, rng=nothing)

    rng = isnothing(rng) ? Random.default_rng() : rng
    r = dynamics.r
    df = exp(-r * T)

    RT = typeof(S0 * one(r))
    payoffs = Vector{RT}(undef, npaths)

    for i in 1:npaths
        path = simulate_heston(S0, T, nsteps, dynamics; rng=rng)
        payoffs[i] = payoff(pf, path)
    end

    price_est = df * mean(payoffs)
    stderr = df * std(payoffs) / sqrt(npaths)

    return MCResult(price_est, stderr, npaths, price_est - 1.96*stderr, price_est + 1.96*stderr)
end

# ============================================================================
# Monte Carlo Greeks (Pathwise Method)
# ============================================================================

"""
    mc_delta(S0, T, payoff, dynamics; npaths=10000, nsteps=252, backend=current_backend())

Compute delta using pathwise differentiation with AD.
"""
function mc_delta(S0, T, payoff::AbstractPayoff, dynamics::GBMDynamics;
                  npaths::Int=10000, nsteps::Int=252, backend=current_backend())

    function price_fn(s0)
        # Use fixed seed for consistent paths across perturbations
        rng = Random.MersenneTwister(42)
        result = mc_price(s0, T, payoff, dynamics; npaths=npaths, nsteps=nsteps,
                         antithetic=false, rng=rng)
        return result.price
    end

    g = gradient(x -> price_fn(x[1]), [S0]; backend=backend)
    return g[1]
end

"""
    mc_greeks(S0, T, payoff, dynamics; npaths=10000, nsteps=252, backend=current_backend())

Compute delta and vega using pathwise differentiation.
"""
function mc_greeks(S0, T, payoff::AbstractPayoff, dynamics::GBMDynamics;
                   npaths::Int=10000, nsteps::Int=252, backend=current_backend())

    function price_fn(params)
        s0, sigma = params
        dyn = GBMDynamics(dynamics.r, sigma)
        rng = Random.MersenneTwister(42)
        result = mc_price(s0, T, payoff, dyn; npaths=npaths, nsteps=nsteps,
                         antithetic=false, rng=rng)
        return result.price
    end

    g = gradient(price_fn, [S0, dynamics.sigma]; backend=backend)
    return (delta=g[1], vega=g[2])
end

# ============================================================================
# Longstaff-Schwartz American Option Pricing
# ============================================================================

"""
    AmericanPut

American put option for LSM pricing.
"""
struct AmericanPut <: AbstractPayoff
    K::Float64
end

"""
    AmericanCall

American call option for LSM pricing.
"""
struct AmericanCall <: AbstractPayoff
    K::Float64
end

# Intrinsic value
intrinsic(p::AmericanPut, S) = max(p.K - S, 0.0)
intrinsic(p::AmericanCall, S) = max(S - p.K, 0.0)

"""
    lsm_price(S0, T, option, dynamics; npaths=10000, nsteps=50, rng=nothing)

Price an American option using the Longstaff-Schwartz Monte Carlo method.

# Algorithm
1. Simulate paths forward
2. Work backwards from expiry
3. At each step, regress continuation value on polynomial basis of spot
4. Exercise when intrinsic value > continuation value

# Arguments
- `S0` - Initial spot price
- `T` - Time to maturity
- `option` - AmericanPut or AmericanCall
- `dynamics` - GBMDynamics
- `npaths` - Number of simulation paths
- `nsteps` - Number of exercise dates
- `rng` - Random number generator

# Returns
MCResult with American option price and standard error.

# Reference
Longstaff & Schwartz (2001), "Valuing American Options by Simulation"
"""
function lsm_price(S0, T, option::Union{AmericanPut,AmericanCall}, dynamics::GBMDynamics;
                   npaths::Int=10000, nsteps::Int=50, rng=nothing)

    rng = isnothing(rng) ? Random.default_rng() : rng
    r = dynamics.r
    dt = T / nsteps
    df = exp(-r * dt)  # discount factor per step

    # Generate all paths upfront (npaths x nsteps+1 matrix)
    paths = Matrix{Float64}(undef, npaths, nsteps + 1)
    for i in 1:npaths
        path = simulate_gbm(S0, T, nsteps, dynamics; rng=rng)
        paths[i, :] = path
    end

    # Cash flow matrix: when and how much each path receives
    # Initialize with terminal payoff
    cashflows = zeros(npaths)
    exercise_time = fill(nsteps, npaths)  # when each path exercises

    for i in 1:npaths
        cashflows[i] = intrinsic(option, paths[i, end])
    end

    # Work backwards through time
    for t in (nsteps-1):-1:1
        # Spot prices at time t
        S_t = paths[:, t+1]

        # Find in-the-money paths (candidates for early exercise)
        itm = intrinsic.(Ref(option), S_t) .> 0

        if sum(itm) < 3
            continue  # Need enough points for regression
        end

        # Discounted future cashflows for ITM paths
        # Discount from their exercise time to current time
        Y = zeros(sum(itm))
        itm_indices = findall(itm)
        for (j, i) in enumerate(itm_indices)
            steps_to_exercise = exercise_time[i] - t
            Y[j] = cashflows[i] * df^steps_to_exercise
        end

        # Regression: fit continuation value to polynomial basis
        # Using Laguerre-like polynomials: 1, S, S^2 (simple but effective)
        X_itm = S_t[itm]
        X_mat = hcat(ones(length(X_itm)), X_itm, X_itm.^2)

        # Least squares: β = (X'X)^(-1) X'Y
        β = X_mat \ Y

        # Continuation value estimate for ITM paths
        continuation = X_mat * β

        # Immediate exercise value
        immediate = intrinsic.(Ref(option), X_itm)

        # Exercise if immediate > continuation
        for (j, i) in enumerate(itm_indices)
            if immediate[j] > continuation[j]
                cashflows[i] = immediate[j]
                exercise_time[i] = t
            end
        end
    end

    # Discount all cashflows back to time 0
    discounted = [cashflows[i] * df^exercise_time[i] for i in 1:npaths]

    price_est = mean(discounted)
    stderr = std(discounted) / sqrt(npaths)

    return MCResult(price_est, stderr, npaths, price_est - 1.96*stderr, price_est + 1.96*stderr)
end

# ============================================================================
# Exports
# ============================================================================

export AbstractDynamics, GBMDynamics, HestonDynamics
export AbstractPayoff, EuropeanCall, EuropeanPut, AsianCall, AsianPut
export UpAndOutCall, DownAndOutPut, AmericanPut, AmericanCall
export payoff, simulate_gbm, simulate_heston, simulate_gbm_antithetic
export MCResult, mc_price, mc_delta, mc_greeks
export lsm_price, intrinsic

end
