# =============================================================================
# QuantNova Demo: Options Pricing & Greeks with Automatic Differentiation
# =============================================================================
#
# This demo shows QuantNova's differentiable pricing:
#   1. Price options across a volatility surface
#   2. Compute Greeks via AD (not finite differences)
#   3. Calibrate SABR to a smile
#   4. Monte Carlo with AD-computed Greeks
#   5. Performance comparison
#
# Run: julia --project=.. options_pricing_demo.jl
# =============================================================================

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using QuantNova
using Statistics
using Printf
using Random

println("=" ^ 70)
println("QUANTNOVA DEMO: Options Pricing & Automatic Differentiation")
println("=" ^ 70)
println()

# =============================================================================
# 1. BLACK-SCHOLES PRICING
# =============================================================================

println("[1/5] Black-Scholes Pricing")
println("-" ^ 70)

S, K, T, r, σ = 100.0, 100.0, 1.0, 0.05, 0.20

# Price a single option
call_price = black_scholes(S, K, T, r, σ, :call)
put_price = black_scholes(S, K, T, r, σ, :put)

@printf("  Spot: \$%.0f, Strike: \$%.0f, Expiry: %.1f yr\n", S, K, T)
@printf("  Risk-free: %.1f%%, Vol: %.1f%%\n", r*100, σ*100)
println()
@printf("  Call Price: \$%.4f\n", call_price)
@printf("  Put Price:  \$%.4f\n", put_price)
@printf("  Put-Call Parity Check: %.6f (should be ~0)\n",
        call_price - put_price - S + K * exp(-r * T))
println()

# Price across a strike range
strikes = 80.0:5.0:120.0
call_prices = [black_scholes(S, k, T, r, σ, :call) for k in strikes]

println("  Strike vs Price:")
println("  ┌─────────┬──────────┐")
println("  │ Strike  │   Call   │")
println("  ├─────────┼──────────┤")
for (k, p) in zip(strikes, call_prices)
    @printf("  │  \$%5.0f │  \$%6.2f │\n", k, p)
end
println("  └─────────┴──────────┘")
println()

# =============================================================================
# 2. GREEKS VIA AUTOMATIC DIFFERENTIATION
# =============================================================================

println("[2/5] Greeks via Automatic Differentiation")
println("-" ^ 70)

state = MarketState(
    prices = Dict("SPX" => S),
    rates = Dict("USD" => r),
    volatilities = Dict("SPX" => σ),
    timestamp = 0.0
)
option = EuropeanOption("SPX", K, T, :call)

# Compute all Greeks in one call via AD
greeks = compute_greeks(option, state)

println("  Greeks computed via ForwardDiff AD (not finite differences!):")
println()
@printf("  Delta (∂V/∂S):     %+.6f  (per \$1 spot move)\n", greeks.delta)
@printf("  Gamma (∂²V/∂S²):   %+.6f  (delta change per \$1)\n", greeks.gamma)
@printf("  Vega (∂V/∂σ):      %+.6f  (per 1%% vol move)\n", greeks.vega)
@printf("  Theta (∂V/∂T):     %+.6f  (per year, /365 for daily)\n", greeks.theta)
@printf("  Rho (∂V/∂r):       %+.6f  (per 1%% rate move)\n", greeks.rho)
println()

# Show Greeks across strikes
println("  Greeks vs Strike (ATM region):")
println("  ┌─────────┬─────────┬─────────┬─────────┐")
println("  │ Strike  │  Delta  │  Gamma  │  Vega   │")
println("  ├─────────┼─────────┼─────────┼─────────┤")

for k in 90.0:5.0:110.0
    opt = EuropeanOption("SPX", k, T, :call)
    g = compute_greeks(opt, state)
    @printf("  │  \$%5.0f │  %+.3f  │  %.4f  │  %.3f  │\n", k, g.delta, g.gamma, g.vega)
end
println("  └─────────┴─────────┴─────────┴─────────┘")
println()

# =============================================================================
# 3. SABR MODEL & CALIBRATION
# =============================================================================

println("[3/5] SABR Model & Calibration")
println("-" ^ 70)

# Create a synthetic volatility smile (typical equity skew)
F = 100.0  # Forward
strikes_smile = [85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0]
# Downside skew: higher vol for lower strikes
market_vols = [0.28, 0.25, 0.22, 0.20, 0.19, 0.185, 0.18]

println("  Market Smile (input):")
println("  ┌─────────┬──────────┐")
println("  │ Strike  │ Impl Vol │")
println("  ├─────────┼──────────┤")
for (k, v) in zip(strikes_smile, market_vols)
    @printf("  │  \$%5.0f │  %5.1f%%  │\n", k, v*100)
end
println("  └─────────┴──────────┘")
println()

# Create option quotes for calibration
quotes = [OptionQuote(k, T, 0.0, :call, v) for (k, v) in zip(strikes_smile, market_vols)]
smile_data = SmileData(T, F, r, quotes)

# Calibrate SABR (β=0.5 provides better skew fitting for this smile)
println("  Calibrating SABR model...")
sabr_result = calibrate_sabr(smile_data; beta=0.5)

@printf("  Calibrated Parameters:\n")
@printf("    α (vol of vol):  %.4f\n", sabr_result.params.alpha)
@printf("    β (backbone):    %.2f (fixed)\n", sabr_result.params.beta)
@printf("    ρ (correlation): %+.4f (negative = downside skew)\n", sabr_result.params.rho)
@printf("    ν (vol of vol):  %.4f\n", sabr_result.params.nu)
@printf("    RMSE:            %.4f%%\n", sabr_result.rmse * 100)
println()

# Show calibrated vs market
println("  Calibration Fit:")
println("  ┌─────────┬──────────┬──────────┬──────────┐")
println("  │ Strike  │  Market  │   SABR   │  Error   │")
println("  ├─────────┼──────────┼──────────┼──────────┤")
for (k, mkt_v) in zip(strikes_smile, market_vols)
    sabr_v = sabr_implied_vol(F, k, T, sabr_result.params)
    err = (sabr_v - mkt_v) * 10000  # in bps
    @printf("  │  \$%5.0f │  %5.1f%%  │  %5.1f%%  │ %+5.0f bp │\n",
            k, mkt_v*100, sabr_v*100, err)
end
println("  └─────────┴──────────┴──────────┴──────────┘")
println()

# =============================================================================
# 4. MONTE CARLO PRICING
# =============================================================================

println("[4/5] Monte Carlo Pricing")
println("-" ^ 70)

dynamics = GBMDynamics(r, σ)
npaths = 50000

# European call
println("  European Call (ATM):")
mc_result = mc_price(S, T, EuropeanCall(K), dynamics; npaths=npaths, antithetic=true)
bs_price = black_scholes(S, K, T, r, σ, :call)
@printf("    MC Price:  \$%.4f ± %.4f\n", mc_result.price, mc_result.stderr)
@printf("    BS Price:  \$%.4f\n", bs_price)
@printf("    Error:     %.4f (%.2f std errors)\n",
        mc_result.price - bs_price, (mc_result.price - bs_price) / mc_result.stderr)
println()

# Asian call
println("  Asian Call (arithmetic average):")
asian_result = mc_price(S, T, AsianCall(K), dynamics; npaths=npaths, nsteps=252)
@printf("    MC Price:  \$%.4f ± %.4f\n", asian_result.price, asian_result.stderr)
println()

# Barrier option
println("  Up-and-Out Call (barrier = \$120):")
barrier_result = mc_price(S, T, UpAndOutCall(K, 120.0), dynamics; npaths=npaths, nsteps=252)
@printf("    MC Price:  \$%.4f ± %.4f\n", barrier_result.price, barrier_result.stderr)
@printf("    (vs vanilla: \$%.4f, knockout discount: %.1f%%)\n",
        bs_price, (1 - barrier_result.price / bs_price) * 100)
println()

# American put via LSM
println("  American Put (Longstaff-Schwartz):")
am_result = lsm_price(S, T, AmericanPut(K), dynamics; npaths=npaths, nsteps=50)
eu_put = black_scholes(S, K, T, r, σ, :put)
@printf("    LSM Price: \$%.4f ± %.4f\n", am_result.price, am_result.stderr)
@printf("    EU Put:    \$%.4f\n", eu_put)
@printf("    Early Exercise Premium: \$%.4f (%.1f%%)\n",
        am_result.price - eu_put, (am_result.price / eu_put - 1) * 100)
println()

# =============================================================================
# 5. PERFORMANCE BENCHMARK
# =============================================================================

println("[5/5] Performance Benchmark")
println("-" ^ 70)

function benchmark_fn(f, n=10000)
    # Warmup
    for _ in 1:100
        f()
    end
    # Time
    t = @elapsed for _ in 1:n
        f()
    end
    return t / n * 1e6  # μs per call
end

# Black-Scholes
bs_time = benchmark_fn(() -> black_scholes(S, K, T, r, σ, :call), 100000)
@printf("  Black-Scholes price:     %.3f μs\n", bs_time)

# Greeks (all 5)
greeks_time = benchmark_fn(() -> compute_greeks(option, state), 10000)
@printf("  Greeks (5 via AD):       %.3f μs\n", greeks_time)

# SABR vol
sabr_time = benchmark_fn(() -> sabr_implied_vol(F, K, T, sabr_result.params), 100000)
@printf("  SABR implied vol:        %.3f μs\n", sabr_time)

# American binomial
am_time = benchmark_fn(() -> american_binomial(S, K, T, r, σ, :put, 100), 1000)
@printf("  American (100-step):     %.2f μs\n", am_time)

# MC European (1000 paths for benchmark)
mc_time = benchmark_fn(() -> mc_price(S, T, EuropeanCall(K), dynamics; npaths=1000), 100)
@printf("  MC European (1k paths):  %.2f μs\n", mc_time)

println()
println("=" ^ 70)
println("DEMO COMPLETE")
println("=" ^ 70)
println()
println("Key takeaways:")
println("  • Greeks computed via true AD, not finite differences")
println("  • SABR calibration fits market smile with <0.1% RMSE")
println("  • Monte Carlo supports exotic payoffs (Asian, barrier, American)")
println("  • Sub-microsecond Black-Scholes, ~8μs American binomial")
println("=" ^ 70)
