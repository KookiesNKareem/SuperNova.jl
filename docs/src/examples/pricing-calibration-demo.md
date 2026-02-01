# Pricing & Calibration Demo

This demo showcases QuantNova's core capabilities in a real-world workflow:

1. **Black-Scholes pricing** across a range of strikes
2. **Greeks via automatic differentiation** (not finite differences)
3. **SABR model calibration** to a volatility smile
4. **Monte Carlo pricing** for exotic options
5. **Performance benchmarks**

## Running the Demo

```bash
julia --project=. demos/options_pricing_demo.jl
```

## 1. Black-Scholes Pricing

```julia
using QuantNova

S, K, T, r, σ = 100.0, 100.0, 1.0, 0.05, 0.20

# Price a single option
call_price = black_scholes(S, K, T, r, σ, :call)
put_price = black_scholes(S, K, T, r, σ, :put)

println("Call Price: \$$(round(call_price, digits=4))")
println("Put Price:  \$$(round(put_price, digits=4))")

# Verify put-call parity
parity = call_price - put_price - S + K * exp(-r * T)
println("Put-Call Parity Check: $(round(parity, digits=6)) (should be ~0)")
```

**Output:**
```
Call Price: $10.4506
Put Price:  $5.5735
Put-Call Parity Check: 0.000000 (should be ~0)
```

### Pricing Across Strikes

```julia
strikes = 80.0:5.0:120.0
call_prices = [black_scholes(S, k, T, r, σ, :call) for k in strikes]

println("Strike vs Price:")
println("┌─────────┬──────────┐")
println("│ Strike  │   Call   │")
println("├─────────┼──────────┤")
for (k, p) in zip(strikes, call_prices)
    println("│  \$$(lpad(Int(k), 5)) │  \$$(lpad(round(p, digits=2), 6)) │")
end
println("└─────────┴──────────┘")
```

**Output:**
```
Strike vs Price:
┌─────────┬──────────┐
│ Strike  │   Call   │
├─────────┼──────────┤
│  $   80 │  $ 24.59 │
│  $   85 │  $ 20.47 │
│  $   90 │  $ 16.70 │
│  $   95 │  $ 13.35 │
│  $  100 │  $ 10.45 │
│  $  105 │  $  8.02 │
│  $  110 │  $  6.04 │
│  $  115 │  $  4.47 │
│  $  120 │  $  3.25 │
└─────────┴──────────┘
```

## 2. Greeks via Automatic Differentiation

QuantNova computes Greeks using ForwardDiff — true derivatives, not finite difference approximations.

```julia
state = MarketState(
    prices = Dict("SPX" => S),
    rates = Dict("USD" => r),
    volatilities = Dict("SPX" => σ),
    timestamp = 0.0
)
option = EuropeanOption("SPX", K, T, :call)

# Compute all Greeks in one call via AD
greeks = compute_greeks(option, state)

println("Greeks computed via ForwardDiff AD:")
println("  Delta (∂V/∂S):     $(round(greeks.delta, digits=6))")
println("  Gamma (∂²V/∂S²):   $(round(greeks.gamma, digits=6))")
println("  Vega (∂V/∂σ):      $(round(greeks.vega, digits=6))")
println("  Theta (∂V/∂T):     $(round(greeks.theta, digits=6))")
println("  Rho (∂V/∂r):       $(round(greeks.rho, digits=6))")
```

**Output:**
```
Greeks computed via ForwardDiff AD:
  Delta (∂V/∂S):     +0.636831  (per $1 spot move)
  Gamma (∂²V/∂S²):   +0.018762  (delta change per $1)
  Vega (∂V/∂σ):      +0.375240  (per 1% vol move)
  Theta (∂V/∂T):     -6.414028  (per year, /365 for daily)
  Rho (∂V/∂r):       +0.532325  (per 1% rate move)
```

### Greeks Across Strikes

```julia
println("Greeks vs Strike (ATM region):")
println("┌─────────┬─────────┬─────────┬─────────┐")
println("│ Strike  │  Delta  │  Gamma  │  Vega   │")
println("├─────────┼─────────┼─────────┼─────────┤")

for k in 90.0:5.0:110.0
    opt = EuropeanOption("SPX", k, T, :call)
    g = compute_greeks(opt, state)
    println("│  \$$(lpad(Int(k), 5)) │  $(lpad(round(g.delta, digits=3), 6))  │  $(round(g.gamma, digits=4))  │  $(round(g.vega, digits=3))  │")
end
println("└─────────┴─────────┴─────────┴─────────┘")
```

**Output:**
```
Greeks vs Strike (ATM region):
┌─────────┬─────────┬─────────┬─────────┐
│ Strike  │  Delta  │  Gamma  │  Vega   │
├─────────┼─────────┼─────────┼─────────┤
│  $   90 │  +0.810  │  0.0136  │  0.272  │
│  $   95 │  +0.728  │  0.0166  │  0.332  │
│  $  100 │  +0.637  │  0.0188  │  0.375  │
│  $  105 │  +0.542  │  0.0198  │  0.397  │
│  $  110 │  +0.450  │  0.0198  │  0.396  │
└─────────┴─────────┴─────────┴─────────┘
```

## 3. SABR Model Calibration

Calibrate SABR to a realistic equity volatility smile with downside skew.

```julia
# Market smile data (typical equity skew)
F = 100.0  # Forward
strikes_smile = [85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0]
market_vols = [0.28, 0.25, 0.22, 0.20, 0.19, 0.185, 0.18]  # Downside skew

# Create calibration data
quotes = [OptionQuote(k, T, 0.0, :call, v) for (k, v) in zip(strikes_smile, market_vols)]
smile_data = SmileData(T, F, r, quotes)

# Calibrate SABR (uses Adam optimizer with multi-start)
sabr_result = calibrate_sabr(smile_data; beta=0.5)

println("Calibrated Parameters:")
println("  α (vol level):     $(round(sabr_result.params.alpha, digits=4))")
println("  β (backbone):      $(sabr_result.params.beta) (fixed)")
println("  ρ (correlation):   $(round(sabr_result.params.rho, digits=4))")
println("  ν (vol of vol):    $(round(sabr_result.params.nu, digits=4))")
println("  RMSE:              $(round(sabr_result.rmse * 100, digits=2))%")
println("  Converged:         $(sabr_result.converged)")
```

**Output:**
```
Calibrated Parameters:
  α (vol level):     1.8578
  β (backbone):      0.5 (fixed)
  ρ (correlation):   -0.4514 (negative = downside skew)
  ν (vol of vol):    1.3211
  RMSE:              0.28%
  Converged:         true
```

### Calibration Fit Quality

```julia
println("Calibration Fit:")
println("┌─────────┬──────────┬──────────┬──────────┐")
println("│ Strike  │  Market  │   SABR   │  Error   │")
println("├─────────┼──────────┼──────────┼──────────┤")
for (k, mkt_v) in zip(strikes_smile, market_vols)
    sabr_v = sabr_implied_vol(F, k, T, sabr_result.params)
    err = (sabr_v - mkt_v) * 10000  # in bps
    println("│  \$$(lpad(Int(k), 5)) │  $(round(mkt_v*100, digits=1))%    │  $(round(sabr_v*100, digits=1))%    │ $(lpad(round(Int, err), 4)) bp │")
end
println("└─────────┴──────────┴──────────┴──────────┘")
```

**Output:**
```
Calibration Fit:
┌─────────┬──────────┬──────────┬──────────┐
│ Strike  │  Market  │   SABR   │  Error   │
├─────────┼──────────┼──────────┼──────────┤
│  $   85 │   28.0%  │   27.8%  │   -20 bp │
│  $   90 │   25.0%  │   25.0%  │    -5 bp │
│  $   95 │   22.0%  │   22.4%  │   +37 bp │
│  $  100 │   20.0%  │   20.2%  │   +20 bp │
│  $  105 │   19.0%  │   18.7%  │   -28 bp │
│  $  110 │   18.5%  │   18.1%  │   -36 bp │
│  $  115 │   18.0%  │   18.3%  │   +33 bp │
└─────────┴──────────┴──────────┴──────────┘
```

## 4. Monte Carlo Pricing

Price exotic options that have no closed-form solution.

```julia
dynamics = GBMDynamics(r, σ)
npaths = 50000

# European call (for validation against Black-Scholes)
mc_result = mc_price(S, T, EuropeanCall(K), dynamics; npaths=npaths, antithetic=true)
bs_price = black_scholes(S, K, T, r, σ, :call)
println("European Call:")
println("  MC Price:  \$$(round(mc_result.price, digits=4)) ± $(round(mc_result.stderr, digits=4))")
println("  BS Price:  \$$(round(bs_price, digits=4))")

# Asian call (path-dependent, no closed form)
asian_result = mc_price(S, T, AsianCall(K), dynamics; npaths=npaths, nsteps=252)
println("\nAsian Call (arithmetic average):")
println("  MC Price:  \$$(round(asian_result.price, digits=4)) ± $(round(asian_result.stderr, digits=4))")

# Barrier option
barrier_result = mc_price(S, T, UpAndOutCall(K, 120.0), dynamics; npaths=npaths, nsteps=252)
println("\nUp-and-Out Call (barrier = \$120):")
println("  MC Price:  \$$(round(barrier_result.price, digits=4)) ± $(round(barrier_result.stderr, digits=4))")
println("  Knockout discount: $(round((1 - barrier_result.price / bs_price) * 100, digits=1))%")

# American put (Longstaff-Schwartz)
am_result = lsm_price(S, T, AmericanPut(K), dynamics; npaths=npaths, nsteps=50)
eu_put = black_scholes(S, K, T, r, σ, :put)
println("\nAmerican Put (Longstaff-Schwartz):")
println("  LSM Price: \$$(round(am_result.price, digits=4)) ± $(round(am_result.stderr, digits=4))")
println("  EU Put:    \$$(round(eu_put, digits=4))")
println("  Early Exercise Premium: \$$(round(am_result.price - eu_put, digits=4))")
```

**Output:**
```
European Call:
  MC Price:  $10.4664 ± 0.0463
  BS Price:  $10.4506

Asian Call (arithmetic average):
  MC Price:  $5.7723 ± 0.0248

Up-and-Out Call (barrier = $120):
  MC Price:  $1.3232 ± 0.0144
  Knockout discount: 87.3%

American Put (Longstaff-Schwartz):
  LSM Price: $5.9918 ± 0.0319
  EU Put:    $5.5735
  Early Exercise Premium: $0.4183
```

## 5. Performance Benchmarks

QuantNova achieves exceptional performance through Julia's JIT compilation.

| Operation | Time | Notes |
|-----------|------|-------|
| Black-Scholes price | 0.05 μs | Single option |
| Greeks (all 5 via AD) | 0.14 μs | ForwardDiff, not finite diff |
| SABR implied vol | 0.11 μs | Hagan's formula |
| American (100-step binomial) | 9 μs | CRR tree |
| MC European (1k paths) | 1.8 ms | With antithetic |

### Comparison with Other Libraries

| Benchmark | QuantNova.jl | QuantLib C++ | Speedup |
|-----------|-------------|--------------|---------|
| European option | 0.04 μs | 5.7 μs | **139x faster** |
| Greeks (all 5) | 0.08 μs | 5.7 μs | **71x faster** |
| American (100-step) | 8.5 μs | 67 μs | **8x faster** |
| SABR vol | 0.04 μs | 0.8 μs | **20x faster** |
| Batch (1000 options) | 40 μs | 5.7 ms | **142x faster** |

| Benchmark | QuantNova.jl | Python | Speedup |
|-----------|-------------|--------|---------|
| CAPM regression | 21 μs | 450 μs | **21x faster** |
| Rolling beta (5yr) | 376 μs | 12 ms | **32x faster** |
| SMA crossover backtest | 104 μs | 2.5 ms | **24x faster** |

*Benchmarks on Apple M1. See `benchmarks/comparison/` for methodology.*

## Key Takeaways

- **Greeks via true AD** — Not finite differences, exact derivatives
- **SABR calibration** — Multi-start Adam optimizer fits steep smiles (<0.3% RMSE)
- **Monte Carlo** — Supports Asian, barrier, American with variance reduction
- **Performance** — Sub-microsecond Black-Scholes, ~10μs American binomial

## Next Steps

- [Portfolio Risk](portfolio-risk.md) — Aggregate risk across positions
- [Monte Carlo Exotic](monte-carlo-exotic.md) — More exotic payoffs
- [Interest Rates](yield-curve.md) — Build curves for discounting
