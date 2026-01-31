# Quasar

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://KookiesNKareem.github.io/Quasar.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://KookiesNKareem.github.io/Quasar.jl/dev/)
[![Build Status](https://github.com/KookiesNKareem/Quasar.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/KookiesNKareem/Quasar.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/KookiesNKareem/Quasar.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/KookiesNKareem/Quasar.jl)

A differentiable portfolio management library for Julia.

## Features

- **Differentiable by default**: Every computation flows through a unified AD system. Gradients are first-class outputs.
- **Backend-agnostic**: Same code runs on CPU (ForwardDiff) or GPU (Reactant+Enzyme). Write once, deploy anywhere.
- **Research to production**: Pure Julia reference implementations for debugging, optimized backends for production.
- **Composable abstractions**: Small, focused types that combine naturally.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/KookiesNKareem/Quasar.jl")
```

## Quick Start

### Pricing Options

```julia
using Quasar

# Create a stock
stock = Stock(:AAPL, 150.0)

# Create a European call option
option = EuropeanOption(
    stock,
    strike=155.0,
    expiry=0.5,      # 6 months
    rate=0.05,
    volatility=0.2,
    isCall=true
)

# Price using Black-Scholes
price(option)  # Returns option price
```

### Computing Greeks

```julia
# Compute all Greeks via automatic differentiation
greeks = compute_greeks(option)

greeks.delta  # Price sensitivity to underlying
greeks.gamma  # Delta sensitivity to underlying
greeks.vega   # Price sensitivity to volatility
greeks.theta  # Price sensitivity to time
greeks.rho    # Price sensitivity to interest rate
# Uses analytical Black-Scholes formulas (exact, fast)
# AD fallback available for exotic options without closed-form solutions
```

### Portfolio Management

```julia
# Create a portfolio
portfolio = Portfolio(
    [Stock(:AAPL, 150.0), Stock(:GOOGL, 140.0)],
    [100.0, 50.0]  # positions
)

# Get portfolio value
value(portfolio)

# Compute portfolio Greeks (for options)
portfolio_greeks(portfolio)
```

### Risk Measures

```julia
returns = randn(1000) * 0.02  # Simulated returns

# Value at Risk (95% confidence)
compute(VaR(0.95), returns)

# Conditional VaR (Expected Shortfall)
compute(CVaR(0.95), returns)

# Volatility
compute(Volatility(), returns)

# Sharpe Ratio
compute(Sharpe(0.02), returns)  # 2% risk-free rate

# Maximum Drawdown
compute(MaxDrawdown(), cumsum(returns))
```

### Portfolio Optimization

```julia
# Expected returns and covariance matrix
mu = [0.10, 0.12, 0.08]
Sigma = [0.04 0.01 0.005;
         0.01 0.05 0.01;
         0.005 0.01 0.03]

# Mean-variance optimization
result = optimize(mu, Sigma, MeanVariance(target_return=0.10))
result.weights      # Optimal portfolio weights
result.objective    # Achieved variance

# Sharpe ratio maximization
result = optimize(mu, Sigma, SharpeMaximizer(risk_free_rate=0.02))
```

### Monte Carlo Pricing

```julia
# GBM dynamics for European/Asian/Barrier options
dynamics = GBMDynamics(0.05, 0.2)  # r=5%, σ=20%

# European options with variance reduction
result = mc_price(100.0, 1.0, EuropeanCall(105.0), dynamics;
                  npaths=50000, antithetic=true)
result.price   # Monte Carlo estimate
result.stderr  # Standard error

# Asian options (path-dependent)
mc_price(100.0, 1.0, AsianCall(100.0), dynamics; npaths=50000)

# Barrier options
mc_price(100.0, 1.0, UpAndOutCall(100.0, 130.0), dynamics; npaths=50000)

# Heston stochastic volatility
heston = HestonDynamics(0.05, 0.04, 2.0, 0.04, 0.3, -0.7)
mc_price(100.0, 1.0, EuropeanCall(100.0), heston; npaths=50000)
```

### American Options (Longstaff-Schwartz)

```julia
# Price American options using LSM algorithm
dynamics = GBMDynamics(0.05, 0.2)

am_put = lsm_price(100.0, 1.0, AmericanPut(100.0), dynamics;
                   npaths=50000, nsteps=50)
am_put.price   # American option price
am_put.stderr  # Standard error

# Compare to European (American >= European for puts)
eu_put = mc_price(100.0, 1.0, EuropeanPut(100.0), dynamics; npaths=50000)
early_exercise_premium = am_put.price - eu_put.price
```

### Model Calibration

```julia
# Calibrate SABR model to market smile
quotes = [OptionQuote(K, T, 0.0, :call, market_vol) for (K, market_vol) in data]
smile = SmileData(T, forward, rate, quotes)

result = calibrate_sabr(smile; beta=1.0)
result.params.alpha  # Fitted α
result.params.rho    # Fitted ρ (skew)
result.params.nu     # Fitted ν (smile curvature)
result.rmse          # Calibration error

# Price with calibrated SABR
sabr_implied_vol(F, K, T, result.params)
sabr_price(F, K, T, r, result.params, :call)

# Heston calibration to term structure
surface = VolSurface([smile1, smile2, smile3])
result = calibrate_heston(surface)
```

## AD Backends

Quasar supports multiple automatic differentiation backends:

```julia
using Quasar

# Pure Julia (finite differences, for debugging)
set_backend!(PureJuliaBackend())

# ForwardDiff (CPU, forward-mode AD)
set_backend!(ForwardDiffBackend())

# Reactant (GPU, Enzyme via Reactant) - requires Reactant.jl
using Reactant
set_backend!(ReactantBackend())

# Per-call backend override
gradient(f, x; backend=ForwardDiffBackend())
```

## Type Hierarchy

```
AbstractInstrument
├── AbstractEquity
│   └── Stock
├── AbstractDerivative
│   ├── AbstractOption
│   │   └── EuropeanOption
│   └── AbstractFuture
└── AbstractPortfolio
    └── Portfolio{I<:AbstractInstrument}
```

## Traits

Quasar uses Julia's Holy Traits pattern for capability dispatch:

- `Priceable` - Can compute present value given market state
- `Differentiable` - Participates in AD
- `HasGreeks` - Can compute sensitivities (Delta, Gamma, Vega, etc.)
- `Simulatable` - Can be included in Monte Carlo paths

## Notebooks

Interactive tutorials demonstrating key features:

| Notebook | Description |
|----------|-------------|
| [Volatility Smile Calibration](notebooks/volatility_smile_calibration.ipynb) | Calibrate SABR model to market smiles using AD-powered optimization |
| [American Options with LSM](notebooks/american_options_lsm.ipynb) | Price American options with Longstaff-Schwartz Monte Carlo |

## License

MIT License - see LICENSE file for details.
