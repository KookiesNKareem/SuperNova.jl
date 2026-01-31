# API Reference

## Pricing

```@docs
black_scholes
price
```

## Greeks

```@docs
compute_greeks
GreeksResult
```

## Instruments

```@docs
Stock
EuropeanOption
Portfolio
MarketState
```

## AD System

```@docs
gradient
hessian
jacobian
value_and_gradient
current_backend
set_backend!
with_backend
```

## Backends

```@docs
ForwardDiffBackend
PureJuliaBackend
EnzymeBackend
ReactantBackend
```

## Monte Carlo

```@docs
GBMDynamics
HestonDynamics
mc_price
mc_price_qmc
mc_delta
mc_greeks
lsm_price
```

## Payoffs

```@docs
EuropeanCall
EuropeanPut
AsianCall
AsianPut
UpAndOutCall
DownAndOutPut
AmericanPut
AmericanCall
```

## Risk Measures

```@docs
VaR
CVaR
Volatility
Sharpe
MaxDrawdown
compute
```

## Optimization

```@docs
MeanVariance
SharpeMaximizer
CVaRObjective
KellyCriterion
optimize
OptimizationResult
```

## Stochastic Volatility Models

```@docs
SABRParams
sabr_implied_vol
sabr_price
HestonParams
heston_price
```

## Calibration

```@docs
OptionQuote
SmileData
VolSurface
calibrate_sabr
calibrate_heston
CalibrationResult
```
