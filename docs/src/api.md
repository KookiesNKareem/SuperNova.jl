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
value
portfolio_greeks
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
optimize
OptimizationResult
```

!!! note "Planned Features"
    `CVaRObjective` and `KellyCriterion` types are defined but `optimize()` methods are not yet implemented.

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

## Interest Rates

### Yield Curves

```@docs
RateCurve
DiscountCurve
ZeroCurve
ForwardCurve
NelsonSiegelCurve
SvenssonCurve
discount
zero_rate
forward_rate
instantaneous_forward
fit_nelson_siegel
fit_svensson
```

### Interpolation

```@docs
LinearInterp
LogLinearInterp
CubicSplineInterp
```

### Bootstrapping

```@docs
DepositRate
FuturesRate
SwapRate
bootstrap
```

### Bonds

```@docs
Bond
ZeroCouponBond
FixedRateBond
FloatingRateBond
yield_to_maturity
duration
modified_duration
convexity
dv01
accrued_interest
clean_price
dirty_price
```

### Short-Rate Models

```@docs
ShortRateModel
Vasicek
CIR
HullWhite
bond_price
short_rate
simulate_short_rate
```

### Interest Rate Derivatives

```@docs
Caplet
Floorlet
Cap
Floor
Swaption
black_caplet
black_floorlet
black_cap
black_floor
```
