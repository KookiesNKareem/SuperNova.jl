
# API Reference {#API-Reference}



## Pricing {#Pricing}
<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.Instruments.black_scholes' href='#QuantNova.Instruments.black_scholes'><span class="jlbinding">QuantNova.Instruments.black_scholes</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
black_scholes(S, K, T, r, σ, optiontype)
```


Black-Scholes option pricing formula.

**Arguments**
- `S` - Current price of underlying
  
- `K` - Strike price
  
- `T` - Time to expiration (years)
  
- `r` - Risk-free rate
  
- `σ` - Volatility
  
- `optiontype` - :call or :put
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/Instruments.jl#L95-L107" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.Instruments.price' href='#QuantNova.Instruments.price'><span class="jlbinding">QuantNova.Instruments.price</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
price(instrument, market_state)
```


Compute the current price of an instrument given market state.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/Instruments.jl#L35-L39" target="_blank" rel="noreferrer">source</a></Badge>

</details>


## Greeks {#Greeks}
<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.Instruments.compute_greeks' href='#QuantNova.Instruments.compute_greeks'><span class="jlbinding">QuantNova.Instruments.compute_greeks</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
compute_greeks(option, market_state; backend=current_backend())
```


Compute option Greeks. Uses analytical formulas when available (preferred), falls back to AD for exotic options without closed-form solutions.

**Arguments**
- `option` - The option to compute Greeks for
  
- `market_state` - Current market conditions
  
- `backend` - AD backend to use for fallback computation (default: current_backend())
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/Greeks.jl#L70-L80" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.Instruments.GreeksResult' href='#QuantNova.Instruments.GreeksResult'><span class="jlbinding">QuantNova.Instruments.GreeksResult</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
GreeksResult
```


Container for option Greeks including first-order, second-order, and cross-derivatives.

**First-order Greeks**
- `delta` - dV/dS (sensitivity to underlying price)
  
- `vega` - dV/dσ (sensitivity to volatility, scaled per 1% move)
  
- `theta` - -dV/dT (time decay per year)
  
- `rho` - dV/dr (sensitivity to rates, scaled per 1% move)
  

**Second-order Greeks**
- `gamma` - d²V/dS² (rate of change of delta)
  
- `vanna` - d²V/dSdσ (sensitivity of delta to volatility)
  
- `volga` - d²V/dσ² (sensitivity of vega to volatility, a.k.a. vomma)
  
- `charm` - d²V/dSdT (sensitivity of delta to time, a.k.a. delta decay)
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/Greeks.jl#L9-L25" target="_blank" rel="noreferrer">source</a></Badge>

</details>


## Instruments {#Instruments}
<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.Instruments.Stock' href='#QuantNova.Instruments.Stock'><span class="jlbinding">QuantNova.Instruments.Stock</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
Stock <: AbstractEquity
```


A simple equity instrument.

**Fields**
- `symbol::String` - Ticker symbol
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/Instruments.jl#L10-L17" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.Instruments.EuropeanOption' href='#QuantNova.Instruments.EuropeanOption'><span class="jlbinding">QuantNova.Instruments.EuropeanOption</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
EuropeanOption <: AbstractOption
```


A European-style option (exercise only at expiry).

**Fields**
- `underlying::String` - Symbol of underlying asset
  
- `strike::Float64` - Strike price
  
- `expiry::Float64` - Time to expiration (in years)
  
- `optiontype::Symbol` - :call or :put
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/Instruments.jl#L51-L61" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.PortfolioModule.Portfolio' href='#QuantNova.PortfolioModule.Portfolio'><span class="jlbinding">QuantNova.PortfolioModule.Portfolio</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
Portfolio{I<:AbstractInstrument} <: AbstractPortfolio
```


A collection of financial instruments with associated position weights.

**Fields**
- `instruments::Vector{I}` - The instruments in the portfolio
  
- `weights::Vector{Float64}` - Position sizes (can be shares, contracts, or notional amounts)
  

**Constructors**
- `Portfolio(instruments, weights)` - Create from vectors (type inferred)
  
- `Portfolio{I}(instruments, weights)` - Create with explicit instrument type
  

**Example**

```julia
# Create a portfolio of options
call = EuropeanOption("AAPL", 150.0, 1.0, :call)
put = EuropeanOption("AAPL", 140.0, 1.0, :put)
portfolio = Portfolio([call, put], [100.0, -50.0])  # Long 100 calls, short 50 puts

# Price the portfolio
state = MarketState(
    prices=Dict("AAPL" => 150.0),
    rates=Dict("USD" => 0.05),
    volatilities=Dict("AAPL" => 0.2),
    timestamp=0.0
)
total_value = value(portfolio, state)
```


See also: [`value`](@ref), [`portfolio_greeks`](/api#QuantNova.PortfolioModule.portfolio_greeks)


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/Portfolio.jl#L13-L44" target="_blank" rel="noreferrer">source</a></Badge>

</details>


::: warning Missing docstring.

Missing docstring for `value`. Check Documenter&#39;s build log for details.

:::
<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.PortfolioModule.portfolio_greeks' href='#QuantNova.PortfolioModule.portfolio_greeks'><span class="jlbinding">QuantNova.PortfolioModule.portfolio_greeks</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
portfolio_greeks(portfolio, market_state)
```


Compute aggregated Greeks for the portfolio. Only includes instruments that have Greeks (options).


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/Portfolio.jl#L98-L103" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.Core.MarketState' href='#QuantNova.Core.MarketState'><span class="jlbinding">QuantNova.Core.MarketState</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
MarketState{P,R,V,T}
```


Immutable snapshot of market conditions.

**Fields**
- `prices::P` - Current prices by symbol
  
- `rates::R` - Interest rates by currency
  
- `volatilities::V` - Implied volatilities by symbol
  
- `timestamp::T` - Time of snapshot
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/Core.jl#L126-L136" target="_blank" rel="noreferrer">source</a></Badge>

</details>


## AD System {#AD-System}
<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.AD.gradient' href='#QuantNova.AD.gradient'><span class="jlbinding">QuantNova.AD.gradient</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
gradient(f, x; backend=current_backend())
```


Compute the gradient of `f` at `x` using the specified backend. Validates input for NaN/Inf values before computing.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/AD.jl#L111-L116" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.AD.hessian' href='#QuantNova.AD.hessian'><span class="jlbinding">QuantNova.AD.hessian</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
hessian(f, x; backend=current_backend())
```


Compute the Hessian of `f` at `x` using the specified backend.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/AD.jl#L178-L182" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.AD.jacobian' href='#QuantNova.AD.jacobian'><span class="jlbinding">QuantNova.AD.jacobian</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
jacobian(f, x; backend=current_backend())
```


Compute the Jacobian of `f` at `x` using the specified backend.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/AD.jl#L215-L219" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.AD.value_and_gradient' href='#QuantNova.AD.value_and_gradient'><span class="jlbinding">QuantNova.AD.value_and_gradient</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
value_and_gradient(f, x; backend=current_backend())
```


Compute the value and gradient of `f` at `x` in a single pass. Returns `(f(x), ∇f(x))`.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/AD.jl#L250-L255" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.AD.current_backend' href='#QuantNova.AD.current_backend'><span class="jlbinding">QuantNova.AD.current_backend</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
current_backend()
```


Return the currently active AD backend.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/AD.jl#L52-L56" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.AD.set_backend!' href='#QuantNova.AD.set_backend!'><span class="jlbinding">QuantNova.AD.set_backend!</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
set_backend!(backend::ADBackend)
```


Set the global AD backend.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/AD.jl#L59-L63" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.AD.with_backend' href='#QuantNova.AD.with_backend'><span class="jlbinding">QuantNova.AD.with_backend</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
with_backend(f, backend::ADBackend)
```


Execute `f` with `backend` as the active backend, then restore the original.

**Example**

```julia
with_backend(EnzymeBackend()) do
    gradient(loss, params)
end
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/AD.jl#L69-L80" target="_blank" rel="noreferrer">source</a></Badge>

</details>


## Backends {#Backends}
<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.AD.ForwardDiffBackend' href='#QuantNova.AD.ForwardDiffBackend'><span class="jlbinding">QuantNova.AD.ForwardDiffBackend</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
ForwardDiffBackend <: ADBackend
```


CPU-based forward-mode AD using ForwardDiff.jl. Best for low-dimensional problems and nested derivatives (Greeks).


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/AD.jl#L19-L24" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.AD.PureJuliaBackend' href='#QuantNova.AD.PureJuliaBackend'><span class="jlbinding">QuantNova.AD.PureJuliaBackend</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
PureJuliaBackend <: ADBackend
```


Reference implementation using finite differences. Slow but always works. Useful for debugging and testing.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/AD.jl#L11-L16" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.AD.EnzymeBackend' href='#QuantNova.AD.EnzymeBackend'><span class="jlbinding">QuantNova.AD.EnzymeBackend</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
EnzymeBackend <: ADBackend
```


CPU/GPU AD using Enzyme.jl (LLVM-based differentiation). Supports both forward and reverse mode.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/AD.jl#L35-L40" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.AD.ReactantBackend' href='#QuantNova.AD.ReactantBackend'><span class="jlbinding">QuantNova.AD.ReactantBackend</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
ReactantBackend <: ADBackend
```


GPU-accelerated AD using Reactant.jl + Enzyme. Best for high-dimensional problems (portfolio optimization).


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/AD.jl#L27-L32" target="_blank" rel="noreferrer">source</a></Badge>

</details>


## Monte Carlo {#Monte-Carlo}
<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.MonteCarlo.GBMDynamics' href='#QuantNova.MonteCarlo.GBMDynamics'><span class="jlbinding">QuantNova.MonteCarlo.GBMDynamics</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
GBMDynamics(r, σ)
```


Geometric Brownian Motion: dS = r_S_dt + σ_S_dW


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/MonteCarlo.jl#L21-L25" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.MonteCarlo.HestonDynamics' href='#QuantNova.MonteCarlo.HestonDynamics'><span class="jlbinding">QuantNova.MonteCarlo.HestonDynamics</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
HestonDynamics(r, v0, κ, θ, ξ, ρ)
```


Heston stochastic volatility model.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/MonteCarlo.jl#L31-L35" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.MonteCarlo.mc_price' href='#QuantNova.MonteCarlo.mc_price'><span class="jlbinding">QuantNova.MonteCarlo.mc_price</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
mc_price(S0, T, payoff, dynamics; npaths=10000, nsteps=252, antithetic=true, rng=nothing)
```


Price a derivative using Monte Carlo simulation.

**Arguments**
- `S0` - Initial spot price
  
- `T` - Time to maturity
  
- `payoff` - Payoff structure
  
- `dynamics` - Price dynamics (GBM or Heston)
  
- `npaths` - Number of simulation paths
  
- `nsteps` - Time steps per path
  
- `antithetic` - Use antithetic variates for variance reduction
  
- `rng` - Random number generator (optional)
  

**Returns**

MCResult with price, standard error, and confidence interval.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/MonteCarlo.jl#L398-L415" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.MonteCarlo.mc_price_qmc' href='#QuantNova.MonteCarlo.mc_price_qmc'><span class="jlbinding">QuantNova.MonteCarlo.mc_price_qmc</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
mc_price_qmc(S0, T, payoff, dynamics; npaths=10000, nsteps=252)
```


Price a derivative using Quasi-Monte Carlo (Sobol sequences).

This is deterministic and differentiable with Enzyme. Better convergence than pseudo-random MC: O(1/N) vs O(1/√N).


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/MonteCarlo.jl#L347-L354" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.MonteCarlo.mc_delta' href='#QuantNova.MonteCarlo.mc_delta'><span class="jlbinding">QuantNova.MonteCarlo.mc_delta</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
mc_delta(S0, T, payoff, dynamics; npaths=10000, nsteps=252, backend=current_backend())
```


Compute delta using pathwise differentiation with AD.

Automatically uses QMC (Sobol sequences) when backend is EnzymeBackend, since Enzyme cannot differentiate through pseudo-random number generators.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/MonteCarlo.jl#L471-L478" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.MonteCarlo.mc_greeks' href='#QuantNova.MonteCarlo.mc_greeks'><span class="jlbinding">QuantNova.MonteCarlo.mc_greeks</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
mc_greeks(S0, T, payoff, dynamics; npaths=10000, nsteps=252, backend=current_backend())
```


Compute delta and vega using pathwise differentiation.

Automatically uses QMC (Sobol sequences) when backend is EnzymeBackend, since Enzyme cannot differentiate through pseudo-random number generators.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/MonteCarlo.jl#L503-L510" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.MonteCarlo.lsm_price' href='#QuantNova.MonteCarlo.lsm_price'><span class="jlbinding">QuantNova.MonteCarlo.lsm_price</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



**Arguments**
- `S0` - Initial spot price
  
- `T` - Time to maturity
  
- `option` - AmericanPut or AmericanCall
  
- `dynamics` - GBMDynamics
  
- `npaths` - Number of simulation paths
  
- `nsteps` - Number of exercise dates
  
- `rng` - Random number generator
  

**Returns**

MCResult with American option price and standard error.

**Reference**

Longstaff &amp; Schwartz (2001), &quot;Valuing American Options by Simulation&quot;


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/MonteCarlo.jl#L580-L596" target="_blank" rel="noreferrer">source</a></Badge>

</details>


## Payoffs {#Payoffs}
<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.MonteCarlo.EuropeanCall' href='#QuantNova.MonteCarlo.EuropeanCall'><span class="jlbinding">QuantNova.MonteCarlo.EuropeanCall</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
EuropeanCall(K)
```


European call option payoff: max(S_T - K, 0).

**Arguments**
- `K::Float64` - Strike price
  

**Example**

```julia
payoff_fn = EuropeanCall(100.0)
result = mc_price(payoff_fn, dynamics, 1.0, 10000)
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/MonteCarlo.jl#L59-L72" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.MonteCarlo.EuropeanPut' href='#QuantNova.MonteCarlo.EuropeanPut'><span class="jlbinding">QuantNova.MonteCarlo.EuropeanPut</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
EuropeanPut(K)
```


European put option payoff: max(K - S_T, 0).

**Arguments**
- `K::Float64` - Strike price
  

**Example**

```julia
payoff_fn = EuropeanPut(100.0)
result = mc_price(payoff_fn, dynamics, 1.0, 10000)
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/MonteCarlo.jl#L77-L90" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.MonteCarlo.AsianCall' href='#QuantNova.MonteCarlo.AsianCall'><span class="jlbinding">QuantNova.MonteCarlo.AsianCall</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
AsianCall(K)
```


Asian call option with arithmetic average: max(avg(S) - K, 0).

The payoff is based on the arithmetic average of the price path, not just the terminal value.

**Arguments**
- `K::Float64` - Strike price
  

**Example**

```julia
payoff_fn = AsianCall(100.0)
result = mc_price(payoff_fn, dynamics, 1.0, 10000; nsteps=252)
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/MonteCarlo.jl#L95-L111" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.MonteCarlo.AsianPut' href='#QuantNova.MonteCarlo.AsianPut'><span class="jlbinding">QuantNova.MonteCarlo.AsianPut</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
AsianPut(K)
```


Asian put option with arithmetic average: max(K - avg(S), 0).

The payoff is based on the arithmetic average of the price path, not just the terminal value.

**Arguments**
- `K::Float64` - Strike price
  

**Example**

```julia
payoff_fn = AsianPut(100.0)
result = mc_price(payoff_fn, dynamics, 1.0, 10000; nsteps=252)
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/MonteCarlo.jl#L116-L132" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.MonteCarlo.UpAndOutCall' href='#QuantNova.MonteCarlo.UpAndOutCall'><span class="jlbinding">QuantNova.MonteCarlo.UpAndOutCall</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
UpAndOutCall(K, barrier)
```


Up-and-out barrier call option.

The option becomes worthless (knocks out) if the underlying price ever touches or exceeds the barrier during the option&#39;s life. Otherwise, pays max(S_T - K, 0) at expiry.

**Arguments**
- `K::Float64` - Strike price
  
- `barrier::Float64` - Upper barrier level (must be &gt; K for typical usage)
  

**Example**

```julia
payoff_fn = UpAndOutCall(100.0, 120.0)  # Knocks out if S >= 120
result = mc_price(payoff_fn, dynamics, 1.0, 10000; nsteps=252)
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/MonteCarlo.jl#L137-L155" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.MonteCarlo.DownAndOutPut' href='#QuantNova.MonteCarlo.DownAndOutPut'><span class="jlbinding">QuantNova.MonteCarlo.DownAndOutPut</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
DownAndOutPut(K, barrier)
```


Down-and-out barrier put option.

The option becomes worthless (knocks out) if the underlying price ever touches or falls below the barrier during the option&#39;s life. Otherwise, pays max(K - S_T, 0) at expiry.

**Arguments**
- `K::Float64` - Strike price
  
- `barrier::Float64` - Lower barrier level (must be &lt; K for typical usage)
  

**Example**

```julia
payoff_fn = DownAndOutPut(100.0, 80.0)  # Knocks out if S <= 80
result = mc_price(payoff_fn, dynamics, 1.0, 10000; nsteps=252)
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/MonteCarlo.jl#L161-L179" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.MonteCarlo.AmericanPut' href='#QuantNova.MonteCarlo.AmericanPut'><span class="jlbinding">QuantNova.MonteCarlo.AmericanPut</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
AmericanPut
```


American put option for LSM pricing.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/MonteCarlo.jl#L543-L547" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.MonteCarlo.AmericanCall' href='#QuantNova.MonteCarlo.AmericanCall'><span class="jlbinding">QuantNova.MonteCarlo.AmericanCall</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
AmericanCall
```


American call option for LSM pricing.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/MonteCarlo.jl#L552-L556" target="_blank" rel="noreferrer">source</a></Badge>

</details>


## Risk Measures {#Risk-Measures}
<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.Risk.VaR' href='#QuantNova.Risk.VaR'><span class="jlbinding">QuantNova.Risk.VaR</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
VaR <: AbstractRiskMeasure
```


Value at Risk at specified confidence level.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/Risk.jl#L13-L17" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.Risk.CVaR' href='#QuantNova.Risk.CVaR'><span class="jlbinding">QuantNova.Risk.CVaR</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
CVaR <: AbstractRiskMeasure
```


Conditional Value at Risk (Expected Shortfall) at specified confidence level.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/Risk.jl#L27-L31" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.Risk.Volatility' href='#QuantNova.Risk.Volatility'><span class="jlbinding">QuantNova.Risk.Volatility</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
Volatility <: AbstractRiskMeasure
```


Standard deviation of returns.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/Risk.jl#L41-L45" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.Risk.Sharpe' href='#QuantNova.Risk.Sharpe'><span class="jlbinding">QuantNova.Risk.Sharpe</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
Sharpe <: AbstractRiskMeasure
```


Sharpe ratio (excess return / volatility).

**Fields**
- `rf::Float64` - Annualized risk-free rate (e.g., 0.05 for 5%)
  
- `periods_per_year::Int` - Number of return periods per year (252 for daily, 52 for weekly, 12 for monthly)
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/Risk.jl#L48-L56" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.Risk.MaxDrawdown' href='#QuantNova.Risk.MaxDrawdown'><span class="jlbinding">QuantNova.Risk.MaxDrawdown</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
MaxDrawdown <: AbstractRiskMeasure
```


Maximum peak-to-trough decline.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/Risk.jl#L64-L68" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.Risk.compute' href='#QuantNova.Risk.compute'><span class="jlbinding">QuantNova.Risk.compute</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
compute(measure::AbstractRiskMeasure, returns)
```


Compute the risk measure for given returns.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/Risk.jl#L71-L75" target="_blank" rel="noreferrer">source</a></Badge>

</details>


## Optimization {#Optimization}
<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.Optimization.MeanVariance' href='#QuantNova.Optimization.MeanVariance'><span class="jlbinding">QuantNova.Optimization.MeanVariance</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
MeanVariance
```


Mean-variance optimization objective (Markowitz).


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/Optimization.jl#L1773-L1777" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.Optimization.SharpeMaximizer' href='#QuantNova.Optimization.SharpeMaximizer'><span class="jlbinding">QuantNova.Optimization.SharpeMaximizer</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
SharpeMaximizer
```


Maximize Sharpe ratio (non-convex).


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/Optimization.jl#L1783-L1787" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.Optimization.optimize' href='#QuantNova.Optimization.optimize'><span class="jlbinding">QuantNova.Optimization.optimize</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
optimize(objective; kwargs...)
```


Optimize portfolio weights for given objective.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/Optimization.jl#L2553-L2557" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.Optimization.OptimizationResult' href='#QuantNova.Optimization.OptimizationResult'><span class="jlbinding">QuantNova.Optimization.OptimizationResult</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
OptimizationResult
```


Result of a portfolio optimization.

**Fields**
- `weights::Vector{Float64}` - Optimal portfolio weights (typically sum to 1)
  
- `objective::Float64` - Final objective function value (e.g., variance for MVO)
  
- `converged::Bool` - Whether the optimization converged successfully
  
- `iterations::Int` - Number of iterations used
  

**Example**

```julia
returns = [0.10, 0.08, 0.12]
cov_matrix = [0.04 0.01 0.02;
              0.01 0.03 0.01;
              0.02 0.01 0.05]

result = optimize(MeanVariance(returns, cov_matrix); target_return=0.10)
if result.converged
    println("Optimal weights: ", result.weights)
end
```


See also: [`optimize`](/api#QuantNova.Optimization.optimize), [`MeanVariance`](/api#QuantNova.Optimization.MeanVariance)


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/Optimization.jl#L2166-L2191" target="_blank" rel="noreferrer">source</a></Badge>

</details>


::: tip Planned Features

`CVaRObjective` and `KellyCriterion` types are defined but `optimize()` methods are not yet implemented.

:::

## Stochastic Volatility Models {#Stochastic-Volatility-Models}
<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.Models.SABRParams' href='#QuantNova.Models.SABRParams'><span class="jlbinding">QuantNova.Models.SABRParams</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
SABRParams{T1,T2,T3,T4}
```


SABR stochastic volatility model parameters.

**Fields**
- `alpha::T1` - Initial volatility level (α &gt; 0)
  
- `beta::T2` - CEV exponent (0 ≤ β ≤ 1, often fixed at 0.5 for rates, 1.0 for equities)
  
- `rho::T3` - Correlation between spot and vol (-1 &lt; ρ &lt; 1)
  
- `nu::T4` - Vol of vol (ν &gt; 0)
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/Models.jl#L6-L16" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.Models.sabr_implied_vol' href='#QuantNova.Models.sabr_implied_vol'><span class="jlbinding">QuantNova.Models.sabr_implied_vol</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
sabr_implied_vol(F, K, T, params::SABRParams)
```


Compute SABR implied volatility using Hagan&#39;s approximation formula.

**Arguments**
- `F` - Forward price
  
- `K` - Strike price
  
- `T` - Time to expiry (in years)
  
- `params` - SABR model parameters
  

**Returns**

Implied Black volatility for the given strike.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/Models.jl#L24-L37" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.Models.sabr_price' href='#QuantNova.Models.sabr_price'><span class="jlbinding">QuantNova.Models.sabr_price</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
sabr_price(F, K, T, r, params::SABRParams, optiontype::Symbol)
```


Price a European option using SABR implied vol + Black-76.

**Arguments**
- `F` - Forward price
  
- `K` - Strike price
  
- `T` - Time to expiry (in years)
  
- `r` - Risk-free rate
  
- `params` - SABR model parameters
  
- `optiontype` - :call or :put
  

**Returns**

Option price


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/Models.jl#L124-L139" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.Models.HestonParams' href='#QuantNova.Models.HestonParams'><span class="jlbinding">QuantNova.Models.HestonParams</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
HestonParams{T1,T2,T3,T4,T5}
```


Heston stochastic volatility model parameters.

**Fields**
- `v0::T1` - Initial variance (v0 &gt; 0)
  
- `theta::T2` - Long-term variance / mean reversion level (θ &gt; 0)
  
- `kappa::T3` - Mean reversion speed (κ &gt; 0)
  
- `sigma::T4` - Volatility of variance (σ &gt; 0)
  
- `rho::T5` - Correlation between spot and variance (-1 &lt; ρ &lt; 1)
  

The Feller condition 2κθ &gt; σ² ensures the variance stays positive.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/Models.jl#L149-L162" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.Models.heston_price' href='#QuantNova.Models.heston_price'><span class="jlbinding">QuantNova.Models.heston_price</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
heston_price(S, K, T, r, q, params::HestonParams, optiontype::Symbol; N=128)
```


Price a European option under the Heston model using numerical integration.

**Arguments**
- `S` - Spot price
  
- `K` - Strike price
  
- `T` - Time to expiry (in years)
  
- `r` - Risk-free rate
  
- `q` - Continuous dividend yield (default 0.0)
  
- `params` - Heston model parameters
  
- `optiontype` - :call or :put
  
- `N` - Number of integration points (default 128)
  

**Returns**

Option price

**Notes**

Uses the Gil-Pelaez / Carr-Madan approach with trapezoidal integration.

**Example**

```julia
params = HestonParams(0.04, 0.04, 1.5, 0.3, -0.7)
# Without dividends
price = heston_price(100.0, 100.0, 1.0, 0.05, params, :call)
# With 2% dividend yield
price = heston_price(100.0, 100.0, 1.0, 0.05, 0.02, params, :call)
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/Models.jl#L212-L241" target="_blank" rel="noreferrer">source</a></Badge>

</details>


## Calibration {#Calibration}
<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.Calibration.OptionQuote' href='#QuantNova.Calibration.OptionQuote'><span class="jlbinding">QuantNova.Calibration.OptionQuote</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
OptionQuote
```


A single option quote from the market.

**Fields**
- `strike::Float64` - Strike price
  
- `expiry::Float64` - Time to expiry (in years)
  
- `price::Float64` - Market price (can be 0 if only using implied_vol)
  
- `optiontype::Symbol` - :call or :put
  
- `implied_vol::Float64` - Market implied volatility
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/Calibration.jl#L85-L96" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.Calibration.SmileData' href='#QuantNova.Calibration.SmileData'><span class="jlbinding">QuantNova.Calibration.SmileData</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
SmileData
```


Market data for a single expiry (a volatility smile).

**Fields**
- `expiry::Float64` - Time to expiry (in years)
  
- `forward::Float64` - Forward price
  
- `rate::Float64` - Risk-free interest rate
  
- `quotes::Vector{OptionQuote}` - Option quotes at this expiry
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/Calibration.jl#L105-L115" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.Calibration.VolSurface' href='#QuantNova.Calibration.VolSurface'><span class="jlbinding">QuantNova.Calibration.VolSurface</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
VolSurface
```


Market data for multiple expiries (full volatility surface).

**Fields**
- `spot::Float64` - Current spot price
  
- `rate::Float64` - Risk-free interest rate
  
- `smiles::Vector{SmileData}` - Smile data for each expiry
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/Calibration.jl#L365-L374" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.Calibration.calibrate_sabr' href='#QuantNova.Calibration.calibrate_sabr'><span class="jlbinding">QuantNova.Calibration.calibrate_sabr</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
calibrate_sabr(smile::SmileData; beta=0.5, max_iter=1000, tol=1e-8, lr=0.01, backend=current_backend())
```


Calibrate SABR model to a single expiry volatility smile.

Uses Adam optimizer with multi-start initialization for robust global optimization. Tries multiple initial values for ν (vol-of-vol) and picks the best fit. β is typically fixed (0.5 for rates, 1.0 for equities).

**Arguments**
- `smile` - Market smile data for a single expiry
  
- `beta` - CEV exponent (fixed during calibration)
  
- `max_iter` - Maximum optimizer iterations (distributed across multi-start runs)
  
- `tol` - Convergence tolerance for gradient norm and loss plateau detection
  
- `lr` - Learning rate for Adam optimizer
  
- `backend` - AD backend for gradient computation (default: current_backend())
  

**Returns**

CalibrationResult with fitted SABRParams containing:
- `params` - Calibrated SABRParams (α, β, ρ, ν)
  
- `rmse` - Root mean squared error in implied vol terms
  
- `converged` - Whether optimizer converged (gradient norm &lt; tol OR loss plateau for 20 iterations)
  
- `iterations` - Total iterations across all multi-start runs
  

**Example**

```julia
quotes = [OptionQuote(K, 1.0, 0.0, :call, market_vol) for (K, market_vol) in market_data]
smile = SmileData(1.0, 100.0, 0.05, quotes)
result = calibrate_sabr(smile; beta=0.5)

# Check fit quality
if result.rmse > 0.01
    @warn "Poor fit, RMSE = $(result.rmse)"
end

# With explicit GPU backend
result = calibrate_sabr(smile; backend=ReactantBackend())
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/Calibration.jl#L148-L186" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.Calibration.calibrate_heston' href='#QuantNova.Calibration.calibrate_heston'><span class="jlbinding">QuantNova.Calibration.calibrate_heston</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
calibrate_heston(surface::VolSurface; max_iter=2000, tol=1e-8, lr=0.001, backend=current_backend())
```


Calibrate Heston model to a full volatility surface.

Uses gradient descent with automatic differentiation. Fits a single set of Heston parameters across all expiries in the surface.

**Arguments**
- `surface` - Market volatility surface data
  
- `max_iter` - Maximum gradient descent iterations
  
- `tol` - Convergence tolerance on loss improvement
  
- `lr` - Learning rate
  
- `backend` - AD backend for gradient computation (default: current_backend())
  

**Returns**

CalibrationResult with fitted HestonParams.

**Example**

```julia
smiles = [SmileData(T, F, r, quotes) for (T, F, quotes) in market_data]
surface = VolSurface(100.0, 0.05, smiles)
result = calibrate_heston(surface)

# With explicit GPU backend
result = calibrate_heston(surface; backend=ReactantBackend())
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/Calibration.jl#L381-L408" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.Calibration.CalibrationResult' href='#QuantNova.Calibration.CalibrationResult'><span class="jlbinding">QuantNova.Calibration.CalibrationResult</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
CalibrationResult{P}
```


Result of model calibration.

**Fields**
- `params::P` - Fitted model parameters
  
- `loss::Float64` - Final objective value
  
- `converged::Bool` - Whether optimization converged
  
- `iterations::Int` - Number of iterations taken
  
- `rmse::Float64` - Root mean squared error (in volatility terms)
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/Calibration.jl#L123-L134" target="_blank" rel="noreferrer">source</a></Badge>

</details>


## Interest Rates {#Interest-Rates}

### Yield Curves {#Yield-Curves}
<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.InterestRates.RateCurve' href='#QuantNova.InterestRates.RateCurve'><span class="jlbinding">QuantNova.InterestRates.RateCurve</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
RateCurve
```


Abstract base type for interest rate curves.

All rate curves support the following operations:
- `discount(curve, T)` - Get discount factor to time T
  
- `zero_rate(curve, T)` - Get zero rate to time T
  
- `forward_rate(curve, T1, T2)` - Get forward rate between T1 and T2
  
- `instantaneous_forward(curve, T)` - Get instantaneous forward rate at T
  

**Subtypes**
- [`DiscountCurve`](/api#QuantNova.InterestRates.DiscountCurve) - Curve of discount factors
  
- [`ZeroCurve`](/api#QuantNova.InterestRates.ZeroCurve) - Curve of zero rates
  
- [`ForwardCurve`](/api#QuantNova.InterestRates.ForwardCurve) - Curve of instantaneous forward rates
  
- [`NelsonSiegelCurve`](/api#QuantNova.InterestRates.NelsonSiegelCurve) - Parametric Nelson-Siegel curve
  
- [`SvenssonCurve`](/api#QuantNova.InterestRates.SvenssonCurve) - Parametric Svensson curve
  

**Example**

```julia
# Create a flat 5% curve
curve = ZeroCurve(0.05)

# Get discount factor and zero rate at 2 years
df = discount(curve, 2.0)      # ≈ 0.9048
r = zero_rate(curve, 2.0)      # = 0.05
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/InterestRates.jl#L543-L570" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.InterestRates.DiscountCurve' href='#QuantNova.InterestRates.DiscountCurve'><span class="jlbinding">QuantNova.InterestRates.DiscountCurve</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
DiscountCurve(times, discount_factors; interp=LogLinearInterp())
```


Curve of discount factors P(0,T). Interpolates in log-space by default to ensure discount factors remain positive.

**Arguments**
- `times::Vector{Float64}` - Maturities in years
  
- `discount_factors::Vector{Float64}` - Discount factors (must be positive)
  
- `interp::InterpolationMethod` - Interpolation method (default: LogLinearInterp)
  

**Constructors**
- `DiscountCurve(times, dfs)` - From vectors
  
- `DiscountCurve(rate)` - Flat curve at given rate
  

**Example**

```julia
curve = DiscountCurve([0.0, 1.0, 5.0], [1.0, 0.95, 0.78])
df = discount(curve, 2.5)  # Interpolated discount factor
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/InterestRates.jl#L573-L593" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.InterestRates.ZeroCurve' href='#QuantNova.InterestRates.ZeroCurve'><span class="jlbinding">QuantNova.InterestRates.ZeroCurve</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
ZeroCurve(times, zero_rates; interp=LinearInterp())
```


Curve of continuously compounded zero rates.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/InterestRates.jl#L608-L612" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.InterestRates.ForwardCurve' href='#QuantNova.InterestRates.ForwardCurve'><span class="jlbinding">QuantNova.InterestRates.ForwardCurve</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
ForwardCurve(times, forward_rates; interp=LinearInterp())
```


Curve of instantaneous forward rates.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/InterestRates.jl#L625-L629" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.InterestRates.NelsonSiegelCurve' href='#QuantNova.InterestRates.NelsonSiegelCurve'><span class="jlbinding">QuantNova.InterestRates.NelsonSiegelCurve</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
NelsonSiegelCurve(β0, β1, β2, τ)
```


Nelson-Siegel parametric yield curve model.

The zero rate at maturity T is given by:

```julia
r(T) = β₀ + β₁ * (1 - exp(-T/τ)) / (T/τ) + β₂ * ((1 - exp(-T/τ)) / (T/τ) - exp(-T/τ))
```


**Parameters**
- `β0` - Long-term rate level (asymptotic rate as T → ∞)
  
- `β1` - Short-term component (controls slope at origin)
  
- `β2` - Medium-term component (controls curvature/hump)
  
- `τ` - Decay factor (time at which medium-term component reaches maximum)
  

**Example**

```julia
curve = NelsonSiegelCurve(0.05, -0.02, 0.01, 2.0)
zero_rate(curve, 5.0)  # Get 5-year zero rate
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/InterestRates.jl#L841-L862" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.InterestRates.SvenssonCurve' href='#QuantNova.InterestRates.SvenssonCurve'><span class="jlbinding">QuantNova.InterestRates.SvenssonCurve</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
SvenssonCurve(β0, β1, β2, β3, τ1, τ2)
```


Svensson extension of Nelson-Siegel with an additional hump component.

The zero rate at maturity T is given by:

```julia
r(T) = β₀ + β₁ * g1(T/τ1) + β₂ * h1(T/τ1) + β₃ * h2(T/τ2)
```


where:
- g1(x) = (1 - exp(-x)) / x
  
- h1(x) = g1(x) - exp(-x)
  
- h2(x) = g1(x, τ2) - exp(-T/τ2)
  

**Parameters**
- `β0` - Long-term rate level
  
- `β1` - Short-term component
  
- `β2` - First medium-term component (hump at τ1)
  
- `β3` - Second medium-term component (hump at τ2)
  
- `τ1` - First decay factor
  
- `τ2` - Second decay factor
  

**Example**

```julia
curve = SvenssonCurve(0.05, -0.02, 0.01, 0.005, 2.0, 5.0)
zero_rate(curve, 10.0)  # Get 10-year zero rate
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/InterestRates.jl#L875-L902" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.InterestRates.discount' href='#QuantNova.InterestRates.discount'><span class="jlbinding">QuantNova.InterestRates.discount</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
discount(curve, T) -> Float64
```


Discount factor from time 0 to time T.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/InterestRates.jl#L649-L653" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.InterestRates.zero_rate' href='#QuantNova.InterestRates.zero_rate'><span class="jlbinding">QuantNova.InterestRates.zero_rate</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
zero_rate(curve, T) -> Float64
```


Continuously compounded zero rate to time T.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/InterestRates.jl#L674-L678" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.InterestRates.forward_rate' href='#QuantNova.InterestRates.forward_rate'><span class="jlbinding">QuantNova.InterestRates.forward_rate</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
forward_rate(curve, T1, T2) -> Float64
```


Simply compounded forward rate between T1 and T2.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/InterestRates.jl#L696-L700" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.InterestRates.instantaneous_forward' href='#QuantNova.InterestRates.instantaneous_forward'><span class="jlbinding">QuantNova.InterestRates.instantaneous_forward</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
instantaneous_forward(curve, T) -> Float64
```


Instantaneous forward rate at time T: f(T) = -d/dT ln(P(0,T))


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/InterestRates.jl#L708-L712" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.InterestRates.fit_nelson_siegel' href='#QuantNova.InterestRates.fit_nelson_siegel'><span class="jlbinding">QuantNova.InterestRates.fit_nelson_siegel</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
fit_nelson_siegel(maturities, rates; initial_guess=nothing) -> NelsonSiegelCurve
```


Fit a Nelson-Siegel curve to observed zero rates using least squares.

**Arguments**
- `maturities` - Vector of maturities (in years)
  
- `rates` - Vector of observed zero rates
  
- `initial_guess` - Optional (β0, β1, β2, τ) starting point
  

**Returns**

A fitted NelsonSiegelCurve

**Example**

```julia
mats = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
rates = [0.02, 0.022, 0.025, 0.028, 0.032, 0.035, 0.037]
curve = fit_nelson_siegel(mats, rates)
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/InterestRates.jl#L984-L1003" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.InterestRates.fit_svensson' href='#QuantNova.InterestRates.fit_svensson'><span class="jlbinding">QuantNova.InterestRates.fit_svensson</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
fit_svensson(maturities, rates; initial_guess=nothing) -> SvenssonCurve
```


Fit a Svensson curve to observed zero rates using least squares.

**Arguments**
- `maturities` - Vector of maturities (in years)
  
- `rates` - Vector of observed zero rates
  
- `initial_guess` - Optional (β0, β1, β2, β3, τ1, τ2) starting point
  

**Returns**

A fitted SvenssonCurve


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/InterestRates.jl#L1035-L1047" target="_blank" rel="noreferrer">source</a></Badge>

</details>


### Interpolation {#Interpolation}
<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.InterestRates.LinearInterp' href='#QuantNova.InterestRates.LinearInterp'><span class="jlbinding">QuantNova.InterestRates.LinearInterp</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
LinearInterp <: InterpolationMethod
```


Linear interpolation between data points.

Simple and stable, but can produce kinks in forward rates. Best for zero rate curves where smoothness is less critical.

**Example**

```julia
curve = ZeroCurve(times, rates; interp=LinearInterp())
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/InterestRates.jl#L414-L426" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.InterestRates.LogLinearInterp' href='#QuantNova.InterestRates.LogLinearInterp'><span class="jlbinding">QuantNova.InterestRates.LogLinearInterp</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
LogLinearInterp <: InterpolationMethod
```


Log-linear interpolation (linear in log-space).

The default for discount curves. Ensures discount factors remain positive and produces smoother forward rates than linear interpolation.

**Example**

```julia
curve = DiscountCurve(times, dfs; interp=LogLinearInterp())
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/InterestRates.jl#L429-L441" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.InterestRates.CubicSplineInterp' href='#QuantNova.InterestRates.CubicSplineInterp'><span class="jlbinding">QuantNova.InterestRates.CubicSplineInterp</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
CubicSplineInterp <: InterpolationMethod
```


Natural cubic spline interpolation.

Produces the smoothest curves with continuous first and second derivatives. Best for applications requiring smooth forward rates (e.g., HJM models).

**Fields**
- `coeffs::Vector{NTuple{4,Float64}}` - Spline coefficients (a, b, c, d) per segment
  

**Example**

```julia
curve = ZeroCurve(times, rates; interp=CubicSplineInterp())
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/InterestRates.jl#L444-L459" target="_blank" rel="noreferrer">source</a></Badge>

</details>


### Bootstrapping {#Bootstrapping}
<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.InterestRates.DepositRate' href='#QuantNova.InterestRates.DepositRate'><span class="jlbinding">QuantNova.InterestRates.DepositRate</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



Deposit rate: simple rate for short maturities


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/InterestRates.jl#L1156" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.InterestRates.FuturesRate' href='#QuantNova.InterestRates.FuturesRate'><span class="jlbinding">QuantNova.InterestRates.FuturesRate</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



Futures rate: convexity-adjusted forward rate


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/InterestRates.jl#L1162" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.InterestRates.SwapRate' href='#QuantNova.InterestRates.SwapRate'><span class="jlbinding">QuantNova.InterestRates.SwapRate</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



Swap rate: par swap rate


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/InterestRates.jl#L1171" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.InterestRates.bootstrap' href='#QuantNova.InterestRates.bootstrap'><span class="jlbinding">QuantNova.InterestRates.bootstrap</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
bootstrap(instruments; interp=LogLinearInterp()) -> DiscountCurve
```


Bootstrap a discount curve from market instruments using sequential stripping.

The function iteratively solves for discount factors that reprice each instrument, starting from the shortest maturity. Instruments should be provided in order of increasing maturity.

**Arguments**
- `instruments::Vector{<:MarketInstrument}` - Market quotes (sorted by maturity)
  
- `interp::InterpolationMethod` - Interpolation for intermediate points
  

**Supported Instruments**
- [`DepositRate`](/api#QuantNova.InterestRates.DepositRate) - Money market deposits (short end)
  
- [`FuturesRate`](/api#QuantNova.InterestRates.FuturesRate) - Interest rate futures (middle)
  
- [`SwapRate`](/api#QuantNova.InterestRates.SwapRate) - Par swap rates (long end)
  

**Returns**

A [`DiscountCurve`](/api#QuantNova.InterestRates.DiscountCurve) that reprices all input instruments.

**Example**

```julia
instruments = [
    DepositRate(0.25, 0.02),   # 3-month deposit at 2%
    DepositRate(0.5, 0.022),   # 6-month deposit at 2.2%
    SwapRate(2.0, 0.028),      # 2-year swap at 2.8%
    SwapRate(5.0, 0.032),      # 5-year swap at 3.2%
]
curve = bootstrap(instruments)
```


See also: [`DepositRate`](/api#QuantNova.InterestRates.DepositRate), [`FuturesRate`](/api#QuantNova.InterestRates.FuturesRate), [`SwapRate`](/api#QuantNova.InterestRates.SwapRate)


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/InterestRates.jl#L1179-L1212" target="_blank" rel="noreferrer">source</a></Badge>

</details>


### Bonds {#Bonds}
<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.InterestRates.Bond' href='#QuantNova.InterestRates.Bond'><span class="jlbinding">QuantNova.InterestRates.Bond</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
Bond
```


Abstract base type for fixed income instruments.

All bonds support the following operations:
- `price(bond, curve)` - Present value using a discount curve
  
- `price(bond, yield)` - Present value at a given yield
  
- `yield_to_maturity(bond, price)` - Solve for yield given price
  
- `duration(bond, yield)` - Macaulay duration
  
- `modified_duration(bond, yield)` - Modified duration
  
- `convexity(bond, yield)` - Convexity
  
- `dv01(bond, yield)` - Dollar value of 1 basis point
  

**Subtypes**
- [`ZeroCouponBond`](/api#QuantNova.InterestRates.ZeroCouponBond) - Zero-coupon (discount) bond
  
- [`FixedRateBond`](/api#QuantNova.InterestRates.FixedRateBond) - Fixed-rate coupon bond
  
- [`FloatingRateBond`](/api#QuantNova.InterestRates.FloatingRateBond) - Floating-rate bond
  

**Example**

```julia
bond = FixedRateBond(5.0, 0.04, 2)  # 5-year, 4% semi-annual
curve = ZeroCurve(0.05)
pv = price(bond, curve)
ytm = yield_to_maturity(bond, 95.0)
dur = duration(bond, ytm)
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/InterestRates.jl#L1529-L1556" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.InterestRates.ZeroCouponBond' href='#QuantNova.InterestRates.ZeroCouponBond'><span class="jlbinding">QuantNova.InterestRates.ZeroCouponBond</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
ZeroCouponBond(maturity, face_value=100.0)
```


Zero-coupon bond paying face value at maturity.

**Arguments**
- `maturity::Float64` - Time to maturity in years
  
- `face_value::Float64` - Face (par) value (default: 100.0)
  

**Example**

```julia
zcb = ZeroCouponBond(5.0)  # 5-year zero
price(zcb, 0.05)  # ≈ 78.12 at 5% yield
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/InterestRates.jl#L1559-L1573" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.InterestRates.FixedRateBond' href='#QuantNova.InterestRates.FixedRateBond'><span class="jlbinding">QuantNova.InterestRates.FixedRateBond</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
FixedRateBond(maturity, coupon_rate, frequency=2, face_value=100.0)
```


Fixed-rate coupon bond. Coupon rate is annual, paid at given frequency.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/InterestRates.jl#L1580-L1584" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.InterestRates.FloatingRateBond' href='#QuantNova.InterestRates.FloatingRateBond'><span class="jlbinding">QuantNova.InterestRates.FloatingRateBond</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
FloatingRateBond(maturity, spread, frequency=4, face_value=100.0)
```


Floating-rate bond paying reference rate + spread.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/InterestRates.jl#L1594-L1598" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.InterestRates.yield_to_maturity' href='#QuantNova.InterestRates.yield_to_maturity'><span class="jlbinding">QuantNova.InterestRates.yield_to_maturity</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
yield_to_maturity(bond, market_price; tol=1e-10) -> Float64
```


Solve for yield given market price using Newton-Raphson.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/InterestRates.jl#L1644-L1648" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.InterestRates.duration' href='#QuantNova.InterestRates.duration'><span class="jlbinding">QuantNova.InterestRates.duration</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
duration(bond, yield) -> Float64
```


Macaulay duration: weighted average time to cash flows.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/InterestRates.jl#L1668-L1672" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.InterestRates.modified_duration' href='#QuantNova.InterestRates.modified_duration'><span class="jlbinding">QuantNova.InterestRates.modified_duration</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
modified_duration(bond, yield) -> Float64
```


Modified duration: -1/P * dP/dy


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/InterestRates.jl#L1678-L1682" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.InterestRates.convexity' href='#QuantNova.InterestRates.convexity'><span class="jlbinding">QuantNova.InterestRates.convexity</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
convexity(bond, yield) -> Float64
```


Convexity: 1/P * d²P/dy²


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/InterestRates.jl#L1687-L1691" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.InterestRates.dv01' href='#QuantNova.InterestRates.dv01'><span class="jlbinding">QuantNova.InterestRates.dv01</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
dv01(bond, yield) -> Float64
```


Dollar value of 1 basis point: price change for 1bp yield move.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/InterestRates.jl#L1697-L1701" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.InterestRates.accrued_interest' href='#QuantNova.InterestRates.accrued_interest'><span class="jlbinding">QuantNova.InterestRates.accrued_interest</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
accrued_interest(bond, settlement_time) -> Float64
```


Accrued interest from last coupon to settlement.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/InterestRates.jl#L1707-L1711" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.InterestRates.clean_price' href='#QuantNova.InterestRates.clean_price'><span class="jlbinding">QuantNova.InterestRates.clean_price</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



Clean price = dirty price - accrued interest


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/InterestRates.jl#L1720" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.InterestRates.dirty_price' href='#QuantNova.InterestRates.dirty_price'><span class="jlbinding">QuantNova.InterestRates.dirty_price</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



Dirty price = full price including accrued


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/InterestRates.jl#L1724" target="_blank" rel="noreferrer">source</a></Badge>

</details>


### Short-Rate Models {#Short-Rate-Models}
<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.InterestRates.ShortRateModel' href='#QuantNova.InterestRates.ShortRateModel'><span class="jlbinding">QuantNova.InterestRates.ShortRateModel</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
ShortRateModel
```


Base type for short-rate interest rate models.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/InterestRates.jl#L1731-L1735" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.InterestRates.Vasicek' href='#QuantNova.InterestRates.Vasicek'><span class="jlbinding">QuantNova.InterestRates.Vasicek</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
Vasicek(κ, θ, σ, r0)
```


Vasicek model: dr = κ(θ - r)dt + σdW

Parameters:
- κ: mean reversion speed
  
- θ: long-term mean rate
  
- σ: volatility
  
- r0: initial short rate
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/InterestRates.jl#L1738-L1748" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.InterestRates.CIR' href='#QuantNova.InterestRates.CIR'><span class="jlbinding">QuantNova.InterestRates.CIR</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
CIR(κ, θ, σ, r0)
```


Cox-Ingersoll-Ross model: dr = κ(θ - r)dt + σ√r dW

Feller condition: 2κθ &gt; σ² ensures r stays positive.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/InterestRates.jl#L1756-L1762" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.InterestRates.HullWhite' href='#QuantNova.InterestRates.HullWhite'><span class="jlbinding">QuantNova.InterestRates.HullWhite</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
HullWhite(κ, σ, curve)
```


Hull-White model: dr = (θ(t) - κr)dt + σdW

Time-dependent θ(t) calibrated to fit initial term structure.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/InterestRates.jl#L1777-L1783" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.InterestRates.bond_price' href='#QuantNova.InterestRates.bond_price'><span class="jlbinding">QuantNova.InterestRates.bond_price</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
bond_price(model, T) -> Float64
```


Zero-coupon bond price P(0,T) under the short rate model.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/InterestRates.jl#L1792-L1796" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.InterestRates.short_rate' href='#QuantNova.InterestRates.short_rate'><span class="jlbinding">QuantNova.InterestRates.short_rate</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
short_rate(model, t) -> (mean, variance)
```


Expected short rate and variance at time t.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/InterestRates.jl#L1834-L1838" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.InterestRates.simulate_short_rate' href='#QuantNova.InterestRates.simulate_short_rate'><span class="jlbinding">QuantNova.InterestRates.simulate_short_rate</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
simulate_short_rate(model, T, n_steps, n_paths) -> Matrix
```


Simulate short rate paths. Returns [n_steps+1 × n_paths] matrix.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/InterestRates.jl#L1858-L1862" target="_blank" rel="noreferrer">source</a></Badge>

</details>


### Interest Rate Derivatives {#Interest-Rate-Derivatives}
<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.InterestRates.Caplet' href='#QuantNova.InterestRates.Caplet'><span class="jlbinding">QuantNova.InterestRates.Caplet</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



Caplet: call option on forward rate


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/InterestRates.jl#L2115" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.InterestRates.Floorlet' href='#QuantNova.InterestRates.Floorlet'><span class="jlbinding">QuantNova.InterestRates.Floorlet</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



Floorlet: put option on forward rate


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/InterestRates.jl#L2124" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.InterestRates.Cap' href='#QuantNova.InterestRates.Cap'><span class="jlbinding">QuantNova.InterestRates.Cap</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



Cap: portfolio of caplets


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/InterestRates.jl#L2133" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.InterestRates.Floor' href='#QuantNova.InterestRates.Floor'><span class="jlbinding">QuantNova.InterestRates.Floor</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



Floor: portfolio of floorlets


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/InterestRates.jl#L2142" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.InterestRates.Swaption' href='#QuantNova.InterestRates.Swaption'><span class="jlbinding">QuantNova.InterestRates.Swaption</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
Swaption(expiry, swap_maturity, strike, is_payer, notional)
```


European swaption - option to enter a swap.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/InterestRates.jl#L2151-L2155" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.InterestRates.black_caplet' href='#QuantNova.InterestRates.black_caplet'><span class="jlbinding">QuantNova.InterestRates.black_caplet</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
black_caplet(caplet, curve, volatility) -> Float64
```


Price a caplet using Black&#39;s formula.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/InterestRates.jl#L2168-L2172" target="_blank" rel="noreferrer">source</a></Badge>

</details>


::: warning Missing docstring.

Missing docstring for `black_floorlet`. Check Documenter&#39;s build log for details.

:::
<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.InterestRates.black_cap' href='#QuantNova.InterestRates.black_cap'><span class="jlbinding">QuantNova.InterestRates.black_cap</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
black_cap(cap, curve, volatilities) -> Float64
```


Price a cap as sum of caplets. volatilities can be scalar (flat) or vector (per caplet).


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/InterestRates.jl#L2211-L2216" target="_blank" rel="noreferrer">source</a></Badge>

</details>


::: warning Missing docstring.

Missing docstring for `black_floor`. Check Documenter&#39;s build log for details.

:::

## Market Data {#Market-Data}
<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.MarketData.AbstractMarketData' href='#QuantNova.MarketData.AbstractMarketData'><span class="jlbinding">QuantNova.MarketData.AbstractMarketData</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
AbstractMarketData
```


Base type for all market data containers.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/MarketData.jl#L12-L16" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.MarketData.AbstractPriceHistory' href='#QuantNova.MarketData.AbstractPriceHistory'><span class="jlbinding">QuantNova.MarketData.AbstractPriceHistory</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
AbstractPriceHistory <: AbstractMarketData
```


Base type for historical price data (OHLCV).


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/MarketData.jl#L19-L23" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.MarketData.AbstractDataAdapter' href='#QuantNova.MarketData.AbstractDataAdapter'><span class="jlbinding">QuantNova.MarketData.AbstractDataAdapter</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
AbstractDataAdapter
```


Base type for data loading adapters.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/MarketData.jl#L26-L30" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.MarketData.PriceHistory' href='#QuantNova.MarketData.PriceHistory'><span class="jlbinding">QuantNova.MarketData.PriceHistory</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
PriceHistory <: AbstractPriceHistory
```


Container for OHLCV (Open, High, Low, Close, Volume) price data.

**Fields**
- `symbol::String` - Ticker symbol
  
- `timestamps::Vector{DateTime}` - Timestamps for each bar
  
- `open::Vector{Float64}` - Opening prices
  
- `high::Vector{Float64}` - High prices
  
- `low::Vector{Float64}` - Low prices
  
- `close::Vector{Float64}` - Closing prices
  
- `volume::Vector{Float64}` - Trading volume
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/MarketData.jl#L37-L50" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.MarketData.returns' href='#QuantNova.MarketData.returns'><span class="jlbinding">QuantNova.MarketData.returns</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
returns(ph::PriceHistory; type=:simple)
```


Compute returns from price history.

**Arguments**
- `type` - :simple for (P_t - P_{t-1})/P_{t-1}, :log for log(P_t/P_{t-1})
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/MarketData.jl#L96-L103" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.MarketData.resample' href='#QuantNova.MarketData.resample'><span class="jlbinding">QuantNova.MarketData.resample</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
resample(ph::PriceHistory, frequency::Symbol) -> PriceHistory
```


Resample price history to a different frequency.

**Arguments**
- `frequency` - :daily, :weekly, :monthly
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/MarketData.jl#L455-L462" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.MarketData.align' href='#QuantNova.MarketData.align'><span class="jlbinding">QuantNova.MarketData.align</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
align(histories::Vector{PriceHistory}) -> Vector{PriceHistory}
```


Align multiple price histories to common timestamps (inner join).


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/MarketData.jl#L509-L513" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.MarketData.fetch_prices' href='#QuantNova.MarketData.fetch_prices'><span class="jlbinding">QuantNova.MarketData.fetch_prices</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
fetch_prices(symbol::String; range="1y", interval="1d",
             startdt=nothing, enddt=nothing, autoadjust=true) -> PriceHistory
```


Fetch historical price data from Yahoo Finance.

**Arguments**
- `symbol` - Ticker symbol (e.g., &quot;AAPL&quot;, &quot;MSFT&quot;, &quot;SPY&quot;)
  
- `range` - Time range: &quot;1d&quot;, &quot;5d&quot;, &quot;1mo&quot;, &quot;3mo&quot;, &quot;6mo&quot;, &quot;1y&quot;, &quot;2y&quot;, &quot;5y&quot;, &quot;10y&quot;, &quot;ytd&quot;, &quot;max&quot;
  
- `interval` - Data interval: &quot;1m&quot;, &quot;2m&quot;, &quot;5m&quot;, &quot;15m&quot;, &quot;30m&quot;, &quot;60m&quot;, &quot;90m&quot;, &quot;1h&quot;, &quot;1d&quot;, &quot;5d&quot;, &quot;1wk&quot;, &quot;1mo&quot;, &quot;3mo&quot;
  
- `startdt` - Start date (overrides range if provided), format: &quot;YYYY-MM-DD&quot; or Date
  
- `enddt` - End date, format: &quot;YYYY-MM-DD&quot; or Date
  
- `autoadjust` - Whether to use adjusted prices (default: true)
  

**Examples**

```julia
# Get 1 year of daily AAPL data
prices = fetch_prices("AAPL")

# Get 5 years of weekly data
prices = fetch_prices("AAPL", range="5y", interval="1wk")

# Get data for a specific date range
prices = fetch_prices("AAPL", startdt="2020-01-01", enddt="2024-01-01")
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/MarketData.jl#L229-L254" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.MarketData.fetch_multiple' href='#QuantNova.MarketData.fetch_multiple'><span class="jlbinding">QuantNova.MarketData.fetch_multiple</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
fetch_multiple(symbols::Vector{String}; align_dates=true, kwargs...) -> Vector{PriceHistory}
```


Fetch historical price data for multiple symbols.

**Arguments**
- `symbols` - Vector of ticker symbols
  
- `align_dates` - Whether to align all histories to common dates (default: true)
  
- `kwargs...` - Arguments passed to `fetch_prices`
  

**Examples**

```julia
# Fetch and align multiple stocks
histories = fetch_multiple(["AAPL", "MSFT", "GOOGL"])

# Access individual histories
aapl, msft, googl = histories
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/MarketData.jl#L283-L301" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.MarketData.fetch_returns' href='#QuantNova.MarketData.fetch_returns'><span class="jlbinding">QuantNova.MarketData.fetch_returns</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
fetch_returns(symbol::String; type=:simple, kwargs...) -> Vector{Float64}
```


Convenience function to fetch prices and compute returns directly.

**Arguments**
- `symbol` - Ticker symbol
  
- `type` - :simple or :log returns
  
- `kwargs...` - Arguments passed to `fetch_prices`
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/MarketData.jl#L312-L321" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.MarketData.fetch_return_matrix' href='#QuantNova.MarketData.fetch_return_matrix'><span class="jlbinding">QuantNova.MarketData.fetch_return_matrix</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
fetch_return_matrix(symbols::Vector{String}; type=:simple, kwargs...) -> Matrix{Float64}
```


Fetch aligned returns for multiple symbols as a matrix.

Returns an (n_periods x n_assets) matrix suitable for portfolio optimization.

**Examples**

```julia
# Get return matrix for portfolio optimization
R = fetch_return_matrix(["AAPL", "MSFT", "GOOGL", "AMZN"], range="2y")
# R is (n_days-1) x 4 matrix
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/MarketData.jl#L327-L340" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.MarketData.to_backtest_format' href='#QuantNova.MarketData.to_backtest_format'><span class="jlbinding">QuantNova.MarketData.to_backtest_format</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
to_backtest_format(histories::Vector{PriceHistory}) -> (timestamps, price_series)
```


Convert aligned PriceHistory objects to backtest-compatible format.

Returns a tuple of:
- `timestamps::Vector{DateTime}` - Common timestamps
  
- `price_series::Dict{Symbol,Vector{Float64}}` - Close prices keyed by symbol
  

**Example**

```julia
histories = fetch_multiple(["AAPL", "MSFT", "GOOGL"], range="1y")
timestamps, prices = to_backtest_format(histories)
result = backtest(strategy, timestamps, prices)
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/MarketData.jl#L355-L370" target="_blank" rel="noreferrer">source</a></Badge>

</details>


## Simulation {#Simulation}
<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.Simulation.SimulationState' href='#QuantNova.Simulation.SimulationState'><span class="jlbinding">QuantNova.Simulation.SimulationState</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
SimulationState{T}
```


Point-in-time snapshot of simulation state.

**Fields**
- `timestamp::DateTime` - Current simulation time
  
- `cash::T` - Cash balance
  
- `positions::Dict{Symbol,T}` - Asset positions (symbol =&gt; quantity)
  
- `prices::Dict{Symbol,T}` - Current market prices
  
- `metadata::Dict{Symbol,Any}` - Extensible storage for custom data
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/Simulation.jl#L13-L24" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.Simulation.MarketSnapshot' href='#QuantNova.Simulation.MarketSnapshot'><span class="jlbinding">QuantNova.Simulation.MarketSnapshot</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
MarketSnapshot
```


Market data at a single point in time (yielded by drivers).


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/Simulation.jl#L189-L193" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.Simulation.AbstractDriver' href='#QuantNova.Simulation.AbstractDriver'><span class="jlbinding">QuantNova.Simulation.AbstractDriver</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
AbstractDriver
```


Base type for market data drivers used in simulations.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/Simulation.jl#L182-L186" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.Simulation.HistoricalDriver' href='#QuantNova.Simulation.HistoricalDriver'><span class="jlbinding">QuantNova.Simulation.HistoricalDriver</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
HistoricalDriver <: AbstractDriver
```


Driver that replays historical price data.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/Simulation.jl#L199-L203" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.Simulation.Order' href='#QuantNova.Simulation.Order'><span class="jlbinding">QuantNova.Simulation.Order</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
Order
```


A trade order to be executed.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/Simulation.jl#L71-L75" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.Simulation.Fill' href='#QuantNova.Simulation.Fill'><span class="jlbinding">QuantNova.Simulation.Fill</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
Fill
```


Result of executing an order.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/Simulation.jl#L88-L92" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.Simulation.AbstractExecutionModel' href='#QuantNova.Simulation.AbstractExecutionModel'><span class="jlbinding">QuantNova.Simulation.AbstractExecutionModel</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
AbstractExecutionModel
```


Base type for execution models that translate orders into fills.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/Simulation.jl#L105-L109" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.Simulation.InstantFill' href='#QuantNova.Simulation.InstantFill'><span class="jlbinding">QuantNova.Simulation.InstantFill</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
InstantFill <: AbstractExecutionModel
```


Instant execution at current market price with no slippage.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/Simulation.jl#L112-L116" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.Simulation.SlippageModel' href='#QuantNova.Simulation.SlippageModel'><span class="jlbinding">QuantNova.Simulation.SlippageModel</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
SlippageModel <: AbstractExecutionModel
```


Linear slippage based on bid-ask spread.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/Simulation.jl#L119-L123" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.Simulation.MarketImpactModel' href='#QuantNova.Simulation.MarketImpactModel'><span class="jlbinding">QuantNova.Simulation.MarketImpactModel</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
MarketImpactModel <: AbstractExecutionModel
```


Slippage with additional price impact based on order size.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/Simulation.jl#L130-L134" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.Simulation.execute' href='#QuantNova.Simulation.execute'><span class="jlbinding">QuantNova.Simulation.execute</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
execute(model, order, prices; timestamp=now())
```


Execute an order using the given execution model.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/Simulation.jl#L143-L147" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.Simulation.SimulationResult' href='#QuantNova.Simulation.SimulationResult'><span class="jlbinding">QuantNova.Simulation.SimulationResult</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
SimulationResult{T}
```


Complete result of running a simulation.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/Simulation.jl#L235-L239" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.Simulation.simulate' href='#QuantNova.Simulation.simulate'><span class="jlbinding">QuantNova.Simulation.simulate</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
simulate(driver, initial_state; execution_model=InstantFill())
```


Run simulation over the driver&#39;s time series.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/Simulation.jl#L249-L253" target="_blank" rel="noreferrer">source</a></Badge>

</details>


## Backtesting {#Backtesting}
<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.Backtesting.AbstractStrategy' href='#QuantNova.Backtesting.AbstractStrategy'><span class="jlbinding">QuantNova.Backtesting.AbstractStrategy</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
AbstractStrategy
```


Base type for trading strategies used by the backtesting engine.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/Backtesting.jl#L13-L17" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.Backtesting.BuyAndHoldStrategy' href='#QuantNova.Backtesting.BuyAndHoldStrategy'><span class="jlbinding">QuantNova.Backtesting.BuyAndHoldStrategy</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
BuyAndHoldStrategy <: AbstractStrategy
```


Invest in target weights once and hold.

**Fields**
- `target_weights::Dict{Symbol,Float64}` - Target allocation (must sum to 1.0)
  
- `invested::Base.RefValue{Bool}` - Track if initial investment made
  

**Example**

```julia
strategy = BuyAndHoldStrategy(Dict(:AAPL => 0.6, :GOOGL => 0.4))
orders = generate_orders(strategy, state)
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/Backtesting.jl#L36-L50" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.Backtesting.RebalancingStrategy' href='#QuantNova.Backtesting.RebalancingStrategy'><span class="jlbinding">QuantNova.Backtesting.RebalancingStrategy</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
RebalancingStrategy <: AbstractStrategy
```


Periodically rebalance to target weights.

**Fields**
- `target_weights::Dict{Symbol,Float64}` - Target allocation (must sum to 1.0)
  
- `rebalance_frequency::Symbol` - One of :daily, :weekly, :monthly
  
- `tolerance::Float64` - Rebalance if off by more than this fraction
  
- `last_rebalance::Base.RefValue{Union{Nothing,DateTime}}` - Last rebalance time
  

**Example**

```julia
strategy = RebalancingStrategy(
    target_weights=Dict(:AAPL => 0.5, :GOOGL => 0.5),
    rebalance_frequency=:monthly,
    tolerance=0.05
)
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/Backtesting.jl#L89-L108" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.Backtesting.VolatilityTargetStrategy' href='#QuantNova.Backtesting.VolatilityTargetStrategy'><span class="jlbinding">QuantNova.Backtesting.VolatilityTargetStrategy</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
VolatilityTargetStrategy <: AbstractStrategy
```


Wraps another strategy and scales positions to target a specific volatility level.

Uses exponentially weighted moving average (EWMA) to estimate realized volatility, then scales all positions to hit the target annualized volatility.

**Fields**
- `base_strategy::AbstractStrategy` - The underlying strategy to wrap
  
- `target_vol::Float64` - Target annualized volatility (e.g., 0.15 for 15%)
  
- `lookback::Int` - Days for volatility estimation (default: 20)
  
- `decay::Float64` - EWMA decay factor (default: 0.94, ~20-day half-life)
  
- `max_leverage::Float64` - Maximum leverage allowed (default: 1.0 = no leverage)
  
- `min_leverage::Float64` - Minimum leverage (default: 0.1 = 10% invested)
  
- `rebalance_threshold::Float64` - Only rebalance if leverage changes by this much
  

**Example**

```julia
base = RebalancingStrategy(target_weights=Dict(:AAPL => 0.5, :MSFT => 0.5), ...)
strategy = VolatilityTargetStrategy(base, target_vol=0.15, max_leverage=1.5)
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/Backtesting.jl#L1076-L1098" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.Backtesting.SignalStrategy' href='#QuantNova.Backtesting.SignalStrategy'><span class="jlbinding">QuantNova.Backtesting.SignalStrategy</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
SignalStrategy <: AbstractStrategy
```


A flexible strategy where users provide a signal function that computes target weights.

The signal function receives the StrategyContext and current SimulationState, and should return a Dict{Symbol,Float64} of target weights.

**Fields**
- `signal_fn` - Function `(ctx::StrategyContext, state::SimulationState) -> Dict{Symbol,Float64}`
  
- `symbols::Vector{Symbol}` - Assets to trade
  
- `rebalance_frequency::Symbol` - :daily, :weekly, or :monthly
  
- `min_weight::Float64` - Minimum weight per asset (default: 0.0)
  
- `max_weight::Float64` - Maximum weight per asset (default: 1.0)
  
- `lookback::Int` - Price history lookback (default: 252)
  

**Example**

```julia
# Custom momentum signal
function my_signal(ctx, state)
    weights = Dict{Symbol,Float64}()
    for sym in ctx.symbols
        rets = get_returns(ctx, sym, 20)
        if length(rets) >= 20
            momentum = sum(rets)
            weights[sym] = max(0, momentum)  # Long only if positive momentum
        else
            weights[sym] = 0.0
        end
    end
    # Normalize
    total = sum(values(weights))
    if total > 0
        for sym in keys(weights)
            weights[sym] /= total
        end
    end
    return weights
end

strategy = SignalStrategy(my_signal, [:AAPL, :MSFT, :GOOGL])
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/Backtesting.jl#L258-L300" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.Backtesting.MomentumStrategy' href='#QuantNova.Backtesting.MomentumStrategy'><span class="jlbinding">QuantNova.Backtesting.MomentumStrategy</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
MomentumStrategy <: AbstractStrategy
```


Trend-following strategy that goes long assets with positive momentum.

Uses past returns over a lookback period to rank assets and allocate to top performers.

**Fields**
- `symbols::Vector{Symbol}` - Assets to trade
  
- `lookback::Int` - Lookback period for momentum calculation (default: 20)
  
- `top_n::Int` - Number of top assets to hold (default: all with positive momentum)
  
- `rebalance_frequency::Symbol` - :daily, :weekly, or :monthly
  

**Example**

```julia
# Hold top 3 momentum stocks, rebalance monthly
strategy = MomentumStrategy(
    [:AAPL, :MSFT, :GOOGL, :AMZN, :META],
    lookback=60,
    top_n=3,
    rebalance_frequency=:monthly
)
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/Backtesting.jl#L392-L416" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.Backtesting.MeanReversionStrategy' href='#QuantNova.Backtesting.MeanReversionStrategy'><span class="jlbinding">QuantNova.Backtesting.MeanReversionStrategy</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
MeanReversionStrategy <: AbstractStrategy
```


Contrarian strategy that buys underperformers and sells outperformers.

Uses z-scores of recent returns to identify assets that have deviated from their mean and are likely to revert.

**Fields**
- `symbols::Vector{Symbol}` - Assets to trade
  
- `lookback::Int` - Lookback for mean/std calculation (default: 20)
  
- `entry_threshold::Float64` - Z-score threshold for entry (default: 1.5)
  
- `rebalance_frequency::Symbol` - :daily, :weekly, or :monthly
  

**Example**

```julia
# Mean reversion with 2 std dev entry threshold
strategy = MeanReversionStrategy(
    [:AAPL, :MSFT, :GOOGL],
    lookback=20,
    entry_threshold=2.0,
    rebalance_frequency=:daily
)
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/Backtesting.jl#L516-L540" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.Backtesting.CompositeStrategy' href='#QuantNova.Backtesting.CompositeStrategy'><span class="jlbinding">QuantNova.Backtesting.CompositeStrategy</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
CompositeStrategy <: AbstractStrategy
```


Combines multiple strategies with specified weights.

Each sub-strategy generates target weights, which are then combined according to the strategy weights.

**Fields**
- `strategies::Vector{<:AbstractStrategy}` - Sub-strategies
  
- `strategy_weights::Vector{Float64}` - Weight for each strategy (must sum to 1.0)
  
- `symbols::Vector{Symbol}` - All symbols across strategies
  

**Example**

```julia
# 60% momentum, 40% mean reversion
momentum = MomentumStrategy(symbols, lookback=60)
mean_rev = MeanReversionStrategy(symbols, lookback=20)

strategy = CompositeStrategy(
    [momentum, mean_rev],
    [0.6, 0.4]
)
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/Backtesting.jl#L650-L674" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.Backtesting.generate_orders' href='#QuantNova.Backtesting.generate_orders'><span class="jlbinding">QuantNova.Backtesting.generate_orders</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
generate_orders(strategy, state) -> Vector{Order}
```


Generate orders based on strategy logic and current state.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/Backtesting.jl#L20-L24" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.Backtesting.should_rebalance' href='#QuantNova.Backtesting.should_rebalance'><span class="jlbinding">QuantNova.Backtesting.should_rebalance</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
should_rebalance(strategy, state) -> Bool
```


Check if strategy should rebalance at current state.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/Backtesting.jl#L27-L31" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.Backtesting.BacktestResult' href='#QuantNova.Backtesting.BacktestResult'><span class="jlbinding">QuantNova.Backtesting.BacktestResult</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
BacktestResult
```


Complete results from running a backtest.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/Backtesting.jl#L776-L780" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.Backtesting.backtest' href='#QuantNova.Backtesting.backtest'><span class="jlbinding">QuantNova.Backtesting.backtest</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
backtest(strategy, timestamps, price_series; kwargs...)
```


Run a full backtest simulation.

**Arguments**
- `strategy::AbstractStrategy` - Trading strategy
  
- `timestamps::Vector{DateTime}` - Time series dates
  
- `price_series::Dict{Symbol,Vector{Float64}}` - Price data per asset
  
- `initial_cash::Float64=100_000.0` - Starting capital
  
- `execution_model::AbstractExecutionModel=InstantFill()` - How orders execute
  
- `cost_model::Union{Nothing,AbstractCostModel}=nothing` - Transaction cost model
  
- `adv::Dict{Symbol,Float64}=Dict()` - Average daily volume by symbol (for market impact)
  

**Returns**

`BacktestResult` with equity curve, trades, and performance metrics.

**Example with transaction costs**

```julia
using QuantNova

# Create cost model
costs = CompositeCostModel([
    ProportionalCostModel(rate_bps=1.0),
    SpreadCostModel(half_spread_bps=5.0)
])

# Run backtest with costs
result = backtest(strategy, timestamps, prices,
    initial_cash=100_000.0,
    cost_model=costs
)

# Check cost impact
println("Gross return: ", result.metrics[:gross_return])
println("Net return: ", result.metrics[:total_return])
println("Total costs: ", result.total_costs)
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/Backtesting.jl#L810-L848" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.Backtesting.compute_backtest_metrics' href='#QuantNova.Backtesting.compute_backtest_metrics'><span class="jlbinding">QuantNova.Backtesting.compute_backtest_metrics</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
compute_backtest_metrics(equity_curve, returns)
```


Compute standard backtest performance metrics.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/Backtesting.jl#L1014-L1018" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.Backtesting.WalkForwardConfig' href='#QuantNova.Backtesting.WalkForwardConfig'><span class="jlbinding">QuantNova.Backtesting.WalkForwardConfig</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
WalkForwardConfig
```


Configuration for walk-forward backtesting.

**Fields**
- `train_period::Int` - Number of days for training/optimization window
  
- `test_period::Int` - Number of days for out-of-sample testing
  
- `step_size::Int` - Days to advance between windows (default: test_period)
  
- `min_train_periods::Int` - Minimum training periods before first test
  
- `expanding::Bool` - If true, training window expands; if false, it rolls
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/Backtesting.jl#L1225-L1236" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.Backtesting.WalkForwardPeriod' href='#QuantNova.Backtesting.WalkForwardPeriod'><span class="jlbinding">QuantNova.Backtesting.WalkForwardPeriod</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
WalkForwardPeriod
```


Results from a single walk-forward period.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/Backtesting.jl#L1259-L1263" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.Backtesting.WalkForwardResult' href='#QuantNova.Backtesting.WalkForwardResult'><span class="jlbinding">QuantNova.Backtesting.WalkForwardResult</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
WalkForwardResult
```


Complete results from walk-forward backtesting.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/Backtesting.jl#L1273-L1277" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.Backtesting.walk_forward_backtest' href='#QuantNova.Backtesting.walk_forward_backtest'><span class="jlbinding">QuantNova.Backtesting.walk_forward_backtest</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
walk_forward_backtest(
    optimizer_fn,
    symbols,
    timestamps,
    price_series;
    config=WalkForwardConfig(),
    initial_cash=100_000.0,
    execution_model=InstantFill()
) -> WalkForwardResult
```


Run walk-forward backtesting with rolling optimization windows.

**Arguments**
- `optimizer_fn` - Function `(train_returns::Matrix, symbols::Vector) -> Dict{Symbol,Float64}`                  that takes training returns and returns optimal weights
  
- `symbols::Vector{String}` - Asset symbols in order
  
- `timestamps::Vector{DateTime}` - Full timestamp series
  
- `price_series::Dict{Symbol,Vector{Float64}}` - Price data per asset
  
- `config::WalkForwardConfig` - Walk-forward configuration
  
- `initial_cash::Float64` - Starting capital
  
- `execution_model` - Execution model for backtesting
  

**Example**

```julia
# Define optimizer function
function my_optimizer(returns, symbols)
    μ = vec(mean(returns, dims=1))
    Σ = cov(returns)
    result = optimize(MinimumVariance(Σ))
    return Dict(Symbol(symbols[i]) => result.weights[i] for i in eachindex(symbols))
end

# Run walk-forward
result = walk_forward_backtest(
    my_optimizer,
    ["AAPL", "MSFT", "GOOGL"],
    timestamps,
    prices,
    config=WalkForwardConfig(train_period=252, test_period=21)
)
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/Backtesting.jl#L1288-L1330" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.Backtesting.compute_extended_metrics' href='#QuantNova.Backtesting.compute_extended_metrics'><span class="jlbinding">QuantNova.Backtesting.compute_extended_metrics</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
compute_extended_metrics(returns; rf=0.0, benchmark_returns=nothing)
```


Compute extended performance metrics including Sortino, Information Ratio, etc.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/Backtesting.jl#L1454-L1458" target="_blank" rel="noreferrer">source</a></Badge>

</details>


## Transaction Costs {#Transaction-Costs}
<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.TransactionCosts.AbstractCostModel' href='#QuantNova.TransactionCosts.AbstractCostModel'><span class="jlbinding">QuantNova.TransactionCosts.AbstractCostModel</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
AbstractCostModel
```


Base type for transaction cost models.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/TransactionCosts.jl#L9-L13" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.TransactionCosts.FixedCostModel' href='#QuantNova.TransactionCosts.FixedCostModel'><span class="jlbinding">QuantNova.TransactionCosts.FixedCostModel</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
FixedCostModel <: AbstractCostModel
```


Flat fee per trade regardless of size.

**Fields**
- `cost_per_trade::Float64` - Fixed cost per trade (e.g., 5.0)
  

**Example**

```julia
model = FixedCostModel(5.0)  # $5 per trade
cost = compute_cost(model, 10000.0, 150.0, 1e6)  # Returns 5.0
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/TransactionCosts.jl#L32-L45" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.TransactionCosts.ProportionalCostModel' href='#QuantNova.TransactionCosts.ProportionalCostModel'><span class="jlbinding">QuantNova.TransactionCosts.ProportionalCostModel</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
ProportionalCostModel <: AbstractCostModel
```


Commission as a percentage of trade value.

**Fields**
- `rate_bps::Float64` - Commission rate in basis points (1 bp = 0.01%)
  
- `min_cost::Float64` - Minimum cost per trade
  

**Example**

```julia
model = ProportionalCostModel(rate_bps=5.0, min_cost=1.0)  # 5 bps with $1 minimum
cost = compute_cost(model, 10000.0, 150.0, 1e6)  # Returns 5.0 (0.05% of 10000)
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/TransactionCosts.jl#L63-L77" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.TransactionCosts.TieredCostModel' href='#QuantNova.TransactionCosts.TieredCostModel'><span class="jlbinding">QuantNova.TransactionCosts.TieredCostModel</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
TieredCostModel <: AbstractCostModel
```


Volume-based tiered pricing (common for institutional traders).

**Fields**
- `tiers::Vector{Tuple{Float64,Float64}}` - (threshold, rate_bps) pairs, sorted ascending
  
- `min_cost::Float64` - Minimum cost per trade
  

Tiers are cumulative: first tier applies up to first threshold, etc.

**Example**

```julia
# 10 bps up to $10k, 5 bps from $10k-$100k, 2 bps above $100k
model = TieredCostModel([
    (10_000.0, 10.0),
    (100_000.0, 5.0),
    (Inf, 2.0)
])
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/TransactionCosts.jl#L98-L118" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.TransactionCosts.SpreadCostModel' href='#QuantNova.TransactionCosts.SpreadCostModel'><span class="jlbinding">QuantNova.TransactionCosts.SpreadCostModel</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
SpreadCostModel <: AbstractCostModel
```


Bid-ask spread cost model.

**Fields**
- `half_spread_bps::Float64` - Half the bid-ask spread in basis points
  
- `spread_estimator::Symbol` - How to estimate spread (:fixed, :volatility_based)
  
- `vol_multiplier::Float64` - For volatility-based: spread = vol * multiplier
  

**Example**

```julia
# Fixed 5 bps half-spread (10 bps round-trip)
model = SpreadCostModel(half_spread_bps=5.0)

# Volatility-based spread estimation
model = SpreadCostModel(spread_estimator=:volatility_based, vol_multiplier=0.1)
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/TransactionCosts.jl#L153-L171" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.TransactionCosts.AlmgrenChrissModel' href='#QuantNova.TransactionCosts.AlmgrenChrissModel'><span class="jlbinding">QuantNova.TransactionCosts.AlmgrenChrissModel</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
AlmgrenChrissModel <: AbstractCostModel
```


Square-root market impact model based on Almgren-Chriss framework.

Market impact = σ * √(order_shares / ADV) * price * order_shares

This is the industry-standard model for estimating permanent and temporary price impact from trading.

**Fields**
- `volatility::Float64` - Daily volatility (annualized σ / √252)
  
- `participation_rate::Float64` - Fraction of ADV you&#39;re willing to trade
  
- `temporary_impact::Float64` - Temporary impact coefficient (default: 0.1)
  
- `permanent_impact::Float64` - Permanent impact coefficient (default: 0.1)
  

**Example**

```julia
model = AlmgrenChrissModel(
    volatility=0.02,           # 2% daily vol
    participation_rate=0.1,    # Trade up to 10% of ADV
    temporary_impact=0.1,
    permanent_impact=0.1
)
```


**Reference**

Almgren, R., &amp; Chriss, N. (2000). Optimal execution of portfolio transactions.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/TransactionCosts.jl#L205-L233" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.TransactionCosts.CompositeCostModel' href='#QuantNova.TransactionCosts.CompositeCostModel'><span class="jlbinding">QuantNova.TransactionCosts.CompositeCostModel</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
CompositeCostModel <: AbstractCostModel
```


Combines multiple cost models (e.g., commission + spread + market impact).

**Example**

```julia
model = CompositeCostModel([
    ProportionalCostModel(rate_bps=1.0),      # 1 bp commission
    SpreadCostModel(half_spread_bps=5.0),     # 5 bp half-spread
    AlmgrenChrissModel(volatility=0.02)       # Market impact
])
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/TransactionCosts.jl#L276-L289" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.TransactionCosts.CostAwareExecutionModel' href='#QuantNova.TransactionCosts.CostAwareExecutionModel'><span class="jlbinding">QuantNova.TransactionCosts.CostAwareExecutionModel</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
CostAwareExecutionModel
```


Execution model that tracks detailed transaction costs.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/TransactionCosts.jl#L458-L462" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.TransactionCosts.compute_cost' href='#QuantNova.TransactionCosts.compute_cost'><span class="jlbinding">QuantNova.TransactionCosts.compute_cost</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
compute_cost(model, order_value, price, volume; kwargs...) -> Float64
```


Compute transaction cost for a trade.

**Arguments**
- `order_value` - Absolute dollar value of the order
  
- `price` - Current price per share
  
- `volume` - Average daily volume (shares) - used for market impact
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/TransactionCosts.jl#L16-L25" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.TransactionCosts.execute_with_costs' href='#QuantNova.TransactionCosts.execute_with_costs'><span class="jlbinding">QuantNova.TransactionCosts.execute_with_costs</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
execute_with_costs(model, symbol, order_value, price) -> (exec_price, breakdown)
```


Execute a trade and return execution price with cost breakdown.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/TransactionCosts.jl#L480-L484" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.TransactionCosts.TradeCostBreakdown' href='#QuantNova.TransactionCosts.TradeCostBreakdown'><span class="jlbinding">QuantNova.TransactionCosts.TradeCostBreakdown</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
TradeCostBreakdown
```


Detailed breakdown of costs for a single trade.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/TransactionCosts.jl#L312-L316" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.TransactionCosts.CostTracker' href='#QuantNova.TransactionCosts.CostTracker'><span class="jlbinding">QuantNova.TransactionCosts.CostTracker</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
CostTracker
```


Tracks transaction costs across a backtest.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/TransactionCosts.jl#L327-L331" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.TransactionCosts.record_trade!' href='#QuantNova.TransactionCosts.record_trade!'><span class="jlbinding">QuantNova.TransactionCosts.record_trade!</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
record_trade!(tracker, breakdown)
```


Record a trade&#39;s costs in the tracker.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/TransactionCosts.jl#L347-L351" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.TransactionCosts.cost_summary' href='#QuantNova.TransactionCosts.cost_summary'><span class="jlbinding">QuantNova.TransactionCosts.cost_summary</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
cost_summary(tracker) -> Dict{Symbol,Float64}
```


Get summary statistics from cost tracker.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/TransactionCosts.jl#L366-L370" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.TransactionCosts.compute_turnover' href='#QuantNova.TransactionCosts.compute_turnover'><span class="jlbinding">QuantNova.TransactionCosts.compute_turnover</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
compute_turnover(weights_history::Vector{Dict{Symbol,Float64}}) -> Float64
```


Compute total portfolio turnover from a history of weights.

Turnover = Σ |w_t - w_{t-1}| / 2

Returns annualized turnover if daily data (multiply by 252).


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/TransactionCosts.jl#L391-L399" target="_blank" rel="noreferrer">source</a></Badge>



```julia
compute_turnover(positions_history, prices_history) -> Float64
```


Compute turnover from position and price histories.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/TransactionCosts.jl#L422-L426" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.TransactionCosts.compute_net_returns' href='#QuantNova.TransactionCosts.compute_net_returns'><span class="jlbinding">QuantNova.TransactionCosts.compute_net_returns</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
compute_net_returns(gross_returns, costs, portfolio_values) -> Vector{Float64}
```


Compute net returns after transaction costs.

**Arguments**
- `gross_returns` - Returns before costs
  
- `costs` - Transaction costs per period
  
- `portfolio_values` - Portfolio value at start of each period
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/TransactionCosts.jl#L578-L587" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.TransactionCosts.estimate_break_even_sharpe' href='#QuantNova.TransactionCosts.estimate_break_even_sharpe'><span class="jlbinding">QuantNova.TransactionCosts.estimate_break_even_sharpe</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
estimate_break_even_sharpe(cost_bps, volatility; periods_per_year=252) -> Float64
```


Estimate minimum Sharpe ratio needed to break even after costs.

A strategy with Sharpe below this threshold will likely lose money after costs.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/TransactionCosts.jl#L604-L610" target="_blank" rel="noreferrer">source</a></Badge>

</details>


## Scenario Analysis {#Scenario-Analysis}
<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.ScenarioAnalysis.StressScenario' href='#QuantNova.ScenarioAnalysis.StressScenario'><span class="jlbinding">QuantNova.ScenarioAnalysis.StressScenario</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
StressScenario{T<:Real}
```


Defines a stress scenario with shocks to different asset classes.

**Fields**
- `name::String` - Human-readable scenario name
  
- `description::String` - Detailed description of the historical event
  
- `shocks::Dict{Symbol,T}` - Asset class to percent change mapping (e.g., -0.50 for -50%)
  
- `duration_days::Int` - Historical duration of the stress event
  

**Type Parameter**

The type parameter `T` allows AD (automatic differentiation) to work through scenario calculations. For standard usage, `T=Float64`. For gradient computation via ForwardDiff, `T` will be `ForwardDiff.Dual`.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/ScenarioAnalysis.jl#L13-L28" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.ScenarioAnalysis.ScenarioImpact' href='#QuantNova.ScenarioAnalysis.ScenarioImpact'><span class="jlbinding">QuantNova.ScenarioAnalysis.ScenarioImpact</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
ScenarioImpact
```


Result of applying a stress scenario.

**Fields**
- `scenario_name::String` - Name of the applied scenario
  
- `initial_value::Float64` - Portfolio value before stress
  
- `stressed_value::Float64` - Portfolio value after stress
  
- `pnl::Float64` - Profit/loss from the scenario
  
- `pct_change::Float64` - Percentage change in portfolio value
  
- `asset_impacts::Dict{Symbol,Float64}` - Per-asset P&amp;L breakdown
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/ScenarioAnalysis.jl#L36-L48" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.ScenarioAnalysis.CRISIS_SCENARIOS' href='#QuantNova.ScenarioAnalysis.CRISIS_SCENARIOS'><span class="jlbinding">QuantNova.ScenarioAnalysis.CRISIS_SCENARIOS</span></a> <Badge type="info" class="jlObjectType jlConstant" text="Constant" /></summary>



```julia
CRISIS_SCENARIOS
```


Dictionary of built-in historical crisis scenarios for stress testing.

Available scenarios:
- `:financial_crisis_2008` - Global financial crisis, S&amp;P 500 fell ~57% from peak
  
- `:covid_crash_2020` - Rapid market crash in March 2020
  
- `:dot_com_bust_2000` - Tech bubble burst, NASDAQ fell ~78%
  
- `:black_monday_1987` - Single-day crash of 22.6%
  
- `:rate_shock_2022` - Fed aggressive tightening, bonds and stocks fell together
  
- `:stagflation_1970s` - High inflation with economic stagnation
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/ScenarioAnalysis.jl#L62-L74" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.ScenarioAnalysis.apply_scenario' href='#QuantNova.ScenarioAnalysis.apply_scenario'><span class="jlbinding">QuantNova.ScenarioAnalysis.apply_scenario</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
apply_scenario(scenario, state, asset_classes)
```


Apply a stress scenario to a simulation state, returning a new stressed state.

**Arguments**
- `scenario::StressScenario` - The stress scenario to apply
  
- `state::SimulationState{T}` - Current portfolio state
  
- `asset_classes::Dict{Symbol,Symbol}` - Mapping of asset symbols to asset classes
  

**Returns**
- `SimulationState{T}` - New state with stressed prices
  

**Example**

```julia
state = SimulationState(
    timestamp=DateTime(2024, 1, 1),
    cash=50_000.0,
    positions=Dict(:SPY => 100.0, :TLT => 50.0),
    prices=Dict(:SPY => 450.0, :TLT => 100.0)
)
asset_classes = Dict(:SPY => :equity, :TLT => :bond)
scenario = CRISIS_SCENARIOS[:financial_crisis_2008]
stressed_state = apply_scenario(scenario, state, asset_classes)
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/ScenarioAnalysis.jl#L118-L143" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.ScenarioAnalysis.scenario_impact' href='#QuantNova.ScenarioAnalysis.scenario_impact'><span class="jlbinding">QuantNova.ScenarioAnalysis.scenario_impact</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
scenario_impact(scenario, state, asset_classes)
```


Compute the impact of a stress scenario on a portfolio.

**Arguments**
- `scenario::StressScenario` - The stress scenario to evaluate
  
- `state::SimulationState` - Current portfolio state
  
- `asset_classes::Dict{Symbol,Symbol}` - Mapping of asset symbols to asset classes
  

**Returns**
- `ScenarioImpact` - Detailed impact analysis including P&amp;L and per-asset breakdown
  

**Example**

```julia
state = SimulationState(
    timestamp=DateTime(2024, 1, 1),
    cash=50_000.0,
    positions=Dict(:SPY => 100.0, :TLT => 50.0),
    prices=Dict(:SPY => 450.0, :TLT => 100.0)
)
asset_classes = Dict(:SPY => :equity, :TLT => :bond)
scenario = CRISIS_SCENARIOS[:financial_crisis_2008]
impact = scenario_impact(scenario, state, asset_classes)
# impact.pnl < 0 (loss during crisis)
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/ScenarioAnalysis.jl#L174-L200" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.ScenarioAnalysis.compare_scenarios' href='#QuantNova.ScenarioAnalysis.compare_scenarios'><span class="jlbinding">QuantNova.ScenarioAnalysis.compare_scenarios</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
compare_scenarios(scenarios, state, asset_classes)
```


Compare impact of multiple scenarios on a portfolio.

**Arguments**
- `scenarios::Vector{StressScenario}` - List of scenarios to compare
  
- `state::SimulationState` - Current portfolio state
  
- `asset_classes::Dict{Symbol,Symbol}` - Mapping of asset symbols to asset classes
  

**Returns**
- `Vector{ScenarioImpact}` - Impact results for each scenario
  

**Example**

```julia
scenarios = [
    CRISIS_SCENARIOS[:financial_crisis_2008],
    CRISIS_SCENARIOS[:covid_crash_2020]
]
impacts = compare_scenarios(scenarios, state, asset_classes)
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/ScenarioAnalysis.jl#L235-L256" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.ScenarioAnalysis.worst_case_scenario' href='#QuantNova.ScenarioAnalysis.worst_case_scenario'><span class="jlbinding">QuantNova.ScenarioAnalysis.worst_case_scenario</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
worst_case_scenario(scenarios, state, asset_classes)
```


Find the scenario with worst portfolio impact.

**Arguments**
- `scenarios::Vector{StressScenario}` - List of scenarios to evaluate
  
- `state::SimulationState` - Current portfolio state
  
- `asset_classes::Dict{Symbol,Symbol}` - Mapping of asset symbols to asset classes
  

**Returns**
- `ScenarioImpact` - Impact of the worst-case scenario
  

**Example**

```julia
scenarios = [
    CRISIS_SCENARIOS[:financial_crisis_2008],
    CRISIS_SCENARIOS[:covid_crash_2020]
]
worst = worst_case_scenario(scenarios, state, asset_classes)
println("Worst case: $(worst.scenario_name) with $(worst.pct_change * 100)% loss")
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/ScenarioAnalysis.jl#L265-L287" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.ScenarioAnalysis.SensitivityResult' href='#QuantNova.ScenarioAnalysis.SensitivityResult'><span class="jlbinding">QuantNova.ScenarioAnalysis.SensitivityResult</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
SensitivityResult
```


Result of a single sensitivity point.

**Fields**
- `shock::Float64` - The shock level applied (e.g., -0.10 for -10%)
  
- `portfolio_value::Float64` - Portfolio value after applying the shock
  
- `pnl::Float64` - Profit/loss from the shock
  
- `pct_change::Float64` - Percentage change in portfolio value
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/ScenarioAnalysis.jl#L302-L312" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.ScenarioAnalysis.sensitivity_analysis' href='#QuantNova.ScenarioAnalysis.sensitivity_analysis'><span class="jlbinding">QuantNova.ScenarioAnalysis.sensitivity_analysis</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
sensitivity_analysis(state, asset_classes, target_class; shock_range)
```


Analyze portfolio sensitivity to shocks in a specific asset class.

**Arguments**
- `state::SimulationState` - Current portfolio state
  
- `asset_classes::Dict{Symbol,Symbol}` - Mapping of asset symbols to asset classes
  
- `target_class::Symbol` - Asset class to shock (e.g., :equity, :bond)
  
- `shock_range::AbstractRange` - Range of shock levels to test (default: -0.50:0.05:0.50)
  

**Returns**
- `Vector{SensitivityResult}` - Results for each shock level
  

**Example**

```julia
results = sensitivity_analysis(state, asset_classes, :equity; shock_range=-0.50:0.10:0.50)
for r in results
    println("Shock: $(r.shock*100)% -> Value: $(r.portfolio_value)")
end
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/ScenarioAnalysis.jl#L320-L341" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.ScenarioAnalysis.ProjectionResult' href='#QuantNova.ScenarioAnalysis.ProjectionResult'><span class="jlbinding">QuantNova.ScenarioAnalysis.ProjectionResult</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
ProjectionResult
```


Results from Monte Carlo portfolio projection.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/ScenarioAnalysis.jl#L373-L377" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.ScenarioAnalysis.monte_carlo_projection' href='#QuantNova.ScenarioAnalysis.monte_carlo_projection'><span class="jlbinding">QuantNova.ScenarioAnalysis.monte_carlo_projection</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
monte_carlo_projection(state, expected_returns, volatilities; kwargs...)
```


Project portfolio forward using Monte Carlo simulation.

**Arguments**
- `state::SimulationState` - Current portfolio state
  
- `expected_returns::Dict{Symbol,Float64}` - Annual expected returns per asset
  
- `volatilities::Dict{Symbol,Float64}` - Annual volatilities per asset
  
- `correlation::Float64=0.3` - Pairwise correlation (simplified)
  
- `horizon_days::Int=252` - Projection horizon in days
  
- `n_simulations::Int=10000` - Number of Monte Carlo paths
  
- `rng` - Random number generator (optional)
  

**Returns**

`ProjectionResult` with distribution of terminal values and risk metrics.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/ScenarioAnalysis.jl#L404-L420" target="_blank" rel="noreferrer">source</a></Badge>

</details>


## Factor Models {#Factor-Models}
<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.FactorModels.RegressionResult' href='#QuantNova.FactorModels.RegressionResult'><span class="jlbinding">QuantNova.FactorModels.RegressionResult</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
RegressionResult
```


Results from factor regression.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/FactorModels.jl#L10-L14" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.FactorModels.factor_regression' href='#QuantNova.FactorModels.factor_regression'><span class="jlbinding">QuantNova.FactorModels.factor_regression</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
factor_regression(returns, factors; factor_names=nothing, rf=0.0)
```


Regress strategy returns on factor returns.

returns = α + Σ(βᵢ * factorᵢ) + ε

Returns RegressionResult with alpha, betas, and statistics.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/FactorModels.jl#L26-L34" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.FactorModels.capm_regression' href='#QuantNova.FactorModels.capm_regression'><span class="jlbinding">QuantNova.FactorModels.capm_regression</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
capm_regression(returns, market_returns; rf=0.0)
```


Capital Asset Pricing Model regression.

Returns (alpha, beta, r_squared, alpha_tstat, alpha_pvalue)


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/FactorModels.jl#L106-L112" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.FactorModels.FamaFrenchResult' href='#QuantNova.FactorModels.FamaFrenchResult'><span class="jlbinding">QuantNova.FactorModels.FamaFrenchResult</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
FamaFrenchResult
```


Results from Fama-French factor analysis.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/FactorModels.jl#L131-L135" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.FactorModels.fama_french_regression' href='#QuantNova.FactorModels.fama_french_regression'><span class="jlbinding">QuantNova.FactorModels.fama_french_regression</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
fama_french_regression(returns, mkt, smb, hml; mom=nothing, rf=0.0)
```


Fama-French 3-factor (or 4-factor with momentum) regression.

Factors should be factor returns (not cumulative).


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/FactorModels.jl#L148-L154" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.FactorModels.construct_market_factor' href='#QuantNova.FactorModels.construct_market_factor'><span class="jlbinding">QuantNova.FactorModels.construct_market_factor</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
construct_market_factor(returns_matrix, weights=nothing) -> Vector{Float64}
```


Construct market factor from asset returns matrix (n_periods × n_assets). Default: equal-weighted market return.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/FactorModels.jl#L188-L193" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.FactorModels.construct_long_short_factor' href='#QuantNova.FactorModels.construct_long_short_factor'><span class="jlbinding">QuantNova.FactorModels.construct_long_short_factor</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
construct_long_short_factor(returns_matrix, signal; quantile=0.3) -> Vector{Float64}
```


Construct long-short factor: long top quantile, short bottom quantile.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/FactorModels.jl#L203-L207" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.FactorModels.AttributionResult' href='#QuantNova.FactorModels.AttributionResult'><span class="jlbinding">QuantNova.FactorModels.AttributionResult</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
AttributionResult
```


Factor-based return attribution.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/FactorModels.jl#L236-L240" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.FactorModels.return_attribution' href='#QuantNova.FactorModels.return_attribution'><span class="jlbinding">QuantNova.FactorModels.return_attribution</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
return_attribution(returns, factors, betas, alpha; factor_names=nothing)
```


Decompose total return into factor contributions + alpha + residual.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/FactorModels.jl#L248-L252" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.FactorModels.rolling_beta' href='#QuantNova.FactorModels.rolling_beta'><span class="jlbinding">QuantNova.FactorModels.rolling_beta</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
rolling_beta(returns, factor; window=60) -> Vector{Float64}
```


Compute rolling beta to a factor.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/FactorModels.jl#L279-L283" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.FactorModels.rolling_alpha' href='#QuantNova.FactorModels.rolling_alpha'><span class="jlbinding">QuantNova.FactorModels.rolling_alpha</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
rolling_alpha(returns, factor; window=60, rf=0.0) -> Vector{Float64}
```


Compute rolling alpha vs a factor (annualized).


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/FactorModels.jl#L298-L302" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.FactorModels.StyleAnalysisResult' href='#QuantNova.FactorModels.StyleAnalysisResult'><span class="jlbinding">QuantNova.FactorModels.StyleAnalysisResult</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
StyleAnalysisResult
```


Results from returns-based style analysis.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/FactorModels.jl#L324-L328" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.FactorModels.style_analysis' href='#QuantNova.FactorModels.style_analysis'><span class="jlbinding">QuantNova.FactorModels.style_analysis</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
style_analysis(returns, style_returns; style_names=nothing)
```


Returns-based style analysis (Sharpe 1992). Constrained regression: weights ≥ 0, sum to 1.

Finds style portfolio that best replicates manager returns.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/FactorModels.jl#L336-L343" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.FactorModels.tracking_error' href='#QuantNova.FactorModels.tracking_error'><span class="jlbinding">QuantNova.FactorModels.tracking_error</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
tracking_error(returns, benchmark_returns) -> Float64
```


Annualized tracking error (volatility of active returns).


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/FactorModels.jl#L400-L404" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.FactorModels.information_ratio' href='#QuantNova.FactorModels.information_ratio'><span class="jlbinding">QuantNova.FactorModels.information_ratio</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
information_ratio(returns, benchmark_returns) -> Float64
```


Annualized information ratio = active return / tracking error.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/FactorModels.jl#L410-L414" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.FactorModels.up_capture_ratio' href='#QuantNova.FactorModels.up_capture_ratio'><span class="jlbinding">QuantNova.FactorModels.up_capture_ratio</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
up_capture_ratio(returns, benchmark_returns) -> Float64
```


Up capture = mean(portfolio | benchmark &gt; 0) / mean(benchmark | benchmark &gt; 0)


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/FactorModels.jl#L420-L424" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.FactorModels.down_capture_ratio' href='#QuantNova.FactorModels.down_capture_ratio'><span class="jlbinding">QuantNova.FactorModels.down_capture_ratio</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
down_capture_ratio(returns, benchmark_returns) -> Float64
```


Down capture = mean(portfolio | benchmark &lt; 0) / mean(benchmark | benchmark &lt; 0) Lower is better (lose less when market falls).


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/FactorModels.jl#L431-L436" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.FactorModels.capture_ratio' href='#QuantNova.FactorModels.capture_ratio'><span class="jlbinding">QuantNova.FactorModels.capture_ratio</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
capture_ratio(returns, benchmark_returns) -> Float64
```


Capture ratio = up_capture / down_capture.
> 
> 1 means manager adds value (captures more upside, less downside).
> 



<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/FactorModels.jl#L443-L448" target="_blank" rel="noreferrer">source</a></Badge>

</details>


## Visualization {#Visualization}
<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.Visualization.AbstractVisualization' href='#QuantNova.Visualization.AbstractVisualization'><span class="jlbinding">QuantNova.Visualization.AbstractVisualization</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
AbstractVisualization
```


Base type for visualization objects.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/Visualization.jl#L9-L13" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.Visualization.VisualizationSpec' href='#QuantNova.Visualization.VisualizationSpec'><span class="jlbinding">QuantNova.Visualization.VisualizationSpec</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
VisualizationSpec
```


Lazy container for visualization configuration. Not rendered until displayed.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/Visualization.jl#L16-L20" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.Visualization.LinkedContext' href='#QuantNova.Visualization.LinkedContext'><span class="jlbinding">QuantNova.Visualization.LinkedContext</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
LinkedContext
```


Shared state for linked interactive plots. All plots sharing a context will synchronize their cursors, zoom levels, and selections.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/Visualization.jl#L168-L173" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.Visualization.visualize' href='#QuantNova.Visualization.visualize'><span class="jlbinding">QuantNova.Visualization.visualize</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
visualize(data; kwargs...)
visualize(data, view::Symbol; kwargs...)
visualize(data, views::Vector{Symbol}; kwargs...)
```


Create a visualization specification for the given data.

**Arguments**
- `data`: Data to visualize (BacktestResult, OptimizationResult, etc.)
  
- `view`: Specific view to render (e.g., `:equity`, `:drawdown`, `:frontier`)
  
- `views`: Multiple views to render as linked panels
  
- `theme`: Override theme (`:light`, `:dark`, or custom Dict)
  
- `backend`: Override backend (`:gl`, `:wgl`, `:cairo`)
  

**Examples**

```julia
result = backtest(strategy, data)
visualize(result)                    # Default view
visualize(result, :drawdown)         # Specific view
visualize(result, [:equity, :drawdown])  # Multiple linked views
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/Visualization.jl#L102-L123" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.Visualization.available_views' href='#QuantNova.Visualization.available_views'><span class="jlbinding">QuantNova.Visualization.available_views</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
available_views(data)
```


Return the list of available visualization views for the given data type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/Visualization.jl#L157-L161" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.Visualization.set_theme!' href='#QuantNova.Visualization.set_theme!'><span class="jlbinding">QuantNova.Visualization.set_theme!</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
set_theme!(theme::Symbol)
```


Set the global visualization theme. Options: `:light`, `:dark`.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/Visualization.jl#L80-L84" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.Visualization.get_theme' href='#QuantNova.Visualization.get_theme'><span class="jlbinding">QuantNova.Visualization.get_theme</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
get_theme()
```


Get the current theme dictionary.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/Visualization.jl#L95-L99" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.Visualization.LIGHT_THEME' href='#QuantNova.Visualization.LIGHT_THEME'><span class="jlbinding">QuantNova.Visualization.LIGHT_THEME</span></a> <Badge type="info" class="jlObjectType jlConstant" text="Constant" /></summary>



```julia
LIGHT_THEME
```


Default light theme configuration for visualizations.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/Visualization.jl#L28-L32" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.Visualization.DARK_THEME' href='#QuantNova.Visualization.DARK_THEME'><span class="jlbinding">QuantNova.Visualization.DARK_THEME</span></a> <Badge type="info" class="jlObjectType jlConstant" text="Constant" /></summary>



```julia
DARK_THEME
```


Default dark theme configuration for visualizations.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/Visualization.jl#L46-L50" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.Visualization.COLORS' href='#QuantNova.Visualization.COLORS'><span class="jlbinding">QuantNova.Visualization.COLORS</span></a> <Badge type="info" class="jlObjectType jlConstant" text="Constant" /></summary>



```julia
COLORS
```


Semantic color palette used across visualizations.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/Visualization.jl#L65-L69" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.Visualization.render' href='#QuantNova.Visualization.render'><span class="jlbinding">QuantNova.Visualization.render</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
render(spec::VisualizationSpec)
```


Render a visualization specification to produce a plot.

This function is implemented by the Makie extension (QuantNovaMakieExt). To use it, load Makie or one of its backends (CairoMakie, GLMakie, WGLMakie).

**Examples**

```julia
using QuantNova
using CairoMakie  # or GLMakie, WGLMakie

result = backtest(strategy, data)
spec = visualize(result, :equity)
fig = render(spec)
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/Visualization.jl#L189-L206" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.Visualization.Row' href='#QuantNova.Visualization.Row'><span class="jlbinding">QuantNova.Visualization.Row</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
Row(items...; weight=1)
```


A row in a dashboard layout.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/Visualization.jl#L218-L222" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.Visualization.Dashboard' href='#QuantNova.Visualization.Dashboard'><span class="jlbinding">QuantNova.Visualization.Dashboard</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
Dashboard
```


A multi-panel dashboard layout.

**Example**

```julia
dashboard = Dashboard(
    title = "Strategy Monitor",
    theme = :dark,
    layout = [
        Row(visualize(result, :equity), weight=2),
        Row(visualize(result, :drawdown), visualize(result, :returns)),
    ]
)
serve(dashboard)
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/Visualization.jl#L230-L247" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.Visualization.serve' href='#QuantNova.Visualization.serve'><span class="jlbinding">QuantNova.Visualization.serve</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
serve(item; port=8080)
```


Serve a visualization or dashboard in the browser. Requires WGLMakie and Bonito to be loaded.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/Visualization.jl#L258-L263" target="_blank" rel="noreferrer">source</a></Badge>

</details>


::: warning Missing docstring.

Missing docstring for `save`. Check Documenter&#39;s build log for details.

:::

## Statistics {#Statistics}
<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.Statistics.sharpe_ratio' href='#QuantNova.Statistics.sharpe_ratio'><span class="jlbinding">QuantNova.Statistics.sharpe_ratio</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
sharpe_ratio(returns; rf=0.0, periods_per_year=252) -> Float64
```


Compute annualized Sharpe ratio.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/Statistics.jl#L11-L15" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.Statistics.sharpe_std_error' href='#QuantNova.Statistics.sharpe_std_error'><span class="jlbinding">QuantNova.Statistics.sharpe_std_error</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
sharpe_std_error(returns, sharpe; periods_per_year=252) -> Float64
```


Standard error of Sharpe ratio estimate. Lo (2002) formula. SE(SR) ≈ sqrt((1 + 0.5*SR²) / n)


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/Statistics.jl#L22-L27" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.Statistics.sharpe_confidence_interval' href='#QuantNova.Statistics.sharpe_confidence_interval'><span class="jlbinding">QuantNova.Statistics.sharpe_confidence_interval</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
sharpe_confidence_interval(returns; rf=0.0, confidence=0.95, periods_per_year=252)
```


Confidence interval for Sharpe ratio.

Returns (sharpe, lower, upper, std_error)


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/Statistics.jl#L34-L40" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.Statistics.sharpe_t_stat' href='#QuantNova.Statistics.sharpe_t_stat'><span class="jlbinding">QuantNova.Statistics.sharpe_t_stat</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
sharpe_t_stat(returns; rf=0.0, benchmark_sharpe=0.0) -> Float64
```


T-statistic for Sharpe ratio vs benchmark (default: testing if SR &gt; 0).


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/Statistics.jl#L55-L59" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.Statistics.sharpe_pvalue' href='#QuantNova.Statistics.sharpe_pvalue'><span class="jlbinding">QuantNova.Statistics.sharpe_pvalue</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
sharpe_pvalue(returns; rf=0.0, benchmark_sharpe=0.0, alternative=:greater) -> Float64
```


P-value for Sharpe ratio hypothesis test.
- :greater - test if SR &gt; benchmark (one-tailed)
  
- :two_sided - test if SR ≠ benchmark
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/Statistics.jl#L66-L72" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.Statistics.probabilistic_sharpe_ratio' href='#QuantNova.Statistics.probabilistic_sharpe_ratio'><span class="jlbinding">QuantNova.Statistics.probabilistic_sharpe_ratio</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
probabilistic_sharpe_ratio(returns, benchmark_sharpe; rf=0.0) -> Float64
```


Probability that true Sharpe exceeds benchmark, accounting for skewness/kurtosis. Bailey &amp; López de Prado (2012).


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/Statistics.jl#L94-L99" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.Statistics.deflated_sharpe_ratio' href='#QuantNova.Statistics.deflated_sharpe_ratio'><span class="jlbinding">QuantNova.Statistics.deflated_sharpe_ratio</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
deflated_sharpe_ratio(returns, n_trials; rf=0.0, expected_max_sr=nothing) -> Float64
```


Deflated Sharpe Ratio - adjusts for multiple testing (strategy selection bias). Bailey &amp; López de Prado (2014).

`n_trials` = number of strategies/parameters tested before selecting this one.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/Statistics.jl#L121-L128" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.Statistics.compare_sharpe_ratios' href='#QuantNova.Statistics.compare_sharpe_ratios'><span class="jlbinding">QuantNova.Statistics.compare_sharpe_ratios</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
compare_sharpe_ratios(returns_a, returns_b; rf=0.0, method=:jobson_korkie)
```


Test if strategy A has significantly higher Sharpe than strategy B.

Methods:
- :jobson_korkie - Jobson-Korkie (1981) test
  
- :bootstrap - Bootstrap test (more robust)
  

Returns (z_stat, pvalue, sharpe_diff)


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/Statistics.jl#L154-L164" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.Statistics.minimum_backtest_length' href='#QuantNova.Statistics.minimum_backtest_length'><span class="jlbinding">QuantNova.Statistics.minimum_backtest_length</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
minimum_backtest_length(sharpe_target, n_trials; confidence=0.95) -> Int
```


Minimum backtest length needed to achieve target Sharpe with given confidence, accounting for multiple testing. Bailey et al. (2015).

More trials = higher expected max SR under null = need more data to prove significance.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/Statistics.jl#L229-L236" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.Statistics.probability_of_backtest_overfitting' href='#QuantNova.Statistics.probability_of_backtest_overfitting'><span class="jlbinding">QuantNova.Statistics.probability_of_backtest_overfitting</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
probability_of_backtest_overfitting(in_sample_sr, out_sample_sr, n_trials) -> Float64
```


Estimate probability that backtest is overfit. High if in-sample &gt;&gt; out-sample and many trials.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/Statistics.jl#L264-L269" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.Statistics.combinatorial_purged_cv_pbo' href='#QuantNova.Statistics.combinatorial_purged_cv_pbo'><span class="jlbinding">QuantNova.Statistics.combinatorial_purged_cv_pbo</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
combinatorial_purged_cv_pbo(returns, n_paths; train_frac=0.5) -> Float64
```


Probability of Backtest Overfitting via Combinatorial Purged Cross-Validation. Bailey et al. (2017). Simplified implementation.

Returns probability that strategy is overfit.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/Statistics.jl#L288-L295" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.Statistics.information_coefficient' href='#QuantNova.Statistics.information_coefficient'><span class="jlbinding">QuantNova.Statistics.information_coefficient</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
information_coefficient(predictions, outcomes) -> Float64
```


Correlation between predictions and outcomes. Key alpha metric.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/Statistics.jl#L333-L337" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.Statistics.hit_rate' href='#QuantNova.Statistics.hit_rate'><span class="jlbinding">QuantNova.Statistics.hit_rate</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
hit_rate(predictions, outcomes) -> Float64
```


Fraction of correct directional predictions.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/Statistics.jl#L342-L346" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuantNova.Statistics.hit_rate_significance' href='#QuantNova.Statistics.hit_rate_significance'><span class="jlbinding">QuantNova.Statistics.hit_rate_significance</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
hit_rate_significance(hit_rate, n_predictions; benchmark=0.5) -> (z_stat, pvalue)
```


Test if hit rate is significantly above benchmark (default 50%).


<Badge type="info" class="source-link" text="source"><a href="https://github.com/KookiesNKareem/QuantNova.jl/blob/bbad8749074a64663c1f33d1605b9be8b670e516/src/Statistics.jl#L352-L356" target="_blank" rel="noreferrer">source</a></Badge>

</details>

