module Simulation

# TODO: Add SimulationConfig for configuration parameters
# TODO: Add trade execution result types
# TODO: Add time-stepping interface

using Dates

# ============================================================================
# Core Types
# ============================================================================

"""
    SimulationState{T}

Point-in-time snapshot of simulation state.

# Fields
- `timestamp::DateTime` - Current simulation time
- `cash::T` - Cash balance
- `positions::Dict{Symbol,T}` - Asset positions (symbol => quantity)
- `prices::Dict{Symbol,T}` - Current market prices
- `metadata::Dict{Symbol,Any}` - Extensible storage for custom data
"""
struct SimulationState{T<:Real}
    timestamp::DateTime
    cash::T
    positions::Dict{Symbol,T}
    prices::Dict{Symbol,T}
    metadata::Dict{Symbol,Any}

    function SimulationState{T}(timestamp, cash, positions, prices, metadata) where T
        new{T}(timestamp, cash, positions, prices, metadata)
    end
end

# Convenience constructor
function SimulationState(;
    timestamp::DateTime,
    cash::T,
    positions::Dict{Symbol,T},
    prices::Dict{Symbol,T},
    metadata::Dict{Symbol,Any}=Dict{Symbol,Any}()
) where T<:Real
    SimulationState{T}(timestamp, cash, positions, prices, metadata)
end

# ============================================================================
# Portfolio Value Computation
# ============================================================================

"""
    portfolio_value(state::SimulationState)

Compute total portfolio value (cash + positions * prices).
"""
function portfolio_value(state::SimulationState{T}) where T
    total = state.cash
    for (sym, qty) in state.positions
        if haskey(state.prices, sym)
            total += qty * state.prices[sym]
        end
    end
    return total
end

# ============================================================================
# Orders and Fills
# ============================================================================

"""
    Order

A trade order to be executed.
"""
struct Order
    symbol::Symbol
    quantity::Float64  # Positive for buy, negative for sell
    side::Symbol       # :buy or :sell (redundant but explicit)

    function Order(symbol::Symbol, quantity::Real, side::Symbol)
        side in (:buy, :sell) || error("side must be :buy or :sell")
        qty = side == :sell ? -abs(quantity) : abs(quantity)
        new(symbol, Float64(qty), side)
    end
end

"""
    Fill

Result of executing an order.
"""
struct Fill
    symbol::Symbol
    quantity::Float64
    price::Float64
    cost::Float64      # Total cost including fees/slippage
    timestamp::DateTime
end

# ============================================================================
# Execution Models
# ============================================================================

abstract type AbstractExecutionModel end

"""
    InstantFill <: AbstractExecutionModel

Instant execution at current market price with no slippage.
"""
struct InstantFill <: AbstractExecutionModel end

"""
    SlippageModel <: AbstractExecutionModel

Linear slippage based on bid-ask spread.
"""
struct SlippageModel <: AbstractExecutionModel
    spread_bps::Float64  # Half-spread in basis points

    SlippageModel(; spread_bps::Float64=10.0) = new(spread_bps)
end

"""
    MarketImpactModel <: AbstractExecutionModel

Slippage with additional price impact based on order size.
"""
struct MarketImpactModel <: AbstractExecutionModel
    spread_bps::Float64
    impact_bps_per_unit::Float64  # Additional bps per unit traded

    MarketImpactModel(; spread_bps::Float64=10.0, impact_bps_per_unit::Float64=0.1) =
        new(spread_bps, impact_bps_per_unit)
end

"""
    execute(model, order, prices; timestamp=now())

Execute an order using the given execution model.
"""
function execute(::InstantFill, order::Order, prices::Dict{Symbol,<:Real};
                 timestamp::DateTime=now())
    price = prices[order.symbol]
    cost = abs(order.quantity) * price
    Fill(order.symbol, order.quantity, price, cost, timestamp)
end

function execute(model::SlippageModel, order::Order, prices::Dict{Symbol,<:Real};
                 timestamp::DateTime=now())
    mid_price = prices[order.symbol]
    # Buy at ask (mid + spread), sell at bid (mid - spread)
    slippage_mult = order.side == :buy ? (1 + model.spread_bps / 10000) : (1 - model.spread_bps / 10000)
    exec_price = mid_price * slippage_mult
    cost = abs(order.quantity) * exec_price
    Fill(order.symbol, order.quantity, exec_price, cost, timestamp)
end

function execute(model::MarketImpactModel, order::Order, prices::Dict{Symbol,<:Real};
                 timestamp::DateTime=now())
    mid_price = prices[order.symbol]
    qty = abs(order.quantity)

    # Base spread + linear impact
    total_bps = model.spread_bps + model.impact_bps_per_unit * qty
    slippage_mult = order.side == :buy ? (1 + total_bps / 10000) : (1 - total_bps / 10000)
    exec_price = mid_price * slippage_mult
    cost = qty * exec_price
    Fill(order.symbol, order.quantity, exec_price, cost, timestamp)
end

# ============================================================================
# Exports
# ============================================================================

export SimulationState, portfolio_value
export Order, Fill
export AbstractExecutionModel, InstantFill, SlippageModel, MarketImpactModel
export execute

end
