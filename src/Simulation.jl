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
# Exports
# ============================================================================

export SimulationState, portfolio_value

end
