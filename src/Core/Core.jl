module Core

# ============================================================================
# Abstract Type Hierarchy
# ============================================================================

"""
    AbstractInstrument

Root type for all financial instruments.
"""
abstract type AbstractInstrument end

"""
    AbstractEquity <: AbstractInstrument

Abstract type for equity instruments (stocks, ETFs).
"""
abstract type AbstractEquity <: AbstractInstrument end

"""
    AbstractDerivative <: AbstractInstrument

Abstract type for derivative instruments.
"""
abstract type AbstractDerivative <: AbstractInstrument end

"""
    AbstractOption <: AbstractDerivative

Abstract type for option contracts.
"""
abstract type AbstractOption <: AbstractDerivative end

"""
    AbstractFuture <: AbstractDerivative

Abstract type for futures contracts.
"""
abstract type AbstractFuture <: AbstractDerivative end

"""
    AbstractPortfolio

Abstract type for portfolio containers.
"""
abstract type AbstractPortfolio end

"""
    AbstractRiskMeasure

Abstract type for risk measures (VaR, CVaR, etc.).
"""
abstract type AbstractRiskMeasure end

"""
    AbstractADBackend

Abstract type for automatic differentiation backends.
"""
abstract type AbstractADBackend end

# ============================================================================
# Exports
# ============================================================================

export AbstractInstrument, AbstractEquity, AbstractDerivative, AbstractOption, AbstractFuture
export AbstractPortfolio, AbstractRiskMeasure, AbstractADBackend

end
