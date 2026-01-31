using Test
using Quasar

@testset "Core Abstract Types" begin
    @testset "Type hierarchy exists" begin
        @test AbstractInstrument <: Any
        @test AbstractEquity <: AbstractInstrument
        @test AbstractDerivative <: AbstractInstrument
        @test AbstractOption <: AbstractDerivative
        @test AbstractPortfolio <: Any
        @test AbstractRiskMeasure <: Any
        @test AbstractADBackend <: Any
    end
end
