using Test
using Quasar
using Dates

@testset "Simulation" begin
    @testset "SimulationState" begin
        state = SimulationState(
            timestamp=DateTime(2024, 1, 1),
            cash=100_000.0,
            positions=Dict(:AAPL => 100.0, :GOOGL => 50.0),
            prices=Dict(:AAPL => 150.0, :GOOGL => 140.0)
        )

        @test state.timestamp == DateTime(2024, 1, 1)
        @test state.cash == 100_000.0
        @test state.positions[:AAPL] == 100.0
        @test state.prices[:AAPL] == 150.0

        # Portfolio value computation
        @test portfolio_value(state) == 100_000.0 + 100*150 + 50*140  # 122,000
    end
end
