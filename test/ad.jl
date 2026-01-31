using Test
using Quasar

@testset "AD Backend System" begin
    @testset "Backend types exist" begin
        @test PureJuliaBackend <: ADBackend
        @test ForwardDiffBackend <: ADBackend
        @test ReactantBackend <: ADBackend
        @test EnzymeBackend <: ADBackend
        @test EnzymeBackend().mode == :reverse
        @test EnzymeBackend(:forward).mode == :forward
    end

    @testset "Backend selection" begin
        # Default backend should be ForwardDiff (most stable)
        @test current_backend() isa ForwardDiffBackend

        # Can change backend
        set_backend!(PureJuliaBackend())
        @test current_backend() isa PureJuliaBackend

        # Reset for other tests
        set_backend!(ForwardDiffBackend())
    end

    @testset "with_backend context manager" begin
        original = current_backend()
        @test original isa ForwardDiffBackend

        result = with_backend(PureJuliaBackend()) do
            @test current_backend() isa PureJuliaBackend
            42
        end

        @test result == 42
        @test current_backend() isa ForwardDiffBackend  # restored
    end

    @testset "Gradient computation" begin
        f(x) = sum(x.^2)
        x = [1.0, 2.0, 3.0]

        # ForwardDiff backend
        set_backend!(ForwardDiffBackend())
        g = gradient(f, x)
        @test g ≈ [2.0, 4.0, 6.0]

        # PureJulia backend (finite differences)
        set_backend!(PureJuliaBackend())
        g_fd = gradient(f, x)
        @test g_fd ≈ [2.0, 4.0, 6.0] atol=1e-6
    end

    @testset "value_and_gradient" begin
        f(x) = sum(x.^2)
        x = [1.0, 2.0, 3.0]

        set_backend!(ForwardDiffBackend())
        val, grad = value_and_gradient(f, x)

        @test val ≈ 14.0  # 1 + 4 + 9
        @test grad ≈ [2.0, 4.0, 6.0]

        # PureJulia backend
        set_backend!(PureJuliaBackend())
        val2, grad2 = value_and_gradient(f, x)
        @test val2 ≈ 14.0
        @test grad2 ≈ [2.0, 4.0, 6.0] atol=1e-6

        set_backend!(ForwardDiffBackend())  # reset
    end

    @testset "enable_gpu!" begin
        # Should error when no GPU backend loaded
        @test_throws ErrorException enable_gpu!()
        @test_throws ErrorException enable_gpu!(:enzyme)
        @test_throws ErrorException enable_gpu!(:reactant)

        set_backend!(ForwardDiffBackend())  # reset
    end
end
