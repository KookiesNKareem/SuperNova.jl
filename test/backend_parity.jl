using Test
using Quasar

@testset "Backend Parity" begin
    f(x) = sum(x.^2) + prod(x)
    x = [1.0, 2.0, 3.0]

    # Reference: ForwardDiff
    fd_grad = gradient(f, x; backend=ForwardDiffBackend())
    fd_val, fd_grad2 = value_and_gradient(f, x; backend=ForwardDiffBackend())

    @testset "PureJulia matches ForwardDiff" begin
        pj_grad = gradient(f, x; backend=PureJuliaBackend())
        @test pj_grad ≈ fd_grad atol=1e-6

        pj_val, pj_grad2 = value_and_gradient(f, x; backend=PureJuliaBackend())
        @test pj_val ≈ fd_val
        @test pj_grad2 ≈ fd_grad2 atol=1e-6
    end

    # Enzyme tests only if Enzyme is available
    @testset "Enzyme matches ForwardDiff" begin
        if isdefined(Main, :Enzyme)
            enz_grad = gradient(f, x; backend=EnzymeBackend())
            @test enz_grad ≈ fd_grad atol=1e-10
        else
            @test_skip "Enzyme not available"
        end
    end

    # Reactant tests only if Reactant is available
    @testset "Reactant matches ForwardDiff" begin
        if isdefined(Main, :Reactant)
            react_grad = gradient(f, x; backend=ReactantBackend())
            @test react_grad ≈ fd_grad atol=1e-10
        else
            @test_skip "Reactant not available"
        end
    end
end
