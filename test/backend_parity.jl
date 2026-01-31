using Test
using Enzyme
using Quasar

@testset "Backend Parity" begin
    f(x) = sum(x.^2) + prod(x)
    x = [1.0, 2.0, 3.0]

    # Reference: ForwardDiff
    fd_grad = Quasar.gradient(f, x; backend=ForwardDiffBackend())
    fd_val, fd_grad2 = Quasar.value_and_gradient(f, x; backend=ForwardDiffBackend())

    @testset "PureJulia matches ForwardDiff" begin
        pj_grad = Quasar.gradient(f, x; backend=PureJuliaBackend())
        @test pj_grad ≈ fd_grad atol=1e-6

        pj_val, pj_grad2 = Quasar.value_and_gradient(f, x; backend=PureJuliaBackend())
        @test pj_val ≈ fd_val
        @test pj_grad2 ≈ fd_grad2 atol=1e-6
    end

    @testset "Enzyme matches ForwardDiff" begin
        enz_grad = Quasar.gradient(f, x; backend=EnzymeBackend())
        @test enz_grad ≈ fd_grad atol=1e-10

        enz_val, enz_grad2 = Quasar.value_and_gradient(f, x; backend=EnzymeBackend())
        @test enz_val ≈ fd_val
        @test enz_grad2 ≈ fd_grad2 atol=1e-10
    end

    @testset "Hessian Parity" begin
        h(x) = sum(x.^2) + x[1]*x[2]*x[3]  # Has off-diagonal terms

        fd_hess = Quasar.hessian(h, x; backend=ForwardDiffBackend())

        pj_hess = Quasar.hessian(h, x; backend=PureJuliaBackend())
        @test pj_hess ≈ fd_hess atol=1e-4  # Finite diff less precise

        enz_hess = Quasar.hessian(h, x; backend=EnzymeBackend())
        @test enz_hess ≈ fd_hess atol=1e-10
    end

    @testset "Jacobian Parity" begin
        g(x) = [x[1]^2 + x[2], x[2]*x[3], x[1] + x[2] + x[3]]

        fd_jac = Quasar.jacobian(g, x; backend=ForwardDiffBackend())

        pj_jac = Quasar.jacobian(g, x; backend=PureJuliaBackend())
        @test pj_jac ≈ fd_jac atol=1e-6

        enz_jac = Quasar.jacobian(g, x; backend=EnzymeBackend())
        @test enz_jac ≈ fd_jac atol=1e-10
    end
end
