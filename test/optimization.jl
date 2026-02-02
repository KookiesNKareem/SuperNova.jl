using Test
using QuantNova
using LinearAlgebra

@testset "Optimization" begin
    @testset "Mean-Variance" begin
        # Simple 2-asset case
        expected_returns = [0.10, 0.15]  # 10% and 15% expected returns
        cov_matrix = [0.04 0.01; 0.01 0.09]  # 20% and 30% vol, 0.167 correlation

        # Optimize for minimum variance
        result = optimize(
            MeanVariance(expected_returns, cov_matrix),
            target_return=0.12
        )

        @test length(result.weights) == 2
        @test sum(result.weights) ≈ 1.0 atol=1e-8  # Weights sum to 1
        @test all(result.weights .>= -1e-8)  # No short selling (within tolerance)

        # Check return constraint is met (within numerical tolerance for gradient descent)
        @test dot(result.weights, expected_returns) ≈ 0.12 atol=1e-4

        # Check it's on efficient frontier (higher return = higher risk beyond min-variance point)
        result_high = optimize(
            MeanVariance(expected_returns, cov_matrix),
            target_return=0.14
        )

        var_12 = result.weights' * cov_matrix * result.weights
        var_14 = result_high.weights' * cov_matrix * result_high.weights

        @test var_14 > var_12  # Higher return = higher variance on efficient frontier
    end

    @testset "Gradient-based optimization" begin
        # Non-convex objective: maximize Sharpe ratio
        expected_returns = [0.10, 0.15, 0.12]
        cov_matrix = [
            0.04 0.01 0.02;
            0.01 0.09 0.01;
            0.02 0.01 0.05
        ]

        result = optimize(
            SharpeMaximizer(expected_returns, cov_matrix, rf=0.02)
        )

        @test length(result.weights) == 3
        @test sum(result.weights) ≈ 1.0 atol=1e-6
        @test result.objective > 0  # Positive Sharpe ratio
    end

    @testset "CMA-ES Solver" begin
        μ = [0.10, 0.08, 0.12, 0.07, 0.09]
        Σ = [
            0.04  0.01  0.02  0.005 0.01
            0.01  0.03  0.01  0.008 0.005
            0.02  0.01  0.05  0.01  0.015
            0.005 0.008 0.01  0.025 0.007
            0.01  0.005 0.015 0.007 0.035
        ]
        rf = 0.02

        @testset "Construction" begin
            solver = CMAESSolver()
            @test solver.sigma == 0.3
            @test solver.max_iter == 1000
            @test solver.parallel == false
            @test solver.track_history == false

            solver2 = CMAESSolver(popsize=50, sigma=0.5, max_iter=500, parallel=true, track_history=true)
            @test solver2.popsize == 50
            @test solver2.sigma == 0.5
            @test solver2.parallel == true
            @test solver2.track_history == true
        end

        @testset "Sharpe Maximization" begin
            function neg_sharpe(w)
                ret = dot(w, μ)
                vol = sqrt(w' * Σ * w + 1e-12)
                return -(ret - rf) / vol
            end

            constraints = [FullInvestmentConstraint(), LongOnlyConstraint()]
            x0 = fill(0.2, 5)

            result = solve_cmaes(neg_sharpe, x0; constraints=constraints,
                                 solver=CMAESSolver(max_iter=300, seed=42))

            @test length(result.x) == 5
            @test sum(result.x) ≈ 1.0 atol=1e-4
            @test all(result.x .>= -1e-6)  # Long only
            @test -result.objective > 0.5  # Reasonable Sharpe
        end

        @testset "History Tracking" begin
            function neg_sharpe(w)
                ret = dot(w, μ)
                vol = sqrt(w' * Σ * w + 1e-12)
                return -(ret - rf) / vol
            end

            constraints = [FullInvestmentConstraint(), LongOnlyConstraint()]
            x0 = fill(0.2, 5)

            result = solve_cmaes(neg_sharpe, x0; constraints=constraints,
                                 solver=CMAESSolver(max_iter=100, track_history=true, seed=42))

            @test result.history !== nothing
            @test length(result.history.objectives) > 0
            @test length(result.history.step_sizes) > 0
            # Objective should decrease over time
            @test result.history.objectives[end] <= result.history.objectives[1]
        end

        @testset "Box Constraints" begin
            function neg_sharpe(w)
                ret = dot(w, μ)
                vol = sqrt(w' * Σ * w + 1e-12)
                return -(ret - rf) / vol
            end

            box_constraints = [
                FullInvestmentConstraint(),
                BoxConstraint(fill(0.05, 5), fill(0.40, 5))
            ]
            x0 = fill(0.2, 5)

            result = solve_cmaes(neg_sharpe, x0; constraints=box_constraints,
                                 solver=CMAESSolver(max_iter=300, seed=42))

            @test all(result.x .>= 0.05 - 1e-4)
            @test all(result.x .<= 0.40 + 1e-4)
            @test sum(result.x) ≈ 1.0 atol=1e-4
        end
    end

    @testset "Differential Evolution Solver" begin
        μ = [0.10, 0.08, 0.12, 0.07, 0.09]
        Σ = [
            0.04  0.01  0.02  0.005 0.01
            0.01  0.03  0.01  0.008 0.005
            0.02  0.01  0.05  0.01  0.015
            0.005 0.008 0.01  0.025 0.007
            0.01  0.005 0.015 0.007 0.035
        ]
        rf = 0.02

        @testset "Construction" begin
            solver = DESolver()
            @test solver.F == 0.8
            @test solver.CR == 0.9
            @test solver.strategy == :rand1bin
            @test solver.parallel == false
            @test solver.track_history == false

            solver2 = DESolver(F=0.5, CR=0.7, strategy=:best1bin, parallel=true)
            @test solver2.F == 0.5
            @test solver2.CR == 0.7
            @test solver2.strategy == :best1bin
        end

        @testset "Different Strategies" begin
            function neg_sharpe(w)
                ret = dot(w, μ)
                vol = sqrt(w' * Σ * w + 1e-12)
                return -(ret - rf) / vol
            end

            constraints = [FullInvestmentConstraint(), LongOnlyConstraint()]
            x0 = fill(0.2, 5)

            for strategy in [:rand1bin, :best1bin, :rand2bin, :best2bin]
                result = solve_de(neg_sharpe, x0; constraints=constraints,
                                  solver=DESolver(max_iter=100, strategy=strategy, seed=42))

                @test length(result.x) == 5
                @test sum(result.x) ≈ 1.0 atol=1e-4
                @test all(result.x .>= -1e-6)
            end
        end

        @testset "History Tracking" begin
            function neg_sharpe(w)
                ret = dot(w, μ)
                vol = sqrt(w' * Σ * w + 1e-12)
                return -(ret - rf) / vol
            end

            constraints = [FullInvestmentConstraint(), LongOnlyConstraint()]
            x0 = fill(0.2, 5)

            result = solve_de(neg_sharpe, x0; constraints=constraints,
                              solver=DESolver(max_iter=50, track_history=true, seed=42))

            @test result.history !== nothing
            @test length(result.history.objectives) > 0
            @test length(result.history.mean_fitness) > 0
        end
    end

    @testset "Covariance Validation and Regularization" begin
        @testset "Valid Covariance" begin
            Σ = [0.04 0.01; 0.01 0.09]
            result = validate_cov_matrix(Σ)
            @test result.valid
            @test isempty(result.errors)
        end

        @testset "Non-Symmetric Matrix" begin
            Σ = [0.04 0.02; 0.01 0.09]
            result = validate_cov_matrix(Σ)
            @test !result.valid
            @test any(e -> occursin("symmetric", e), result.errors)
        end

        @testset "Non-PSD Matrix" begin
            Σ = [1.0 2.0; 2.0 1.0]  # Not PSD (eigenvalues: 3, -1)
            result = validate_cov_matrix(Σ)
            @test !result.valid
            @test any(e -> occursin("positive semi-definite", e), result.errors)
        end

        @testset "Regularization" begin
            # Create ill-conditioned matrix
            Σ = [1.0 0.999; 0.999 1.0]

            # Ledoit-Wolf shrinkage
            Σ_reg = regularize_covariance(Σ; method=:ledoit_wolf)
            eigenvals = eigvals(Symmetric(Σ_reg))
            @test minimum(eigenvals) > 1e-8
            @test maximum(eigenvals) / minimum(eigenvals) < 1e7

            # Eigenvalue clipping
            Σ_clip = regularize_covariance(Σ; method=:eigenvalue_clip)
            eigenvals_clip = eigvals(Symmetric(Σ_clip))
            @test minimum(eigenvals_clip) > 1e-8

            # Diagonal loading
            Σ_load = regularize_covariance(Σ; method=:diagonal_load)
            eigenvals_load = eigvals(Symmetric(Σ_load))
            @test minimum(eigenvals_load) > 1e-8
        end

        @testset "ensure_valid_covariance" begin
            # Well-conditioned matrix should pass through unchanged
            Σ_good = [0.04 0.01; 0.01 0.09]
            Σ_out = ensure_valid_covariance(Σ_good)
            @test Σ_out ≈ Σ_good atol=1e-10

            # Ill-conditioned matrix should be regularized
            Σ_bad = [1.0 0.9999; 0.9999 1.0]
            Σ_fixed = ensure_valid_covariance(Σ_bad; regularize=true)
            eigenvals = eigvals(Symmetric(Σ_fixed))
            @test minimum(eigenvals) > 1e-8
        end
    end

    @testset "Auto-Solver Selection" begin
        μ = [0.10, 0.08, 0.12]
        Σ = [0.04 0.01 0.02; 0.01 0.03 0.01; 0.02 0.01 0.05]

        @testset "Sharpe uses CMA-ES by default" begin
            sm = SharpeMaximizer(μ, Σ, rf=0.02)
            result = optimize(sm)  # Should use CMA-ES
            @test result.converged || result.iterations > 0
            @test sum(result.weights) ≈ 1.0 atol=1e-4
            @test result.objective > 0  # Positive Sharpe
        end

        @testset "RiskParity uses CMA-ES by default" begin
            rp = RiskParity(Σ)
            result = optimize(rp)  # Should use CMA-ES
            @test result.converged || result.iterations > 0
            @test sum(result.weights) ≈ 1.0 atol=1e-4
        end

        @testset "CVaR uses DE by default" begin
            cvar = CVaRObjective(μ, Σ, alpha=0.95)
            result = optimize(cvar; target_return=0.09)  # Should use DE
            @test result.converged || result.iterations > 0
            @test sum(result.weights) ≈ 1.0 atol=1e-4
        end

        @testset "Explicit solver override" begin
            sm = SharpeMaximizer(μ, Σ, rf=0.02)

            # Use DE instead of default CMA-ES
            result_de = optimize(sm; solver=DESolver(max_iter=100, seed=42))
            @test sum(result_de.weights) ≈ 1.0 atol=1e-4

            # Use projected gradient
            result_pg = optimize(sm; solver=ProjectedGradientSolver(max_iter=1000))
            @test sum(result_pg.weights) ≈ 1.0 atol=1e-4
        end
    end
end
