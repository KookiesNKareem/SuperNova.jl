using Test
using Quasar

@testset "Quasar.jl" begin
    include("core.jl")
    include("ad.jl")
    include("instruments.jl")
    include("portfolio.jl")
    include("risk.jl")
    include("optimization.jl")
    include("calibration.jl")
    include("montecarlo.jl")
    include("full_pipeline.jl")
    include("backend_parity.jl")
end
