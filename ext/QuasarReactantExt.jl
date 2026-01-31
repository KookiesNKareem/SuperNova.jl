module QuasarReactantExt

using Quasar
using Quasar.AD: ReactantBackend, _gradient, _hessian, _jacobian, _value_and_gradient
using Quasar.Core: ADBackend

using Reactant

# ============================================================================
# Reactant Backend Implementation
# ============================================================================

# Cache for compiled functions
const COMPILED_CACHE = Dict{UInt, Any}()

function Quasar.AD._gradient(b::ReactantBackend, f, x)
    x_react = x isa Reactant.ConcreteRArray ? x : Reactant.ConcreteRArray(x)

    # Create traced version for compilation
    x_traced = Reactant.TracedRArray(x_react)

    # Compile gradient function
    grad_fn = Reactant.@compile Reactant.Compiler.gradient(f, x_traced)

    result = grad_fn(x_react)
    return Array(result)
end

function Quasar.AD._hessian(b::ReactantBackend, f, x)
    # Hessian via nested gradient
    x_react = x isa Reactant.ConcreteRArray ? x : Reactant.ConcreteRArray(x)

    function grad_f(y)
        Reactant.Compiler.gradient(f, y)
    end

    x_traced = Reactant.TracedRArray(x_react)
    hess_fn = Reactant.@compile Reactant.Compiler.jacobian(grad_f, x_traced)

    result = hess_fn(x_react)
    return Array(result)
end

function Quasar.AD._jacobian(b::ReactantBackend, f, x)
    x_react = x isa Reactant.ConcreteRArray ? x : Reactant.ConcreteRArray(x)
    x_traced = Reactant.TracedRArray(x_react)

    jac_fn = Reactant.@compile Reactant.Compiler.jacobian(f, x_traced)

    result = jac_fn(x_react)
    return Array(result)
end

function Quasar.AD._value_and_gradient(b::ReactantBackend, f, x)
    x_react = x isa Reactant.ConcreteRArray ? x : Reactant.ConcreteRArray(x)
    x_traced = Reactant.TracedRArray(x_react)

    vg_fn = Reactant.@compile Reactant.Compiler.value_and_gradient(f, x_traced)

    val, grad = vg_fn(x_react)
    return (Array(val)[], Array(grad))
end

function __init__()
    # Set default device based on availability
    if Reactant.has_cuda()
        Reactant.set_default_backend("gpu")
        @info "Reactant: CUDA GPU detected"
    end
end

end # module
