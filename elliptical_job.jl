using SpecialFunctions
using QuantumOptics
using ArgParse
using Cubature
using DelimitedFiles


σ1 = .1
σ2 = 1
amp = .001

start = time()
println(start)
flush(stdout)

function fidelity(ρ, σ)
    tr(sqrt(sqrt(ρ) * σ * sqrt(ρ)))^(1/2)
end

function gaussian(σ1, σ2)
    function func(ρ, ϕ)
        x = ρ*cos(ϕ)
        y = ρ*sin(ϕ)
        amp*exp(-x^2/σ1^2 + -y^2/σ2^2)
    end
end

function H_odf(ρ, ϕ, t, zernike_recon, U, ψ, orders, ω)
    sum([U * cos(-order*ω*t + ψ - gaussian(σ1, σ2)(ρ, ϕ-ω*t)) for order in orders])
end

function infidelity_across_disk(F1, F2)
    function infidelity_polar(ρ, ϕ)
        ψ1 = F1(ρ, ϕ).data
        ψ2 = F2(ρ, ϕ).data
        infid = 1 - real(fidelity(ψ1, ψ2))
        return infid, ψ1, ψ2
    end
end

function sequential_exact_evolution_evaluator_factory(ψ0, T, maxm, U, θ, ω, b)
    """Apply all the zernike coefficients given, in order, for time T each."""
    orders = range(0, maxm, step=1)
    function evaluator(ρ, ϕ)
        ψ = ψ0
        H(t, _) = H_odf(ρ, ϕ, t, 0, U, θ, orders, ω)*sigmaz(b), [], []
        _, ψ = @skiptimechecks timeevolution.master_dynamic(T, ψ, H)
        ψ = last(ψ)
    end
end

function gaussian_spin_profile(ρ, ϕ)
    ψ0 = 1/sqrt(2) * (spindown(b) + spinup(b))
    H(t, _) = gaussian(σ1, σ2)(ρ, ϕ) * sigmaz(b), [], []
    evolution_time = π/(2*amp)
    step_size = evolution_time/1
    T = [0.0:step_size:evolution_time;];
    _, ψ = timeevolution.master_dynamic(T, ψ0, H)
    last(ψ)
end

Γ = 1/62
ω = 2*π*180E3
θ = -π/2;
max_order = 20
b = SpinBasis(1//2)
ψ0 = 1/sqrt(2) * (spindown(b) + spinup(b))
U = 2 * π * 10E3
evolution_time = π/(2*U*amp)
step_size = evolution_time/1
T = [0.0:step_size:evolution_time;];
sequential_exact_evolution = sequential_exact_evolution_evaluator_factory(ψ0, T, max_order, U, θ, ω, b)
x = parse(Float64, ARGS[1])
y = parse(Float64, ARGS[2])
ρ = sqrt(x^2 + y^2)
ϕ = atan(y, x)
infid, ψ1, ψ2 = infidelity_across_disk(sequential_exact_evolution, gaussian_spin_profile)(ρ, ϕ)
writedlm("infid$x,$y.csv",  infid, ',')
writedlm("seq$x,$y.csv",  ψ1, ',')
writedlm("gauss$x,$y.csv",  ψ2, ',')
stop = time()
println(stop)
println(stop-start)
flush(stdout)
