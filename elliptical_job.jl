using SpecialFunctions
using QuantumOptics
using ArgParse
using Cubature
using DelimitedFiles
using DifferentialEquations
using Sundials

σ1 = .1
σ2 = 1  
amp= BigFloat(.1) #bringing this up to .1 fixed it, I have no idea why...

start = time()
println(start)
flush(stdout)

function fidelity(ρ, σ)
    #ρ = ρ/norm(ρ)
    #σ = σ/norm(σ)
    f = abs(conj(transpose(ρ))*σ)^2
    print(f)
    f
end

function gaussian(σ1, σ2)
    function func(ρ, ϕ)
        x = ρ*cos(ϕ)
        y = ρ*sin(ϕ)
        amp*exp(-x^2/σ1^2 + -y^2/σ2^2)
    end
end

function H_odf(ρ, ϕ, t, zernike_recon, U, ψ, order1, order2, ω)
    total = 0
    if order1 ≤ order2
        total += amp*data[order2+1, order1+1] * Z(order2, order1, ρ, ϕ-ω*t)
    end
    U * cos(-order1*ω*t + ψ + total)
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
        for order1 in orders
            for order2 in range(0, maxn, step=1)
                H(t, _) = H_odf(ρ, ϕ, t, 0, U, θ, order1, order2, ω)*sigmaz(b)
                _, ψ = timeevolution.schroedinger_dynamic(T, ψ, H; alg=DifferentialEquations.Tsit5())
                ψ = last(ψ)
            end
        end
        ψ
    end
end

function gaussian_spin_profile(ρ, ϕ)
    ψ0 = 1/sqrt(2) * (spindown(b) + spinup(b))
    H(t, _) = gaussian(σ1, σ2)(ρ, ϕ) * sigmaz(b)
    evolution_time = π/(2*amp)
    step_size = evolution_time/1
    T = [0.0:step_size:evolution_time;];
    _, ψ = timeevolution.schroedinger_dynamic(T, ψ0, H)
    last(ψ)
end

function R(n::Int64, m::Int64, ρ::Float64)
    if (n - m) % 2 != 0
        0
    else
        function summand(k)
            n = big(n)
            k = big(k)
            (-1)^k * factorial(n-k)/(factorial(k)*factorial(Int((n+m)/2) - k)*factorial(Int((n-m)/2) - k))*(ρ)^(n-2*k)
        end
        mapreduce(summand, +, Array(range(0, stop=Int((n-m)/2), step=1)))
    end
end

function Z(n, m, ρ, θ)
    if m < 0
        R(n, abs(m), ρ) * sin(abs(m) * θ)
    else
        R(n, m, ρ) * cos(m * θ)
    end
end

function integrand(n, m)
    function rtn(coor)
        ρ = coor[1]
        θ = coor[2]
        x = ρ * cos(θ)
        y = ρ * sin(θ)
        Z(n, m, ρ, θ) * exp(-x^2/σ1^2 - y^2/σ2^2) * ρ
    end
    rtn
end

function neumann(m)
    if m == 0
        2
    else
        1
    end
end

function cond_eval(n, m)
    if -n ≤ m ≤ n
        (2*n+2)/(π*neumann(m)) * hcubature(integrand(n, m), [0., 0.], [1., 2*π], maxevals=10000)[1]
    else
        0
    end
end

maxn = 1
max_order = 1
data = hcat([[c[1] for c in [cond_eval(n, m) for n in range(0, maxn, step=1)]] for m in range(0, max_order, step=1)]...)


ω = 2*π*180E3
θ = -π/2;
# From numerical experiments it seems like 40 is sufficient to match the pattern for .1, 1., to an accuracy of .003.
b = SpinBasis(1//2)
ψ0 = 1/sqrt(2) * (spindown(b) + spinup(b))
U = BigFloat(2 * π * 10E3)
#evolution_time =.0005
#U = π/(2*evolution_time*amp)
evolution_time = π/(2*U*amp)
#ω = 2*π*180E3/(2 * π * 10E3) * U
step_size = evolution_time/1
T = [0.0:step_size:evolution_time;];
sequential_exact_evolution = sequential_exact_evolution_evaluator_factory(ψ0, T, max_order, U, θ, ω, b)
x = parse(Float64, ARGS[1])
y = parse(Float64, ARGS[2])
ρ = sqrt(x^2 + y^2)
ϕ = atan(y, x)
infid, ψ1, ψ2 = infidelity_across_disk(sequential_exact_evolution, gaussian_spin_profile)(ρ, ϕ)
writedlm("2infid$x,$y.csv",  infid, ',')
writedlm("2seq$x,$y.csv",  ψ1, ',')
writedlm("2gauss$x,$y.csv",  ψ2, ',')
stop = time()
println(stop)
println(stop-start)
flush(stdout)
