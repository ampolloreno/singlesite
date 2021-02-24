using SpecialFunctions
using QuantumOptics
using ArgParse
using Cubature
using DelimitedFiles
start = time()
println(start)
flush(stdout)
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


function plot_triangles_across_unit_disk(f, x, y)
    ret = []
    for (i, xx) in enumerate(x)
        print(i)
        ρ = sqrt(xx^2 + y[i]^2)
        ϕ = atan(y[i], xx)
        res = f(ρ, ϕ)
        push!(ret, res)
        end
    ret
end

function fidelity(ρ, σ)
    tr(sqrt(sqrt(ρ) * σ * sqrt(ρ)))^(1/2)
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

function unpack_zernike(zernike_coefficients_even, zernike_coefficients_odd, ρ, ω, t)
    Pevens = []
    eventuples = []
    for (m, zeven) in enumerate(zernike_coefficients_even)
        meven = m - 1
        push!(Pevens, [zeven[i] * R(i-1, meven, ρ) for i in Array(range(1, length(zeven), step=1))])
    end
    for i in range(1, length(Pevens), step=1)
        rotationeven = - (i - 1) * ω * t
        total = sum(Pevens[i])
        push!(eventuples, (i - 1, total, rotationeven))
    end
    eventuples # No odd tuples, so we'll leave it simple. (In fact, only m=0 again but we'll test this.)
end



function gaussian(σ1, σ2)
    function func(ρ, ϕ)
        x = ρ*cos(ϕ)
        y = ρ*sin(ϕ)
        amp*exp(-x^2/σ1^2 + -y^2/σ2^2)
    end
end
function integrand(n, m)
    function rtn(coor)
        ρ = coor[1]
        θ = coor[2]
        x = ρ * cos(θ)
        y = ρ * sin(θ)
        amp*Z(n, m, ρ, θ) * exp(-x^2/σ1^2 - y^2/σ2^2) * ρ
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
data = hcat([[c[1] for c in [cond_eval(n, m) for n in range(0, 40, step=1)]] for m in range(0, 30, step=1)]...);
function recon(ρ, ϕ)
    total = 0
    for (n,x) in enumerate(eachrow(data))
        for (m,y) in enumerate(x)
            if m ≤ n
                total += y*Z(n-1, m-1, ρ, ϕ)
            end
        end
    end
    total
end

function H_odf(ρ, ϕ, t, zernike_recon, U, ψ, μ, ω)
    U * cos(-μ*t + ψ + gaussian(σ1, σ2)(ρ, ϕ))
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
        for order in orders
            println(order)
            flush(stdout)
            μ = order * ω
            H(t, _) = H_odf(ρ, ϕ, t, recon, U, θ, μ, ω)*sigmaz(b), [], []
            _, ψ = @skiptimechecks timeevolution.master_dynamic(T, ψ, H)
            ψ = last(ψ)
        end
        ψ
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

amp = .1
Γ = 1/62
σ1 = .1
σ2 = 1
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