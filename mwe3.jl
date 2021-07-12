using SpecialFunctions
using QuantumOptics
using ArgParse
using Cubature
using DelimitedFiles

interionic_spacing = .1
up_modifier = sqrt(3)/2 * interionic_spacing
over_modifer = 1/2 * interionic_spacing
points_inside_circle = []
digits = 2
radius = .5
function gen_points(pt, points_inside_circle, x, y)
    pt = [round(pt[1], digits=digits), round(pt[2], digits=digits)]
    if pt in points_inside_circle || pt[1]^2 + pt[2]^2 > radius^2
        return
    else
        push!(points_inside_circle, pt)
        push!(x, pt[1])
        push!(y, pt[2])
        gen_points([pt[1] + over_modifer, pt[2] - up_modifier], points_inside_circle, x, y)
        gen_points([pt[1] - over_modifer, pt[2] - up_modifier], points_inside_circle, x, y)
        gen_points([pt[1] - over_modifer, pt[2] + up_modifier], points_inside_circle, x, y)
        gen_points([pt[1] + over_modifer, pt[2] + up_modifier], points_inside_circle, x, y)
        gen_points([pt[1] + interionic_spacing, pt[2]], points_inside_circle, x, y)
        gen_points([pt[1] + interionic_spacing, pt[2]], points_inside_circle, x, y)
        gen_points([pt[1] - interionic_spacing, pt[2]], points_inside_circle, x, y)
        return points_inside_circle, x, y
    end
end
pairs, xs, ys = gen_points([0, 0], [], [], [])
function circleShape(h, k, r)
    θ = LinRange(0, 2*π, 500)
    h .+ r*sin.(θ), k .+ r*cos.(θ)
end

σ1 = .1
σ2 = 1
amp= .001  #bringing this up to .1 fixed it, I have no idea why...

start = time()
println(start)
flush(stdout)

function fidelity(ρ, σ)
    ρ = ρ/norm(ρ)
    σ = σ/norm(σ)
    f = abs(conj(transpose(ρ))*σ)^2
    f
end

function gaussian(σ1, σ2)
    function func(ρ, ϕ)
        x = ρ*cos(ϕ)
        y = ρ*sin(ϕ)
        exp(-x^2/σ1^2 + -y^2/σ2^2)
    end
end

function H_odf(ρ, ϕ, t, zernike_recon, U, ψ, order1, order2, ω)
    U * cos(-order1*ω*t + ψ + amp*gaussian(σ1, σ2)(ρ, ϕ-ω*t))
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
            H(t, _) = H_odf(ρ, ϕ, t, 0, U, θ, order1, 0, ω)*sigmaz(b)
            _, ψ = timeevolution.schroedinger_dynamic(T, ψ, H;)# dtmin=1e-3)#; dtmin=1e-5, dt=1.1e-4)#;maxiters=1e5)# abstol=1e-10, reltol=1e-8)
            ψ = last(ψ)
        end
        ψ
    end
end

function gaussian_spin_profile(ρ, ϕ)
    ψ0 = 1/sqrt(2) * (spindown(b) + spinup(b))
    H(t, _) = gaussian(σ1, σ2)(ρ, ϕ) * sigmaz(b)
    evolution_time = π/(2)
    step_size = evolution_time/1
    T = [0.0:step_size:evolution_time;];
    T = [0, evolution_time]
    _, ψ = timeevolution.schroedinger_dynamic(T, ψ0, H)#;maxiters=1e5)# abstol=1e-10, reltol=1e-8)
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


maxn = 32
max_order = 60
#data = hcat([[c[1] for c in [cond_eval(n, m) for n in range(0, maxn, step=1)]] for m in range(0, max_order, step=1)]...)


ω = 2*π*180E3
θ = -π/2;
# From numerical experiments it seems like 40 is sufficient to match the pattern for .1, 1., to an accuracy of .003.
b = SpinBasis(1//2)
ψ0 = 1/sqrt(2) * (spindown(b) + spinup(b))
U = BigFloat(2 * π * 10E3)
evolution_time = π/(2*U*amp)
#step_size = evolution_time/1
#T = [0.0:step_size:evolution_time;];
#evolution_time = round(evolution_time, sigdigits=3)
T = [0, evolution_time]
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
