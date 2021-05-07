using SpecialFunctions
using QuantumOptics
using ArgParse
using Cubature
using DelimitedFiles
using OrdinaryDiffEq

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


σ1 = .1
σ2 = 1
amp= .1  #bringing this up to .1 fixed it, I have no idea why...

start = time()
println(start)
flush(stdout)

function fidelity(ρ, σ)
    #ρ = ρ/norm(ρ)
    #σ = σ/norm(σ)
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

function H_odf(ρ, ϕ, t, zernike_recon, U, ψ, order1, ω)
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
    function evaluator(ρ, ϕ)
        ψ = ψ0
        orders = range(0, maxm, step=1)
        for order1 in orders
            function H(t, _)
                total = 0
                H_odf(ρ, ϕ, t, 0, U, θ, order1, ω)*sigmaz(b)
            end
            _, ψ = timeevolution.schroedinger_dynamic(T, ψ, H)#; alg=OrdinaryDiffEq.Rodas4P(autodiff=false))
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

max_order = 20 


ω = 2*π*180E3
θ = -π/2;
b = SpinBasis(1//2)
ψ0 = 1/sqrt(2) * (spindown(b) + spinup(b))
U = BigFloat(2 * π * 10E3)
evolution_time = π/(2*U*amp)
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
