using GeoParams, Plots, Printf, MathTeXEngine, BenchmarkTools
import LinearAlgebra:norm
# Makie.update_theme!(fonts = (regular = texfont(), bold = texfont(:bold), italic = texfont(:italic)))

const cmy = 356.25*3600*24*100

function main()

    # Unit system
    CharDim    = SI_units(length=1000m, temperature=1000C, stress=1e7Pa, viscosity=1e20Pas)

    # Physical parameters
    Ly         = nondimensionalize(2e4m, CharDim)
    T0         = nondimensionalize(673K, CharDim)
    τxy0       = nondimensionalize(550e6Pa, CharDim)
    Vτ0        = nondimensionalize(1.0Pa*m/s, CharDim)
    Ẇ0         = nondimensionalize(5e-5Pa/s, CharDim)
    ε0         = nondimensionalize(8e-14s^-1, CharDim)
    ρ          = nondimensionalize(2800kg/m^3, CharDim)
    Cp         = nondimensionalize(1050J/kg/K, CharDim)
    k          = nondimensionalize(2.5J/s/m/K, CharDim)
    ΔT         = nondimensionalize(20K, CharDim)
    σ          = Ly/40

    # Numerical parameters
    Ncy        = 100
    Nt         = 100
    Δy         = Ly/Ncy
    yc         = LinRange(-Ly/2-Δy/2, Ly/2+Δy/2, Ncy+2)
    yv         = LinRange(-Ly/2,      Ly/2,      Ncy+1)
    Δt         = nondimensionalize(2.5e10s, CharDim)

    # Allocate arrays
    Tc         = T0 .+ ΔT.*exp.(-yc.^2/2σ^2)  
    Tc0        = copy(Tc) 
    Tv         = 0.5*(Tc[1:end-1] .+ Tc[2:end])
    ∂T∂y       =   zeros(Ncy+1)
    qT         = T0*ones(Ncy+1)
    τxy        =   zeros(Ncy+1) 
    ε̇xy        = ε0*ones(Ncy+1)
    ε̇ii        = ε0*ones(Ncy+1) 
    η_phys     =   zeros(Ncy+1)
    η          =   zeros(Ncy+1)
    ΔτV        =   zeros(Ncy)
    η_mm       =   zeros(Ncy)
    Vx         =   zeros(Ncy+2);  Vx .= ε0.*yc
    RT         =   zeros(Ncy+2)
    RV         =   zeros(Ncy+2)
    ∂T∂τ       =   zeros(Ncy+2)
    ∂V∂τ       =   zeros(Ncy+2)

    # Monitoring
    probes    = (Ẇ0 = zeros(Nt), τxy0 = zeros(Nt), Vx0  = zeros(Nt))

    # Configure viscosity model
    flow_nd  = SetDislocationCreep("Diabase | Caristan (1982)", CharDim)    # non-dimensionalized
    # flow_dim = dimensionalize( flow, CharDim)       # dimensionalized

    # Setup up viscosity model
    for i in eachindex(ε̇ii)
        η[i]     = compute_viscosity_εII(flow_nd, ε̇ii[i], (;T=Tv[i]))
    end
    @show minimum(dimensionalize(η, Pas, CharDim ) )
    @show maximum(dimensionalize(η, Pas, CharDim ) )

    # BC
    BC  = :Robin1
    VxS =  ε0*yv[1]
    VxN =  ε0*yv[end]

    # PT solver
    niter = 25000
    θV    = 0.02
    θT    = 0.4
    nout  = 500
    ϵ     = 1e-7

    for it=1:Nt
        # History
        @. Tc0  = Tc
        # PT steps
        @. η_mm  = min.(η[1:end-1], η[2:end])
        @. ΔτV   = Δy^2/η_mm/2.1/10
        ΔτT      = Δy^2/(k/ρ/Cp)/2.1/10

        # a = @allocated begin

        @views for iter=1:niter

            # Kinematics
            Vx[1]   = -Vx[2]     + 2VxS
            if BC==:Dirichlet
                Vx[end] = -Vx[end-1] + 2VxN
            elseif BC==:Neumann
                Vx[end] = Vx[end-1] + τxy0*Δy/η[end]
            elseif BC==:Robin1  
                Vx[end] = Vx[end-1] + Δy*sqrt(2*Ẇ0/η[end])
            elseif BC==:Robin2  
                Vx[end] = sqrt(Vx[end-1]^2 + Δy*2*Vτ0/η[end])
            end
            @. ε̇xy     = 0.5*(Vx[2:end] - Vx[1:end-1])/Δy
            @. ε̇ii     = sqrt(ε̇xy^2)
            @. Tv      = 0.5*(Tc[1:end-1] + Tc[2:end])
            Tc[1]      = Tc[2]
            Tc[end]    = Tc[end-1]
            @. ∂T∂y    = (Tc[2:end] - Tc[1:end-1])/Δy

            # Stress
            for i in eachindex(ε̇ii)
                η_phys[i]     = compute_viscosity_εII(flow_nd, ε̇ii[i], (;T=Tv[i]))
            end
            @. η        = η_phys
            @. τxy      =  2 * η * ε̇xy
            @. qT       =     -k * ∂T∂y

            # Residuals
            @. RT[2:end-1] = -(Tc[2:end-1] - Tc0[2:end-1]) / Δt - 1.0/(ρ*Cp) * (qT[2:end] - qT[1:end-1])/Δy + 1.0/(ρ*Cp) * 0.5*(ε̇xy[1:end-1]*τxy[1:end-1] + ε̇xy[2:end]*τxy[2:end])
            @. RV[2:end-1] =  (τxy[2:end]  - τxy[1:end-1]) / Δy 

            # Damp residuals
            @. ∂V∂τ = RV + (1.0 - θV)*∂V∂τ
            @. ∂T∂τ = RT + (1.0 - θT)*∂T∂τ

            # Update solutions
            @. Vx[2:end-1] += ΔτV * ∂V∂τ[2:end-1]
            @. Tc[2:end-1] += ΔτT * ∂T∂τ[2:end-1]

            if mod(iter, nout) == 0
                errT = norm(RT)/sqrt(length(RT))
                errV = norm(RV)/sqrt(length(RV))
                (Δy/2/maximum(Vx) < Δt) ? Δt = Δy/2/maximum(Vx) : Δt=Δt
                @printf("Iteration %05d --- Time step %4d --- Δt = %2.2e --- ΔtC = %2.2e \n", iter, it, ustrip(dimensionalize(Δt, s, CharDim)), ustrip(dimensionalize(Δy/2/maximum(Vx), s, CharDim)))
                @printf("fT = %2.4e\n", errT)
                @printf("fV = %2.4e\n", errV)
                (errT < ϵ && errV < ϵ) && break
            end

        end 
        probes.Ẇ0[it]   = τxy[end]*ε̇xy[end]
        probes.τxy0[it] = τxy[end]
        probes.Vx0[it]  = 0.5*(Vx[end] + Vx[end-1])

        # # Visualisation
        if mod(it, 10)==0
            p1=plot(ustrip.(dimensionalize(Tc, K, CharDim) .- 273K), ustrip.(dimensionalize(yc, m, CharDim)./1e3) )
            p2=plot(ustrip.(dimensionalize(Vx, m/s, CharDim))*cmy, ustrip.(dimensionalize(yc, m, CharDim)./1e3) )
            p3=plot(log10.(ustrip.(dimensionalize(η, Pas, CharDim))), ustrip.(dimensionalize(yv, m, CharDim)./1e3) )
            p4=plot( 1:it, ustrip.(probes.Ẇ0[1:it]  ./probes.Ẇ0[1]  ), label="ẆBC"   )
            p4=plot!( 1:it, ustrip.(probes.τxy0[1:it]./probes.τxy0[1]), label="τxyBC" )
            p4=plot!( 1:it, ustrip.(probes.Vx0[1:it] ./probes.Vx0[1] ), label="VxBC"  )
            display(plot(p1,p2,p3,p4))
        end
    end
end

main()