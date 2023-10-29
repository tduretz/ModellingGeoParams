using GeoParams, Plots, Printf, MathTeXEngine, BenchmarkTools, Statistics
import LinearAlgebra:norm
# Makie.update_theme!(fonts = (regular = texfont(), bold = texfont(:bold), italic = texfont(:italic)))

const cmy = 356.25*3600*24*100
const ky  = 356.25*3600*24*1e3

function GershgorinMechanics1D( η, Δy )
    return maximum( (η[1:end-1] .+ η[2:end])./Δy^2 + η[2:end]./Δy^2 .+ η[1:end-1]./Δy^2)
end

function ResidualMechanics!(RV, Vx, flow_nd, η, ηe, η_phys, Δy, τxy, τxy0, ε̇xy, ε̇ii, Tv, rhs )
    @. ε̇xy     = 0.5*(Vx[2:end] - Vx[1:end-1])/Δy
    @. ε̇ii     = sqrt(ε̇xy^2)
    # Stress
    for i in eachindex(ε̇ii)
        η_phys[i]   = compute_viscosity_εII(flow_nd, ε̇ii[i], (;T=Tv[i]))
    end
    @. η        = (1.0/η_phys + 1.0/(ηe))^(-1)
    @. τxy      =  2 * η * (ε̇xy + τxy0/(2*ηe))
    @. RV[2:end-1] =  (τxy[2:end]  - τxy[1:end-1]) / Δy 
end

function GershgorinThermics1D( k, ρ, Cp, Δy, Δt )
    return maximum( (1.0 ./ Δt + 2k/ρ/Cp/Δy^2) + k/ρ/Cp/Δy^2 + k/ρ/Cp/Δy^2)
end

function ResidualThermics!(RT, Tc, Tc0, Δt, ρ, Cp, k, Δy, ∂T∂y, qT, ε̇xy, τxy, rhs )
    @. ∂T∂y    = (Tc[2:end] - Tc[1:end-1])/Δy
    @. qT       =     -k * ∂T∂y
    @. RT[2:end-1] = -(Tc[2:end-1] - rhs*Tc0[2:end-1]) / Δt - 1.0/(ρ*Cp) * (qT[2:end] - qT[1:end-1])/Δy + rhs/(ρ*Cp) * 0.5*(ε̇xy[1:end-1]*τxy[1:end-1] + ε̇xy[2:end]*τxy[2:end])
end

function main()

    # Unit system
    CharDim    = SI_units(length=1000m, temperature=1000C, stress=1e7Pa, viscosity=1e20Pas)

    # Physical parameters
    Ly         = nondimensionalize(2e4m, CharDim)
    T0         = nondimensionalize(500C, CharDim)
    τxyi       = nondimensionalize(250e6Pa, CharDim)
    Vτ0        = nondimensionalize(1.0Pa*m/s, CharDim)
    Ẇ0         = nondimensionalize(5e-5Pa/s, CharDim)
    ε0         = nondimensionalize(5e-14s^-1, CharDim)
    ρ          = nondimensionalize(3000kg/m^3, CharDim)
    Cp         = nondimensionalize(1050J/kg/K, CharDim)
    k          = nondimensionalize(1.5J/s/m/K, CharDim)
    ΔT         = nondimensionalize(20K, CharDim)
    G          = nondimensionalize(5e10Pa, CharDim)
    σ          = Ly/40
    t          = 0.
    tmax       = nondimensionalize((150ky)s, CharDim)

    # Numerical parameters
    Ncy        = 100
    Nt         = 1000
    Δy         = Ly/Ncy
    yc         = LinRange(-Ly/2-Δy/2, Ly/2+Δy/2, Ncy+2)
    yv         = LinRange(-Ly/2,      Ly/2,      Ncy+1)
    Δt         = nondimensionalize(7.5e10s, CharDim)

    # Allocate arrays
    Tc         = T0 .+ ΔT.*exp.(-yc.^2/2σ^2)  
    Tc0        = copy(Tc) 
    Tv         = 0.5*(Tc[1:end-1] .+ Tc[2:end])
    ∂T∂y       =   zeros(Ncy+1)
    qT         = T0*ones(Ncy+1)
    τxy        =   zeros(Ncy+1) 
    τxy0       =   zeros(Ncy+1)
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


    Vxit       =   zeros(Ncy+2)
    KδV1       =   zeros(Ncy+2)
    Tcit       =   zeros(Ncy+2)
    KδT1       =   zeros(Ncy+2)


    δV         =   zeros(Ncy+2)
    KδV        =   zeros(Ncy+2)
    δT         =   zeros(Ncy+2)
    KδT        =   zeros(Ncy+2)
    
    # Monitoring
    probes    = (iters = zeros(Nt), t = zeros(Nt), Ẇ0 = zeros(Nt), τxyi = zeros(Nt), Vx0 = zeros(Nt), maxT = zeros(Nt))
    
    # Configure viscosity model
    flow_nd0 = DislocationCreep(;
    Name = "Dry Olivine | Hirth & Kohlstedt (2003)",
    n = 3.5NoUnits,
    r = 0.0NoUnits,
    A = 1.1e5MPa^(-7 // 2) / s,
    E = 530.0kJ / mol,
    V = 14e-6m^3 / mol,
    Apparatus = AxialCompression,
        # Name = "Diabase | Caristan (1982)",
        # n = 3.05NoUnits,
        # A = 6.0e-2MPa^(-61 // 20) / s,
        # E = 276kJ / mol,
        # V = 0m^3 / mol,
        # r = 0NoUnits,
        # Apparatus = AxialCompression,
    )
    flow_nd  = Transform_DislocationCreep(flow_nd0, CharDim)

    ηe = G*Δt

    # Setup up viscosity model
    for i in eachindex(ε̇ii)
        η[i]     = compute_viscosity_εII(flow_nd, ε̇ii[i], (;T=Tv[i]))
    end
    @show minimum(dimensionalize(η, Pas, CharDim ) )
    @show maximum(dimensionalize(η, Pas, CharDim ) )

    # BC
    BC  = :Dirichlet
    VxS =  ε0*yv[1]
    VxN =  ε0*yv[end]

    # PT solver
    niter = 25000
    θV    = 0.04
    θT    = 0.2
    nout  = 500
    ϵ     = 1e-8

    for it=1:Nt
        # History
        @. Tc0  = Tc
        @. τxy0 = τxy
        # PT steps
        @. η_mm  = min.(η[1:end-1], η[2:end])
        @. ΔτV   = Δy^2/η_mm/2.1/5
        ΔτT      = Δy^2/(k/ρ/Cp)/2.1

        # (Δy/2/maximum(Vx) < Δtc) ? Δt = Δy/2/maximum(Vx) : Δt=Δt 
        ηe = G*Δt
        t       += Δt 

        # if (t>tmax)
        #     ε1  = ε0*exp(-(t-tmax)/tmax)
        #     VxS = ε1*yv[1]*exp(-(t-tmax)/tmax)
        #     VxN = ε1*yv[end]*exp(-(t-tmax)/tmax)
        #     Vx .= ε1.*yc
        #     @show (ε0, ε1)
        # end

        # DYREL
        λmaxV = GershgorinMechanics1D( η, Δy )*4
        λminV = 1.0
        h     = 1.0
        h_ρV  = 4/(λminV + λmaxV)
        ch_ρV = 4*sqrt(λminV*λmaxV)/(λminV + λmaxV)

        λmaxT = GershgorinThermics1D( k, ρ, Cp, Δy, Δt ) *4
        λminT = 1.0
        h     = 1.0
        h_ρT  = 4/(λminT + λmaxT)
        ch_ρT = 4*sqrt(λminT*λmaxT)/(λminT + λmaxT)

        iters = 0
        @views for iter=1:niter

            iters += 1
            Vxit .= Vx
            Tcit .= Tc

            # Kinematics
            Vx[1]   = -Vx[2]     + 2VxS
            if BC==:Dirichlet
                Vx[end] = -Vx[end-1] + 2VxN
            elseif BC==:Neumann
                Vx[end] = Vx[end-1] + τxyi*Δy/η[end]
            elseif BC==:Robin1  
                Vx[end] = Vx[end-1] + Δy*sqrt(2*Ẇ0/η[end])
            elseif BC==:Robin2  
                Vx[end] = sqrt(Vx[end-1]^2 + Δy*2*Vτ0/η[end])
            end

            Tc[1]      = Tc[2]
            Tc[end]    = Tc[end-1]
            @. Tv      = 0.5*(Tc[1:end-1] + Tc[2:end])

            # Residuals
            ResidualMechanics!(RV, Vx, flow_nd, η, ηe, η_phys, Δy, τxy, τxy0, ε̇xy, ε̇ii, Tv, 1.0 )
            ResidualThermics!(RT, Tc, Tc0, Δt, ρ, Cp, k, Δy, ∂T∂y, qT, ε̇xy, τxy, 1.0 )
           
            # # Damp residuals
            # @. ∂V∂τ = RV + (1.0 - θV)*∂V∂τ
            # @. ∂T∂τ = RT + (1.0 - θT)*∂T∂τ

            # # Update solutions
            # @. Vx[2:end-1] += ΔτV * ∂V∂τ[2:end-1]
            # @. Tc[2:end-1] += ΔτT * ∂T∂τ[2:end-1]

            ∂T∂τ           .= (2-ch_ρT)/(2+ch_ρT).*∂T∂τ + 2*h_ρT/(2+ch_ρT).*RT
            δT             .= h.*∂T∂τ
            Tc[2:end-1]   .+= δT[2:end-1]
     
            ∂V∂τ           .= (2-ch_ρV)/(2+ch_ρV).*∂V∂τ + 2*h_ρV/(2+ch_ρV).*RV
            δV             .= h.*∂V∂τ
            Vx[2:end-1]   .+= δV[2:end-1]

            if mod(iter, nout) == 0 

                # Joldes et al. (2011)
                ResidualMechanics!(KδV1, Vx, flow_nd, η, ηe, η_phys, Δy, τxy, τxy0, ε̇xy, ε̇ii, Tv, 0.0 )
                ResidualMechanics!(KδV, Vxit, flow_nd, η, ηe, η_phys, Δy, τxy, τxy0, ε̇xy, ε̇ii, Tv, 0.0 )
                λminV =  abs(sum(.-δV.*(KδV1.-KδV))/sum(δV.*δV) / 1.0 )
                λmaxV = GershgorinMechanics1D( η, Δy ) * 4
                h_ρV  = 4/(λminV + λmaxV)
                ch_ρV = 4*sqrt(λminV*λmaxV)/(λminV + λmaxV)

                ResidualThermics!(KδT1, Tc, Tc0, Δt, ρ, Cp, k, Δy, ∂T∂y, qT, ε̇xy, τxy, 0.0 )
                ResidualThermics!(KδT, Tcit, Tc0, Δt, ρ, Cp, k, Δy, ∂T∂y, qT, ε̇xy, τxy, 0.0 )
                λminT =  abs(sum(.-δT.*(KδT1.-KδT))/sum(δT.*δT) / 1.0 )
                λmaxT = GershgorinThermics1D( k, ρ, Cp, Δy, Δt )  * 4
                h_ρT  = 4/(λminT + λmaxT)
                ch_ρT = 4*sqrt(λminT*λmaxT)/(λminT + λmaxT)

                @show (λminV, λmaxV)
                @show (λminT, λmaxT)

                ResidualMechanics!(RV, Vx, flow_nd, η, ηe, η_phys, Δy, τxy, τxy0, ε̇xy, ε̇ii, Tv, 1.0 )
                ResidualThermics!(RT, Tc, Tc0, Δt, ρ, Cp, k, Δy, ∂T∂y, qT, ε̇xy, τxy, 1.0 )    

                errT = norm(RT)/sqrt(length(RT))
                errV = norm(RV)/sqrt(length(RV))
                @printf("Iteration %05d --- Time step %4d --- Δt = %2.2e --- ΔtC = %2.2e \n", iter, it, ustrip(dimensionalize(Δt, s, CharDim)), ustrip(dimensionalize(Δy/2/maximum(Vx), s, CharDim)))
                @printf("fT = %2.4e\n", errT)
                @printf("fV = %2.4e\n", errV)
                @show (minimum(dimensionalize(Tc, C, CharDim)), maximum(dimensionalize(Tc, C, CharDim)))
                if ( isnan(errT) || isnan(errV) ) error() end
                ( errT < ϵ && errV < ϵ ) && break
            end
        end
        probes.t[it]     = t
        probes.Ẇ0[it]    = τxy[end]*ε̇xy[end]
        probes.τxyi[it]  = τxy[end]
        probes.Vx0[it]   = 0.5*(Vx[end] + Vx[end-1])
        probes.maxT[it]  = maximum(Tc)
        probes.iters[it] = iters

        # Visualisation
        if mod(it, 10)==0 || it==1
            p1=plot(ustrip.(dimensionalize(Tc, C, CharDim)), ustrip.(dimensionalize(yc, m, CharDim)./1e3) )
            p2=plot(ustrip.(dimensionalize(Vx, m/s, CharDim))*cmy, ustrip.(dimensionalize(yc, m, CharDim)./1e3) )
            p3=plot(log10.(ustrip.(dimensionalize(η, Pas, CharDim))), ustrip.(dimensionalize(yv, m, CharDim)./1e3) )
            # p4=plot( 1:it, ustrip.(probes.Ẇ0[1:it]  ./probes.Ẇ0[1]  ), label="ẆBC"   )
            # p4=plot!( 1:it, ustrip.(probes.τxyi[1:it]./probes.τxyi[1]), label="τxyBC" )
            # p4=plot!( 1:it, ustrip.(probes.Vx0[1:it] ./probes.Vx0[1] ), label="VxBC"  )
            p4=plot( ustrip.(dimensionalize(probes.t[1:it], s, CharDim))/ky, ustrip.(dimensionalize(probes.maxT[1:it], C, CharDim) ), label="max T"  )
            dTdt = (probes.maxT[2:it] .- probes.maxT[1:it-1]) ./ (probes.t[2:it] .- probes.t[1:it-1])
            tc   = 0.5.*(probes.t[2:it] .+ probes.t[1:it-1])
            p5=plot( ustrip.(dimensionalize(tc, s, CharDim))/ky, ustrip.(dimensionalize(dTdt, K/Myrs, CharDim) ), label="max T"  )
            p6=plot( 1:it, probes.iters[1:it], label="iters", title = "$(mean(probes.iters[1:it]))"   )
            display(plot(p1,p2,p3,p4,p5,p6))
        end
    end
end

main()