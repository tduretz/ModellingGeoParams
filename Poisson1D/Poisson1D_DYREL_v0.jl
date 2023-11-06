using GeoParams, Plots, Printf, MathTeXEngine, BenchmarkTools, Statistics
import LinearAlgebra:norm
# Makie.update_theme!(fonts = (regular = texfont(), bold = texfont(:bold), italic = texfont(:italic)))

const cmy = 356.25*3600*24*100
const ky  = 356.25*3600*24*1e3

function GershgorinThermics1D( k, ρ, Cp, Δy, Δt )
    return maximum( (1.0 ./ Δt + 2k/ρ/Cp/Δy^2) + k/ρ/Cp/Δy^2 + k/ρ/Cp/Δy^2)
end

function ResidualThermics!(RT, Tc, Tc0, Qr, Δt, ρ, Cp, k, Δy, ∂T∂y, qT, rhs )
    @. ∂T∂y        = (Tc[2:end] - Tc[1:end-1])/Δy
    @. qT          = -k * ∂T∂y
    @. RT[2:end-1] = -(Tc[2:end-1] - rhs*Tc0[2:end-1]) / Δt - 1.0/(ρ*Cp) * (qT[2:end] - qT[1:end-1])/Δy + rhs/(ρ*Cp) *  Qr[2:end-1]
end

function main()

    # Unit system
    CharDim    = SI_units(length=1000m, temperature=1000C, stress=1e7Pa, viscosity=1e20Pas)

    # Physical parameters
    Ly         = nondimensionalize(2e4m, CharDim)
    T0         = nondimensionalize(500C, CharDim)
    ρ          = nondimensionalize(3000kg/m^3, CharDim)
    Cp         = nondimensionalize(1050J/kg/K, CharDim)
    k          = nondimensionalize(1.5J/s/m/K, CharDim)
    Qr0        = nondimensionalize(1.e-3J/s/m^3, CharDim)
    ΔT         = nondimensionalize(20K, CharDim)
    σ          = Ly/40
    t          = 0.

    # Numerical parameters
    Ncy        = 1000
    Nt         = 1
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
    RT         =   zeros(Ncy+2)
    Qr         =   zeros(Ncy+2) .+ Qr0.*exp.(-(yc.-2.2).^2/2σ^2) 
    ∂T∂τ       =   zeros(Ncy+2)
 
    Tcit       =   zeros(Ncy+2)
    KδT1       =   zeros(Ncy+2)
    δT         =   zeros(Ncy+2)
    KδT        =   zeros(Ncy+2)
    
    # Monitoring
    probes    = (iters = zeros(Nt), t = zeros(Nt), Ẇ0 = zeros(Nt), τxyi = zeros(Nt), Vx0 = zeros(Nt), maxT = zeros(Nt))

    # BC

    # PT solver
    niter  = 25000
    nout   = 10
    ϵ      = 1e-12
    GershT = 1.0

    for it=1:Nt
        # History
        @. Tc0  = Tc
        
        # PT steps
        ΔτT      = Δy^2/(k/ρ/Cp)/2.1        
        t       += Δt 

        # DYREL
        λmaxT = GershgorinThermics1D( k, ρ, Cp, Δy, Δt )*GershT
        λminT = 1.0/Δt
        h     = 1.0
        h_ρT  = 4/(λminT + λmaxT)
        ch_ρT = 4*sqrt(λminT*λmaxT)/(λminT + λmaxT)

        @show (λminT, λmaxT)

        iters = 0
        @views for iter=1:niter

            iters += 1
            Tcit .= Tc

            Tc[1]      = Tc[2]
            Tc[end]    = Tc[end-1]
            @. Tv      = 0.5*(Tc[1:end-1] + Tc[2:end])

            # Residuals
            ResidualThermics!(RT, Tc, Tc0, Qr, Δt, ρ, Cp, k, Δy, ∂T∂y, qT, 1.0 )

            ∂T∂τ           .= (2-ch_ρT)/(2+ch_ρT).*∂T∂τ + 2*h_ρT/(2+ch_ρT).*RT
            δT             .= h.*∂T∂τ
            Tc[2:end-1]   .+= δT[2:end-1]
            
            if mod(iter, nout) == 0 

                ResidualThermics!(KδT1, Tc, Tc0, Qr, Δt, ρ, Cp, k, Δy, ∂T∂y, qT, 0.0 )
                ResidualThermics!(KδT, Tcit, Tc0, Qr, Δt, ρ, Cp, k, Δy, ∂T∂y, qT, 0.0 )
                λminT =  abs(sum(.-δT.*(KδT1.-KδT))/sum(δT.*δT) / 1.0 )
                λmaxT = GershgorinThermics1D( k, ρ, Cp, Δy, Δt )  * GershT
                h_ρT  = 4/(λminT + λmaxT)
                ch_ρT = 4*sqrt(λminT*λmaxT)/(λminT + λmaxT)

                ResidualThermics!(RT, Tc, Tc0, Qr, Δt, ρ, Cp, k, Δy, ∂T∂y, qT, 1.0 )    

                errT = norm(RT)/sqrt(length(RT))
                @printf("Iteration %05d --- Time step %4d --- Δt = %2.2e \n", iter, it, ustrip(dimensionalize(Δt, s, CharDim)))
                @printf("fT = %2.4e\n", errT)
                @show (λminT, λmaxT)
                if ( isnan(errT) ) error() end
                ( errT < ϵ  ) && break
            end
        end
        probes.t[it]     = t
        probes.maxT[it]  = maximum(Tc)
        probes.iters[it] = iters

        # Visualisation
        if mod(it, 10)==0 || it==1
            p1=plot(ustrip.(dimensionalize(Tc, C, CharDim)), ustrip.(dimensionalize(yc, m, CharDim)./1e3), title = "$(mean(probes.iters[1:it])) iterations" )
            display(plot(p1))
        end
    end
end

main()