using GeoParams, Plots, Printf, MathTeXEngine, BenchmarkTools, Statistics
import LinearAlgebra:norm
# Makie.update_theme!(fonts = (regular = texfont(), bold = texfont(:bold), italic = texfont(:italic)))

const cmy = 356.25*3600*24*100
const ky  = 356.25*3600*24*1e3

function GershgorinThermics2D( k, ρ, Cp, Δx, Δy, Δt, transient )
    return maximum( (transient ./ Δt .+ k.y[:,1:end-1]/ρ/Cp/Δy^2 .+ k.y[:,2:end-0]/ρ/Cp/Δy^2 .+ k.x[1:end-1,:]/ρ/Cp/Δx^2 .+ k.x[2:end-0,:]/ρ/Cp/Δx^2) + k.y[:,1:end-1]/ρ/Cp/Δy^2 + k.y[:,2:end-0]/ρ/Cp/Δy^2 + k.x[1:end-1,:]/ρ/Cp/Δx^2 + k.x[2:end-0,:]/ρ/Cp/Δx^2)
end

function ResidualThermics2D!(RT, Tc, Tc0, Qr, Δt, ρ, Cp, k, Δx, Δy, ∂T∂x, ∂T∂y, qTx, qTy, rhs, T_South, T_North, transient )
    
    Tc[:,1]   .= -Tc[:,2]     .+ 2*T_South*rhs
    Tc[:,end] .= -Tc[:,end-1] .+ 2*T_North*rhs
    Tc[1,:]   .= Tc[2,:]
    Tc[end,:] .= Tc[end-1,:]

    @. ∂T∂x        = (Tc[2:end,2:end-1] - Tc[1:end-1,2:end-1])/Δx
    @. qTx         = -k.x .* ∂T∂x
    @. ∂T∂y        = (Tc[2:end-1,2:end] - Tc[2:end-1,1:end-1])/Δy
    @. qTy         = -k.y * ∂T∂y
    @. RT[2:end-1,2:end-1] = -transient*(Tc[2:end-1,2:end-1] - rhs*Tc0[2:end-1,2:end-1]) / Δt - 1.0/(ρ*Cp) * (qTx[2:end,:] - qTx[1:end-1,:])/Δx - 1.0/(ρ*Cp) * (qTy[:,2:end] - qTy[:,1:end-1])/Δy + rhs/(ρ*Cp) *  Qr[2:end-1,2:end-1]
end

function MainPoisson2D()

    # Unit system
    CharDim    = SI_units(length=1000m, temperature=1000C, stress=1e7Pa, viscosity=1e20Pas)

    # Physical parameters
    Lx         = nondimensionalize(2e4m, CharDim)
    Ly         = nondimensionalize(2e4m, CharDim)
    T0         = nondimensionalize(500C, CharDim)
    ρ          = nondimensionalize(3000kg/m^3, CharDim)
    Cp         = nondimensionalize(1050J/kg/K, CharDim)
    k0         = nondimensionalize(1.5J/s/m/K, CharDim)
    Qr0        = nondimensionalize(1.e-3J/s/m^3, CharDim)
    ΔT         = nondimensionalize(20K, CharDim)
    σ          = Ly/40
    t          = 0.

    # Transient problem?
    transient  = 0

    # BCs
    T_North    = nondimensionalize(500C, CharDim)
    T_South    = nondimensionalize(520C, CharDim)

    # Numerical parameters
    Ncx        = 40
    Ncy        = 40
    Nt         = 1
    Δx         = Lx/Ncx
    Δy         = Ly/Ncy
    xc         = LinRange(-Lx/2-Δx/2, Lx/2+Δx/2, Ncx+2)
    yc         = LinRange(-Ly/2-Δy/2, Ly/2+Δy/2, Ncy+2)
    Δt         = nondimensionalize(2e11s, CharDim)

    # Allocate arrays
    Tc         = T0 .+ ΔT.*exp.(-(yc').^2/(2σ^2) .- xc.^2/(2σ^2) )  
    Tc0        = copy(Tc) 
    ∂T∂x       =   zeros(Ncx+1, Ncy+0)
    qTx        = T0*ones(Ncx+1, Ncy+0)
    ∂T∂y       =   zeros(Ncx+0, Ncy+1)
    qTy        = T0*ones(Ncx+0, Ncy+1)
    RT         =   zeros(Ncx+2, Ncy+2)
    Qr         =   zeros(Ncx+2, Ncy+2) .+ Qr0.*exp.(-((yc').-2.2).^2/2σ^2 .- xc.^2/(2σ^2)) 
    ∂T∂τ       =   zeros(Ncx+2, Ncy+2)
    KδT1       =   zeros(Ncx+2, Ncy+2)
    δT         =   zeros(Ncx+2, Ncy+2)
    KδT        =   zeros(Ncx+2, Ncy+2)
    kv         = k0.*ones(Ncx+1, Ncy+1)
    k          = (x=0.5.*(kv[:,1:end-1] .+ kv[:,2:end-0]), y=0.5.*(kv[1:end-1,:] .+ kv[2:end-0,:]))
    
    # Monitoring
    probes    = (iters = zeros(Nt), t = zeros(Nt), Ẇ0 = zeros(Nt), τxyi = zeros(Nt), Vx0 = zeros(Nt), maxT = zeros(Nt))
    # PT solver
    niter  = 25000
    nout   = 10
    ϵ      = 1e-11
    GershT = 1.0

    for it=1:Nt
        # History
        @. Tc0  = Tc
        
        # PT steps
        ΔτT      = Δy^2/(k0/ρ/Cp)/2.1        
        t       += Δt 

        # DYREL
        λmaxT = GershgorinThermics2D( k, ρ, Cp, Δx, Δy, Δt, transient )*GershT
        λminT = 1.0

        CFL   = 0.99
        hT    = 2/sqrt(λmaxT)*CFL
        cT    = 2*sqrt(λminT)

        iters = 0
        @views for iter=1:niter
            iters  += 1

            # BCs
            Tc[:,1]   .= -Tc[:,2]     .+ 2*T_South
            Tc[:,end] .= -Tc[:,end-1] .+ 2*T_North
            Tc[1,:]   .= Tc[2,:]
            Tc[end,:] .= Tc[end-1,:]

            # Residuals
            ResidualThermics2D!(RT, Tc, Tc0, Qr, Δt, ρ, Cp, k, Δx, Δy, ∂T∂x, ∂T∂y, qTx, qTy, 1.0, T_South, T_North, transient )

            @. ∂T∂τ                  = (2-cT*hT)/(2+cT*hT)*∂T∂τ + 2*hT/(2+cT*hT)*RT
            @. δT                    = hT*∂T∂τ
            @. Tc[2:end-1,2:end-1] .+= δT[2:end-1,2:end-1]
            
            if mod(iter, nout) == 0 
                ResidualThermics2D!(KδT1, Tc,     Tc0, Qr, Δt, ρ, Cp, k, Δx, Δy, ∂T∂x, ∂T∂y, qTx, qTy, 0.0, T_South, T_North, transient )
                ResidualThermics2D!(KδT,  Tc.-δT, Tc0, Qr, Δt, ρ, Cp, k, Δx, Δy, ∂T∂x, ∂T∂y, qTx, qTy, 0.0, T_South, T_North, transient )
                λminT =  abs(sum(.-δT.*(KδT1.-KδT))/sum(δT.*δT) / 1.0 )
                λmaxT = GershgorinThermics2D( k, ρ, Cp, Δx, Δy, Δt, transient )*GershT                
                hT    = 2/sqrt(λmaxT)*CFL  
                cT    = 2*sqrt(λminT)
    
                ResidualThermics2D!(RT, Tc, Tc0, Qr, Δt, ρ, Cp, k, Δx, Δy, ∂T∂x, ∂T∂y, qTx, qTy, 1.0, T_South, T_North, transient )

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
            p1=heatmap(ustrip.(dimensionalize(xc, m, CharDim)./1e3), ustrip.(dimensionalize(yc, m, CharDim)./1e3), ustrip.(dimensionalize(Tc, C, CharDim))', title = "$(mean(probes.iters[1:it])) iterations" )
            display(plot(p1))
        end
    end
end

MainPoisson2D()