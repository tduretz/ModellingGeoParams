using GeoParams, Plots, Printf, MathTeXEngine, BenchmarkTools
import LinearAlgebra:norm

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
    ΔPf        = nondimensionalize(1e6Pa, CharDim)
    σ          = Ly/40
    ε0         = nondimensionalize(8e-14s^-1, CharDim)
    ϕ0         = 1e-3
    ρ          = nondimensionalize(2800kg/m^3, CharDim)
    Cp         = nondimensionalize(1050J/kg/K, CharDim)
    kf         = nondimensionalize(1e-20m^2, CharDim)
    G          = nondimensionalize(2e10Pa, CharDim)
    μs         = nondimensionalize(1e22Pa*s, CharDim)
    μf         = nondimensionalize(1e10Pa*s, CharDim)
    ηϕ         = nondimensionalize(1e25Pa*s, CharDim)

    # Numerical parameters
    Ncy        = 100
    Nt         = 1
    Δy         = Ly/Ncy
    yc         = LinRange(-Ly/2-Δy/2, Ly/2+Δy/2, Ncy+2)
    yv         = LinRange(-Ly/2,      Ly/2,      Ncy+1)
    Δt         = nondimensionalize(2.5e10s, CharDim)
    ηe         = G*Δt
    ηve        = 1.0/(1.0/μs + 1.0/ηe)

    # Allocate arrays
    Pt         = zeros(Ncy+1) 
    Pf         = ΔPf.*exp.(-yv.^2/2σ^2)
    ϕv         = ϕ0*ones(Ncy+1) 
    ϕv0        = ϕ0*ones(Ncy+1)
    ϕc         = ϕ0*ones(Ncy+2)
    Tv         = T0*ones(Ncy+1)
    ∂T∂y       =   zeros(Ncy+1)
    qT         = T0*ones(Ncy+1)
    τxy        =   zeros(Ncy+1) 
    τyy        =   zeros(Ncy+1) 
    τxy0       =   zeros(Ncy+1) 
    τyy0       =   zeros(Ncy+1)
    ε̇xy        = ε0*ones(Ncy+1)
    ε̇yy        =   zeros(Ncy+1)
    ε̇ii        = ε0*ones(Ncy+1)
    η_phys     =   zeros(Ncy+1)
    η          =   zeros(Ncy+1)
    ΔτV        =   zeros(Ncy)
    η_mm       =   zeros(Ncy)
    Vx         =   zeros(Ncy+2);  Vx .= ε0.*yc
    Vy         =   zeros(Ncy+2);
    # qDx        =   zeros(Ncy+2);  
    qDy        =   zeros(Ncy+2);
    Vyf        =   zeros(Ncy+2);
    RPt        =   zeros(Ncy+1)
    RPf        =   zeros(Ncy+1)
    RVx        =   zeros(Ncy+2)
    RVy        =   zeros(Ncy+2)
    ∂Pt∂τ      =   zeros(Ncy+1)
    ∂Pf∂τ      =   zeros(Ncy+1)
    ∂Vx∂τ      =   zeros(Ncy+2)
    ∂Vy∂τ      =   zeros(Ncy+2)

    # Monitoring
    probes    = (Ẇ0 = zeros(Nt), τxy0 = zeros(Nt), Vx0  = zeros(Nt))
    η        .= μs
   
    # BC
    BC  = :Dirichlet
    VxS =  ε0*yv[1]
    VxN =  ε0*yv[end]
    VyS =  0.0
    VyN =  0.0

    # PT solver
    niter = 1000
    θVx   = 0.01
    θPf   = 1.0
    nout  = 500
    ϵ     = 1e-7

    for it=1:Nt
        # History
        @. τxy0  = τxy
        @. τyy0  = τyy
        @. ϕv0   = ϕv
        # PT steps
        @. η_mm  = min(η[1:end-1], η[2:end])
        @. ΔτV   = Δy^2/η_mm/2.1
        ΔτPf  = min(Δy^2/(kf/μf)/2.1)
        ΔτPt  = η/Δy/10

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
            Vy[1]   = - Vy[2]
            Vy[end] = - Vy[end-1]
            @. ε̇xy     =  0.5*(Vx[2:end] - Vx[1:end-1])/Δy
            @. ε̇yy     = 2//3*(Vy[2:end] - Vy[1:end-1])/Δy # deviatoric
            @. ε̇ii     = sqrt(ε̇xy^2 + ε̇yy.^2)
            @. ϕv      = ϕv0 + ((Vy[2:end] - Vy[1:end-1])/Δy)*Δt

            # Stress
            @. τxy     =  2 * ηve * (ε̇xy - τxy0/2/ηe) 
            @. τyy     =  2 * ηve * (ε̇yy - τyy0/2/ηe)
            @. qDy[2:end-1] = -kf/μf*(Pf[2:end] - Pf[1:end-1])/Δy  

            # Residuals
            @. RPt = - (Vy[2:end]  - Vy[1:end-1] )/Δy - (Pt - Pf)/ηϕ/(1-ϕv)
            @. RPf = - (qDy[2:end] - qDy[1:end-1])/Δy + (Pt - Pf)/ηϕ/(1-ϕv)
            @. RVx[2:end-1] =   (τxy[2:end] - τxy[1:end-1])/Δy 

            # Damp residuals
            @. ∂Vx∂τ = RVx + (1.0 - θVx)*∂Vx∂τ
            @. ∂Pf∂τ = RPf + (1.0 - θPf)*∂Pf∂τ
            @. ∂Pt∂τ = ∂Pt∂τ

            # Update solutions
            @. Vx[2:end-1] += ΔτV * ∂Vx∂τ[2:end-1]
            @. Pf += ΔτPf * ∂Pf∂τ /1e9
            @. Pt += ΔτPt * ∂Pt∂τ 

            # if mod(iter, nout) == 0 || iter==1
                errPt = norm(RPt)/sqrt(length(RPt))
                errPf = norm(RPf)/sqrt(length(RPf))
                errVx = norm(RVx)/sqrt(length(RVx))
            #     (Δy/2/maximum(Vx) < Δt) ? Δt = Δy/2/maximum(Vx) : Δt=Δt
                @printf("Iteration %05d --- Time step %4d --- Δt = %2.2e --- ΔtC = %2.2e \n", iter, it, ustrip(dimensionalize(Δt, s, CharDim)), ustrip(dimensionalize(Δy/2/maximum(Vx), s, CharDim)))
                @printf("fPt = %2.4e\n", errPt)
                @printf("fPf = %2.4e\n", errPf)
                @printf("fVx = %2.4e\n", errVx)
                # (errPf < ϵ && errVx < ϵ) && break 
            # end

        end 

    #     probes.Ẇ0[it]   = τxy[end]*ε̇xy[end]
    #     probes.τxy0[it] = τxy[end]
    #     probes.Vx0[it]  = 0.5*(Vx[end] + Vx[end-1])

        # Visualisation
        if mod(it, 10)==0 || it==1

            p = plot(Vx, yc)
           
            display(p)
        end
    end
end

main()