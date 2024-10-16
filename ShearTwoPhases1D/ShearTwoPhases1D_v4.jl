using GeoParams, Plots, Printf, MathTeXEngine, BenchmarkTools
import LinearAlgebra:norm

# add poroelasticity
# add dilation
# dilation of 0.05 works with viscoplasticity: massive stress drop

function PlasticDilationFactor(mt, mf, dt, B, K_d, alpha, eta_phi, phi)
    @. mt .= K_d .* alpha .* dt .* eta_phi .* (B - 1) ./ (-2 * B .* K_d .* alpha .* dt + B .* K_d .* dt + B .* alpha .^ 2 .* eta_phi .* phi - B .* alpha .^ 2 .* eta_phi + K_d .* alpha .* dt - alpha .* eta_phi .* phi + alpha .* eta_phi)
    @. mf .= B .* K_d .* dt .* eta_phi .* (1 - alpha) ./ (-2 * B .* K_d .* alpha .* dt + B .* K_d .* dt + B .* alpha .^ 2 .* eta_phi .* phi - B .* alpha .^ 2 .* eta_phi + K_d .* alpha .* dt - alpha .* eta_phi .* phi + alpha .* eta_phi)
end

const cmy = 356.25*3600*24*100

function main()

    # Unit system
    CharDim    = SI_units(length=1000m, temperature=1000C, stress=1e7Pa, viscosity=1e20Pas)

    # Physical parameters
    Ly      = nondimensionalize(2e4m, CharDim)
    τxy0    = nondimensionalize(-50e6Pa, CharDim)
    Vτ0     = nondimensionalize(1.0Pa*m/s, CharDim)
    Ẇ0      = nondimensionalize(5e-5Pa/s, CharDim)
    ΔPf     = nondimensionalize(1e6Pa, CharDim)
    σ       = Ly/40
    ε0      = nondimensionalize(8e-14s^-1, CharDim)
    ϕ0      = 1e-3
    kf      = nondimensionalize(1e-20m^2, CharDim)
    G       = nondimensionalize(2e10Pa, CharDim)
    ηs      = nondimensionalize(1e22Pa*s, CharDim)
    ηf      = nondimensionalize(1e10Pa*s, CharDim)
    ηϕ      = nondimensionalize(1e25Pa*s, CharDim)
    τ_y     = nondimensionalize(5e7Pa, CharDim)                                  # yield stress. If do_DP=true, τ_y stand for the cohesion: c*cos(ϕ)
    sinϕ    = sind(35) 
    cosϕ    = cosd(35)                                                    # sinus of the friction angle
    sinψ    = sind(0.05)     # sinus of the dilation angle 
    ηvp     = nondimensionalize(1e20Pa*s, CharDim) # regularisation "viscosity"
    Ks      = nondimensionalize(18e10Pa, CharDim)                                 # solid compressibility
    Kf      = nondimensionalize(1e10Pa, CharDim)                                 # fluid compressibility
    Pti     = nondimensionalize((10e3*2800*9.81)Pa, CharDim)
    Pfi     = nondimensionalize((10e3*1000*9.81)Pa, CharDim)
   
    # Numerical parameters
    Ncy        = 100
    Nt         = 1000
    Δy         = Ly/Ncy
    yc         = LinRange(-Ly/2-Δy/2, Ly/2+Δy/2, Ncy+2)
    yv         = LinRange(-Ly/2,      Ly/2,      Ncy+1)
    Δt         = nondimensionalize(0.05e10s, CharDim)

    # Allocate arrays
    Pt         = zeros(Ncy+1)
    Pf         = zeros(Ncy+1) 
    Pt1        = zeros(Ncy+1)
    Pf1        = zeros(Ncy+1)
    Pt0        = zeros(Ncy+1) 
    Pf0        = copy(Pf)
    Rϕ         =   zeros(Ncy+1)
    ϕv         = ϕ0*ones(Ncy+1) 
    ϕv0        = ϕ0*ones(Ncy+1)
    τII        =   zeros(Ncy+1)
    τxy        = τxy0*ones(Ncy+1)
    τxx        =   zeros(Ncy+1)
    τzz        =   zeros(Ncy+1) 
    τyy        =   zeros(Ncy+1) 
    τxy0       =   zeros(Ncy+1) 
    τyy0       =   zeros(Ncy+1)
    τxx0       =   zeros(Ncy+1)
    τzz0       =   zeros(Ncy+1)
    ε̇xy        = ε0*ones(Ncy+1)
    ε̇xx        =   zeros(Ncy+1)
    ε̇zz        =   zeros(Ncy+1)
    ∇v         =   zeros(Ncy+1)
    ε̇yy        =   zeros(Ncy+1)
    ε̇ii        = ε0*ones(Ncy+1)
    α          =   zeros(Ncy+1)
    Kd         =   zeros(Ncy+1)
    Kϕ         =   zeros(Ncy+1)
    B          =   zeros(Ncy+1)
    mt         =   zeros(Ncy+1)
    mf         =   zeros(Ncy+1)
    F          =   zeros(Ncy+1)
    λ          =   zeros(Ncy+1)
    ηϕ         =   zeros(Ncy+1)
    ηv         =   zeros(Ncy+1)
    ηe         =   zeros(Ncy+1)
    ηve        =   zeros(Ncy+1)
    ΔτV        =   zeros(Ncy)
    η_mm       =   zeros(Ncy)
    Vx         =   zeros(Ncy+2);  Vx .= ε0.*yc
    Vy         =   zeros(Ncy+2);
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
    probes      = (Ẇ0 = zeros(Nt), τxy0 = zeros(Nt), Vx0  = zeros(Nt), τII = zeros(Nt))

    # Initial condition
    @. ηv       = ηs
    @. ηe       = G*Δt
    @. Pt       = Pti + 0.0 *ΔPf *exp(-yv.^2/2σ^2) 
    @. Pf       = Pfi + 1.0 *ΔPf *exp(-yv.^2/2σ^2)
    # @. ηe[abs(yv)<1.0 ] /= 1.1
    # @. ηv[abs(yv)<1.0 ] /= 10
    @. ηve      = 1.0/(1.0/ηv + 1.0/ηe)
    VxS =  ε0*yv[1]
    VxN =  ε0*yv[end]
    VyS =  ε0/10000*yv[1]
    VyN =  ε0/10000*yv[end]

    σxxB       = nondimensionalize(-8e7Pa, CharDim) # Courbe B - Vermeer
    σyyB       = nondimensionalize(-2e7Pa, CharDim) # Courbe B - Vermeer
    σzzB       = σxxB
    PB         = -(σxxB + σyyB + σzzB)/3.0
    τxx       .= PB + σxxB
    τyy       .= PB + σyyB
    τzz       .= PB + σzzB
    τxy       .= 0.0
   
    # BC
    BC = (
        VxTypeN = :Dirichlet,
        VyTypeN = :Dirichlet,
    )

    # PT solver
    niter = 10000
    θVx   = 0.05
    θPf   = 1.0
    nout  = 1000
    ϵ     = 1e-7

    for it=1:Nt

        # History
        @. Pt0   = Pt
        @. Pf0   = Pf
        @. τxy0  = τxy
        @. τxx0  = τxx
        @. τyy0  = τyy
        @. τzz0  = τzz
        @. ϕv0   = ϕv

        # PT steps
        @. η_mm  = min(ηve[1:end-1], ηve[2:end])
        @. ΔτV   = Δy^2/η_mm/2.1 /1000
        ΔτPf  = minimum(Δy^2/(kf/ηf)/2.1) /1e13
        ΔτPt  = ηv/Δy/1e5
        Δτϕ   = 2.1.*ηve./Ncy/100

        @views for iter=1:niter

            # Poro-V-E
            @. ηϕ = ηs/ ϕv 
            @. Kϕ = G/ϕv
            @. Kd = (1 - ϕv)*(1/Ks + 1/Kϕ)^(-1)
            @. α  = 1 - Kd/Ks
            @. B  = (1/Kd - 1/Ks) / ((1/Kd - 1/Ks) +  ϕv*(1/Kf - 1/Ks))

            # Kinematics
            Vx[1]   = -Vx[2]     + 2VxS
            if BC.VxTypeN==:Dirichlet
                Vx[end] = -Vx[end-1] + 2VxN
            elseif BC.VxTypeN==:Neumann
                Vx[end] = Vx[end-1] + τxy0*Δy/ηv[end]
            elseif BC.VxTypeN==:Robin1  
                Vx[end] = Vx[end-1] + Δy*sqrt(2*Ẇ0/ηv[end])
            elseif BC.VxTypeN==:Robin2  
                Vx[end] = sqrt(Vx[end-1]^2 + Δy*2*Vτ0/ηv[end])
            end
            if BC.VyTypeN == :Dirichlet
                Vy[end] = - Vy[end-1] + 2*VyN
            elseif BC.VyTypeN == :Neumann   
                # Vy[end] =  Vy[end-1] + Δy/(4//3*arrays.ηv[end] + arrays.Kb[end]*Δt)*(BC.σyyN + Pt0[end] - τ0.yy[end]./arrays.ηe[end] * arrays.ηve[end])
            end
            Vy[1]   = - Vy[2] + 2*VyS

            @. ∇v      =       (Vy[2:end] - Vy[1:end-1])/Δy
            @. ε̇xy     =  1//2*(Vx[2:end] - Vx[1:end-1])/Δy
            @. ε̇yy     =  2//3*∇v
            @. ε̇xx     = -1//3*∇v
            @. ε̇zz     = -1//3*∇v
            @. ε̇ii     = sqrt(ε̇xy^2 + 0.5*(ε̇xx.^2 + ε̇yy.^2 + ε̇zz.^2))

            # Stress
            @. τxy     =  2 * (1.0-ϕv) * ηve * (ε̇xy + τxy0/2/ηe) 
            @. τxx     =  2 * (1.0-ϕv) * ηve * (ε̇xx + τxx0/2/ηe)
            @. τyy     =  2 * (1.0-ϕv) * ηve * (ε̇yy + τyy0/2/ηe)
            @. τzz     =  2 * (1.0-ϕv) * ηve * (ε̇zz + τzz0/2/ηe)
            @. τII     =  sqrt( 0.5*(τxx^2 + τyy^2 + τzz^2) + τxy^2)
            @. qDy[2:end-1] = -kf/ηf*(Pf[2:end] - Pf[1:end-1])/Δy  

            # Plasticity
            @. F =  τII - (Pt-Pf)*sinϕ - τ_y*cosϕ
            PlasticDilationFactor(mt, mf, Δt, B, Kd, α, ηϕ, ϕv)
            @. λ = (F>0) * F/( (1.0-ϕv)* ηve + ηvp + (mt-mf)*sinϕ*sinψ)
            @. τxy     =  2 * (1.0-ϕv) * ηve * (ε̇xy + τxy0/2/ηe - 0.5*λ*τxy/τII) 
            @. τxx     =  2 * (1.0-ϕv) * ηve * (ε̇xx + τxx0/2/ηe - 0.5*λ*τxx/τII)
            @. τyy     =  2 * (1.0-ϕv) * ηve * (ε̇yy + τyy0/2/ηe - 0.5*λ*τyy/τII)
            @. τzz     =  2 * (1.0-ϕv) * ηve * (ε̇zz + τzz0/2/ηe - 0.5*λ*τzz/τII)
            @. τII     =  sqrt( 0.5*(τxx^2 + τyy^2 + τzz^2) + τxy^2)
            @. Pt1     = Pt + sinψ*λ * mt
            @. Pf1     = Pf + sinψ*λ * mf
            @. F =  τII - (Pt1-Pf1)*sinϕ - τ_y*cosϕ - λ*ηvp
            
            # Residuals
            @. RPt          = - ( (Vy[2:end]  - Vy[1:end-1] )/Δy + (Pt - Pf)/ηϕ/(1-ϕv) + 1.0/Kd*((Pt -Pt0)/Δt - α    *(Pf -Pf0)/Δt) )
            @. RPf          = - ( (qDy[2:end] - qDy[1:end-1])/Δy - (Pt - Pf)/ηϕ/(1-ϕv) - α  /Kd*((Pt -Pt0)/Δt - 1.0/B*(Pf -Pf0)/Δt) )
            @. RVx[2:end-1] =   (τxy[2:end] - τxy[1:end-1])/Δy 
            @. RVy[2:end-1] =   (τyy[2:end] - τyy[1:end-1])/Δy - (Pt1[2:end] - Pt1[1:end-1])/Δy 
            @. Rϕ           = - ( (ϕv - ϕv0)/Δt - ((Pf1 -Pf0)-(Pt1 -Pf0))/Δt/Kϕ + (Pf1 -Pt1 )/ηϕ - λ*sinψ )

            # Damp residuals
            @. ∂Vx∂τ = RVx + (1.0 - θVx)*∂Vx∂τ
            @. ∂Vy∂τ = RVy + (1.0 - θVx)*∂Vy∂τ
            @. ∂Pf∂τ = RPf + (1.0 - θPf)*∂Pf∂τ
            @. ∂Pt∂τ = RPt

            # Update solutions
            @. Vx[2:end-1] += ΔτV * ∂Vx∂τ[2:end-1]
            @. Vy[2:end-1] += ΔτV * ∂Vy∂τ[2:end-1] 
            @. Pf          += ΔτPf * ∂Pf∂τ  
            @. Pt          += ΔτPt * ∂Pt∂τ
            @. ϕv          += Δτϕ  * Rϕ 

            if mod(iter, nout) == 0 || iter==1
                errPt = norm(RPt)/sqrt(length(RPt))
                errPf = norm(RPf)/sqrt(length(RPf))
                errVx = norm(RVx)/sqrt(length(RVx))
                errVy = norm(RVy)/sqrt(length(RVy))
                errϕ  = norm(Rϕ)/sqrt(length(Rϕ))
            #     (Δy/2/maximum(Vx) < Δt) ? Δt = Δy/2/maximum(Vx) : Δt=Δt
                @printf("Iteration %05d --- Time step %4d --- Δt = %2.2e --- ΔtC = %2.2e --- ΔτPf = %2.2e --- F = %2.2e \n", iter, it, ustrip(dimensionalize(Δt, s, CharDim)), ustrip(dimensionalize(Δy/2/maximum(Vx), s, CharDim)), ustrip(dimensionalize(ΔτPf, s, CharDim)), maximum(F))  
                @printf("fPt = %2.4e\n", errPt)
                @printf("fPf = %2.4e\n", errPf)
                @printf("fVx = %2.4e\n", errVx)
                @printf("fVy = %2.4e\n", errVy)
                @printf("fϕ  = %2.4e\n", errϕ)
                if isnan(errVx) error("Nans!") end
                (errPf < ϵ && errVx < ϵ) && break 
            end

        end 

        Pf .= Pf1
        Pt .= Pt1

    #     probes.Ẇ0[it]   = τxy[end]*ε̇xy[end]
    #     probes.τxy0[it] = τxy[end]
    #     probes.Vx0[it]  = 0.5*(Vx[end] + Vx[end-1])
        probes.τII[it] = mean(τII)

        # Visualisation
        if mod(it, 1)==0 || it==1

            yc_viz  = ustrip.(dimensionalize(yc, m, CharDim))./1e3
            yv_viz  = ustrip.(dimensionalize(yv, m, CharDim))./1e3
            Vx_viz  = ustrip.(dimensionalize(Vx, m/s, CharDim)).*cmy
            Vy_viz  = ustrip.(dimensionalize(Vy, m/s, CharDim)).*cmy
            Pt_viz  = ustrip.(dimensionalize(Pt, Pa, CharDim))./1e6
            Pf_viz  = ustrip.(dimensionalize(Pf, Pa, CharDim))./1e6
            τ_t_viz = ustrip.(dimensionalize(probes.τII[1:it], Pa, CharDim))./1e6

            p1 = plot(Vx_viz, yc_viz, title="Vx")
            p2 = plot(Vy_viz, yc_viz, title="Vy")
            p3 = plot(Pt_viz, yv_viz, title="Pt")
            p4 = plot(Pf_viz, yv_viz, title="Pf")
            p5 = plot(ϕv,     yv_viz, title="ϕ") #, xlim=(9e-4, 1.1e-3)
            p6 = plot(1:it,   τ_t_viz, title="Stress-strain")
            display(plot(p1, p2, p3, p4, p5, p6))
        end
    end
end

main()