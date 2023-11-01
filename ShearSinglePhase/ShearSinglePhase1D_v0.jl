using GeoParams, Plots, Printf, MathTeXEngine, BenchmarkTools, LinearAlgebra, StaticArrays
import LinearAlgebra:norm
# Makie.update_theme!(fonts = (regular = texfont(), bold = texfont(:bold), italic = texfont(:italic)))

const cmy = 356.25*3600*24*100

function PrincipalStress!(σ1, τxx, τzz, τxz, P)
    for i in eachindex(τxz)
        σ = @SMatrix[-P[i]+τxx[i] τxz[i]; τxz[i] -P[i]+τzz[i]]
        v = eigvecs(σ)
        σ1.x[i] = v[1,1]
        σ1.z[i] = v[2,1]
    end
end

function main()

    # Unit system
    CharDim    = SI_units(length=1000m, temperature=1000C, stress=1e7Pa, viscosity=1e20Pas)

    # Physical parameters
    # σxxi       = nondimensionalize( -25e3Pa, CharDim) # Courbe A - Vermeer
    # σyyi       = nondimensionalize(-100e3Pa, CharDim) # Courbe A - Vermeer
    σxxi       = nondimensionalize(-400e3Pa, CharDim) # Courbe B - Vermeer
    σyyi       = nondimensionalize(-100e3Pa, CharDim) # Courbe B - Vermeer
    σzzi       = σxxi
    Pi         = -(σxxi + σxxi + σzzi)/3.0
    τxxi       = Pi + σxxi
    τyyi       = Pi + σyyi
    τzzi       = Pi + σzzi
    τxyi       = 0.0

    E          = nondimensionalize(45MPa, CharDim)
    ν          = 0.2
    Ly         = nondimensionalize(2e4m, CharDim)
    Ẇ0         = nondimensionalize(5e-5Pa/s, CharDim)
    σ          = Ly/40
    ε0         = nondimensionalize(1e-9s^-1, CharDim)
    G          = E/2.0/(1+ν)
    Kbulk      = E/3.0/(1-2ν)
    Coh0       = nondimensionalize(0Pa, CharDim)*40
    μs         = nondimensionalize(1e52Pa*s, CharDim)
    ϕ          = 43*π/180.
    ψ          = 15.0*π/180.   
    ηvp        = nondimensionalize(1*1e11Pa*s, CharDim)     

    # Numerical parameters
    Ncy        = 100
    Nt         = 2000
    Δy         = Ly/Ncy
    yc         = LinRange(-Ly/2-Δy/2, Ly/2+Δy/2, Ncy+2)
    yv         = LinRange(-Ly/2,      Ly/2,      Ncy+1)
    Δt         = nondimensionalize(1e4s, CharDim)
    ηe         = G*Δt

    # Allocate arrays
    Pt         =  Pi*ones(Ncy+1) 
    Ptc        =  Pi*ones(Ncy+1) 
    Pt0        =  Pi*ones(Ncy+1)
    τxx        =  τxxi*ones(Ncy+1)
    τxy        =  τxyi*ones(Ncy+1)
    τyy        =  τyyi*ones(Ncy+1)    
    τzz        =  τzzi*ones(Ncy+1)  
    τxx0       =  τxxi*ones(Ncy+1)
    τxy0       =  τxyi*ones(Ncy+1)
    τyy0       =  τyyi*ones(Ncy+1)
    τzz0       =  τzzi*ones(Ncy+1)
    Kb         = Kbulk*ones(Ncy+1); Kb[50] = 2 *Kbulk #.- Coh0 .* exp.(-yv.^2/2σ^2)
    Coh        =  zeros((Ncy+1));    
    F          =  zeros((Ncy+1))
    Fc         =  zeros((Ncy+1))
    λ̇          =  zeros((Ncy+1))
    λ̇rel       =  zeros((Ncy+1))
    ispl       =  zeros(Int, (Ncy+1))
    ηve        =  zeros((Ncy+1)); ηve        .= 1.0./(1.0/μs .+ 1.0/ηe)
    ηvep       =  zeros((Ncy+1))
    ε̇xy        =  ε0*ones(Ncy+1)
    ε̇yy        =    zeros(Ncy+1)
    ε̇iiᵉᶠᶠ     =    zeros(Ncy+1)
    τii        =    zeros(Ncy+1)
    η          =    zeros(Ncy+1)
    ΔτV        =    zeros(Ncy)
    ΔτPt       =    zeros(Ncy+1)
    η_mm       =    zeros(Ncy)
    Vx         =    zeros(Ncy+2);  Vx .= ε0.*yc
    Vy         =    zeros(Ncy+2);
    RPt        =    zeros(Ncy+1)
    RVx        =    zeros(Ncy+2)
    RVy        =    zeros(Ncy+2)
    ∂Pt∂τ      =    zeros(Ncy+1)
    ∂Vx∂τ      =    zeros(Ncy+2)
    ∂Vy∂τ      =    zeros(Ncy+2)
    σ1         = (x=zeros(size(τxx)), z=zeros(size(τxx)) )
 
    ε̇xy_pl     =   zeros(Ncy+1)
    ε̇yy_pl     =   zeros(Ncy+1)
    ε̇xy_el     =   zeros(Ncy+1)
    ε̇yy_el     =   zeros(Ncy+1)
    ε̇xy_net    =   zeros(Ncy+1)
    ε̇yy_net    =   zeros(Ncy+1)

    ∇v         =   zeros(Ncy+1)
    ∇v_el      =   zeros(Ncy+1)
    ∇v_pl      =   zeros(Ncy+1)
    ∇v_net     =   zeros(Ncy+1)

    # Monitoring
    probes    = (Ẇ0 = zeros(Nt), τxy0 = zeros(Nt), σyy0 = zeros(Nt), Vx0  = zeros(Nt), τii)
    η        .= μs
   
    # BC
    BC_Vy = :Neumann
    VxS   =  ε0*yv[1]
    VxN   =  ε0*yv[end]
    VyS   =  0.0
    VyN   =  0.0

    # PT solver
    niter = 100000
    θVx   = 0.2
    θVy   = 0.2
    θPt   = 1.0
    nout  = 1000
    ϵ     = 1e-14
    rel   = 1e-2
    errPt, errVx, errVy = 0., 0., 0.

    for it=1:Nt
        # History
        @. τxy0  = τxy
        @. τxx0  = τxx
        @. τyy0  = τyy
        @. τzz0  = τzz
        @. Pt0   = Pt

        @views for iter=1:niter

            # Kinematics
            # Vx BC
            Vx[1]   = -Vx[2]     + 2VxS
            Vx[end] = -Vx[end-1] + 2VxN
            # Vy BC
            Vy[1]   = - Vy[2]     + 2VyS
            if BC_Vy == :Dirichlet
                Vy[end] = - Vy[end-1] + 2VyN
            elseif BC_Vy == :Neumann   
                Vy[end] =  Vy[end-1] + Δy/(4//3*ηve[end] + Kb[end]*Δt)*(σyyi + Pt0[end] - τyy0[end]/ηe*ηve[end])
            end
            @. ε̇xy     =  0.5*(Vx[2:end] - Vx[1:end-1])/Δy
            @. ε̇yy     = 2//3*(Vy[2:end] - Vy[1:end-1])/Δy # deviatoric

            # Stress
            @. τxy     =  2 * ηve * (ε̇xy + τxy0/2/ηe) 
            @. τyy     =  2 * ηve * (ε̇yy + τyy0/2/ηe) 
            @. τxx     =  2 * ηve * (0.0 + τxx0/2/ηe)
            @. τzz     =  2 * ηve * (0.0 + τzz0/2/ηe)
            @. τii     = sqrt(τxy^2 + 0.5*(τyy.^2 + τxx.^2 + τzz.^2))

            # Plasticity
            @. F    = τii - Coh*cos(ϕ) - Pt*sin(ϕ)
            @. Ptc  = Pt
            @. ηvep = ηve
            for pl in eachindex(F)
                λ̇[pl] = 0.0
                ε̇iiᵉᶠᶠ[pl]   = sqrt( (ε̇xy[pl] + τxy0[pl]/2/ηe)^2 + 0.5*( (0.0 + τxx0[pl]/2/ηe)^2 + ((ε̇yy[pl] + τyy0[pl]/2/ηe)).^2 + ((0.0 + τzz0[pl]/2/ηe)).^2 ) ) 
                if F[pl] > 0.
                    ispl[pl] = 1
                    λ̇[pl]    = F[pl] / (ηvp + ηve[pl] + Kb[pl]*Δt*sin(ϕ)*sin(ψ))
                    λ̇rel[pl] = (1.0-rel)*λ̇rel[pl] + rel*λ̇[pl]   
                    Ptc[pl]  = Pt[pl] + Kb[pl]*Δt*sin(ψ)*λ̇rel[pl]
                    ηvep[pl] = (Coh[pl]*cos(ϕ) + Ptc[pl]*sin(ϕ) + ηvp*λ̇rel[pl]) / 2.0 / ε̇iiᵉᶠᶠ[pl]
                    τxy[pl]  =  2 * ηve[pl] * (ε̇xy[pl] + τxy0[pl]/2/ηe - τxy[pl]/τii[pl]/2*λ̇rel[pl] ) 
                    τyy[pl]  =  2 * ηve[pl] * (ε̇yy[pl] + τyy0[pl]/2/ηe - τyy[pl]/τii[pl]/2*λ̇rel[pl] )
                    τxx[pl]  =  2 * ηve[pl] * (0.0     + τxx0[pl]/2/ηe - τxx[pl]/τii[pl]/2*λ̇rel[pl] )
                    τzz[pl]  =  2 * ηve[pl] * (0.0     + τzz0[pl]/2/ηe - τzz[pl]/τii[pl]/2*λ̇rel[pl] )
                    τii[pl]  = sqrt(τxy[pl]^2 + 0.5*(τyy[pl]^2 + τxx[pl]^2 + τzz[pl]^2))
                    Fc[pl]   = τii[pl] - Coh[pl]*cos(ϕ) - Ptc[pl]*sin(ϕ) - ηvp*λ̇rel[pl]
                end
            end

            # Check
            @. ∇v      = (Vy[2:end] - Vy[1:end-1])/Δy
            @. ε̇yy_el  =  (τyy - τyy0)/2/ηe
            @. ε̇yy_pl  =   τyy/τii/2*λ̇rel
            @. ε̇xy_el  =  (τxy - τxy0)/(2*ηe)
            @. ε̇xy_pl  =   τxy/τii/2*λ̇rel
            @. ∇v_el   =  -(Ptc - Pt0)/Kb/Δt
            @. ∇v_pl   =  λ̇rel*sin(ψ)
            @. ε̇xy_net = ε̇xy - ε̇xy_el - ε̇xy_pl
            @. ε̇yy_net = ε̇yy - ε̇yy_el - ε̇yy_pl
            @. ∇v_net  = ∇v  - ∇v_el  - ∇v_pl

            # PT time steps
            @. η_mm = min.(ηvep[1:end-1], ηvep[2:end]); 
            @. ΔτV   = Δy^2/(η_mm)/2.1 /4
            @. ΔτPt  = ηvep/Δy/Kb/Δt/50
            
            # Residuals
            @. RPt          = - Kb*Δt*(Vy[2:end]  - Vy[1:end-1] )/Δy - (Pt - Pt0)
            @. RVx[2:end-1] =   (τxy[2:end] - τxy[1:end-1])/Δy 
            @. RVy[2:end-1] =   (τyy[2:end] - τyy[1:end-1])/Δy - (Ptc[2:end] - Ptc[1:end-1])/Δy
            
            # Damp residuals
            @. ∂Vx∂τ = RVx + (1.0 - θVx)*∂Vx∂τ
            @. ∂Vy∂τ = RVy + (1.0 - θVy)*∂Vy∂τ
            @. ∂Pt∂τ = RPt + (1.0 - θPt)*∂Pt∂τ

            # Update solutions
            @. Vx[2:end-1] += ΔτV * ∂Vx∂τ[2:end-1]
            @. Vy[2:end-1] += ΔτV * ∂Vy∂τ[2:end-1]
            @. Pt += ΔτPt * ∂Pt∂τ 

            if mod(iter, nout) == 0 || iter==1
                errPt = norm(RPt)/sqrt(length(RPt))
                errVx = norm(RVx)/sqrt(length(RVx))
                errVy = norm(RVy)/sqrt(length(RVy))
                @printf("Iteration %05d --- Time step %4d --- Δt = %2.2e --- ΔtC = %2.2e --- εxy = %2.2e --- max(F) = %2.2e --- max(Fc) = %2.2e \n", iter, it, ustrip(dimensionalize(Δt, s, CharDim)), ustrip(dimensionalize(Δy/2/maximum(Vx), s, CharDim)), ε0*it*Δt, maximum(F), maximum(Fc))
                @printf("Exy_net = %2.2e --- Eyy_net = %2.2e --- Div net = %2.2e\n", mean(abs.(ε̇xy_net)), mean(abs.(ε̇yy_net)), mean(abs.(∇v_net)) )
                @printf("Exy_el  = %2.2e --- Exy_pl  = %2.2e --- Exy net = %2.2e\n", mean(abs.(ε̇xy_el)), mean(abs.(ε̇xy_pl)), mean(abs.(ε̇xy_net)) )
                @printf("fPt = %2.4e\n", errPt)
                @printf("fVx = %2.4e\n", errVx)
                @printf("fVy = %2.4e\n", errVy)
                ( errVx < ϵ && errVy < ϵ) && break 
                (isnan(errPt) || isnan(errVx) || isnan(errVx)) && error("NaNs")        
            end
        end

        @. Pt = Ptc

        # if (errPt > ϵ || errVx > ϵ || errVx > ϵ) error("non converged") end

        probes.Ẇ0[it]         = τxy[end]*ε̇xy[end]
        probes.τxy0[it]       = τxy[end]
        probes.Vx0[it]        = 0.5*(Vx[end] + Vx[end-1])
        @show probes.σyy0[it] = τyy[end] - Pt[end]
        
        # Visualisation
        if mod(it, 10)==0 || it==1 

            PrincipalStress!(σ1, τxx, τyy, τxy, Pt)

            p1=plot( title = "Total pressure", xlabel = L"$P$ [kPa]", ylabel = L"$y$ [km]" )
            p1=plot!(ustrip.(dimensionalize(Pt, Pa, CharDim))/1e3, ustrip.(dimensionalize(yv, m, CharDim)./1e3) )
            p1=plot!(ustrip.(dimensionalize(Pt[ispl.==1], Pa, CharDim))/1e3, ustrip.(dimensionalize(yv[ispl.==1], m, CharDim)./1e3), linewidth=5 )

            p2=plot(title = "Velocity", xlabel = L"$Vx$ [cm/y]", ylabel = L"$y$ [km]" )
            p2=plot!(ustrip.(dimensionalize(Vx, m/s, CharDim))*cmy, ustrip.(dimensionalize(yc, m, CharDim)./1e3) )

            p3=plot(title = "σ1 angle", xlabel = "angle", ylabel = L"$y$ [km]" )
            p3=plot!(ustrip.(-atand.(σ1.z[:] ./ σ1.x[:])), ustrip.(dimensionalize(yv, m, CharDim)./1e3), xlimits=(10,60) )

            p4=plot(title = "Probes", xlabel = "Strain", ylabel = L"[-]" )
            # p4=plot!(1:it, ustrip.(dimensionalize(probes.τxy0[1:it], Pa, CharDim))/1e3, label="τxy" )
            app_fric =  ustrip.(-probes.τxy0[1:it]./probes.σyy0[1:it])/2
            p4=plot!((1:it)*ε0*Δt, app_fric, label="-τxy/σyyBC/2", title=@sprintf("max = %1.4f", maximum(app_fric)) )
            # p4=plot!(1:it, ustrip.(dimensionalize(-probes.σyy0[1:it], Pa, CharDim))/1e3, label="σyyBC" )

            display(plot(p1,p2,p3,p4))
            
        end
    end
end

@show main()

# function getellipsepoints(cx, cy, rx, ry, θ)
# 	t = range(0, 2*pi, length=100)
# 	ellipse_x_r = @. rx * cos(t)
# 	ellipse_y_r = @. ry * sin(t)
# 	R = [cos(θ) sin(θ); -sin(θ) cos(θ)]
# 	r_ellipse = [ellipse_x_r ellipse_y_r] * R
# 	x = @. cx + r_ellipse[:,1]
# 	y = @. cy + r_ellipse[:,2]
# 	(x,y)
# end

# cx = 1  # x-position of the center
# cy = 2  # y-position of the center
# rx = 5  # major radius
# ry = 2  # minor radius
# θ  = π/3 # angle to x axis

# #plot
# lines!(ax2, getellipsepoints(cx, cy, rx, ry, θ)...)