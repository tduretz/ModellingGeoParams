using GeoParams, Plots, Printf, MathTeXEngine, BenchmarkTools, LinearAlgebra, StaticArrays, ForwardDiff
import LinearAlgebra:norm
# Makie.update_theme!(fonts = (regular = texfont(), bold = texfont(:bold), italic = texfont(:italic)))

# with Yury 24/01/24

const cmy = 356.25*3600*24*100

@inline function Residuals!(R, D, τ, τ0, Pt, Pt0, ε̇, ∇v, V, BC, arrays, yield, rel, NL, Δt, Δy)
    # Kinematics
    # Vx BC
    V.x[1]    = -V.x[2]     + 2*BC.VxS
    V.x[end]  = -V.x[end-1] + 2*BC.VxN
    # Vy
    V.y[1]   = - V.y[2]     + 2*BC.VyS
    if BC.VyTypeN == :Dirichlet
        V.y[end] = - V.y[end-1] + 2*BC.VyN
    elseif BC.VyTypeN == :Neumann   
        V.y[end] =  V.y[end-1] + Δy/(4//3*arrays.ηve[end] + arrays.Kb[end]*Δt)*(BC.σyyN + Pt0[end] - τ0.yy[end]./arrays.ηe[end] * arrays.ηve[end])
    end
    Kinematics!(ε̇, ∇v, V, Δy)
    Rheology!(τ, Pt, ε̇, ∇v, τ0, Pt0, Δt, arrays, yield, rel, NL)
    # Residuals
    ResidualMomentumX!(R.x, τ.xy, Δy)
    ResidualMomentumY!(R.y, τ.yy, Pt, Δy)
    R.x ./= D.x
    R.y ./= D.y
end

@inline @views function GershgorinMomentumX!(λmaxlocx, D, η, arr, Δt, Δy)
    λmaxlocx[2:end-1] .= 2*(η[1:end-1] .+ η[2:end])./Δy^2
    λmaxlocx ./= D.x
end

@inline @views function GershgorinMomentumY!(λmaxlocy, D, η, arr, Δt, Δy)
    λmaxlocy[2:end-1] .= 8.0/3.0.*(η[1:end-1] .+ η[2:end])./Δy^2 .+ 2.0.*Δt.*(arr.Kb[1:end-1] .+ arr.Kb[2:end-0])./Δy^2
    λmaxlocy ./= D.y
end

@inline @views function DiagPCMomentumX!(Dx, η, arr, Δt, Δy)
    Dx[2:end-1] .= 1*(η[1:end-1] .+ η[2:end])./Δy^2
end

@inline @views function DiagPCMomentumY!(Dy, η, arr, Δt, Δy)
    Dy[2:end-1] .= 1*(4.0/3.0.*(η[1:end-1].+η[2:end])./Δy^2 .+ 1.0.*Δt.*(arr.Kb[1:end-1] .+ arr.Kb[2:end-0])./Δy^2)
end

@inline @views function ResidualMomentumX!(RVx, τxy, Δy)
    @. RVx[2:end-1] =   (τxy[2:end] - τxy[1:end-1])/Δy 
end

@inline @views function ResidualMomentumY!(RVy, τyy, Pt, Δy)
    @. RVy[2:end-1] =   (τyy[2:end] - τyy[1:end-1])/Δy - (Pt[2:end] - Pt[1:end-1])/Δy
end

@inline @views function PrincipalStress!(σ1, τ, P)
    for i in eachindex(τ.xy)
        σ = @SMatrix[-P[i]+τ.xx[i] τ.xy[i]; τ.xy[i] -P[i]+τ.yy[i]]
        v = eigvecs(σ)
        σ1.x[i] = v[1,2]
        σ1.z[i] = v[2,2]
    end
end

@inline @views  function Kinematics!(ε̇, ∇v, V, Δy)
    @. ε̇.xy     =  0.5*(V.x[2:end] - V.x[1:end-1])/Δy
    @. ε̇.yy     = 2//3*(V.y[2:end] - V.y[1:end-1])/Δy # deviatoric
    @. ∇v.tot   = (V.y[2:end] - V.y[1:end-1])/Δy
end

Lode(τII, J3) = -3.0*sqrt(3.0)/2.0*J3/τII^3

function Yield_MCAS95(τ, P, ϕ, c, θt, ηvp, λ̇ )
    τII = sqrt(0.5*(τ[1]^2 + τ[2]^2 + τ[3]^2) + τ[4]^2)
    J3  = τ[1]*τ[2]*τ[3] + τ[3]*τ[4]^2 # + 2*τ[4]*τ[5]*τ[6] + τ[1]*τ[6]^2 + τ[2]*τ[5]^2
    L   = Lode(τII,J3)
    L> 1.0 ? L= 1.0 : nothing
    L<-1.0 ? L=-1.0 : nothing
    θ   =  1.0/3.0*asin(L)
    if abs(θ)>θt
        sgnθ = sign(θ)
        A = 1/3*cos(θt)*(3+tan(θt)*tan(3*θt) + 1/sqrt(3)*sgnθ*(tan(3*θt)-3*tan(θt))*sin(ϕ))
        B = 1/(3*cos(3*θt))*(sgnθ*sin(θt) + 1/sqrt(3)*sin(ϕ)*cos(θt))
        k = A - B*sin(3*θ)
    else
        k   = cos(θ) - 1/sqrt(3)*sin(ϕ)*sin(θ)
    end
    F   = k*τII - P*sin(ϕ) - c*cos(ϕ) - ηvp*λ̇
    return F
end

function Fλ(λ̇, ∂Q∂τ, ϕ, ψ, c, θt, ηvp, ηve, ηe, Kb, Δt, τxx0, τyy0, τzz0, τxy0, P0, ε̇xx, ε̇yy, ε̇zz, ε̇xy, ∇v)
    τxx  =  2 * ηve * (0.0 + τxx0/2/ηe - λ̇*∂Q∂τ[1])
    τyy  =  2 * ηve * (ε̇yy + τyy0/2/ηe - λ̇*∂Q∂τ[2]) 
    τzz  =  2 * ηve * (0.0 + τzz0/2/ηe - λ̇*∂Q∂τ[3])
    τxy  =  2 * ηve * (ε̇xy + τxy0/2/ηe - λ̇*∂Q∂τ[4]) 
    P    = P0 - Kb*Δt*(∇v - λ̇*sin(ψ))
    F    = Yield_MCAS95([τxx; τyy; τzz; τxy], P, ϕ, c, θt, ηvp, λ̇ )
    return F
end

@inline @views function Rheology!(τ, Pt, ε̇, ∇v, τ0, Pt0, Δt, arrays, yield, rel, NL)
    
    type, Coh, ϕ, ψ, θt, ηvp = yield
    Kb, ηe, ηve, ηvep, F, Fc, λ̇, λ̇rel, ispl = arrays

    α    = LinRange(0.1, 1.0, 5)
    Fmin = zero(α)

    # Stress
    @. τ.xx     =  2 * ηve * (0.0  + τ0.xx/2/ηe)
    @. τ.yy     =  2 * ηve * (ε̇.yy + τ0.yy/2/ηe) 
    @. τ.zz     =  2 * ηve * (0.0  + τ0.zz/2/ηe)
    @. τ.xy     =  2 * ηve * (ε̇.xy + τ0.xy/2/ηe) 
    @. τ.II     = sqrt(τ.xy.^2 + 0.5*(τ.yy.^2 + τ.xx.^2 + τ.zz.^2))
    @. Pt       = Pt0 - Kb*Δt*∇v.tot
    @. ηvep     = ηve

    if type==:MC
        for i in eachindex(F)
            F[i] = Yield_MCAS95( [τ.xx[i]; τ.yy[i]; τ.zz[i]; τ.xy[i]], Pt[i], ϕ, Coh[i], θt, ηvp, 0. )
        end
    else
        @. F    = τ.II - Coh*cos(ϕ) - Pt*sin(ϕ) 
    end

    for pl in axes(F,1)
        
        if F[pl] > 0.
            # λ̇[pl] = 0.0
            # ε̇.IIᵉᶠᶠ[pl]   = sqrt( (ε̇.xy[pl] + τ0.xy[pl]/2/ηe[pl])^2 + 0.5*( (0.0 + τ0.xx[pl]/2/ηe[pl])^2 + ((ε̇.yy[pl] + τ0.yy[pl]/2/ηe[pl])).^2 + ((0.0 + τ0.zz[pl]/2/ηe[pl])).^2 ) ) 
            # ispl[pl]  = 1
            # if type==:MC 
            #     F0   = 0.0
            #     iter = 0
            #     𝐹τ   = τ -> Yield_MCAS95(τ, Pt[pl], ϕ, Coh[pl], θt, ηvp, 0.0 )
                # 𝐹𝜆   = λ̇ -> Fλ(λ̇, ∂F∂τ, ϕ, ψ, Coh[pl], θt, ηvp, ηve[pl], ηe[pl], Kb[pl], Δt, τ0.xx[pl], τ0.yy[pl], τ0.zz[pl], τ0.xy[pl], Pt0[pl], 0.0, ε̇.yy[pl], 0.0, ε̇.xy[pl], ∇v.tot[pl])
                # ∂F∂τ = ForwardDiff.gradient( 𝐹τ, [τ.xx[pl]; τ.yy[pl]; τ.zz[pl]; τ.xy[pl]])
            #     for _=1:10
            #         iter +=1
            #         Fc[pl] = Fλ(λ̇[pl], ∂F∂τ, ϕ, ψ, Coh[pl], θt, ηvp, ηve[pl], ηe[pl], Kb[pl], Δt, τ0.xx[pl], τ0.yy[pl], τ0.zz[pl], τ0.xy[pl], Pt0[pl], 0.0, ε̇.yy[pl], 0.0, ε̇.xy[pl], ∇v.tot[pl])
            #         iter==1 ? F0 = Fc[pl] :  nothing
            #         abs(Fc[pl]) < 1e-7 ? break : nothing
            #         ∂F∂λ̇  = ForwardDiff.derivative(𝐹𝜆, λ̇[pl])
            #         Δλ̇    = Fc[pl]/∂F∂λ̇
            #         Fmin .= 𝐹𝜆.(λ̇[pl] .- α.*Δλ̇)
            #         _,imin = findmin(abs.(Fmin))
            #         λ̇[pl] -= α[imin]*Δλ̇ 
            #     end
            # else 
            λ̇[pl] += Fc[pl] *100
            if type==:MC
                𝐹τ   = τ -> Yield_MCAS95(τ, Pt[pl], ϕ, Coh[pl], θt, ηvp, 0.0 )
                ∂F∂τ = ForwardDiff.gradient( 𝐹τ, [τ.xx[pl]; τ.yy[pl]; τ.zz[pl]; τ.xy[pl]])
            else
                ∂F∂τ   = [τ.xx[pl]; τ.yy[pl]; τ.zz[pl]; τ.xy[pl]]./τ.II[pl]/2.
            end
            τ.xx[pl] = 2 * ηve[pl] * (0.0      + τ0.xx[pl]/2/ηe[pl] - ∂F∂τ[1]*λ̇[pl])
            τ.yy[pl] = 2 * ηve[pl] * (ε̇.yy[pl] + τ0.yy[pl]/2/ηe[pl] - ∂F∂τ[2]*λ̇[pl]) 
            τ.zz[pl] = 2 * ηve[pl] * (0.0      + τ0.zz[pl]/2/ηe[pl] - ∂F∂τ[3]*λ̇[pl])
            τ.xy[pl] = 2 * ηve[pl] * (ε̇.xy[pl] + τ0.xy[pl]/2/ηe[pl] - ∂F∂τ[4]*λ̇[pl]) 
            Pt[pl]   = Pt0[pl] - Kb[pl]*Δt*(∇v.tot[pl] - sin(ψ)*λ̇[pl])
            ηvep[pl] = τ.II[pl] / 2.0 / ε̇.IIᵉᶠᶠ[pl]
        else
            λ̇[pl]       = 0.0
        end
    end

    if type==:MC
        for i in eachindex(F)
            Fc[i] = Yield_MCAS95( [τ.xx[i]; τ.yy[i]; τ.zz[i]; τ.xy[i]], Pt[i], ϕ, Coh[i], θt, ηvp, λ̇[i] )
        end
    else
        @. Fc    = τ.II - Coh*cos(ϕ) - Pt*sin(ϕ) - λ̇*ηvp
    end

end

function main()

    # Unit system
    CharDim    = SI_units(length=1000m, temperature=1000C, stress=1e7Pa, viscosity=1e15Pas)

    # Physical parameters
    #σxxB       = nondimensionalize( -25e3Pa, CharDim) # Courbe A - Vermeer
    #σyyB       = nondimensionalize(-100e3Pa, CharDim) # Courbe A - Vermeer
    σxxB       = nondimensionalize(-400e3Pa, CharDim) # Courbe B - Vermeer
    σyyB       = nondimensionalize(-100e3Pa, CharDim) # Courbe B - Vermeer
    σzzB       = 0*σxxB
    PB         = -(σxxB + σyyB + σzzB)/3.0
    τxxB       = PB + σxxB
    τyyB       = PB + σyyB
    τzzB       = PB + σzzB
    τxyB       = 0.0

    E          = nondimensionalize(20MPa, CharDim)
    ν          = 0.0
    Ly         = nondimensionalize(4e4m, CharDim)
    Ẇ0         = nondimensionalize(5e-5Pa/s, CharDim)
    σ          = Ly/40
    ε0         = nondimensionalize(1e-9s^-1, CharDim)
    G          = E/2.0/(1+ν)
    Kbulk      = E/3.0/(1-2ν) 
    μs         = nondimensionalize(1e52Pa*s, CharDim)
    yield      = ( 
        type       = :MC, # :DP or :MC
        Coh0       = nondimensionalize(1e5Pa, CharDim),
        ϕ          = 40.0*π/180.,
        ψ          = 10.0*π/180.,    
        θt         = 25.0*π/180.,
        ηvp        = nondimensionalize(1*1e8Pa*s, CharDim),
    )
    
    # Numerical parameters
    Ncy        = 100
    Nt         = 10000
    Δy         = Ly/Ncy
    yc         = LinRange(-Ly/2-Δy/2, Ly/2+Δy/2, Ncy+2)
    yv         = LinRange(-Ly/2,      Ly/2,      Ncy+1)
    Δt         = nondimensionalize(1e4s, CharDim)
   
    # Allocate arrays
    Pt         =  PB*ones(Ncy+1) 
    Pt0        =  PB*ones(Ncy+1)
    τ  = (xx= τxxB*ones(Ncy+1), xy=τxyB*ones(Ncy+1), yy=τyyB*ones(Ncy+1), zz=τzzB*ones(Ncy+1), II=zeros(Ncy+1))
    τ0 = (xx= τxxB*ones(Ncy+1), xy=τxyB*ones(Ncy+1), yy=τyyB*ones(Ncy+1), zz=τzzB*ones(Ncy+1), II=zeros(Ncy+1))
    Coh        =  yield.Coh0.*ones((Ncy+1));    
    arrays = ( #Kb, ηe, ηve, ηvep, F, Fc, λ̇, λ̇rel, ispl
        Kb         = Kbulk*ones(Ncy+1),
        ηe         =  zeros((Ncy+1)),
        ηve        =  zeros((Ncy+1)), 
        ηvep       =  zeros((Ncy+1)),
        F          =  zeros((Ncy+1)),
        Fc         =  zeros((Ncy+1)),
        λ̇          =  zeros((Ncy+1)),
        λ̇rel       =  zeros((Ncy+1)),
        ispl       =  zeros(Int, (Ncy+1)),
    )
    @. arrays.ηe   = G*Δt
    arrays.Kb[50]  = 2 *Kbulk #.- Coh0 .* exp.(-yv.^2/2σ^2)
    @. arrays.ηve  = 1.0 /(1.0/μs + 1.0/arrays.ηe)
    @. arrays.ηvep = arrays.ηve
    ε̇ = ( xy=ε0*ones(Ncy+1), yy=zeros(Ncy+1), IIᵉᶠᶠ=zeros(Ncy+1), xy_pl=zeros(Ncy+1), yy_pl=zeros(Ncy+1), 
    xy_el=zeros(Ncy+1), yy_el=zeros(Ncy+1), xy_net=zeros(Ncy+1), yy_net=zeros(Ncy+1) )
    ∇v = ( tot=zeros(Ncy+1), el=zeros(Ncy+1), pl=zeros(Ncy+1), net=zeros(Ncy+1) )
    η        =    zeros(Ncy+1)
    V     = (x=zeros(Ncy+2), y=zeros(Ncy+2))
    Vit   = (x=zeros(Ncy+2), y=zeros(Ncy+2))
    R     = (x=zeros(Ncy+2), y=zeros(Ncy+2))
    ∂Vx∂τ =    zeros(Ncy+2)
    ∂Vy∂τ =    zeros(Ncy+2)
    σ3    = (x=zeros(size(τ.xx)), z=zeros(size(τ.xx)) )

    # Monitoring
    probes    = (ẆB =zeros(Nt), τxyB=zeros(Nt),  σyyB=zeros(Nt), theta=zeros(Nt), VxB=zeros(Nt), τII=zeros(Nt),)
    η        .= μs
    η_maxloc  = copy( arrays.ηve )
    η_minloc  = copy( arrays.ηve )
   
    # BC
    # VyTypeN = :Dirichlet
    BC = (VyTypeN = :Neumann, VxS=ε0*yv[1], VxN=ε0*yv[end], VyS=1.0, VyN=0.0, σyyN=σyyB)

    # PT solver
    niter = 100000
    nout  = 100
    ϵ     = 1e-9
    rel   = 1e-2
    errPt, errVx, errVy = 0., 0., 0.
    KδV   = (x = zeros(Ncy+2), y = zeros(Ncy+2))
    KδVit = (x = zeros(Ncy+2), y = zeros(Ncy+2))
    λmax  = (x = zeros(Ncy+2), y = zeros(Ncy+2))
    λmax  = (x = zeros(Ncy+2), y = zeros(Ncy+2))
    h     = (x = zeros(Ncy+2), y = zeros(Ncy+2))
    D     = (x =  ones(Ncy+2), y =  ones(Ncy+2))
    iters = 0

    for it=1:Nt
        # History
        @. τ0.xy = τ.xy
        @. τ0.xx = τ.xx
        @. τ0.yy = τ.yy
        @. τ0.zz = τ.zz
        @. Pt0   = Pt

        # DYREL adapt
        CFL    = 0.99
        cfact  = 0.95
        λmin   = 1.0
        c      = 2*sqrt(λmin)
        @. η_minloc[2:end-1]  = min.(arrays.ηve[1:end-2], min(arrays.ηve[2:end-1], arrays.ηve[3:end])) 
        @. η_maxloc[2:end-1]  = max.(arrays.ηve[1:end-2], max(arrays.ηve[2:end-1], arrays.ηve[3:end]))  
        DiagPCMomentumX!(D.x, η_maxloc, arrays, Δt, Δy)
        DiagPCMomentumY!(D.y, η_maxloc, arrays, Δt, Δy)
        GershgorinMomentumX!(λmax.x, D, η_maxloc, arrays, Δt, Δy)
        GershgorinMomentumY!(λmax.y, D, η_maxloc, arrays, Δt, Δy)
        h.x[2:end-1] .= 2.0./sqrt.(λmax.x[2:end-1]).*CFL
        h.y[2:end-1] .= 2.0./sqrt.(λmax.y[2:end-1]).*CFL

        for iter=1:niter

            iters += 1
            Vit.x .= V.x
            Vit.y .= V.y

            Residuals!(R, D, τ, τ0, Pt, Pt0, ε̇, ∇v, V, BC, arrays, yield, rel, true, Δt, Δy)
        
            # Check
            @. ∇v.tot   = (V.y[2:end] - V.y[1:end-1])/Δy
            @. ε̇.yy_el  =  (τ.yy - τ0.yy)/2/arrays.ηe
            @. ε̇.yy_pl  =   τ.yy/τ.II/2*arrays.λ̇rel
            @. ε̇.xy_el  =  (τ.xy - τ0.xy)/(2*arrays.ηe)
            @. ε̇.xy_pl  =   τ.xy/τ.II/2*arrays.λ̇rel
            @. ∇v.el    =  -(Pt - Pt0)/arrays.Kb/Δt
            @. ∇v.pl    =  arrays.λ̇rel*sin(yield.ψ)
            @. ε̇.xy_net = ε̇.xy - ε̇.xy_el - ε̇.xy_pl
            @. ε̇.yy_net = ε̇.yy - ε̇.yy_el - ε̇.yy_pl
            @. ∇v.net   = ∇v.tot  - ∇v.el  - ∇v.pl
            
            # DYREL
            @. ∂Vx∂τ         = (2-c*h.x)/(2+c*h.x)*∂Vx∂τ + 2*h.x/(2+c*h.x)*R.x
            @. V.x[2:end-1] += h.x[2:end-1]*∂Vx∂τ[2:end-1]
            
            @. ∂Vy∂τ         = (2-c*h.y)/(2+c*h.y)*∂Vy∂τ + 2*h.y/(2+c*h.y)*R.y
            @. V.y[2:end-1] += h.y[2:end-1]*∂Vy∂τ[2:end-1]

            if mod(iter, nout) == 0 || iter==1
                @show norm(η_maxloc-arrays.ηvep)

                @. η_minloc[2:end-1]  = min.(arrays.ηve[1:end-2], min(arrays.ηve[2:end-1], arrays.ηve[3:end])) 
                @. η_maxloc[2:end-1]  = max.(arrays.ηve[1:end-2], max(arrays.ηve[2:end-1], arrays.ηve[3:end]))  
                @show norm(η_minloc-arrays.ηvep)

                Residuals!(KδV,   D, τ, τ0, Pt, Pt0, ε̇, ∇v, V,   BC, arrays, yield, rel, true, Δt, Δy)
                Residuals!(KδVit, D, τ, τ0, Pt, Pt0, ε̇, ∇v, Vit, BC, arrays, yield, rel, true, Δt, Δy)

                λmin = abs(sum((V.x .- Vit.x).*(KδV.x .- KδVit.x)) + sum((V.y .- Vit.y).*(KδV.y .- KδVit.y)) / (sum((V.x .- Vit.x).^2) + sum((V.y .- Vit.y).^2)) )
                c    = 2.0*sqrt(λmin)*cfact

                DiagPCMomentumX!(D.x, η_maxloc, arrays, Δt, Δy)
                DiagPCMomentumY!(D.y, η_maxloc, arrays, Δt, Δy)
                GershgorinMomentumX!(λmax.x, D, η_maxloc, arrays, Δt, Δy)
                GershgorinMomentumY!(λmax.y, D, η_maxloc, arrays, Δt, Δy)
                h.x[2:end-1] .= 2.0./sqrt.(λmax.x[2:end-1]).*CFL
                h.y[2:end-1] .= 2.0./sqrt.(λmax.y[2:end-1]).*CFL

                Residuals!(R, D, τ, τ0, Pt, Pt0, ε̇, ∇v, V, BC, arrays, yield, rel, true, Δt, Δy)
                errVx = norm(R.x.*D.x)/sqrt(length(R.x))
                errVy = norm(R.y.*D.y)/sqrt(length(R.y))
                @printf("Iteration %05d --- Time step %4d --- Δt = %2.2e --- ΔtC = %2.2e --- εxy = %2.2e --- max(F) = %2.2e --- max(Fc) = %2.2e \n", iter, it, ustrip(dimensionalize(Δt, s, CharDim)), ustrip(dimensionalize(Δy/2/maximum(V.x), s, CharDim)), ε0*it*Δt, maximum(arrays.F), maximum(arrays.Fc))
                @printf("Exy_net = %2.2e --- Eyy_net = %2.2e --- Div net = %2.2e\n", mean(abs.(ε̇.xy_net)), mean(abs.(ε̇.yy_net)), mean(abs.(∇v.net)) )
                @printf("Exy_el  = %2.2e --- Exy_pl  = %2.2e --- Exy net = %2.2e\n", mean(abs.(ε̇.xy_el)), mean(abs.(ε̇.xy_pl)), mean(abs.(ε̇.xy_net)) )
                @printf("fVx = %2.4e\n", errVx)
                @printf("fVy = %2.4e\n", errVy)
                ( errVx < ϵ && errVy < ϵ) && break 
                ( isnan(errVx) || isnan(errVx)) && error("NaNs")        
            end
        end

        # if (errPt > ϵ || errVx > ϵ || errVx > ϵ) error("non converged") end

        probes.ẆB[it]         = τ.xy[end]*ε̇.xy[end]
        probes.τxyB[it]       = τ.xy[end]
        probes.VxB[it]        = 0.5*(V.x[end] + V.x[end-1])
        probes.σyyB[it] = τ.yy[end] - Pt[end]

        PrincipalStress!(σ3, τ, Pt)
        theta = ustrip.(atand.(σ3.z[:] ./ σ3.x[:]))
        probes.theta[it] = theta[50];         
        # Visualisation
        if mod(it, 10)==0 || it==1 

            p1=plot( title = "Total pressure", xlabel = L"$P$ [kPa]", ylabel = L"$y$ [km]" )
            p1=plot!(ustrip.(dimensionalize(Pt, Pa, CharDim))/1e3, ustrip.(dimensionalize(yv, m, CharDim)./1e3) )
            p1=plot!(ustrip.(dimensionalize(Pt[arrays.ispl.==1], Pa, CharDim))/1e3, ustrip.(dimensionalize(yv[arrays.ispl.==1], m, CharDim)./1e3), linewidth=5 )

            p2=plot(title = "Velocity", xlabel = L"$Vx$ [cm/y]", ylabel = L"$y$ [km]" )
            p2=plot!(ustrip.(dimensionalize(V.x, m/s, CharDim))*cmy, ustrip.(dimensionalize(yc, m, CharDim)./1e3), label="Vx" )
            p2=plot!(ustrip.(dimensionalize(V.y, m/s, CharDim))*cmy, ustrip.(dimensionalize(yc, m, CharDim)./1e3), label="Vy" )
            
            p3=plot(title = "σ3 angle", xlabel = "angle", ylabel = L"$y$ [km]" )
            p3=plot!(ustrip.(theta), ustrip.(dimensionalize(yv, m, CharDim)./1e3), xlimits=(0,90) )
            p4=plot(title = "Probes", xlabel = "Strain", ylabel = L"[-]" )
            # p4=plot!(1:it, ustrip.(dimensionalize(probes.τxy0[1:it], Pa, CharDim))/1e3, label="τxy" )
            app_fric      =  ustrip.(-probes.τxyB[1:it]./probes.σyyB[1:it])
            app_fric_theo =  sind.(2*probes.theta[1:it]) .* sin(yield.ϕ) ./ (1 .+ cosd.(2*probes.theta[1:it]) .* sin(yield.ϕ))
            p4=plot!((1:it)*ε0*Δt*2, app_fric, label="-τxy/σyyBC", title=@sprintf("max = %1.4f", maximum(app_fric)) )
            p4=plot!((1:it)*ε0*Δt*2, app_fric_theo, label="theoritical", title=@sprintf("max = %1.4f", maximum(app_fric_theo)) )
            # p4=plot!((1:it)*ε0*Δt, ustrip.(dimensionalize(-probes.σyy0[1:it], Pa, CharDim))/1e3, label="σyyBC" )
            display(plot(p1,p2,p3,p4))
        end
    end
    @show iters
end

@time main()