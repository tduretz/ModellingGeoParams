using GeoParams, Plots, Printf, MathTeXEngine, BenchmarkTools, Statistics
import LinearAlgebra:norm
using IncompleteLU, ILUZero, SparseArrays, LinearAlgebra

# The yield function can be solved both diriectly iteratively
# Single weak inclusion

include("AssembleKuu.jl")

const cmy = 356.25*3600*24*100
const ky  = 356.25*3600*24*1e3

@views function Vert2Cent!(y, x)
    y .= 0.25.*(x[1:end-1,1:end-1] .+ x[2:end-0,1:end-1] .+ x[1:end-1,2:end-0] .+ x[2:end-0,2:end-0])
end

@views function Cent2Vert!(y, x)
    # @show (size(y), size(x))
    # @show (size(y[2:end-1,2:end-1]), size(x))
    Vert2Cent!(y[2:end-1,2:end-1], x)
    y[1,  2:end-1] .= 0.5*(x[1,  2:end] .+ x[1,  1:end-1])
    y[end,2:end-1] .= 0.5*(x[end,2:end] .+ x[end,1:end-1])
    y[2:end-1,1  ] .= 0.5*(x[2:end,1  ] .+ x[1:end-1, 1  ])
    y[2:end-1,end] .= 0.5*(x[2:end,end] .+ x[1:end-1, end])
end

function DiagMechanics2Dx!( Dx, ηc, kc, ηv, Δx, Δy, b, Ncx, Ncy )
    ηW = zeros(Ncx+1, Ncy); ηW[2:end-0,:] .= ηc; ηW[1,:]   = ηW[2,:]
    ηE = zeros(Ncx+1, Ncy); ηE[1:end-1,:] .= ηc; ηE[end,:] = ηE[end-1,:]
    kW = zeros(Ncx+1, Ncy); kW[2:end-0,:] .= kc; ηW[1,:]   = kW[2,:]
    kE = zeros(Ncx+1, Ncy); kE[1:end-1,:] .= kc; ηE[end,:] = kE[end-1,:]
    ηS = zeros(Ncx+1, Ncy); ηS            .= ηv[:,1:end-1]
    ηN = zeros(Ncx+1, Ncy); ηN            .= ηv[:,2:end-0]
    @. Dx[:,2:end-1] = 4//3*ηE/Δx^2 + 4//3*ηW/Δx^2 + ηS/Δy^2 + ηN/Δy^2 + kW/Δx^2 + kE/Δx^2
end

function DiagMechanics2Dy!( Dy, ηc, kc, ηv, Δx, Δy, b, Ncx, Ncy )
    ηS = zeros(Ncx, Ncy+1); ηS[:,2:end-0] .= ηc
    ηN = zeros(Ncx, Ncy+1); ηN[:,1:end-1] .= ηc
    kS = zeros(Ncx, Ncy+1); kS[:,2:end-0] .= kc
    kN = zeros(Ncx, Ncy+1); kN[:,1:end-1] .= kc
    ηW = zeros(Ncx, Ncy+1); ηW            .= ηv[1:end-1,:]
    ηE = zeros(Ncx, Ncy+1); ηE            .= ηv[2:end-0,:]
    @. Dy[2:end-1,:] = 4//3*ηS/Δy^2 + 4//3*ηN/Δy^2 + ηW/Δx^2 + ηE/Δx^2 + kS/Δy^2 + kN/Δy^2
end

function GershgorinMechanics2Dx_Local!( λmaxloc, ηc, kc, ηv, Dx, Δx, Δy, Ncx, Ncy )
    ηW = zeros(Ncx+1, Ncy); ηW[2:end-0,:] .= ηc; ηW[1,:]   = ηW[2,:]
    ηE = zeros(Ncx+1, Ncy); ηE[1:end-1,:] .= ηc; ηE[end,:] = ηE[end-1,:]
    kW = zeros(Ncx+1, Ncy); kW[2:end-0,:] .= kc; kW[1,:]   = kW[2,:]
    kE = zeros(Ncx+1, Ncy); kE[1:end-1,:] .= kc; kE[end,:] = kE[end-1,:]
    ηS = zeros(Ncx+1, Ncy); ηS            .= ηv[:,1:end-1]
    ηN = zeros(Ncx+1, Ncy); ηN            .= ηv[:,2:end-0]
    Cx = zeros(Ncx+1, Ncy);
    Cy = zeros(Ncx+1, Ncy);
    # @. Cx = 2*4//3*ηE/Δx^2 + 2*4//3*ηW/Δx^2 + 2*ηS/Δy^2 + 2*ηN/Δy^2 + 2*kW/Δx^2 + 2*kE/Δx^2
    # @. Cy = abs.(-2//3*ηE+ ηN + kE)/Δx/Δy  + abs.(-2//3*ηE + ηS + kE)/Δx/Δy + abs.(ηN - 2//3*ηW + kW)/Δy/Δx + abs.(ηS- 2//3*ηW + kW)/Δy/Δx  
    dx, dy = Δx, Δy
    @. Cx = (4 // 3) * abs(ηE ./ dx .^ 2) + (4 // 3) * abs(ηW ./ dx .^ 2) + abs(ηN ./ dy .^ 2) + abs(ηS ./ dy .^ 2) + abs(-(-ηN ./ dy - ηS ./ dy) ./ dy + ((4 // 3) * ηE ./ dx + (4 // 3) * ηW ./ dx) ./ dx)
    @. Cy = abs((2 // 3) * ηE ./ (dx .* dy) - ηN ./ (dx .* dy)) + abs((2 // 3) * ηE ./ (dx .* dy) - ηS ./ (dx .* dy)) + abs(ηN ./ (dx .* dy) - 2 // 3 * ηW ./ (dx .* dy)) + abs(ηS ./ (dx .* dy) - 2 // 3 * ηW ./ (dx .* dy))
    @.  λmaxloc = (Cx + Cy + 2/dx)/Dx[:,2:end-1]
end

function GershgorinMechanics2Dy_Local!( λmaxloc, ηc, kc, ηv, Dy, Δx, Δy, Ncx, Ncy )
    ηS = zeros(Ncx, Ncy+1); ηS[:,2:end-0] .= ηc
    ηN = zeros(Ncx, Ncy+1); ηN[:,1:end-1] .= ηc
    kS = zeros(Ncx, Ncy+1); kS[:,2:end-0] .= kc
    kN = zeros(Ncx, Ncy+1); kN[:,1:end-1] .= kc
    ηW = zeros(Ncx, Ncy+1); ηW            .= ηv[1:end-1,:]
    ηE = zeros(Ncx, Ncy+1); ηE            .= ηv[2:end-0,:]
    Cy = zeros(Ncx, Ncy+1);
    Cx = zeros(Ncx, Ncy+1);
    # @. Cy = 2*4//3*ηS/Δy^2 + 2*4//3*ηN/Δy^2 + 2*ηW/Δx^2 + 2*ηE/Δx^2 + 2*kS/Δy^2 + 2*kN/Δy^2
    # @. Cx = abs.(-2//3*ηN+ ηE + kN)/Δx/Δy  + abs.(-2//3*ηN + ηW + kN)/Δx/Δy + abs.(ηE - 2//3*ηS + kS)/Δy/Δx + abs.(ηW - 2//3*ηS + kS)/Δy/Δx  
    dx, dy = Δx, Δy
    @. Cy = abs(ηE ./ dx .^ 2) + abs(ηW ./ dx .^ 2) + (4 // 3) * abs(ηN ./ dy .^ 2) + (4 // 3) * abs(ηS ./ dy .^ 2) + abs(((4 // 3) * ηN ./ dy + (4 // 3) * ηS ./ dy) ./ dy - (-ηE ./ dx - ηW ./ dx) ./ dx)
    @. Cx = abs(ηE ./ (dx .* dy) - 2 // 3 * ηN ./ (dx .* dy)) + abs(ηE ./ (dx .* dy) - 2 // 3 * ηS ./ (dx .* dy)) + abs((2 // 3) * ηN ./ (dx .* dy) - ηW ./ (dx .* dy)) + abs((2 // 3) * ηS ./ (dx .* dy) - ηW ./ (dx .* dy))
    @. λmaxloc .= (Cx + Cy + 2/dy)./Dy[2:end-1,:]
end

@views function ApplyFreeSlip!(Vx, Vy)
    @. Vx[:,  1] = Vx[:,    2]
    @. Vx[:,end] = Vx[:,end-1]
    @. Vy[  1,:] = Vy[    2,:]
    @. Vy[end,:] = Vy[end-1,:]
end

@views function Kinematics!(ε̇, Vx, Vy, Δx, Δy)
    @. ε̇.∇V  = (Vx[2:end-0,2:end-1] - Vx[1:end-1,2:end-1])/Δx + (Vy[2:end-1,2:end-0] - Vy[2:end-1,1:end-1])/Δy
    @. ε̇.xx  = (Vx[2:end-0,2:end-1] - Vx[1:end-1,2:end-1])/Δx - 1//3*ε̇.∇V
    @. ε̇.yy  = (Vy[2:end-1,2:end-0] - Vy[2:end-1,1:end-1])/Δy - 1//3*ε̇.∇V
    @. ε̇.xy  = 1//2*((Vx[:,2:end-0] - Vx[:,1:end-1])/Δy + (Vy[2:end,:] - Vy[1:end-1,:])/Δx )
    Cent2Vert!(ε̇.xxv, ε̇.xx)
    Cent2Vert!(ε̇.yyv, ε̇.yy)
    Cent2Vert!(ε̇.∇Vv, ε̇.∇V)
    Vert2Cent!(ε̇.xyc, ε̇.xy)
end

@views function VonMises(p, τII, J3, c, ϕ, θt, ηvp, λ̇ )
    F = τII - c
    return F
end

@views function DruckerPrager(p, τII, J3, c, ϕ, θt, ηvp, λ̇ )
    F = τII - c*cos(ϕ) - p*sin(ϕ) - ηvp*λ̇
    return F
end

@views function MohrCoulombAbbo(p, τII, J3, c, ϕ, θt, ηvp, λ̇ )
    L  = -3.0*sqrt(3.0)/2.0*J3/τII^3
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
    F = k*τII - c*cos(ϕ) - p*sin(ϕ) - ηvp*λ̇
    return F
end

@views function YieldFunction( yield, p, τII, J3, c, ϕ, θt, ηvp, λ̇ )
    if yield == :vonMises
        F = τII - c
    elseif yield == :DruckerPrager
        F = τII - c*cos(ϕ) - p*sin(ϕ) - ηvp*λ̇
    elseif yield == :MohrCoulombAbbo
        L  = -3.0*sqrt(3.0)/2.0*J3/τII^3
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
        F = k*τII - c*cos(ϕ) - p*sin(ϕ) - ηvp*λ̇
    end
    return F
end

@views function VEP(ϵ̇xx, ϵ̇yy, ϵ̇xy, ϵ̇zz, ∇V, params, ηv, ηe, ξ, λ̇it, Fc)
    c     = params.pl.c
    ϕ     = params.pl.ϕ
    ψ     = params.pl.ψ
    θt    = params.pl.θt
    ηvp   = params.pl.ηvp
    yield = params.pl.yield

    rel           = params.pl.rel
    ηeff          = (1.0/ηv + 1.0/ηe)^-1
    τxx           = 2*ηeff*ϵ̇xx
    τyy           = 2*ηeff*ϵ̇yy
    τxy           = 2*ηeff*ϵ̇xy
    τzz           = -(τxx + τyy)
    p             = -ξ*∇V
    τII           = sqrt( 0.5*(τxx.^2 + τyy^2 + τzz^2) + τxy^2 )
    J3            = τxx*τyy*τzz + τxy^2*τzz
    F             = YieldFunction( yield, p, τII, J3, c, ϕ, θt, ηvp, 0.0 ) 
    if F>0
        if params.pl.Fiter == false
            λ̇             = F/(ηeff + ηvp) # direct analytics
            # λ̇it           = λ̇it0 + Fc/ηeff # iteration (much worse)
            λ̇it           = λ̇*rel + (1.0-rel)*λ̇it
        end        
        τxx           = 2*ηeff*(ϵ̇xx - λ̇it*τxx/2/τII)
        τyy           = 2*ηeff*(ϵ̇yy - λ̇it*τyy/2/τII)
        τxy           = 2*ηeff*(ϵ̇xy - λ̇it*τxy/2/τII)
        τzz           = -(τxx + τyy)
        τII           = sqrt( 0.5*(τxx.^2 + τyy^2 + τzz^2) + τxy^2 )
        J3            = τxx*τyy*τzz + τxy^2*τzz
        F             = YieldFunction( yield, p, τII, J3, c, ϕ, θt, ηvp, λ̇it ) 
        ϵ̇II           = sqrt( 0.5*(ϵ̇xx.^2 + ϵ̇yy^2 + ϵ̇zz^2) + ϵ̇xy^2 )
        ηeff          = τII / 2.0 / ϵ̇II
    else
        F   = 0.
        λ̇it = 0.
    end
    return ηeff, τxx, τyy, τxy, p, F, λ̇it
end

@views function Stress!(τ, τ0, Pt, ε̇, rheo, params, Δt)
    for i in eachindex(τ.xx)
        ηe            = rheo.Gc[i]*Δt
        ηv            = rheo.ηc[i]
        ξ             = rheo.ξc[i]
        λ̇it           = rheo.λ̇itc[i]
        F             = rheo.Fc[i]
        ηeff          = (1.0/rheo.ηc[i] + 1.0/ηe)^-1
        ϵ̇xx           = ε̇.xx[i]  + τ0.xx[i] /(2.0*ηe)
        ϵ̇yy           = ε̇.yy[i]  + τ0.yy[i] /(2.0*ηe)
        ϵ̇xy           = ε̇.xyc[i] + τ0.xyc[i]/(2.0*ηe)
        ϵ̇zz           = -(ϵ̇xx + ϵ̇yy)
        ∇V            = ε̇.∇V[i]
        ηeff, τxx, τyy, τxy, p, F, λ̇it = VEP(ϵ̇xx, ϵ̇yy, ϵ̇xy, ϵ̇zz, ∇V, params, ηv, ηe, ξ, λ̇it, F)
        #------------ update
        rheo.Fc[i]   = F
        rheo.λ̇c[i]    = λ̇it
        τ.xx[i]       = τxx
        τ.yy[i]       = τyy

        # τ.xx[i]       = 2*rheo.Gc[i]*Δt*ε̇.xx[i]
        # τ.yy[i]       = 2*rheo.Gc[i]*Δt*ε̇.yy[i]
        
        # Pt[i]         = p 
        rheo.ηeffc[i] = ηeff
    end
    for i in eachindex(τ.xy)
        ηe            = rheo.Gv[i]*Δt
        ηv            = rheo.ηv[i]
        ξ             = rheo.ξv[i] 
        λ̇it           = rheo.λ̇itv[i]
        F             = rheo.Fv[i]
        ϵ̇xx           = ε̇.xxv[i] + τ0.xxv[i]/(2.0*ηe)
        ϵ̇yy           = ε̇.yyv[i] + τ0.yyv[i]/(2.0*ηe)
        ϵ̇xy           = ε̇.xy[i]  + τ0.xy[i] /(2.0*ηe)
        ϵ̇zz           = -(ϵ̇xx + ϵ̇yy)
        ∇Vv           = ε̇.∇Vv[i]
        ηeff, τxx, τyy, τxy, p, F, λ̇it = VEP(ϵ̇xx, ϵ̇yy, ϵ̇xy, ϵ̇zz, ∇Vv, params, ηv, ηe, ξ, λ̇it, F)
        #------------ update
        rheo.Fv[i]    = F
        rheo.λ̇v[i]    = λ̇it
        τ.xy[i]       = τxy
        rheo.ηeffv[i] = ηeff
    end
end

@views function ResidualMomentum!(Rx, Ry, Vx, Vy, Pt, bx, by, ε̇, rheo, params, Dx, Dy, τ, τ0, Δx, Δy, Δt, rhs)
    ApplyFreeSlip!(Vx, Vy)
    Kinematics!(ε̇, Vx, Vy, Δx, Δy)
    Stress!(τ, τ0, Pt, ε̇, rheo, params, Δt)
    # @show maximum( abs.(τ.xx[2:end,:] - τ.xx[1:end-1,:])/Δx)
    # @show maximum( abs.(τ.yy[:,2:end] - τ.yy[:,1:end-1])/Δy)
    # η = rheo.ηeffc
    @. Rx[2:end-1,2:end-1] = ( (τ.xx[2:end,:] - τ.xx[1:end-1,:])/Δx + (τ.xy[2:end-1,2:end] - τ.xy[2:end-1,1:end-1])/Δy - (Pt[2:end,:] - Pt[1:end-1,:])/Δx  ) + bx[2:end-1,2:end-1]*rhs
    @. Rx                ./= Dx
    @. Ry[2:end-1,2:end-1] = ( (τ.yy[:,2:end] - τ.yy[:,1:end-1])/Δy + (τ.xy[2:end,2:end-1] - τ.xy[1:end-1,2:end-1])/Δx - (Pt[:,2:end] - Pt[:,1:end-1])/Δy  ) + by[2:end-1,2:end-1]*rhs
    @. Ry                ./= Dy
end

@views function ResidualContinuity!(Rp, Vx, Vy, Pt, bp, ε̇, rheo, params, Dp, τ, τ0, Δx, Δy, Δt, rhs, comp)
    Rp .= 0.
    ApplyFreeSlip!(Vx, Vy)
    Kinematics!(ε̇, Vx, Vy, Δx, Δy)
    @. Rp = -ε̇.∇V - comp*Pt/rheo.ηeffc
end

@views function maxloc!(A2, A)
    A2[2:end-1,2:end-1] .= max.(max.(max.(A[1:end-2,2:end-1], A[3:end,2:end-1]), A[2:end-1,2:end-1]), max.(A[2:end-1,1:end-2], A[2:end-1,3:end]))
    A2[[1,end],:] .= A2[[2,end-1],:]; A2[:,[1,end]] .= A2[:,[2,end-1]]
end

@views function minloc!(A2, A)
    A2[2:end-1,2:end-1] .= min.(min.(min.(A[1:end-2,2:end-1], A[3:end,2:end-1]), A[2:end-1,2:end-1]), min.(A[2:end-1,1:end-2], A[2:end-1,3:end]))
    A2[[1,end],:] .= A2[[2,end-1],:]; A2[:,[1,end]] .= A2[:,[2,end-1]]
end

@views function MainMechanicsDiagPC2D(n)

    # Unit system
    CharDim    = SI_units(length=1m, temperature=1C, stress=1Pa, viscosity=1Pas)

    # Physical parameters
    Lx         = nondimensionalize(1/1m, CharDim)
    Ly         = nondimensionalize(1/1m, CharDim)
    η_mat      = nondimensionalize(1Pas, CharDim)
    G_mat      = nondimensionalize(1Pa,  CharDim)
    plastic    = (
        yield  = :MohrCoulombAbbo,
        ϕ      = 30.0*π/180,
        ψ      = 0.0,
        θt     = 25.0*π/180,
        c      = nondimensionalize(20e10Pa, CharDim),
        ηvp    = nondimensionalize(0.05Pas, CharDim),
        rel    = 1.0,
        Fiter  = true
    )
    params     = (pl=plastic, el=(G=G_mat))

    # BCs
    ε̇bg        = nondimensionalize(1/s, CharDim)

    # Numerical parameters
    Ncx        = n*63
    Ncy        = n*63
    Nt         = 1
    Δx         = Lx/Ncx
    Δy         = Ly/Ncy
    Δt         = nondimensionalize(1/4s, CharDim)
    xc         = LinRange(-0*Lx/2-Δx/2, 2Lx/2+Δx/2, Ncx+2)
    yc         = LinRange(-0*Ly/2-Δy/2, 2Ly/2+Δy/2, Ncy+2)
    xv         = LinRange(-0*Lx/2, 2Lx/2, Ncx+1)
    yv         = LinRange(-0*Ly/2, 2Ly/2, Ncy+1)

    # Allocate arrays
    Vx         = zeros(Ncx+1, Ncy+2); Vx .= ε̇bg.*xv .+   0*yc'
    Vy         = zeros(Ncx+2, Ncy+1); Vy .=    0*xc .- ε̇bg*yv'
    Pt         = zeros(Ncx+0, Ncy+0)
    ε̇          = (
        II        = zeros(Ncx+0, Ncy+0),
        xx        = zeros(Ncx+0, Ncy+0),
        yy        = zeros(Ncx+0, Ncy+0),
        xyc       = zeros(Ncx+0, Ncy+0),
        xxv       = zeros(Ncx+1, Ncy+1),
        yyv       = zeros(Ncx+1, Ncy+1),
        xy        = zeros(Ncx+1, Ncy+1),
        ∇V        = zeros(Ncx+0, Ncy+0),
        ∇Vv       = zeros(Ncx+1, Ncy+1),
    )
    τ          = (
        xx     = zeros(Ncx+0, Ncy+0),
        yy     = zeros(Ncx+0, Ncy+0),
        xy     = zeros(Ncx+1, Ncy+1),
        II     = zeros(Ncx+0, Ncy+0),
    )
    τ0         = (
        xx     = zeros(Ncx+0, Ncy+0),
        yy     = zeros(Ncx+0, Ncy+0),
        xyc    = zeros(Ncx+0, Ncy+0),
        xxv    = zeros(Ncx+1, Ncy+1),
        yyv    = zeros(Ncx+1, Ncy+1),
        xy     = zeros(Ncx+1, Ncy+1),
    )
    Rx         = zeros(Ncx+1, Ncy+2)
    Ry         = zeros(Ncx+2, Ncy+1)
    bx         = zeros(Ncx+1, Ncy+2)
    by         = zeros(Ncx+2, Ncy+1)
    ∂Vx∂τ      = zeros(Ncx+1, Ncy+2)
    hVx        = zeros(Ncx+1, Ncy+2) # Time step is local
    KδVx1      = zeros(Ncx+1, Ncy+2)
    δVx        = zeros(Ncx+1, Ncy+2)
    KδVx       = zeros(Ncx+1, Ncy+2)
    ∂Vy∂τ      = zeros(Ncx+2, Ncy+1)
    hVy        = zeros(Ncx+2, Ncy+1) # Time step is local
    KδVy1      = zeros(Ncx+2, Ncy+1)
    δVy        = zeros(Ncx+2, Ncy+1)
    KδVy       = zeros(Ncx+2, Ncy+1)

    ∂Pt∂τ       = zeros(Ncx+0, Ncy+0)
    δPt         = zeros(Ncx+0, Ncy+0)
    KδPt        = zeros(Ncx+0, Ncy+0)
    KδPt1       = zeros(Ncx+0, Ncy+0)
    bp          = zeros(Ncx+0, Ncy+0)
    Dp          =  ones(Ncx+0, Ncy+0)
    Rp          = zeros(Ncx+0, Ncy+0)
    λmaxlocPt   = zeros(Ncx+0, Ncy+0)
    cPt         = zeros(Ncx+0, Ncy+0)
    hPt         = zeros(Ncx+0, Ncy+0)

    rheo       = (
        ηeffv      = η_mat.*ones(Ncx+1, Ncy+1),
        ηeffc      = η_mat.*ones(Ncx+0, Ncy+0),
        ηv         = η_mat.*ones(Ncx+1, Ncy+1),
        ηc         = η_mat.*ones(Ncx+0, Ncy+0),
        Gv         = G_mat.*ones(Ncx+1, Ncy+1),
        Gc         = G_mat.*ones(Ncx+0, Ncy+0),
        ξc         = ones(Ncx+0, Ncy+0)*1000,
        ξv         = ones(Ncx+1, Ncy+1)*1000,
        Fc         = zeros(Ncx+0, Ncy+0),
        Fv         = zeros(Ncx+1, Ncy+1),
        λ̇c         = zeros(Ncx+0, Ncy+0),
        λ̇v         = zeros(Ncx+1, Ncy+1),
        λ̇itc       = zeros(Ncx+0, Ncy+0),
        λ̇itv       = zeros(Ncx+1, Ncy+1),
    )
    Dx         = ones(Ncx+1, Ncy+2)
    Dy         = ones(Ncx+2, Ncy+1)
    λmaxlocVx  = zeros(Ncx+1, Ncy+0)
    λmaxlocVy  = zeros(Ncx+0, Ncy+1)
    ηc_maxloc  = zeros(Ncx+0, Ncy+0)
    ηv_maxloc  = zeros(Ncx+1, Ncy+1)
    ηc_minloc  = zeros(Ncx+0, Ncy+0)
    ηv_minloc  = zeros(Ncx+1, Ncy+1)

    # Multiple circles with various viscosities
    ηi    = (s=η_mat*1e0, w=η_mat*1e0) 
    Gi    = (s=G_mat*5e-4, w=G_mat*5e-4) 
    sx = -0.4
    sy = -0.1
    x_inc = [0.5+sx   0.2 -0.3 -0.4  0.0 -0.3 0.4  0.3  0.35 -0.1] 
    y_inc = [0.5+sy   0.4  0.4 -0.3 -0.2  0.2 -0.2 -0.4 0.2  -0.4]
    r_inc = [sqrt(0.01) 0.09 0.05 0.08 0.08  0.1 0.07 0.08 0.07 0.07] 
    η_inc = [ηi.s]# ηi.w ηi.w ηi.s ηi.w ηi.s ηi.w ηi.s ηi.s ηi.w]
    G_inc = [Gi.s Gi.w Gi.w Gi.s Gi.w Gi.s Gi.w Gi.s Gi.s Gi.w]
    for inc in eachindex(η_inc)
        rheo.ηv[(xv.-x_inc[inc]).^2 .+ (yv'.-y_inc[inc]).^2 .< r_inc[inc]^2 ] .= η_inc[inc]
        rheo.Gv[(xv.-x_inc[inc]).^2 .+ (yv'.-y_inc[inc]).^2 .< r_inc[inc]^2 ] .= G_inc[inc]
    end
    Vert2Cent!(rheo.ηc, rheo.ηv)
    Vert2Cent!(rheo.Gc, rheo.Gv)

    ηc2 = η_mat*ones(Ncx+2, Ncy+2)
    for inc in eachindex(η_inc)
        ηc2[(xc.-x_inc[inc]).^2 .+ (yc'.-y_inc[inc]).^2 .< r_inc[inc]^2 ] .= η_inc[inc]
    end
    Vert2Cent!(rheo.ηv, ηc2)
    
    Gc2 = G_mat*ones(Ncx+2, Ncy+2)
    for inc in eachindex(η_inc)
        Gc2[(xc.-x_inc[inc]).^2 .+ (yc'.-y_inc[inc]).^2 .< r_inc[inc]^2 ] .= G_inc[inc]
    end
    Vert2Cent!(rheo.Gv, Gc2)

    # Effective viscosity
    Stress!(τ, τ0, Pt, ε̇, rheo, params, Δt)

    maxloc!(ηc_maxloc, rheo.ηeffc)
    maxloc!(ηv_maxloc, rheo.ηeffv)
    minloc!(ηv_minloc, rheo.ηeffv)
    minloc!(ηc_minloc, rheo.ηeffc)

    # Monitoring
    probes    = (iters = zeros(Nt), t = zeros(Nt), Ẇ0 = zeros(Nt), τxyi = zeros(Nt), Vx0 = zeros(Nt), maxT = zeros(Nt))
    
    # PT solver
    niter    = 1e5
    nout     = 100
    ϵ        = 1e-6
    CFL      = 0.99#0.999
    cfact    = 1.1
    PC       = :diag 
    ismaxloc = true
    λmin     = 1e-6
    itertot  = 0

    if PC == :diag
        if ismaxloc
            DiagMechanics2Dx!( Dx, ηc_maxloc, rheo.ξc, ηv_maxloc, Δx, Δy, bx, Ncx, Ncy )
            DiagMechanics2Dy!( Dy, ηc_maxloc, rheo.ξc, ηv_maxloc, Δx, Δy, by, Ncx, Ncy )
        else
            DiagMechanics2Dx!( Dx, rheo.ηeffc, rheo.ξc, rheo.ηeffv, Δx, Δy, bx, Ncx, Ncy )
            DiagMechanics2Dy!( Dy, rheo.ηeffc, rheo.ξc, rheo.ηeffv, Δx, Δy, by, Ncx, Ncy )
        end
    end
    for it=1:Nt

        # History
        τ0.xx .= τ.xx
        τ0.yy .= τ.yy
        τ0.xy .= τ.xy
        Cent2Vert!(τ0.xxv, τ0.xx)
        Cent2Vert!(τ0.yyv, τ0.yy)
        Vert2Cent!(τ0.xyc, τ0.xy)
   
        # DYREL
        if ismaxloc
            GershgorinMechanics2Dx_Local!( λmaxlocVx, ηc_maxloc, rheo.ξc, ηv_maxloc, Dx, Δx, Δy, Ncx, Ncy )
        else
            GershgorinMechanics2Dx_Local!( λmaxlocVx, rheo.ηeffc, rheo.ξc, rheo.ηeffv, Dx, Δx, Δy, Ncx, Ncy )
            λmaxlocVx .= maximum(λmaxlocVx)
        end
        hVx[:,2:end-1] .= 2.0./sqrt.(λmaxlocVx)*CFL
        c               = 2.0.*sqrt(λmin)

        # λminlocVy = λminlocVy'
        if ismaxloc
            GershgorinMechanics2Dy_Local!( λmaxlocVy, ηc_maxloc, rheo.ξc, ηv_maxloc, Dy, Δx, Δy, Ncx, Ncy )
        else
            GershgorinMechanics2Dy_Local!( λmaxlocVy, rheo.ηeffc, rheo.ξc, rheo.ηeffv, Dy, Δx, Δy, Ncx, Ncy )
            λmaxlocVy .= maximum(λmaxlocVy)
        end
        hVy[2:end-1,:] .= 2.0./sqrt.(λmaxlocVy)*CFL

        cλ̇   = 1.0
        CFLλ̇ = 0.85
        hλ̇c  = 1.0./(rheo.Gc*Δt)*CFLλ̇
        hλ̇v  = 1.0./(rheo.Gv*Δt)*CFLλ̇

        δλ̇c         = zeros(Ncx+0, Ncy+0)
        δλ̇v         = zeros(Ncx+1, Ncy+1)
        ∂λ̇c∂τ       = zeros(Ncx+0, Ncy+0)
        ∂λ̇v∂τ       = zeros(Ncx+1, Ncy+1)

        λmaxlocPt .=  (0.0./rheo.ηeffc .+ 2.0/Δx .+ 2.0/Δy)        
        hPt .= 2.0./sqrt.(λmaxlocPt)*1.0 
        cPt .= sqrt.(λmaxlocPt).*(0.5./rheo.ηeffc)*0.34

        # cPt .= 2.0./hPt

        # hPt .= 2.0./sqrt.(λmaxlocPt)*1.0
        # # cPt .= sqrt.(λmaxlocPt).*(2.0./rheo.ηeffc)
    
        iters = 0
        @time @views for iter=1:niter
            iters     += 1
            rheo.λ̇itc .= rheo.λ̇c
            rheo.λ̇itv .= rheo.λ̇v

            # Residuals
            ResidualMomentum!(Rx, Ry, Vx, Vy, Pt, bx, by, ε̇, rheo, params, Dx, Dy, τ, τ0, Δx, Δy, Δt, 1.0)
            ResidualContinuity!(Rp, Vx,      Vy,      Pt, bp, ε̇, rheo, params, Dp, τ, τ0, Δx, Δy, Δt, 1.0, 0.0)

            @. ∂Vx∂τ                 = (2-c*hVx)/(2+c*hVx)*∂Vx∂τ + 2*hVx/(2+c*hVx).*Rx
            @. δVx                   = hVx*∂Vx∂τ
            @. Vx[2:end-1,2:end-1]  += δVx[2:end-1,2:end-1]

            @. ∂Vy∂τ                 = (2-c*hVy)/(2+c*hVy)*∂Vy∂τ + 2*hVy/(2+c*hVy).*Ry
            @. δVy                   = hVy*∂Vy∂τ
            @. Vy[2:end-1,2:end-1]  += δVy[2:end-1,2:end-1]

            @. ∂Pt∂τ                 = (2-cPt*hPt)/(2+cPt*hPt)*∂Pt∂τ + 2*hPt/(2+cPt*hPt).*Rp
            @. δPt                   = hPt*∂Pt∂τ
            @. Pt                   += δPt

            if params.pl.Fiter
                @. ∂λ̇c∂τ                 = (2-cλ̇*hλ̇c)/(2+cλ̇*hλ̇c)*∂λ̇c∂τ + 2*hλ̇c/(2+cλ̇*hλ̇c).*rheo.Fc
                @. δλ̇c                   = hλ̇c*∂λ̇c∂τ
                @. rheo.λ̇itc            += δλ̇c
    
                @. ∂λ̇v∂τ                 = (2-cλ̇*hλ̇v)/(2+cλ̇*hλ̇v)*∂λ̇v∂τ + 2*hλ̇v/(2+cλ̇*hλ̇v).*rheo.Fv
                @. δλ̇v                   = hλ̇v*∂λ̇v∂τ
                @. rheo.λ̇itv            += δλ̇v
            end
            
            if mod(iter, nout) == 0 || iter==1

                maxloc!(ηc_maxloc, rheo.ηeffc)
                maxloc!(ηv_maxloc, rheo.ηeffv)
                minloc!(ηv_minloc, rheo.ηeffv)
                minloc!(ηc_minloc, rheo.ηeffc)

                if PC == :diag
                    if ismaxloc
                        DiagMechanics2Dx!( Dx, ηc_maxloc, rheo.ξc, ηv_maxloc, Δx, Δy, bx, Ncx, Ncy )
                        DiagMechanics2Dy!( Dy, ηc_maxloc, rheo.ξc, ηv_maxloc, Δx, Δy, by, Ncx, Ncy )
                    else
                        DiagMechanics2Dx!( Dx, rheo.ηeffc, rheo.ξc, rheo.ηeffv, Δx, Δy, bx, Ncx, Ncy )
                        DiagMechanics2Dy!( Dy, rheo.ηeffc, rheo.ξc, rheo.ηeffv, Δx, Δy, by, Ncx, Ncy )
                    end
                end

                ResidualMomentum!(KδVx1, KδVy1, Vx,      Vy,      Pt, bx, by, ε̇, rheo, params, Dx, Dy, τ, τ0, Δx, Δy, Δt, 1.0)
                ResidualMomentum!(KδVx,  KδVy,  Vx.-δVx, Vy.-δVy, Pt, bx, by, ε̇, rheo, params, Dx, Dy, τ, τ0, Δx, Δy, Δt, 1.0)

                # if ismaxloc
                #     GershgorinMechanics2Dx_Local!( λmaxlocVx, ηc_maxloc, rheo.ξc, ηv_maxloc, Dx, Δx, Δy, Ncx, Ncy )
                # else
                #     GershgorinMechanics2Dx_Local!( λmaxlocVx, rheo.ηeffc, rheo.ξc, rheo.ηeffv, Dx, Δx, Δy, Ncx, Ncy )
                #     λmaxlocVx .= maximum(λmaxlocVx)
                # end                
                # hVx[:,2:end-1] .= 2.0./sqrt.(λmaxlocVx)*CFL

                # if ismaxloc
                #     GershgorinMechanics2Dy_Local!( λmaxlocVy, ηc_maxloc, rheo.ξc, ηv_maxloc, Dy, Δx, Δy, Ncx, Ncy )
                # else
                #     GershgorinMechanics2Dy_Local!( λmaxlocVy, rheo.ηeffc, rheo.ξc, rheo.ηeffv, Dy, Δx, Δy, Ncx, Ncy )
                #     λmaxlocVy .= maximum(λmaxlocVy)
                # end
                # hVy[2:end-1,:] .= 2.0./sqrt.(λmaxlocVy)*CFL

                λmin = abs(sum(.-δVx.*(KδVx1.-KδVx)) + sum(.-δVy.*(KδVy1.-KδVy))/(sum(δVx.*δVx) + sum(δVy.*δVy)) )
                c    = 2.0*sqrt(λmin)*cfact*1.0
                @show (λmin)

                # ResidualContinuity!(KδPt1, Vx,      Vy,      Pt,      bp, ε̇, rheo, params, Dp, τ, τ0, Δx, Δy, Δt, 1.0, 0.0)
                # ResidualContinuity!(KδPt,  Vx.-δVx, Vy.-δVy, Pt, bp, ε̇, rheo, params, Dp, τ, τ0, Δx, Δy, Δt, 1.0, 0.0)
                # λminPt = abs(sum(.-δPt.*(KδPt1.-KδPt))/(sum(δPt.*δPt) ) )
                # @show (λminPt)


                ResidualContinuity!(KδPt1, Vx,      Vy,      Pt,      bp, ε̇, rheo, params, Dp, τ, τ0, Δx, Δy, Δt, 1.0, 0.0)
                ResidualContinuity!(KδPt,  Vx, Vy, Pt.-δPt, bp, ε̇, rheo, params, Dp, τ, τ0, Δx, Δy, Δt, 1.0, 1.0)
                λminPt = abs(sum(.-δPt.*(KδPt1.-KδPt))/(sum(δPt.*δPt) ) )
                @show (λminPt)
                @show hPt[1]
                @show cPt[1]
                # @show 2/sqrt(λminPt)
                # cPt = 2.5*λminPt
                # cPt = 2.0*sqrt(λminPt)*cfact
                # @show hPt/cPt

                # # λmaxlocPt .= rheo.ηeffc .* (2.0/Δx + 2.0/Δy)
                # λmaxlocPt .=  (2.0/Δx + 2.0/Δy)
                # hPt .= 2.0./sqrt.(λmaxlocPt)*CFL
                # cPt .= sqrt.(λmaxlocPt)

                ResidualMomentum!(Rx, Ry, Vx, Vy, Pt, bx, by, ε̇, rheo, params, Dx, Dy, τ, τ0, Δx, Δy, Δt, 1.0)
                ResidualContinuity!(Rp, Vx,      Vy,      Pt, bp, ε̇, rheo, params, Dp, τ, τ0, Δx, Δy, Δt, 1.0, 0.0)

                errVx = norm(Rx.*Dx)/(length(Rx))
                errVy = norm(Ry.*Dy)/(length(Ry))
                errPt = norm(Rp.*Dp)/(length(Rp))
                @printf("Iteration %05d --- Time step %4d \n", iter, it )
                @printf("Rx = %2.4e --- Ry = %2.4e --- Rp = %2.4e --- max(Fc) = %lf --- max(Fv) = %lf  \n", errVx, errVy, errPt, maximum(rheo.Fc), maximum(rheo.Fv))
                #, maximum(λmaxlocVx), log10(maximum(λmaxlocVx)) - log10(λmin))
                # @show (λmin, maximum(λmaxlocVy), log10(maximum(λmaxlocVy)) - log10(λmin))
                # @show (minimum(rheo.λ̇itc), maximum(rheo.λ̇itc))
                if ( isnan(errVx) ) error() end
                if ( errVx > 1e6  ) error() end
                ( errVx < ϵ && errPt < ϵ ) && break
            end
        end
        itertot += iters
        probes.iters[it] = iters

        # @show (minimum(rheo.ηc), maximum(rheo.ηc))
        # @show (minimum(rheo.ηeffc), maximum(rheo.ηeffc))
        # @show (minimum(λmaxlocVx), maximum(λmaxlocVx))
        # @show (minimum(λmaxlocVy), maximum(λmaxlocVy))
        # @show (minimum(hVx), maximum(hVx))
        # @show (minimum(hVy), maximum(hVy))

        # Visualisation
        if mod(it, 1)==0 || it==1
            τxyc = zero(τ.xx)
            Vert2Cent!(τxyc, τ.xy)
            @. τ.II = sqrt( 0.5*( τ.xx^2 + τ.yy^2 + (τ.xx + τ.yy)^2 ) + τxyc^2 )
            @. ε̇.II = sqrt( 0.5*( ε̇.xx^2 + ε̇.yy^2 + (ε̇.xx + ε̇.yy)^2 ) + ε̇.xyc^2 )
            # p1=heatmap(ustrip.(dimensionalize(xv, m, CharDim)./1e3), ustrip.(dimensionalize(yc, m, CharDim)./1e3), ustrip.(Rx)', title = "$(mean(probes.iters[1:it])) iterations", aspect_ratio=1.0 )
            # p2=heatmap(ustrip.(dimensionalize(xc, m, CharDim)./1e3), ustrip.(dimensionalize(yv, m, CharDim)./1e3), (ustrip.(Ry))', title = "$(mean(probes.iters[1:it])) iterations", aspect_ratio=1.0 )
            
            # p1=heatmap(ustrip.(dimensionalize(xc[2:end-1], m, CharDim)./1e3), ustrip.(dimensionalize(yc[2:end-1], m, CharDim)./1e3), ustrip.(dimensionalize(Pt, Pa, CharDim))', title = "$(mean(probes.iters[1:it])) iterations", aspect_ratio=1.0, clims=(-3,3) )
            p1=heatmap(ustrip.(dimensionalize(xc[2:end-1], m, CharDim)./1e3), ustrip.(dimensionalize(yc[2:end-1], m, CharDim)./1e3), ustrip.(dimensionalize(τ.II, Pa, CharDim))', title = "$(mean(probes.iters[1:it])) iterations", aspect_ratio=1.0 )
            # p1=heatmap(ustrip.(dimensionalize(xc[2:end-1], m, CharDim)./1e3), ustrip.(dimensionalize(yc[2:end-1], m, CharDim)./1e3), ustrip.(dimensionalize(rheo.Fc, Pa, CharDim))', title = "$(mean(probes.iters[1:it])) iterations", aspect_ratio=1.0 )
            # p1=heatmap(ustrip.(dimensionalize(xc[2:end-1], m, CharDim)./1e3), ustrip.(dimensionalize(yc[2:end-1], m, CharDim)./1e3), log10.(ustrip.(dimensionalize(ε̇.II, inv(s), CharDim))'), title = "$(mean(probes.iters[1:it])) iterations", aspect_ratio=1.0, clims=(log10(abs(ε̇bg)/10), log10(abs(ε̇bg)*10)) )
            p2=heatmap(ustrip.(dimensionalize(xv, m, CharDim)./1e3), ustrip.(dimensionalize(yv, m, CharDim)./1e3), (ustrip.(dimensionalize(rheo.ηeffv, Pas, CharDim)))', title = "$(mean(probes.iters[1:it])) iterations", aspect_ratio=1.0 )
            p3=heatmap(ustrip.(dimensionalize(xv, m, CharDim)./1e3), ustrip.(dimensionalize(yc, m, CharDim)./1e3), ustrip.(dimensionalize(Vx, m*s^-1, CharDim))', title = "$(mean(probes.iters[1:it])) iterations", aspect_ratio=1.0 )
            p4=heatmap(ustrip.(dimensionalize(xc, m, CharDim)./1e3), ustrip.(dimensionalize(yv, m, CharDim)./1e3), ustrip.(dimensionalize(Vy, m*s^-1, CharDim))', title = "$(mean(probes.iters[1:it])) iterations", aspect_ratio=1.0 )
            display(plot(p1,p2,p3,p4))
        end
    end
    @show itertot, itertot/Ncx
end

MainMechanicsDiagPC2D(1)