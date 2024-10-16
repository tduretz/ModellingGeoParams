using GeoParams, Plots, Printf, MathTeXEngine, BenchmarkTools, Statistics
import LinearAlgebra:norm
using IncompleteLU, ILUZero, SparseArrays, LinearAlgebra

include("AssembleKuu.jl")

const cmy = 356.25*3600*24*100
const ky  = 356.25*3600*24*1e3

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
    kW = zeros(Ncx+1, Ncy); kW[2:end-0,:] .= kc; ηW[1,:]   = kW[2,:]
    kE = zeros(Ncx+1, Ncy); kE[1:end-1,:] .= kc; ηE[end,:] = kE[end-1,:]
    ηS = zeros(Ncx+1, Ncy); ηS            .= ηv[:,1:end-1]
    ηN = zeros(Ncx+1, Ncy); ηN            .= ηv[:,2:end-0]
    Cx = zeros(Ncx+1, Ncy);
    Cy = zeros(Ncx+1, Ncy);
    @. Cx = 2*4//3*ηE/Δx^2 + 2*4//3*ηW/Δx^2 + 2*ηS/Δy^2 + 2*ηN/Δy^2 + 2*kW/Δx^2 + 2*kE/Δx^2
    @. Cy = abs.(-2//3*ηE+ ηN + kE)/Δx/Δy  + abs.(-2//3*ηE + ηS + kE)/Δx/Δy + abs.(ηN - 2//3*ηW + kW)/Δy/Δx + abs.(ηS- 2//3*ηW + kW)/Δy/Δx  
    @.  λmaxloc = (Cx + Cy)/Dx[:,2:end-1]
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
    @. Cy = 2*4//3*ηS/Δy^2 + 2*4//3*ηN/Δy^2 + 2*ηW/Δx^2 + 2*ηE/Δx^2 + 2*kS/Δy^2 + 2*kN/Δy^2
    @. Cx = abs.(-2//3*ηN+ ηE + kN)/Δx/Δy  + abs.(-2//3*ηN + ηW + kN)/Δx/Δy + abs.(ηE - 2//3*ηS + kS)/Δy/Δx + abs.(ηW - 2//3*ηS + kS)/Δy/Δx  
    @. λmaxloc .= (Cx + Cy)./Dy[2:end-1,:]
end

@views function ResidualMomentumX!(Rx, Vx, Vy, Pt, bx, ε̇xx, ε̇xy, ∇v, ηc, kc, ηv, Dx, τxx, τxy, Δx, Δy, rhs)
    @. Vx[:,  1]     = Vx[:,    2]
    @. Vx[:,end]     = Vx[:,end-1]
    @. Vy[  1,:]     = Vy[    2,:]
    @. Vy[end,:]     = Vy[end-1,:]
    @. ∇v            = (Vx[2:end,2:end-1] - Vx[1:end-1,2:end-1])/Δx + (Vy[2:end-1,2:end-0] - Vy[2:end-1,1:end-1])/Δy
    @. ε̇xx           = (Vx[2:end-0,2:end-1] - Vx[1:end-1,2:end-1])/Δx - 1//3*∇v
    @. ε̇xy           = 1//2*((Vx[:,2:end] - Vx[:,1:end-1])/Δy + (Vy[2:end,:] - Vy[1:end-1,:])/Δx )
    @. τxx           = 2*ηc*ε̇xx
    @. τxy           = 2*ηv*ε̇xy
    @. Pt            = -kc*∇v
    @. Rx[2:end-1,2:end-1] = ( (τxx[2:end,:] - τxx[1:end-1,:])/Δx + (τxy[2:end-1,2:end] - τxy[2:end-1,1:end-1])/Δy - (Pt[2:end,:] - Pt[1:end-1,:])/Δx  ) + bx[2:end-1,2:end-1]*rhs
    @. Rx                ./= Dx
end

@views function ResidualMomentumY!(Ry, Vx, Vy, Pt, by, ε̇yy, ε̇xy, ∇v, ηc, kc, ηv, Dy, τyy, τxy, Δx, Δy, rhs)
    @. Vx[:,  1]     = Vx[:,    2]
    @. Vx[:,end]     = Vx[:,end-1]
    @. Vy[  1,:]     = Vy[    2,:]
    @. Vy[end,:]     = Vy[end-1,:]
    @. ∇v            = (Vx[2:end,2:end-1] - Vx[1:end-1,2:end-1])/Δx + (Vy[2:end-1,2:end-0] - Vy[2:end-1,1:end-1])/Δy
    @. ε̇yy           = (Vy[2:end-1,2:end-0] - Vy[2:end-1,1:end-1])/Δy - 1//3*∇v
    @. ε̇xy           = 1//2*((Vx[:,2:end] - Vx[:,1:end-1])/Δy + (Vy[2:end,:] - Vy[1:end-1,:])/Δx )
    @. τyy           = 2*ηc*ε̇yy
    @. τxy           = 2*ηv*ε̇xy
    @. Pt            = -kc*∇v
    @. Ry[2:end-1,2:end-1] = ( (τyy[:,2:end] - τyy[:,1:end-1])/Δy + (τxy[2:end,2:end-1] - τxy[1:end-1,2:end-1])/Δx - (Pt[:,2:end] - Pt[:,1:end-1])/Δy  ) + by[2:end-1,2:end-1]*rhs
    @. Ry                ./= Dy
end

function ResidualContinuity!(Rp, Vx, Vy, Pt, ηb, Δx, Δy)
    @. Rp = -( Pt./ηb + (Vx[2:end,2:end-1] - Vx[1:end-1,2:end-1])/Δx + (Vy[2:end-1,2:end-0] - Vy[2:end-1,1:end-1])/Δy )
end

@views function maxloc!(A2, A)
    A2[2:end-1,2:end-1] .= max.(max.(max.(A[1:end-2,2:end-1], A[3:end,2:end-1]), A[2:end-1,2:end-1]), max.(A[2:end-1,1:end-2], A[2:end-1,3:end]))
    A2[[1,end],:] .= A2[[2,end-1],:]; A2[:,[1,end]] .= A2[:,[2,end-1]]
end

@views function minloc!(A2, A)
    A2[2:end-1,2:end-1] .= min.(min.(min.(A[1:end-2,2:end-1], A[3:end,2:end-1]), A[2:end-1,2:end-1]), min.(A[2:end-1,1:end-2], A[2:end-1,3:end]))
    A2[[1,end],:] .= A2[[2,end-1],:]; A2[:,[1,end]] .= A2[:,[2,end-1]]
end

@views function MainMechanicsDiagPC2D()

    # Unit system
    CharDim    = SI_units(length=1m, temperature=1C, stress=1Pa, viscosity=1Pas)

    # Physical parameters
    Lx         = nondimensionalize(1/1m, CharDim)
    Ly         = nondimensionalize(1/1m, CharDim)
    η_mat      = nondimensionalize(1Pas, CharDim)
    η_inc      = nondimensionalize(1e3Pas, CharDim)

    # BCs
    ε̇bg        = -1.0

    # Numerical parameters
    n          = 2
    Ncx        = n*40
    Ncy        = n*40
    Nt         = 1
    Δx         = Lx/Ncx
    Δy         = Ly/Ncy
    xc         = LinRange(-Lx/2-Δx/2, Lx/2+Δx/2, Ncx+2)
    yc         = LinRange(-Ly/2-Δy/2, Ly/2+Δy/2, Ncy+2)
    xv         = LinRange(-Lx/2, Lx/2, Ncx+1)
    yv         = LinRange(-Ly/2, Ly/2, Ncy+1)

    # Allocate arrays
    Vx         = zeros(Ncx+1, Ncy+2); Vx .= ε̇bg.*xv .+   0*yc'
    Vy         = zeros(Ncx+2, Ncy+1); Vy .=    0*xc .- ε̇bg*yv'
    Pt         = zeros(Ncx+0, Ncy+0)
    ε̇xx        = zeros(Ncx+0, Ncy+0)
    ε̇yy        = zeros(Ncx+0, Ncy+0)
    ε̇xy        = zeros(Ncx+1, Ncy+1)
    ∇v         = zeros(Ncx+0, Ncy+0)
    τxx        = zeros(Ncx+0, Ncy+0)
    τyy        = zeros(Ncx+0, Ncy+0)
    τxy        = zeros(Ncx+1, Ncy+1)
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
    ηv         = η_mat.*ones(Ncx+1, Ncy+1)
    ηc         = η_mat.*ones(Ncx+0, Ncy+0)
    ηb         = ones(size(ηc))*1000
    Dx         = ones(Ncx+1, Ncy+2)
    Dy         = ones(Ncx+2, Ncy+1)
    ηc_maxloc  = η_mat.*ones(Ncx+0, Ncy+0)
    ηv_maxloc  = η_mat.*ones(Ncx+1, Ncy+1)

    # Multiple circles with various viscosities
    ηi    = (s=1e4, w=1e-4) 
    x_inc = [0.0   0.2 -0.3 -0.4  0.0 -0.3 0.4  0.3  0.35 -0.1] 
    y_inc = [0.0   0.4  0.4 -0.3 -0.2  0.2 -0.2 -0.4 0.2  -0.4]
    r_inc = [0.08 0.09 0.05 0.08 0.08  0.1 0.07 0.08 0.07 0.07] 
    η_inc = [ηi.s ηi.w ηi.w ηi.s ηi.w ηi.s ηi.w ηi.s ηi.s ηi.w]
    for inc in eachindex(η_inc)
        ηv[(xv.-x_inc[inc]).^2 .+ (yv'.-y_inc[inc]).^2 .< r_inc[inc]^2 ] .= η_inc[inc]
    end
    ηc                  .= 0.25.*(ηv[1:end-1,1:end-1] .+ ηv[2:end-0,1:end-1] .+ ηv[1:end-1,2:end-0] .+ ηv[2:end-0,2:end-0])
    
    ηc2 = η_mat*ones(Ncx+2, Ncy+2)
    for inc in eachindex(η_inc)
        ηc2[(xc.-x_inc[inc]).^2 .+ (yc'.-y_inc[inc]).^2 .< r_inc[inc]^2 ] .= η_inc[inc]
    end
    ηv                  .= 0.25.*(ηc2[1:end-1,1:end-1] .+ ηc2[2:end-0,1:end-1] .+ ηc2[1:end-1,2:end-0] .+ ηc2[2:end-0,2:end-0])

    ηc_maxloc .= ηc
    ηv_maxloc .= ηv

    maxloc!(ηc_maxloc, ηc_maxloc)
    maxloc!(ηv_maxloc, ηv_maxloc)

    ηc_minloc = η_mat.*ones(Ncx+0, Ncy+0)
    ηv_minloc = η_mat.*ones(Ncx+1, Ncy+1)
    minloc!(ηv_minloc, ηv)
    minloc!(ηc_minloc, ηc)

    λmaxlocVx   = zeros(Ncx+1, Ncy+0)
    λmaxlocVy   = zeros(Ncx+0, Ncy+1)

    # Monitoring
    probes    = (iters = zeros(Nt), t = zeros(Nt), Ẇ0 = zeros(Nt), τxyi = zeros(Nt), Vx0 = zeros(Nt), maxT = zeros(Nt))
    
    # PT solver
    niter    = 1e5
    nout     = 200
    ϵ        = 1e-5
    CFL      = 0.999
    cfact    = 0.9
    PC       = :diag 
    ismaxloc = true

    if PC == :diag
        if ismaxloc
            DiagMechanics2Dx!( Dx, ηc_maxloc, ηb, ηv_maxloc, Δx, Δy, bx, Ncx, Ncy )
            DiagMechanics2Dy!( Dy, ηc_maxloc, ηb, ηv_maxloc, Δx, Δy, by, Ncx, Ncy )
        else
            DiagMechanics2Dx!( Dx, ηc, ηb, ηv, Δx, Δy, bx, Ncx, Ncy )
            DiagMechanics2Dy!( Dy, ηc, ηb, ηv, Δx, Δy, by, Ncx, Ncy )
        end
    end
    for it=1:Nt
   
        λmin = 1.0

        # DYREL
        if ismaxloc
            GershgorinMechanics2Dx_Local!( λmaxlocVx, ηc_maxloc, ηb, ηv_maxloc, Dx, Δx, Δy, Ncx, Ncy )
        else
            GershgorinMechanics2Dx_Local!( λmaxlocVx, ηc, ηb, ηv, Dx, Δx, Δy, Ncx, Ncy )
            λmaxlocVx .= maximum(λmaxlocVx)
        end
        hVx[:,2:end-1] .= 2.0./sqrt.(λmaxlocVx)*CFL
        c               = 2.0.*sqrt(λmin)

        # λminlocVy = λminlocVy'
        if ismaxloc
            GershgorinMechanics2Dy_Local!( λmaxlocVy, ηc_maxloc, ηb, ηv_maxloc, Dy, Δx, Δy, Ncx, Ncy )
        else
            GershgorinMechanics2Dy_Local!( λmaxlocVy, ηc, ηb, ηv, Dy, Δx, Δy, Ncx, Ncy )
            λmaxlocVy .= maximum(λmaxlocVy)
        end
        hVy[2:end-1,:] .= 2.0./sqrt.(λmaxlocVy)*CFL
    
        iters = 0
        @time @views for iter=1:niter
            iters  += 1

            # Residuals
            ResidualMomentumX!(Rx, Vx, Vy, Pt, bx, ε̇xx, ε̇xy, ∇v, ηc, ηb, ηv, Dx, τxx, τxy, Δx, Δy, 1.0)
            ResidualMomentumY!(Ry, Vx, Vy, Pt, by, ε̇yy, ε̇xy, ∇v, ηc, ηb, ηv, Dy, τyy, τxy, Δx, Δy, 1.0)
            
            @. ∂Vx∂τ                 = (2-c*hVx)/(2+c*hVx)*∂Vx∂τ + 2*hVx/(2+c*hVx).*Rx
            @. δVx                   = hVx*∂Vx∂τ
            @. Vx[2:end-1,2:end-1]  += δVx[2:end-1,2:end-1]

            @. ∂Vy∂τ                 = (2-c*hVy)/(2+c*hVy)*∂Vy∂τ + 2*hVy/(2+c*hVy).*Ry
            @. δVy                   = hVy*∂Vy∂τ
            @. Vy[2:end-1,2:end-1]  += δVy[2:end-1,2:end-1]
            
            if mod(iter, nout) == 0 || iter==1

                ResidualMomentumX!(KδVx1, Vx,      Vy,      Pt, bx, ε̇xx, ε̇xy, ∇v, ηc_minloc, ηb, ηv_minloc, Dx, τxx, τxy, Δx, Δy, 0.0) # this allocates because of Vx.-δVx
                ResidualMomentumX!(KδVx,  Vx.-δVx, Vy.-δVy, Pt, bx, ε̇xx, ε̇xy, ∇v, ηc_minloc, ηb, ηv_minloc, Dx, τxx, τxy, Δx, Δy, 0.0)
                # GershgorinMechanics2Dx_Local!( λmaxlocVx, ηc, ηb, ηv, Dx, Δx, Δy, Ncx, Ncy )
                # hVx[:,2:end-1] .= 2.0./sqrt.(λmaxlocVx)*CFL

                ResidualMomentumY!(KδVy1, Vx,      Vy,      Pt, by, ε̇yy, ε̇xy, ∇v, ηc_minloc, ηb, ηv_minloc, Dy, τyy, τxy, Δx, Δy, 0.0)
                ResidualMomentumY!(KδVy,  Vx.-δVx, Vy.-δVy, Pt, by, ε̇yy, ε̇xy, ∇v, ηc_minloc, ηb, ηv_minloc, Dy, τyy, τxy, Δx, Δy, 0.0)
                # GershgorinMechanics2Dy_Local!( λmaxlocVy, ηc, ηb, ηv, Dy, Δx, Δy, Ncx, Ncy )
                # hVy[2:end-1,:] .= 2.0./sqrt.(λmaxlocVy)*CFL

                λmin = abs(sum(.-δVx.*(KδVx1.-KδVx)) + sum(.-δVy.*(KδVy1.-KδVy))/(sum(δVx.*δVx) + sum(δVy.*δVy)) )
                c    = 2.0*sqrt(λmin)*cfact

                ResidualMomentumX!(Rx, Vx, Vy, Pt, bx, ε̇xx, ε̇xy, ∇v, ηc, ηb, ηv, Dx, τxx, τxy, Δx, Δy, 1.0)
                ResidualMomentumY!(Ry, Vx, Vy, Pt, by, ε̇yy, ε̇xy, ∇v, ηc, ηb, ηv, Dy, τyy, τxy, Δx, Δy, 1.0)

                errVx = norm(Rx.*Dx)/(length(Rx))
                errVy = norm(Ry.*Dy)/(length(Ry))
                @printf("Iteration %05d --- Time step %4d \n", iter, it )
                @printf("Rx = %2.4e\n", errVx)
                @printf("Ry = %2.4e\n", errVy)
                @show (λmin, maximum(λmaxlocVx), log10(maximum(λmaxlocVx))-log10(λmin))
                @show (λmin, maximum(λmaxlocVy), log10(maximum(λmaxlocVx))-log10(λmin))
                if ( isnan(errVx) ) error() end
                ( errVx < ϵ ) && break
            end
        end
        probes.iters[it] = iters

        # Visualisation
        if mod(it, 10)==0 || it==1
            p1=heatmap(ustrip.(dimensionalize(xc[2:end-1], m, CharDim)./1e3), ustrip.(dimensionalize(yc[2:end-1], m, CharDim)./1e3), ustrip.(dimensionalize(Pt, Pa, CharDim))', title = "$(mean(probes.iters[1:it])) iterations", aspect_ratio=1.0, clims=(-3,3) )
            p2=heatmap(ustrip.(dimensionalize(xv, m, CharDim)./1e3), ustrip.(dimensionalize(yv, m, CharDim)./1e3), log10.(ustrip.(dimensionalize(ηv, Pas, CharDim)))', title = "$(mean(probes.iters[1:it])) iterations", aspect_ratio=1.0 )
            p3=heatmap(ustrip.(dimensionalize(xv, m, CharDim)./1e3), ustrip.(dimensionalize(yc, m, CharDim)./1e3), ustrip.(dimensionalize(Vx, m*s^-1, CharDim))', title = "$(mean(probes.iters[1:it])) iterations", aspect_ratio=1.0 )
            p4=heatmap(ustrip.(dimensionalize(xc, m, CharDim)./1e3), ustrip.(dimensionalize(yv, m, CharDim)./1e3), ustrip.(dimensionalize(Vy, m*s^-1, CharDim))', title = "$(mean(probes.iters[1:it])) iterations", aspect_ratio=1.0 )
            display(plot(p1,p2,p3,p4))
        end
    end
end

MainMechanicsDiagPC2D()