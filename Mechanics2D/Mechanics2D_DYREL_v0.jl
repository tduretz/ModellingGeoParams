using GeoParams, Plots, Printf, MathTeXEngine, BenchmarkTools, Statistics
import LinearAlgebra:norm
# Makie.update_theme!(fonts = (regular = texfont(), bold = texfont(:bold), italic = texfont(:italic)))

const cmy = 356.25*3600*24*100
const ky  = 356.25*3600*24*1e3

function GershgorinMechanics2Dx_Local!( λmaxloc, ηc, kc, ηv, Δx, Δy, Ncx, Ncy )
    ηW = zeros(Ncx+1, Ncy); ηW[2:end-0,:] .= ηc; ηW[1,:]   = ηW[2,:]
    ηE = zeros(Ncx+1, Ncy); ηE[1:end-1,:] .= ηc; ηE[end,:] = ηE[end-1,:]
    kW = zeros(Ncx+1, Ncy); kW[2:end-0,:] .= kc; ηW[1,:]   = kW[2,:]
    kE = zeros(Ncx+1, Ncy); kE[1:end-1,:] .= kc; ηE[end,:] = kE[end-1,:]
    ηS = zeros(Ncx+1, Ncy); ηS            .= ηv[:,1:end-1]
    ηN = zeros(Ncx+1, Ncy); ηN            .= ηv[:,2:end-0]
    Cx = 2*4//3*ηE/Δx^2 + 2*4//3*ηW/Δx^2 + 2*ηS/Δy^2 + 2*ηN/Δy^2 + 2*kW/Δx^2 + 2*kE/Δx^2
    Cy = abs.(-2//3*ηE+ ηN + kE)/Δx/Δy  + abs.(-2//3*ηE + ηS + kE)/Δx/Δy + abs.(ηN - 2//3*ηW + kW)/Δy/Δx + abs.(ηS- 2//3*ηW + kW)/Δy/Δx  
    λmaxloc .= Cx + Cy
end

function GershgorinMechanics2Dy_Local!( λmaxloc, ηc, kc, ηv, Δx, Δy, Ncx, Ncy )
    ηS = zeros(Ncx, Ncy+1); ηS[:,2:end-0] .= ηc
    ηN = zeros(Ncx, Ncy+1); ηN[:,1:end-1] .= ηc
    kS = zeros(Ncx, Ncy+1); kS[:,2:end-0] .= kc
    kN = zeros(Ncx, Ncy+1); kN[:,1:end-1] .= kc
    ηW = zeros(Ncx, Ncy+1); ηW            .= ηv[1:end-1,:]
    ηE = zeros(Ncx, Ncy+1); ηE            .= ηv[2:end-0,:]
    Cy = 2*4//3*ηS/Δy^2 + 2*4//3*ηN/Δy^2 + 2*ηW/Δx^2 + 2*ηE/Δx^2 + 2*kS/Δy^2 + 2*kN/Δy^2
    # Cx = abs.(2//3*ηN/Δx/Δy - ηE/Δy/Δx) + abs.(2//3*ηN/Δx/Δy - ηW/Δy/Δx) + abs.(ηE/Δy/Δx - 2//3*ηS/Δx/Δy) + abs.(ηW/Δy/Δx - 2//3*ηS/Δx/Δy) 
    Cx = abs.(-2//3*ηN+ ηE + kN)/Δx/Δy  + abs.(-2//3*ηN + ηW + kN)/Δx/Δy + abs.(ηE - 2//3*ηS + kS)/Δy/Δx + abs.(ηW - 2//3*ηS + kS)/Δy/Δx  
    λmaxloc .= Cx + Cy
end

@views function ResidualMomentumX!(Rx, Vx, Vy, Pt, bx, ε̇xx, ε̇xy, ηc, kc, ηv, τxx, τxy, Δx, Δy, rhs)
    @. Vx[:,  1]     = Vx[:,    2]
    @. Vx[:,end]     = Vx[:,end-1]
    @. Vy[  1,:]     = Vy[    2,:]
    @. Vy[end,:]     = Vy[end-1,:]
    @. ε̇xx           = (Vx[2:end-0,2:end-1] - Vx[1:end-1,2:end-1])/Δx - 1//3*((Vx[2:end,2:end-1] - Vx[1:end-1,2:end-1])/Δx + (Vy[2:end-1,2:end-0] - Vy[2:end-1,1:end-1])/Δy)
    @. ε̇xy           = 1//2*((Vx[:,2:end] - Vx[:,1:end-1])/Δy + (Vy[2:end,:] - Vy[1:end-1,:])/Δx )
    @. τxx           = 2*ηc*ε̇xx
    @. τxy           = 2*ηv*ε̇xy
    @. Pt            = -kc*((Vx[2:end,2:end-1] - Vx[1:end-1,2:end-1])/Δx + (Vy[2:end-1,2:end-0] - Vy[2:end-1,1:end-1])/Δy)
    @. Rx[2:end-1,2:end-1] = ( (τxx[2:end,:] - τxx[1:end-1,:])/Δx + (τxy[2:end-1,2:end] - τxy[2:end-1,1:end-1])/Δy - (Pt[2:end,:] - Pt[1:end-1,:])/Δx  ) + bx[2:end-1,2:end-1]*rhs
end

@views function ResidualMomentumY!(Ry, Vx, Vy, Pt, by, ε̇yy, ε̇xy, ηc, kc, ηv, τyy, τxy, Δx, Δy, rhs)
    @. Vx[:,  1]     = Vx[:,    2]
    @. Vx[:,end]     = Vx[:,end-1]
    @. Vy[  1,:]     = Vy[    2,:]
    @. Vy[end,:]     = Vy[end-1,:]
    @. ε̇yy           = (Vy[2:end-1,2:end-0] - Vy[2:end-1,1:end-1])/Δy - 1//3*((Vx[2:end,2:end-1] - Vx[1:end-1,2:end-1])/Δx + (Vy[2:end-1,2:end-0] - Vy[2:end-1,1:end-1])/Δy)
    @. ε̇xy           = 1//2*((Vx[:,2:end] - Vx[:,1:end-1])/Δy + (Vy[2:end,:] - Vy[1:end-1,:])/Δx )
    @. τyy           = 2*ηc*ε̇yy
    @. τxy           = 2*ηv*ε̇xy
    @. Pt            = -kc*((Vx[2:end,2:end-1] - Vx[1:end-1,2:end-1])/Δx + (Vy[2:end-1,2:end-0] - Vy[2:end-1,1:end-1])/Δy)
    @. Ry[2:end-1,2:end-1] = ( (τyy[:,2:end] - τyy[:,1:end-1])/Δy + (τxy[2:end,2:end-1] - τxy[1:end-1,2:end-1])/Δx - (Pt[:,2:end] - Pt[:,1:end-1])/Δy  ) + by[2:end-1,2:end-1]*rhs
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

@views function MainStokes2D()

    # Unit system
    CharDim    = SI_units(length=1m, temperature=1C, stress=1Pa, viscosity=1Pas)

    # Physical parameters
    Lx         = nondimensionalize(1/1*24m, CharDim)
    Ly         = nondimensionalize(1/1*24m, CharDim)
    r          = nondimensionalize(1/1*2.8m, CharDim)
    η_mat      = nondimensionalize(1Pas, CharDim)
    η_inc      = nondimensionalize(0.05Pas, CharDim)

    # BCs
    ε̇bg        = -1.0

    # Numerical parameters
    n          = 4 
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
    τxx        = zeros(Ncx+0, Ncy+0)
    τyy        = zeros(Ncx+0, Ncy+0)
    τxy        = zeros(Ncx+1, Ncy+1)
    Rx         = zeros(Ncx+1, Ncy+2)
    Ry         = zeros(Ncx+2, Ncy+1)
    bx         = zeros(Ncx+1, Ncy+2)
    by         = zeros(Ncx+2, Ncy+1)
    Rp         = zeros(Ncx+0, Ncy+0)
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
    hPt        = zeros(Ncx+0, Ncy+0) # Time step is local
    cPt        = zeros(Ncx+0, Ncy+0) 
    ∂Pt∂τ       = zeros(Ncx,Ncy)
    δPt         = zeros(Ncx,Ncy)
    KδPt1       = zeros(Ncx,Ncy)
    KδPt        = zeros(Ncx,Ncy)
    ηv         = η_mat.*ones(Ncx+1, Ncy+1)
    ηc         = η_mat.*ones(Ncx+0, Ncy+0)
    ηb         = ones(size(ηc))
    ηb         = copy(ηc)*100

    bx[xv.^2 .+ yc'.^2 .< r^2 ] .= 1e-2
    by[yc.^2 .+ yv'.^2 .< r^2 ] .= 1e-2

    ηc_maxloc = η_mat.*ones(Ncx+0, Ncy+0)
    ηv_maxloc = η_mat.*ones(Ncx+1, Ncy+1)
    # ηv[xv.^2 .+ yv'.^2 .< r^2 ] .= η_inc
    ηc       .= 0.25.*(ηv[1:end-1,1:end-1] .+ ηv[2:end-0,1:end-1] .+ ηv[1:end-1,2:end-0] .+ ηv[2:end-0,2:end-0])
    maxloc!(ηv_maxloc, ηv)
    maxloc!(ηc_maxloc, ηc)

    ηc_minloc = η_mat.*ones(Ncx+0, Ncy+0)
    ηv_minloc = η_mat.*ones(Ncx+1, Ncy+1)
    minloc!(ηv_minloc, ηv)
    minloc!(ηc_minloc, ηc)

    λmaxlocVx   = zeros(Ncx+1, Ncy+0)
    λmaxlocVy   = zeros(Ncx+0, Ncy+1)

    # Monitoring
    probes    = (iters = zeros(Nt), t = zeros(Nt), Ẇ0 = zeros(Nt), τxyi = zeros(Nt), Vx0 = zeros(Nt), maxT = zeros(Nt))
    
    # PT solver
    niter  = 50000
    nout   = 50
    ϵ      = 5e-11
    CFL    = 0.999
    cfact  = 1.0

    for it=1:Nt
   
        λmin = 0.077

        # DYREL
        GershgorinMechanics2Dx_Local!( λmaxlocVx, ηc_maxloc, ηb, ηv_maxloc, Δx, Δy, Ncx, Ncy )
        λminVx          = λmin
        hVx[:,2:end-1] .= 2.0./sqrt.(λmaxlocVx)*CFL
        cVx             = 2.0*sqrt(λminVx)

        # λminlocVy = λminlocVy'
        GershgorinMechanics2Dy_Local!( λmaxlocVy, ηc_maxloc, ηb, ηv_maxloc, Δx, Δy, Ncx, Ncy )
        λminVy          = λmin
        hVy[2:end-1,:] .= 2.0./sqrt.(λmaxlocVy)*CFL
        cVy             = 2.0*sqrt(λminVy)
    
        iters = 0
        @time @views for iter=1:niter
            iters  += 1

            # Residuals
            ResidualMomentumX!(Rx, Vx, Vy, Pt, bx, ε̇xx, ε̇xy, ηc, ηb, ηv, τxx, τxy, Δx, Δy, 1.0)
            ResidualMomentumY!(Ry, Vx, Vy, Pt, by, ε̇yy, ε̇xy, ηc, ηb, ηv, τyy, τxy, Δx, Δy, 1.0)
            
            @. ∂Vx∂τ                 = (2-cVx*hVx)/(2+cVx*hVx)*∂Vx∂τ + 2*hVx/(2+cVx*hVx).*Rx
            @. δVx                   = hVx*∂Vx∂τ
            @. Vx[2:end-1,2:end-1]  += δVx[2:end-1,2:end-1]

            @. ∂Vy∂τ                 = (2-cVy*hVy)/(2+cVy*hVy)*∂Vy∂τ + 2*hVy/(2+cVy*hVy).*Ry
            @. δVy                   = hVy*∂Vy∂τ
            @. Vy[2:end-1,2:end-1]  += δVy[2:end-1,2:end-1]
            
            if mod(iter, nout) == 0 || iter==1

                # KδVxy  = zero(KδVx) 
                # KδVxy1 = zero(KδVx1) 
                # KδVyx  = zero(KδVy) 
                # KδVyx1 = zero(KδVy1)

                # ResidualMomentumX!(KδVx1, Vx,      Vy, Pt, ε̇xx, ε̇xy, ηc, ηb, ηv, τxx, τxy, Δx, Δy)
                # ResidualMomentumX!(KδVx,  Vx.-δVx, Vy, Pt, ε̇xx, ε̇xy, ηc, ηb, ηv, τxx, τxy, Δx, Δy)
                # # λminVx =  abs(sum(.-δVx.*(KδVx1.-KδVx))/sum(δVx.*δVx) )
                # ResidualMomentumX!(KδVxy1, Vx, Vy,      Pt, ε̇xx, ε̇xy, ηc, ηb, ηv, τxx, τxy, Δx, Δy)
                # ResidualMomentumX!(KδVxy,  Vx, Vy.-δVy, Pt, ε̇xx, ε̇xy, ηc, ηb, ηv, τxx, τxy, Δx, Δy)
                # # λminVx =  abs(λminVx - abs(sum(.-δVx.*(KδVx1.-KδVx))/sum(δVx.*δVx)) )
                # @show λminVx =  abs(  sum(.-δVx.*( (KδVx1.-KδVx) .+ (KδVxy1.-KδVxy))  )/sum(δVx.*δVx)) 
                
                ResidualMomentumX!(KδVx1, Vx,      Vy,      Pt, bx, ε̇xx, ε̇xy, ηc_minloc, ηb, ηv_minloc, τxx, τxy, Δx, Δy, 0.0)
                ResidualMomentumX!(KδVx,  Vx.-δVx, Vy.-δVy, Pt, bx, ε̇xx, ε̇xy, ηc_minloc, ηb, ηv_minloc, τxx, τxy, Δx, Δy, 0.0)
                @show λminVx =  abs(sum(.-δVx.*(KδVx1.-KδVx))/sum(δVx.*δVx) ) 
                # @show (λminVx,  abs(sum(.-δVx.*(KδVx1.-KδVx))/sum(δVx.*δVx) ) )
                GershgorinMechanics2Dx_Local!( λmaxlocVx, ηc, ηb, ηv, Δx, Δy, Ncx, Ncy )
                # hVx[:,2:end-1] .= 2.0./sqrt.(λmaxlocVx)*CFL
                # cVx             = 2.0*sqrt(λminVx)*cfact
        
                # ResidualMomentumY!(KδVy1, Vx, Vy,      Pt, ε̇yy, ε̇xy, ηc, ηb, ηv, τyy, τxy, Δx, Δy)
                # ResidualMomentumY!(KδVy,  Vx, Vy.-δVy, Pt, ε̇yy, ε̇xy, ηc, ηb, ηv, τyy, τxy, Δx, Δy)
                # # λminVy =  abs(sum(.-δVy.*(KδVy1.-KδVy))/sum(δVy.*δVy) )
                # ResidualMomentumY!(KδVyx1, Vx,      Vy, Pt, ε̇yy, ε̇xy, ηc, ηb, ηv, τyy, τxy, Δx, Δy)
                # ResidualMomentumY!(KδVyx,  Vx.-δVx, Vy, Pt, ε̇yy, ε̇xy, ηc, ηb, ηv, τyy, τxy, Δx, Δy)
                # # λminVy =  abs(λminVy - abs(sum(.-δVy.*(KδVy1.-KδVy))/sum(δVy.*δVy)) )
                # λminVy =  abs(  sum(.-δVy.*( (KδVy1.-KδVy) .+ (KδVyx1.-KδVyx))  )/sum(δVy.*δVy)) 

                ResidualMomentumY!(KδVy1, Vx,      Vy,      Pt, by, ε̇yy, ε̇xy, ηc_minloc, ηb, ηv_minloc, τyy, τxy, Δx, Δy, 0.0)
                ResidualMomentumY!(KδVy,  Vx.-δVx, Vy.-δVy, Pt, by, ε̇yy, ε̇xy, ηc_minloc, ηb, ηv_minloc, τyy, τxy, Δx, Δy, 0.0)
                λminVy =  abs(sum(.-δVy.*(KδVy1.-KδVy))/sum(δVy.*δVy) ) 
                # @show abs(sum(.-δVy.*(KδVy1.-KδVy))/sum(δVy.*δVy) )
                # λminlocVy = λminlocVy'
                GershgorinMechanics2Dy_Local!( λmaxlocVy, ηc, ηb, ηv, Δx, Δy, Ncx, Ncy )
                # hVy[2:end-1,:] .= 2.0./sqrt.(λmaxlocVy)*CFL
                # cVy             = 2.0*sqrt(λminVy)*cfact

                ResidualMomentumX!(Rx, Vx, Vy, Pt, bx, ε̇xx, ε̇xy, ηc, ηb, ηv, τxx, τxy, Δx, Δy, 1.0)
                ResidualMomentumY!(Ry, Vx, Vy, Pt, by, ε̇yy, ε̇xy, ηc, ηb, ηv, τyy, τxy, Δx, Δy, 1.0)

                errVx = norm(Rx)/(length(Rx))
                errVy = norm(Ry)/(length(Ry))
                @printf("Iteration %05d --- Time step %4d \n", iter, it )
                @printf("Rx = %2.4e\n", errVx)
                @printf("Ry = %2.4e\n", errVy)
                @show (λminVx, maximum(λmaxlocVx))
                @show (λminVy, maximum(λmaxlocVy))
                if ( isnan(errVx) ) error() end
                ( errVx < ϵ ) && break
            end
        end
        probes.iters[it] = iters

    #     # Visualisation
    #     if mod(it, 10)==0 || it==1
            p1=heatmap(ustrip.(dimensionalize(xc[2:end-1], m, CharDim)./1e3), ustrip.(dimensionalize(yc[2:end-1], m, CharDim)./1e3), ustrip.(dimensionalize(Pt, Pa, CharDim))', title = "$(mean(probes.iters[1:it])) iterations", aspect_ratio=1.0 )
            p2=heatmap(ustrip.(dimensionalize(xv, m, CharDim)./1e3), ustrip.(dimensionalize(yv, m, CharDim)./1e3), log10.(ustrip.(dimensionalize(ηv, Pas, CharDim)))', title = "$(mean(probes.iters[1:it])) iterations", aspect_ratio=1.0 )
            display(plot(p1,p2))
    #     end
    end
end

MainStokes2D()