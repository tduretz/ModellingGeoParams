using GeoParams, Plots, Printf, MathTeXEngine, BenchmarkTools, Statistics
import LinearAlgebra:norm
plotlyjs()
# Makie.update_theme!(fonts = (regular = texfont(), bold = texfont(:bold), italic = texfont(:italic)))

const cmy = 356.25*3600*24*100
const ky  = 356.25*3600*24*1e3

function GershgorinMechanics2Dx_Local!( λmaxloc, ηc, ηv, Δx, Δy, Ncx, Ncy )
    ηW = zeros(Ncx+1, Ncy); ηW[2:end-0,:] .= ηc; ηW[1,:] = ηW[2,:]
    ηE = zeros(Ncx+1, Ncy); ηE[1:end-1,:] .= ηc; ηE[end,:] = ηE[end-1,:]
    ηS = zeros(Ncx+1, Ncy); ηS            .= ηv[:,1:end-1]
    ηN = zeros(Ncx+1, Ncy); ηN            .= ηv[:,2:end-0]
    Cx = 2*4//3*ηE/Δx^2 + 2*4//3*ηW/Δx^2 + 2*ηS/Δy^2 + 2*ηN/Δy^2
    Cy = abs.(2//3*ηE/Δx/Δy - ηN/Δy/Δx) + abs.(2//3*ηE/Δx/Δy - ηS/Δy/Δx) + abs.(ηN/Δy/Δx - 2//3*ηW/Δx/Δy) + abs.(ηS/Δy/Δx - 2//3*ηW/Δx/Δy) 
    Cp = 2/Δx
    λmaxloc .= Cx + Cy .+ 0*Cp
end

function GershgorinMechanics2Dy_Local!( λmaxloc, ηc, ηv, Δx, Δy, Ncx, Ncy )
    ηS = zeros(Ncx, Ncy+1); ηS[:,2:end-0] .= ηc
    ηN = zeros(Ncx, Ncy+1); ηN[:,1:end-1] .= ηc
    ηW = zeros(Ncx, Ncy+1); ηW            .= ηv[1:end-1,:]
    ηE = zeros(Ncx, Ncy+1); ηE            .= ηv[2:end-0,:]
    Cy = 2*4//3*ηS/Δy^2 + 2*4//3*ηN/Δy^2 + 2*ηW/Δx^2 + 2*ηE/Δx^2
    Cx = abs.(2//3*ηN/Δx/Δy - ηE/Δy/Δx) + abs.(2//3*ηN/Δx/Δy - ηW/Δy/Δx) + abs.(ηE/Δy/Δx - 2//3*ηS/Δx/Δy) + abs.(ηW/Δy/Δx - 2//3*ηS/Δx/Δy) 
    Cp = 2/Δy
    λmaxloc .= Cx + Cy .+ 0*Cp
end

function ResidualMomentumX!(Rx, Vx, Vy, Pt, ε̇xx, ε̇xy, ηc, ηv, τxx, τxy, Δx, Δy)
    @. Vx[:,  1]     = Vx[:,    2]
    @. Vx[:,end]     = Vx[:,end-1]
    @. Vy[  1,:]     = Vy[    2,:]
    @. Vy[end,:]     = Vy[end-1,:]
    @. ε̇xx           = (Vx[2:end-0,2:end-1] - Vx[1:end-1,2:end-1])/Δx - 1//3*((Vx[2:end,2:end-1] - Vx[1:end-1,2:end-1])/Δx + (Vy[2:end-1,2:end-0] - Vy[2:end-1,1:end-1])/Δy)
    @. ε̇xy           = 1//2*((Vx[:,2:end] - Vx[:,1:end-1])/Δy + (Vy[2:end,:] - Vy[1:end-1,:])/Δx )
    @. τxx           = 2*ηc*ε̇xx
    @. τxy           = 2*ηv*ε̇xy
    @. Rx[2:end-1,2:end-1] = (τxx[2:end,:] - τxx[1:end-1,:])/Δx + (τxy[2:end-1,2:end] - τxy[2:end-1,1:end-1])/Δy - (Pt[2:end,:] - Pt[1:end-1,:])/Δx
end

function ResidualMomentumY!(Ry, Vx, Vy, Pt, ε̇yy, ε̇xy, ηc, ηv, τyy, τxy, Δx, Δy)
    @. Vx[:,  1]     = Vx[:,    2]
    @. Vx[:,end]     = Vx[:,end-1]
    @. Vy[  1,:]     = Vy[    2,:]
    @. Vy[end,:]     = Vy[end-1,:]
    @. ε̇yy           = (Vy[2:end-1,2:end-0] - Vy[2:end-1,1:end-1])/Δy - 1//3*((Vx[2:end,2:end-1] - Vx[1:end-1,2:end-1])/Δx + (Vy[2:end-1,2:end-0] - Vy[2:end-1,1:end-1])/Δy)
    @. ε̇xy           = 1//2*((Vx[:,2:end] - Vx[:,1:end-1])/Δy + (Vy[2:end,:] - Vy[1:end-1,:])/Δx )
    @. τyy           = 2*ηc*ε̇yy
    @. τxy           = 2*ηv*ε̇xy
    @. Ry[2:end-1,2:end-1] = (τyy[:,2:end] - τyy[:,1:end-1])/Δy + (τxy[2:end,2:end-1] - τxy[1:end-1,2:end-1])/Δx - (Pt[:,2:end] - Pt[:,1:end-1])/Δy
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

av2x_arit(A) = 0.5.*(A[1:end-1,:].+A[2:end,:])   
av2y_arit(A) = 0.5.*(A[:,1:end-1].+A[:,2:end]) 
av4_arit(A)  = 0.25.*(A[1:end-1,1:end-1].+A[2:end,1:end-1].+A[1:end-1,2:end].+A[2:end,2:end])   
av4_harm(A)  = 1.0./( 0.25.*(1.0./A[1:end-1,1:end-1].+1.0./A[2:end,1:end-1].+1.0./A[1:end-1,2:end].+1.0./A[2:end,2:end]) )  

@views function MainStokes2D()

    # Unit system
    CharDim    = SI_units(length=1m, temperature=1C, stress=1Pa, viscosity=1Pas)

    # Physical parameters
    Lx         = nondimensionalize(1m, CharDim)
    Ly         = nondimensionalize(1m, CharDim)
    r          = nondimensionalize(sqrt(0.025)m, CharDim)
    # Lx         = nondimensionalize(1/1*24m, CharDim)
    # Ly         = nondimensionalize(1/1*24m, CharDim)
    # r          = nondimensionalize(1/1*2.8m, CharDim)
    η_mat      = nondimensionalize(1Pas, CharDim)
    η_inc      = nondimensionalize(500Pas, CharDim)

    # BCs
    ε̇bg        = 1.0

    # Numerical parameters
    Ncx        = 8*10
    Ncy        = 8*10
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
    ηb         = ones(size(ηc) ) #copy(ηc).*10000

    ηc_minloc = η_mat.*ones(Ncx+0, Ncy+0)
    ηc_maxloc = η_mat.*ones(Ncx+0, Ncy+0)
    ηv_maxloc = η_mat.*ones(Ncx+1, Ncy+1)
    ηv[xv.^2 .+ yv'.^2 .< r^2 ] .= η_inc
    ηc[xc[2:end-1].^2 .+ yc[2:end-1]'.^2 .< r^2 ] .= η_inc
    # ηc       .= 0.25.*(ηv[1:end-1,1:end-1] .+ ηv[2:end-0,1:end-1] .+ ηv[1:end-1,2:end-0] .+ ηv[2:end-0,2:end-0])
    maxloc!(ηv_maxloc, ηv)
    maxloc!(ηc_maxloc, ηc)
    minloc!(ηc_minloc, ηc)

    λmaxlocVx   = zeros(Ncx+1, Ncy+0)
    λmaxlocVy   = zeros(Ncx+0, Ncy+1)

    hPt1 = zero(hPt)

    # Monitoring
    probes    = (iters = zeros(Nt), t = zeros(Nt), Ẇ0 = zeros(Nt), τxyi = zeros(Nt), Vx0 = zeros(Nt), maxT = zeros(Nt))
    
    # PT solver
    niter  = 10000
    nout   = 500
    ϵ      = 1e-6
    CFL    = 0.99

    for it=1:Nt
   
        # DYREL
        GershgorinMechanics2Dx_Local!( λmaxlocVx, ηc_maxloc, ηv_maxloc, Δx, Δy, Ncx, Ncy )
        λminVx          = 1.0
        hVx[:,2:end-1] .= 2.0./sqrt.(λmaxlocVx)*CFL
        cVx             = 2*sqrt(λminVx)

        # λminlocVy = λminlocVy'
        GershgorinMechanics2Dy_Local!( λmaxlocVy, ηc_maxloc, ηv_maxloc, Δx, Δy, Ncx, Ncy )
        λminVy          = 1.0   
        hVy[2:end-1,:] .= 2.0./sqrt.(λmaxlocVy)*CFL
        cVy             = 2*sqrt(λminVy)

        λmaxPt      = (1.0./ηb .+ ( 2/Δx +  2/Δy )) ./ ηc.^2 .*1000*sqrt(2) 
        λmaxPt      = (1.0./ηb .+ ( 2/Δx +  2/Δy )) 


        # λmaxPt       =  (ηb[1]./ηc[1])^0.5 ./ (2*4//3*ηc/Δx^2 + 2*4//3*ηc/Δx^2 + 2*ηc/Δy^2 + 2*ηc/Δy^2)  *4 
        
        λminPt      = 1
        hPt        .= 2.0./sqrt.(λmaxPt)*CFL 
        cPt        .= 2*sqrt.(λminPt)

        mode = 1

        λmin = 1.

        hx  = zero(λmaxlocVx)
        hy  = zero(λmaxlocVy)
        hp  = zero(Pt)
        hx .= 1 ./av2y_arit(ηv_maxloc)
        hy .= 1 ./av2x_arit(ηv_maxloc)
        hp .= 1  .*ones(size(λmaxPt))
        @show λmax = max(maximum(λmaxlocVx.*hx), maximum(λmaxlocVy.*hy), maximum(λmaxPt.*hp))

        @show maximum(λmaxlocVx.*hx)
        @show maximum(λmaxlocVy.*hy)
        @show maximum(λmaxPt.*hp)
        λmax = max(maximum(λmaxlocVx), maximum(λmaxlocVy), maximum(λmaxPt))

        @show λmax = Gershgorin_Stokes2D( hx, hy, hp, ηb, ηc, ηv, Δx, Δy )

        h    = 2.0./sqrt.(λmax)*CFL
        c    = 2.0.*sqrt.(λmin)

        iters = 0
        @views for iter=1:niter
            iters  += 1

            # Residuals
            ResidualMomentumX!(Rx, Vx, Vy, Pt, ε̇xx, ε̇xy, ηc, ηv, τxx, τxy, Δx, Δy)
            ResidualMomentumY!(Ry, Vx, Vy, Pt, ε̇yy, ε̇xy, ηc, ηv, τyy, τxy, Δx, Δy)
            ResidualContinuity!(Rp, Vx, Vy, Pt, ηb, Δx, Δy)

            if mode==1
                @. ∂Vx∂τ                 = (2-c*h)/(2+c*h)*∂Vx∂τ + 2*h^2/(2+c*h).*Rx
                @. δVx[:,2:end-1]        = hx*∂Vx∂τ[:,2:end-1]
                @. Vx[2:end-1,2:end-1]  += δVx[2:end-1,2:end-1]

                @. ∂Vy∂τ                 = (2-c*h)/(2+c*h)*∂Vy∂τ + 2*h^2/(2+c*h).*Ry
                @. δVy[2:end-1,:]        = hy*∂Vy∂τ[2:end-1,:] 
                @. Vy[2:end-1,2:end-1]  += δVy[2:end-1,2:end-1]

                @. ∂Pt∂τ                 = (2-c*h)/(2+c*h)*∂Pt∂τ + 2*h^2/(2+c*h).*Rp
                @. δPt                   = hp*∂Pt∂τ
                @. Pt                   += δPt 
            else

                @. ∂Vx∂τ                 = (2-cVx*hVx)/(2+cVx*hVx)*∂Vx∂τ + 2*hVx/(2+cVx*hVx).*Rx
                @. δVx                   = hVx*∂Vx∂τ
                @. Vx[2:end-1,2:end-1]  += δVx[2:end-1,2:end-1]

                @. ∂Vy∂τ                 = (2-cVy*hVy)/(2+cVy*hVy)*∂Vy∂τ + 2*hVy/(2+cVy*hVy).*Ry
                @. δVy                   = hVy*∂Vy∂τ
                @. Vy[2:end-1,2:end-1]  += δVy[2:end-1,2:end-1]
                
                # hPt1          = ηc/sqrt(Δx*Δy)*((Lx*Ly)/130000) 
                
                # @. hPt          = ηc./ηb * (Δx*Lx + Δy*Ly) * min(Lx/Ly, Ly/Lx)^2 
                # hPt1         =  min(Δx^2, Δy^2) ./ (ηb.*(4//3 .*ηc./ηb .+ 1))
                # @. Pt          += hPt*Rp

                # @. δPt                   = hPt*Rp
                # # @show hPt[1]
                # hPt1          = abs.(ηc_minloc./τxx/(8/3)) / 2 
                # @. hPt          = abs.(ηc./(Pt+1e-13)/(8/3)) /1000       

                # @show (hPt[1], hPt1[1])
                # @show (minimum(hPt), minimum(hPt1))
                
                @. ∂Pt∂τ                 = (2-cPt*hPt)/(2+cPt*hPt)*∂Pt∂τ + 2*hPt/(2+cPt*hPt).*Rp
                # @. ∂Pt∂τ                 = Rp
                @. δPt                   = hPt*∂Pt∂τ 
                @. Pt                   += δPt
            end

            if mod(iter, nout) == 0 || iter==1

                # ResidualMomentumX!(KδVx1, Vx,      Vy, Pt, ε̇xx, ε̇xy, ηc, ηv, τxx, τxy, Δx, Δy)
                # ResidualMomentumX!(KδVx,  Vx.-δVx, Vy, Pt, ε̇xx, ε̇xy, ηc, ηv, τxx, τxy, Δx, Δy)
                # λminVx =  abs(sum(.-δVx.*(KδVx1.-KδVx))/sum(δVx.*δVx) / 1.0 )
                # GershgorinMechanics2Dx_Local!( λmaxlocVx, ηc, ηv, Δx, Δy, Ncx, Ncy )
                # hVx[:,2:end-1] .= 2.0./sqrt.(λmaxlocVx)*CFL
                # cVx             = 2*sqrt(λminVx)
        
                # ResidualMomentumY!(KδVy1, Vx, Vy,      Pt, ε̇xx, ε̇xy, ηc, ηv, τxx, τxy, Δx, Δy)
                # ResidualMomentumY!(KδVy,  Vx, Vy.-δVy, Pt, ε̇xx, ε̇xy, ηc, ηv, τxx, τxy, Δx, Δy)
                # λminVy =  abs(sum(.-δVy.*(KδVy1.-KδVy))/sum(δVy.*δVy) / 1.0 )
                # # λminlocVy = λminlocVy'
                # GershgorinMechanics2Dy_Local!( λmaxlocVy, ηc, ηv, Δx, Δy, Ncx, Ncy )
                # hVy[2:end-1,:] .= 2.0./sqrt.(λmaxlocVy)*CFL
                # cVy             = 2*sqrt(λminVy)

                # ResidualContinuity!(KδPt1, Vx, Vy, Pt,      ηb, Δx, Δy)
                # ResidualContinuity!(KδPt,  Vx, Vy, Pt.-δPt, ηb, Δx, Δy)
                # λminPt      = abs(sum(.-δPt.*(KδPt1.-KδPt).*ηb)/sum(δPt.*δPt)  ) 
                # # λmaxPt      = 1.0./ηb .+ ( 2/Δx + 2/Δy ) 
                # # hPt        .= 2.0./sqrt.(λmaxPt)*CFL
                # # cPt        .= 2.0.*sqrt.(λminPt)

                # @show (λminVx, λminVy, λminPt)
                # @show (λmaxlocVx[1], λmaxlocVy[1], λmaxPt[1])

                # @show ((ηc[1]./ηb[1])^0.5)
                # @show (Δx^2)

                # ResidualMomentumX!(KδVx, δVx,      δVy, δPt, ε̇yy, ε̇xy, ηc, ηv, τyy, τxy, Δx, Δy)
                # ResidualMomentumY!(KδVy, δVx, δVy,      δPt, ε̇yy, ε̇xy, ηc, ηv, τyy, τxy, Δx, Δy)
                # ResidualContinuity!(KδPt, δVx, δVy, δPt,      ηb, Δx, Δy)

                # # λmin = abs(( sum(δPt.*(KδPt1.-KδPt)) + sum(δVy.*(KδVy1.-KδVy)) + sum(δVx.*(KδVx1.-KδVx)))/(sum(δPt.*δPt) + sum(δVy.*δVy) + sum(δVx.*δVx))  ) 
                # # λmin = abs(( sum(δPt.*(KδPt)) + sum(δVy.*(KδVy)) + sum(δVx.*(KδVx)))/(sum(δPt.*δPt) + sum(δVy.*δVy) + sum(δVx.*δVx))  ) 

                # λmin = 0.01
                # h    = 2.0./sqrt.(λmax)*CFL
                # c    = 2.0*sqrt.(λmin)
                # @show (λmin)

                # ResidualMomentumX!(Rx, Vx, Vy, Pt, ε̇xx, ε̇xy, ηc, ηv, τxx, τxy, Δx, Δy)
                # ResidualMomentumY!(Ry, Vx, Vy, Pt, ε̇yy, ε̇xy, ηc, ηv, τyy, τxy, Δx, Δy)
                # ResidualContinuity!(Rp, Vx, Vy, Pt, ηb, Δx, Δy)

                errVx = norm(Rx)/sqrt(length(Rx))
                errVy = norm(Ry)/sqrt(length(Ry))
                errPt = norm(Rp)/sqrt(length(Rp))
                @printf("Iteration %05d --- Time step %4d \n", iter, it )
                @printf("Rx = %2.4e\n", errVx)
                @printf("Ry = %2.4e\n", errVy)
                @printf("Rp = %2.4e\n", errPt)
                @show (λminVx, maximum(λmaxlocVx))
                @show (λminVy, maximum(λmaxlocVy))
                if ( isnan(errVx) ) error() end
                ( errVx < ϵ &&  errPt < ϵ  ) && break
            end
        end
        probes.iters[it] = iters
    #     # Visualisation
    #     if mod(it, 10)==0 || it==1
            # p1=heatmap(ustrip.(dimensionalize(xc[2:end-1], m, CharDim)), ustrip.(dimensionalize(yc[2:end-1], m, CharDim)), ustrip.(dimensionalize(hPt, Pa, CharDim))', title = "$(minimum(hPt))", aspect_ratio=1.0 )
            # p2=heatmap(ustrip.(dimensionalize(xc[2:end-1], m, CharDim)), ustrip.(dimensionalize(yc[2:end-1], m, CharDim)), ustrip.(dimensionalize(hPt1, Pa, CharDim))', title = "$(minimum(hPt1))", aspect_ratio=1.0 )

            p1=heatmap(ustrip.(dimensionalize(xc[2:end-1], m, CharDim)), ustrip.(dimensionalize(yc[2:end-1], m, CharDim)), ustrip.(dimensionalize(Pt, Pa, CharDim))', title = "$(minimum(hPt))", aspect_ratio=1.0, colormap=cgrad(:roma, rev=true) )
            p2=heatmap(ustrip.(dimensionalize(xv, m, CharDim)), ustrip.(dimensionalize(yv, m, CharDim)), log10.(ustrip.(dimensionalize(ηv, Pas, CharDim)))', title = "$(mean(probes.iters[1:it])) iterations", aspect_ratio=1.0, colormap=cgrad(:roma, rev=true) )
            # p1=heatmap(ustrip.(dimensionalize(xv, m, CharDim)./1e3), ustrip.(dimensionalize(yc, m, CharDim)./1e3), ustrip.(dimensionalize(Rx, Pa, CharDim))', title = "$(mean(probes.iters[1:it])) iterations", aspect_ratio=1.0 )
            # p2=heatmap(ustrip.(dimensionalize(xc, m, CharDim)./1e3), ustrip.(dimensionalize(yv, m, CharDim)./1e3), ustrip.(dimensionalize(Ry, Pa, CharDim))', title = "$(mean(probes.iters[1:it])) iterations", aspect_ratio=1.0 )
            display(plot(p1,p2))
    #     end
    end
end

MainStokes2D()