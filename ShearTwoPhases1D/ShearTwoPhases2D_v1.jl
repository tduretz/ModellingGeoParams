# Initialisation
using Revise, Plots, Printf, Statistics, LinearAlgebra, GeoParams

# add iterative lamda 

# Macros
@views    av(A) = 0.25*(A[1:end-1,1:end-1].+A[2:end,1:end-1].+A[1:end-1,2:end].+A[2:end,2:end])
@views av_xa(A) =  0.5*(A[1:end-1,:].+A[2:end,:])
@views av_xi(A) =  0.5*(A[1:end-1,2:end-1].+A[2:end,2:end-1])
@views av_ya(A) =  0.5*(A[:,1:end-1].+A[:,2:end])
@views av_yi(A) =  0.5*(A[2:end-1,1:end-1].+A[2:end-1,2:end])
@views function av_xi!(B, A) 
    @. B = 0.5*(A[1:end-1,2:end-1] + A[2:end,2:end-1])
end
@views function av_yi!(B, A) 
    @. B = 0.5*(A[2:end-1,1:end-1] + A[2:end-1,2:end])
end
@views function av_xa!(B, A) 
    @. B = 0.5*(A[1:end-1,:] + A[2:end,:])
end
@views function av_ya!(B, A) 
    @. B = 0.5*(A[:,1:end-1] + A[:,2:end])
end
@views function av!(B, A) 
    @. B = 0.25*(A[1:end-1,1:end-1].+A[2:end,1:end-1].+A[1:end-1,2:end].+A[2:end,2:end])
end

function CorrPt(Pt, DivV, Divq_D, Pt0, Pf0, dt, B, K_d, alpha, eta_phi, phi, lamda, sin_psi)
    @. Pt = (B .* DivV .* K_d .^ 2 .* dt .^ 2 + B .* Divq_D .* K_d .^ 2 .* dt .^ 2 - B .* Divq_D .* K_d .* alpha .* dt .* eta_phi .* phi + B .* Divq_D .* K_d .* alpha .* dt .* eta_phi + B .* K_d .* Pf0 .* alpha .* dt + B .* K_d .* Pt0 .* alpha .* dt - B .* K_d .* Pt0 .* dt - B .* K_d .* alpha .* dt .* eta_phi .* lamda .* sin_psi - B .* Pt0 .* alpha .^ 2 .* eta_phi .* phi + B .* Pt0 .* alpha .^ 2 .* eta_phi - DivV .* K_d .* alpha .* dt .* eta_phi .* phi + DivV .* K_d .* alpha .* dt .* eta_phi - K_d .* Pf0 .* alpha .* dt + K_d .* alpha .* dt .* eta_phi .* lamda .* sin_psi + Pt0 .* alpha .* eta_phi .* phi - Pt0 .* alpha .* eta_phi) ./ (2 * B .* K_d .* alpha .* dt - B .* K_d .* dt - B .* alpha .^ 2 .* eta_phi .* phi + B .* alpha .^ 2 .* eta_phi - K_d .* alpha .* dt + alpha .* eta_phi .* phi - alpha .* eta_phi)
end
function CorrPf(Pf, DivV, Divq_D, Pt0, Pf0, dt, B, K_d, alpha, eta_phi, phi, lamda, sin_psi)
    @. Pf = (B .* DivV .* K_d .^ 2 .* dt .^ 2 - B .* DivV .* K_d .* alpha .* dt .* eta_phi .* phi + B .* DivV .* K_d .* alpha .* dt .* eta_phi + B .* Divq_D .* K_d .^ 2 .* dt .^ 2 - B .* Divq_D .* K_d .* dt .* eta_phi .* phi + B .* Divq_D .* K_d .* dt .* eta_phi + B .* K_d .* Pf0 .* alpha .* dt + B .* K_d .* Pt0 .* alpha .* dt - B .* K_d .* Pt0 .* dt + B .* K_d .* alpha .* dt .* eta_phi .* lamda .* sin_psi - B .* K_d .* dt .* eta_phi .* lamda .* sin_psi - B .* Pf0 .* alpha .^ 2 .* eta_phi .* phi + B .* Pf0 .* alpha .^ 2 .* eta_phi - K_d .* Pf0 .* alpha .* dt + Pf0 .* alpha .* eta_phi .* phi - Pf0 .* alpha .* eta_phi) ./ (2 * B .* K_d .* alpha .* dt - B .* K_d .* dt - B .* alpha .^ 2 .* eta_phi .* phi + B .* alpha .^ 2 .* eta_phi - K_d .* alpha .* dt + alpha .* eta_phi .* phi - alpha .* eta_phi)
end

function TrialPt(Pt, DivV, Divq_D, Pt0, Pf0, dt, B, K_d, alpha, eta_phi, phi)
    @. Pt = (B .* DivV .* K_d .^ 2 .* dt .^ 2 + B .* Divq_D .* K_d .^ 2 .* dt .^ 2 - B .* Divq_D .* K_d .* alpha .* dt .* eta_phi .* phi + B .* Divq_D .* K_d .* alpha .* dt .* eta_phi + B .* K_d .* Pf0 .* alpha .* dt + B .* K_d .* Pt0 .* alpha .* dt - B .* K_d .* Pt0 .* dt - B .* Pt0 .* alpha .^ 2 .* eta_phi .* phi + B .* Pt0 .* alpha .^ 2 .* eta_phi - DivV .* K_d .* alpha .* dt .* eta_phi .* phi + DivV .* K_d .* alpha .* dt .* eta_phi - K_d .* Pf0 .* alpha .* dt + Pt0 .* alpha .* eta_phi .* phi - Pt0 .* alpha .* eta_phi) ./ (2 * B .* K_d .* alpha .* dt - B .* K_d .* dt - B .* alpha .^ 2 .* eta_phi .* phi + B .* alpha .^ 2 .* eta_phi - K_d .* alpha .* dt + alpha .* eta_phi .* phi - alpha .* eta_phi)
end

function TrialPf(Pf, DivV, Divq_D, Pt0, Pf0, dt, B, K_d, alpha, eta_phi, phi)
    @. Pf = (B .* DivV .* K_d .^ 2 .* dt .^ 2 - B .* DivV .* K_d .* alpha .* dt .* eta_phi .* phi + B .* DivV .* K_d .* alpha .* dt .* eta_phi + B .* Divq_D .* K_d .^ 2 .* dt .^ 2 - B .* Divq_D .* K_d .* dt .* eta_phi .* phi + B .* Divq_D .* K_d .* dt .* eta_phi + B .* K_d .* Pf0 .* alpha .* dt + B .* K_d .* Pt0 .* alpha .* dt - B .* K_d .* Pt0 .* dt - B .* Pf0 .* alpha .^ 2 .* eta_phi .* phi + B .* Pf0 .* alpha .^ 2 .* eta_phi - K_d .* Pf0 .* alpha .* dt + Pf0 .* alpha .* eta_phi .* phi - Pf0 .* alpha .* eta_phi) ./ (2 * B .* K_d .* alpha .* dt - B .* K_d .* dt - B .* alpha .^ 2 .* eta_phi .* phi + B .* alpha .^ 2 .* eta_phi - K_d .* alpha .* dt + alpha .* eta_phi .* phi - alpha .* eta_phi)
end

function PlasticDilationFactor(mt, mf, dt, B, K_d, alpha, eta_phi, phi)
    @. mt .= K_d .* alpha .* dt .* eta_phi .* (B - 1) ./ (-2 * B .* K_d .* alpha .* dt + B .* K_d .* dt + B .* alpha .^ 2 .* eta_phi .* phi - B .* alpha .^ 2 .* eta_phi + K_d .* alpha .* dt - alpha .* eta_phi .* phi + alpha .* eta_phi)
    @. mf .= B .* K_d .* dt .* eta_phi .* (1 - alpha) ./ (-2 * B .* K_d .* alpha .* dt + B .* K_d .* dt + B .* alpha .^ 2 .* eta_phi .* phi - B .* alpha .^ 2 .* eta_phi + K_d .* alpha .* dt - alpha .* eta_phi .* phi + alpha .* eta_phi)
end

function CoeffPtTrial(dPtdPt0, dPtdPf0, dPtddivVs, dPtddivqD, dt, B, K_d, alpha, eta_phi, phi)
    @. dPtdPt0   .= (B .* K_d .* alpha .* dt - B .* K_d .* dt - B .* alpha .^ 2 .* eta_phi .* phi + B .* alpha .^ 2 .* eta_phi + alpha .* eta_phi .* phi - alpha .* eta_phi) ./ (2 * B .* K_d .* alpha .* dt - B .* K_d .* dt - B .* alpha .^ 2 .* eta_phi .* phi + B .* alpha .^ 2 .* eta_phi - K_d .* alpha .* dt + alpha .* eta_phi .* phi - alpha .* eta_phi)
    @. dPtdPf0   .= (B .* K_d .* alpha .* dt - K_d .* alpha .* dt) ./ (2 * B .* K_d .* alpha .* dt - B .* K_d .* dt - B .* alpha .^ 2 .* eta_phi .* phi + B .* alpha .^ 2 .* eta_phi - K_d .* alpha .* dt + alpha .* eta_phi .* phi - alpha .* eta_phi)
    @. dPtddivVs .= (B .* K_d .^ 2 .* dt .^ 2 - K_d .* alpha .* dt .* eta_phi .* phi + K_d .* alpha .* dt .* eta_phi) ./ (2 * B .* K_d .* alpha .* dt - B .* K_d .* dt - B .* alpha .^ 2 .* eta_phi .* phi + B .* alpha .^ 2 .* eta_phi - K_d .* alpha .* dt + alpha .* eta_phi .* phi - alpha .* eta_phi)
    @. dPtddivqD .= (B .* K_d .^ 2 .* dt .^ 2 - B .* K_d .* alpha .* dt .* eta_phi .* phi + B .* K_d .* alpha .* dt .* eta_phi) ./ (2 * B .* K_d .* alpha .* dt - B .* K_d .* dt - B .* alpha .^ 2 .* eta_phi .* phi + B .* alpha .^ 2 .* eta_phi - K_d .* alpha .* dt + alpha .* eta_phi .* phi - alpha .* eta_phi)
end

function CoeffPfTrial(dPfdPt0, dPfdPf0, dPfddivVs, dPfddivqD, dt, B, K_d, alpha, eta_phi, phi)
    @. dPfdPt0   .= (B .* K_d .* alpha .* dt - B .* K_d .* dt) ./ (2 * B .* K_d .* alpha .* dt - B .* K_d .* dt - B .* alpha .^ 2 .* eta_phi .* phi + B .* alpha .^ 2 .* eta_phi - K_d .* alpha .* dt + alpha .* eta_phi .* phi - alpha .* eta_phi)
    @. dPfdPf0   .= (B .* K_d .* alpha .* dt - B .* alpha .^ 2 .* eta_phi .* phi + B .* alpha .^ 2 .* eta_phi - K_d .* alpha .* dt + alpha .* eta_phi .* phi - alpha .* eta_phi) ./ (2 * B .* K_d .* alpha .* dt - B .* K_d .* dt - B .* alpha .^ 2 .* eta_phi .* phi + B .* alpha .^ 2 .* eta_phi - K_d .* alpha .* dt + alpha .* eta_phi .* phi - alpha .* eta_phi)
    @. dPfddivVs .= (B .* K_d .^ 2 .* dt .^ 2 - B .* K_d .* alpha .* dt .* eta_phi .* phi + B .* K_d .* alpha .* dt .* eta_phi) ./ (2 * B .* K_d .* alpha .* dt - B .* K_d .* dt - B .* alpha .^ 2 .* eta_phi .* phi + B .* alpha .^ 2 .* eta_phi - K_d .* alpha .* dt + alpha .* eta_phi .* phi - alpha .* eta_phi)
    @. dPfddivqD .= (B .* K_d .^ 2 .* dt .^ 2 - B .* K_d .* dt .* eta_phi .* phi + B .* K_d .* dt .* eta_phi) ./ (2 * B .* K_d .* alpha .* dt - B .* K_d .* dt - B .* alpha .^ 2 .* eta_phi .* phi + B .* alpha .^ 2 .* eta_phi - K_d .* alpha .* dt + alpha .* eta_phi .* phi - alpha .* eta_phi)
end

function EffectiveVolumetricModuli!(dPtd∇Vs, dPtd∇qD, dPfd∇Vs, dPfd∇qD, dt, ϕ, ηϕ, β_dr, K_BW, K_Sk)
    dPtd∇Vs .= dt.*((1.0.- ϕ).*ηϕ) .*(.-K_BW.*β_dr.*((1.0.- ϕ).*ηϕ)  .- K_Sk.*dt)./(.-K_Sk.*(K_BW.*β_dr.*((1.0.- ϕ).*ηϕ)  .+ dt).^2 .+ (β_dr.*((1.0.- ϕ).*ηϕ)  .+ dt).*(K_BW.*β_dr.*((1.0.- ϕ).*ηϕ)  .+ K_Sk.*dt))
    dPtd∇qD .= K_Sk.*dt.*((1.0.- ϕ).*ηϕ) .*(K_BW.*β_dr.*((1.0.- ϕ).*ηϕ)  .+ dt)./(K_Sk.*(K_BW.*β_dr.*((1.0.- ϕ).*ηϕ)  .+ dt).^2 .- (β_dr.*((1.0.- ϕ).*ηϕ)  .+ dt).*(K_BW.*β_dr.*((1.0.- ϕ).*ηϕ)  .+ K_Sk.*dt))
    dPfd∇Vs .= K_Sk.*dt.*((1.0.- ϕ).*ηϕ) .*(K_BW.*β_dr.*((1.0.- ϕ).*ηϕ)  .+ dt)./(K_Sk.*(K_BW.*β_dr.*((1.0.- ϕ).*ηϕ)  .+ dt).^2 .- (β_dr.*((1.0.- ϕ).*ηϕ)  .+ dt).*(K_BW.*β_dr.*((1.0.- ϕ).*ηϕ)  .+ K_Sk.*dt))
    dPfd∇qD .= K_Sk.*dt.*((1.0.- ϕ).*ηϕ) .*(β_dr.*((1.0.- ϕ).*ηϕ)  .+ dt)./(K_Sk.*(K_BW.*β_dr.*((1.0.- ϕ).*ηϕ)  .+ dt).^2 .- (β_dr.*((1.0.- ϕ).*ηϕ)  .+ dt).*(K_BW.*β_dr.*((1.0.- ϕ).*ηϕ)  .+ K_Sk.*dt))  
    return nothing
end

# 2D Stokes routine
@views function Stokes2D_vep()
    Dat = Float64  # Precision (double=Float64 or single=Float32)

    yesPf = 1.0

    # Unit system
    CharDim    = SI_units(length=1000m, temperature=1000C, stress=1e7Pa, viscosity=1e20Pas)
    do_DP   = true               # do_DP=false: Von Mises, do_DP=true: Drucker-Prager (friction angle)
    η_vp    = nondimensionalize(5e18Pa*s, CharDim) # regularisation "viscosity"
    # Physics
    Lx, Ly  = nondimensionalize(1e4m, CharDim), nondimensionalize(1e4m, CharDim) # domain size
    radi    = nondimensionalize(1e3m, CharDim)                                   # inclusion radius
    τ_y     = nondimensionalize(1e7Pa, CharDim)                                  # yield stress. If do_DP=true, τ_y stand for the cohesion: c*cos(ϕ)
    sinϕ    = sind(35)*do_DP                                                     # sinus of the friction angle
    sinψ    = sind(10)*do_DP                                                     # sinus of the dilation angle
    ηi      = nondimensionalize(1e25Pa*s, CharDim)                               # viscous viscosity
    ηf      = nondimensionalize(1e10Pa*s, CharDim)                               # fluid viscosity
    Gi      = nondimensionalize(4e10Pa, CharDim)                                 # elastic shear modulus
    Ginc    = Gi/3#(6.0-4.0*do_DP)                                                 # elastic shear modulus perturbation
    Ks      = nondimensionalize(18e10Pa, CharDim)                                 # solid compressibility
    Kf      = nondimensionalize(1e10Pa, CharDim)                                 # fluid compressibility
    αϕ      = 1.0
    ϕi      = 0.01
    kϕ      = nondimensionalize(1e-16m^2, CharDim)                               # permeability
    np      = 1.0
    ρf      = nondimensionalize(1000kg/m^3, CharDim)
    εbg     = nondimensionalize(1e-14s^-1, CharDim)                               # background strain-rate
    g       = [nondimensionalize(0m/s^2, CharDim); nondimensionalize(0m/s^2, CharDim)]
    # Numerics
    nt      = 50                 # number of time steps
    nx, ny  = 63, 63             # numerical grid resolution
    Vdmp    = 4.0                # convergence acceleration (damping)
    Vsc     = 2.0                # iterative time step limiter
    Ptsc    = 3.0                # iterative time step limiter
    ε       = 1e-10              # nonlinear tolerence
    iterMax = 19500#3e4           # max number of iters
    nout    = 500                # check frequency
    rel     = 0.05
    # Preprocessing
    dx, dy  = Lx/nx, Ly/ny
    dt      = ηi/Gi/80000.0/4   
    # Array initialisation
    ∇qD     = zeros(Dat, nx  ,ny  )
    qDx     = zeros(Dat, nx+1,ny  )
    qDy     = zeros(Dat, nx  ,ny+1)
    kfx     = zeros(Dat, nx+1,ny  ) # k on Vx points
    kfy     = zeros(Dat, nx  ,ny+1) # k on Vy points
    fx      =  ones(Dat, nx+1,ny  ) # k on Vx points
    fy      =  ones(Dat, nx  ,ny+1) # k on Vy points
    kfc     = zeros(Dat, nx  ,ny  )
    B       = zeros(Dat, nx  ,ny  )
    α       = zeros(Dat, nx  ,ny  )
    Kd      = zeros(Dat, nx  ,ny  )
    Kϕ      = zeros(Dat, nx  ,ny  )
    mt      = zeros(Dat, nx  ,ny  )
    mf      = zeros(Dat, nx  ,ny  )
    ηϕ     = zeros(Dat, nx  ,ny  )
    ln1mϕ   = log.(1.0.-ϕi.*ones(Dat, nx  ,ny  ))
    ln1mϕ0  = copy(ln1mϕ)
    ϕ       = ϕi*ones(Dat, nx  ,ny  )
    ϕ0      = ϕi*ones(Dat, nx  ,ny  )
    ϕ1      = ϕi*ones(Dat, nx  ,ny  )
    ϕex     = ϕi*ones(Dat, nx+2,ny+2)
    Pt      = zeros(Dat, nx  ,ny  )
    Pf      = zeros(Dat, nx  ,ny  )
    Pt1     = zeros(Dat, nx  ,ny  ) # include plastic correction
    Pf1     = zeros(Dat, nx  ,ny  ) # include plastic correction
    Pe      = zeros(Dat, nx  ,ny  )
    Pfex    = zeros(Dat, nx+2,ny+2)
    Pt0     = zeros(Dat, nx  ,ny  )
    Pf0     = zeros(Dat, nx  ,ny  )
    RPf     = zeros(Dat, nx  ,ny  )
    RPt     = zeros(Dat, nx  ,ny  )
    RPf1    = zeros(Dat, nx  ,ny  )
    RPt1    = zeros(Dat, nx  ,ny  )
    Rϕ      = zeros(Dat, nx  ,ny  )
    Rϕ1     = zeros(Dat, nx  ,ny  )
    Rϕ2     = zeros(Dat, nx  ,ny  )
    ∇V      = zeros(Dat, nx  ,ny  )
    ∇Vp     = zeros(Dat, nx  ,ny  ) # plastic divergence
    Vx      = zeros(Dat, nx+1,ny  )
    Vy      = zeros(Dat, nx  ,ny+1)
    Exx     = zeros(Dat, nx  ,ny  )
    Eyy     = zeros(Dat, nx  ,ny  )
    Exyv    = zeros(Dat, nx+1,ny+1)
    Exx1    = zeros(Dat, nx  ,ny  )
    Eyy1    = zeros(Dat, nx  ,ny  )
    Exy1    = zeros(Dat, nx  ,ny  )
    Exyv1   = zeros(Dat, nx+1,ny+1)
    τxx     = zeros(Dat, nx  ,ny  )
    τyy     = zeros(Dat, nx  ,ny  )
    τxy     = zeros(Dat, nx  ,ny  )
    τxyv    = zeros(Dat, nx+1,ny+1)
    τxx0    = zeros(Dat, nx  ,ny  )
    τyy0    = zeros(Dat, nx  ,ny  )
    τxy0    = zeros(Dat, nx  ,ny  )
    τxyv0   = zeros(Dat, nx+1,ny+1)
    Tii     = zeros(Dat, nx  ,ny  )
    Eii     = zeros(Dat, nx  ,ny  )
    F       = zeros(Dat, nx  ,ny  )
    Fchk    = zeros(Dat, nx  ,ny  )
    Pla     = zeros(Dat, nx  ,ny  )
    λ       = zeros(Dat, nx  ,ny  )
    λPT     = zeros(Dat, nx  ,ny  )
    dQdτxx  = zeros(Dat, nx  ,ny  )
    dQdτyy  = zeros(Dat, nx  ,ny  )
    dQdτxy  = zeros(Dat, nx  ,ny  )
    Rx      = zeros(Dat, nx-1,ny  )
    Ry      = zeros(Dat, nx  ,ny-1)
    dPtdt   = zeros(Dat, nx  ,ny  )
    dPfdt   = zeros(Dat, nx  ,ny  )
    dϕdt    = zeros(Dat, nx  ,ny  )
    dVxdt   = zeros(Dat, nx-1,ny  )
    dVydt   = zeros(Dat, nx  ,ny-1)
    dtPt    = zeros(Dat, nx  ,ny  )
    dtPf    = zeros(Dat, nx  ,ny  )
    dtϕ     = zeros(Dat, nx  ,ny  )
    aϕ      = zeros(Dat, nx  ,ny  )
    dtVx    = zeros(Dat, nx-1,ny  )
    dtVy    = zeros(Dat, nx  ,ny-1)
    Rog     = zeros(Dat, nx  ,ny  )
    η_v     =    ηi*ones(Dat, nx, ny)
    η_e     = dt*Gi*ones(Dat, nx, ny)
    η_ev    = dt*Gi*ones(Dat, nx+1, ny+1)
    η_ve    =       ones(Dat, nx, ny)
    η_vep   =       ones(Dat, nx, ny)
    η_vepv  =       ones(Dat, nx+1, ny+1)
    # Initial condition
    xc, yc    = LinRange(dx/2, Lx-dx/2, nx), LinRange(dy/2, Ly-dy/2, ny)
    xc, yc    = LinRange(dx/2, Lx-dx/2, nx), LinRange(dy/2, Ly-dy/2, ny)
    xv, yv    = LinRange(0.0, Lx, nx+1), LinRange(0.0, Ly, ny+1)
    (Xvx,Yvx) = ([x for x=xv,y=yc], [y for x=xv,y=yc])
    (Xvy,Yvy) = ([x for x=xc,y=yv], [y for x=xc,y=yv])
    radc      = (xc.-Lx./2).^2 .+ (yc'.-Ly./2).^2
    radv      = (xv.-Lx./2).^2 .+ (yv'.-Ly./2).^2
    radx      = (xv.-Lx./2).^2 .+ (yc'.-Ly./2).^2
    rady      = (xc.-Lx./2).^2 .+ (yv'.-Ly./2).^2
    fx[radx.<radi] .= 0.1
    fy[rady.<radi] .= 0.1
    η_e[radc.<radi] .= dt*Ginc
    av!(η_ev[2:end-1,2:end-1], η_e)
    η_ev[radv.<radi].= dt*Ginc
    av!(η_e, η_ev)
    η_ve   .= (1.0./η_e + 1.0./η_v).^-1
    Vx     .=         εbg.*Xvx
    Vy     .= .-(1.0)*εbg.*Yvy
    
    @. Kϕ = ϕ*η_e/dt
    @. Kd = (1 - ϕ)*(1/Ks + 1/Kϕ)^(-1)
    @. α  = 1 - Kd/Ks
    @. B  = (1/Kd - 1/Ks) / ((1/Kd - 1/Ks) +  ϕ*(1/Kf - 1/Ks))

    ϕx   = ϕi .* ones(Dat, nx+1, ny)
    ϕy   = ϕi .* ones(Dat, nx, ny+1)
    ηx   = zeros(Dat, nx-1, ny)
    ηy   = zeros(Dat, nx, ny-1)
    Exyc = zeros(Dat, nx, ny)
    ε̇II  = zeros(Dat, nx, ny)

    # Time loop
    t = 0.0; evo_t = []; evo_τxx = [] 
    for it = 1:nt
        iter=1; err=2*ε; err_evo1=[]; err_evo2=[]
        # Previous time step
        τxx0 .= τxx; τyy0 .= τyy; τxy0 .= av(τxyv); τxyv0 .= τxyv; λ .= 0.0; λPT .= 0.0
        ln1mϕ0   .= ln1mϕ;   Pf0  .= Pf;  Pt0  .= Pt; ϕ0 .= ϕ
        @printf("it = %d\n", it) 
        @views mem=@allocated while (err>ε && iter<=iterMax)
            # @. ϕ  = 1.0 .- exp.(ln1mϕ)
            @. ηϕ .=  ηi ./ ϕ 
            @. Kϕ = Gi/100/ϕ
            @. Kd = (1 - ϕ)*(1/Ks + 1/Kϕ)^(-1)
            @. α  = 1 - Kd/Ks
            @. B  = (1/Kd - 1/Ks) / ((1/Kd - 1/Ks) +  ϕ*(1/Kf - 1/Ks))
            PlasticDilationFactor(mt, mf, dt, B, Kd, α, ηϕ, ϕ)
            @. η_v  = ηi * exp( -αϕ * ϕ)
            @. η_ve = (1.0/η_e + 1.0/η_v)^-1 
            # Assign new pressure to trial pressure
            @. Pe  = Pt - Pf
            @. Pt1 = Pt
            @. Pf1 = Pf
            # BCs
            ϕex[2:end-1,2:end-1] .= ϕ
            ϕex[1,:] .= 2ϕi.-ϕex[2,:]; ϕex[end,:] .= 2ϕi.-ϕex[end-1,:]; ϕex[:,1] .= 2ϕi.-ϕex[:,2]; ϕex[:,end] .= 2ϕi.-ϕex[:,end-1] 
            Pfex[2:end-1,2:end-1] .= Pf
            Pfex[1,:].= Pfex[2,:]; Pfex[end,:].= Pfex[end-1,:]; Pfex[:,1].= Pfex[:,2]; Pfex[:,end].= Pfex[:,end-1];
            # Porosity-permeability relationship
            av_xi!(ϕx, ϕex)
            av_yi!(ϕy, ϕex)
            kfx    .=  fx.*kϕ./ηf.*(  ϕx./ϕi).^np ./ ((1.0 .- ϕx ) ./ (1.0.-ϕi)) .^(np.-1.0)
            kfy    .=  fy.*kϕ./ηf.*(  ϕy./ϕi).^np ./ ((1.0 .- ϕy ) ./ (1.0.-ϕi)) .^(np.-1.0)
            kfc    .=  0.25.*( kfx[1:end-1,:] .+ kfx[2:end-0,:] .+ kfy[:,1:end-1] .+ kfy[:,2:end-0] )
            # Darcy flux divergence
            @. qDx    = -kfx * ((Pfex[2:end-0,2:end-1] - Pfex[1:end-1,2:end-1])/dx - ρf*g[1])
            @. qDy    = -kfy * ((Pfex[2:end-1,2:end-0] - Pfex[2:end-1,1:end-1])/dy - ρf*g[2])
            @. ∇qD    = (qDx[2:end-0,:] - qDx[1:end-1,:])/dx .+ (qDy[:,2:end-0] - qDy[:,1:end-1])/dy
            # Solid velocity divergence - pressure
            @. ∇V     = (Vx[2:end-0,:] - Vx[1:end-1,:])/dx .+ (Vy[:,2:end-0] - Vy[:,1:end-1])/dy
            # Strain rates
            @. Exx    = (Vx[2:end-0,:] - Vx[1:end-1,:])/dx .- 1.0/3.0*∇V
            @. Eyy    = (Vy[:,2:end-0] - Vy[:,1:end-1])/dy .- 1.0/3.0*∇V
            @. Exyv[2:end-1,2:end-1] = 0.5.*((Vx[2:end-1,2:end-0] - Vx[2:end-1,1:end-1])/dy .+ (Vy[2:end,2:end-1] - Vy[1:end-1,2:end-1])/dx)
            # Visco-elastic strain rates
            Exx1   .=    Exx   .+ τxx0 ./2.0./η_e
            Eyy1   .=    Eyy   .+ τyy0 ./2.0./η_e
            Exyv1  .=    Exyv  .+ τxyv0./2.0./η_ev
            av!(Exyc, Exyv)
            Exy1   .= Exyc .+ τxy0 ./2.0./η_e
            Eii    .= sqrt.(0.5.*(Exx1.^2 .+ Eyy1.^2 .+ (Exx1.+Eyy1).^2) .+ Exy1.^2)
            # # Trial stress
            τxx    .= (1.0.-ϕ).*2.0.*η_ve.*Exx1
            τyy    .= (1.0.-ϕ).*2.0.*η_ve.*Eyy1
            τxy    .= (1.0.-ϕ).*2.0.*η_ve.*Exy1
            Tii    .= sqrt.(0.5.*(τxx.^2 .+ τyy.^2 .+ (τxx.+τyy).^2) .+ τxy.^2)
            # Yield function
            @. F     = Tii - τ_y - (Pt - yesPf*Pf)*sinϕ
            @. Pla    .= 0.0
            @. Pla    .= F .> 0.0
            @. λ      .= Pla.*F./( (1.0.-ϕ).*η_ve .+ η_vp .+ (mt .- yesPf*mf)*sinϕ*sinψ)
            @. λPT    .= rel.*λ .+ (1.0.-rel).*λPT
            # @.  λ      .+= Pla.*F./η_ve/1200
            @. ∇Vp    .= λPT.*sinψ 
            @. dQdτxx .= 0.5.*τxx./Tii
            @. dQdτyy .= 0.5.*τyy./Tii
            @. dQdτxy .=      τxy./Tii
            # Plastic corrections
            @. Pt1    .= Pt + ∇Vp .* mt
            @. Pf1    .= Pf + ∇Vp .* mf
            @. τxx    .= (1.0.-ϕ).*2.0.*η_ve.*(Exx1 .-      λPT.*dQdτxx) 
            @. τyy    .= (1.0.-ϕ).*2.0.*η_ve.*(Eyy1 .-      λPT.*dQdτyy) 
            @. τxy    .= (1.0.-ϕ).*2.0.*η_ve.*(Exy1 .- 0.5.*λPT.*dQdτxy) 
            @. Tii    .= sqrt.(0.5*(τxx.^2 .+ τyy.^2 .+ (τxx.+τyy).^2) .+ τxy.^2)
            @. Fchk   .= Tii .- τ_y .- (Pt1 .- yesPf*Pf1).*sinϕ .- λPT.*η_vp
            @. η_vep  .= Tii./2.0./Eii
            # τxyv[2:end-1,2:end-1] .= av(τxy)
            av!(η_vepv[2:end-1,2:end-1], η_vep); η_vepv[1,:].=η_vepv[2,:]; η_vepv[end,:].=η_vepv[end-1,:]; η_vepv[:,1].=η_vepv[:,2]; η_vepv[:,end].=η_vepv[:,end-1]
            @. τxyv   .= 2.0.*η_vepv.*Exyv1
            # PT timestep
            av_xa!(ηx,  η_vep)
            av_ya!(ηy,  η_vep)
            @. dtVx   .= min(dx,dy)^2.0./ηx./4.1./Vsc
            @. dtVy   .= min(dx,dy)^2.0./ηy./4.1./Vsc
            @. dtPt   .= 4.1.*η_vep./max(nx,ny)./Ptsc
            @. dtPf   .= min(dx,dy).^2.0./kfc/4.1 #/Ptsc
            @. dtϕ    .= 2.1.*η_ve./max(nx,ny)#/Ptsc
            # Residuals
            @. aϕ     = ( (Pt-Pf)/ηϕ + ((Pt-Pt0)-(Pf-Pf0))/dt/Kϕ )/(1-ϕ) 
            # aϕ     .= ((Pt1.-Pf1)./ηϕ .+ ((Pt1.-Pt0).-(Pf1.-Pf0))/dt./Kϕ)./(1.0.-ϕ) .+ ∇Vp

            @. Rϕ     = aϕ - (ln1mϕ - ln1mϕ0)/dt
            @. Rϕ1    = (ϕ - ϕ0)/dt - (((Pf -Pf0)-(Pt -Pt0))/dt/Kϕ + (Pf -Pt )/ηϕ)
            @. Rϕ2    = (ϕ - ϕ0)/dt - (((Pf1-Pf0)-(Pt1-Pt0))/dt/Kϕ + (Pf1-Pt1)/ηϕ + ∇Vp)

            @. RPt    = -( ∇V  + (Pt - Pf )/ηϕ/(1-ϕ) + 1.0/Kd*((Pt -Pt0)/dt - α    *(Pf -Pf0)/dt) )
            @. RPf    = -( ∇qD - (Pt - Pf )/ηϕ/(1-ϕ) - α  /Kd*((Pt -Pt0)/dt - 1.0/B*(Pf -Pf0)/dt) )
            @. RPt1   = -( ∇V  + (Pt1- Pf1)/ηϕ/(1-ϕ) + 1.0/Kd*((Pt1-Pt0)/dt - α    *(Pf1-Pf0)/dt) + λ*sinψ/(1-ϕ) )
            @. RPf1   = -( ∇qD - (Pt1- Pf1)/ηϕ/(1-ϕ) - α  /Kd*((Pt1-Pt0)/dt - 1.0/B*(Pf1-Pf0)/dt) - λ*sinψ/(1-ϕ) )
            @. Rx     = -(Pt1[2:end,:] - Pt1[1:end-1,:])/dx + (τxx[2:end,:] - τxx[1:end-1,:])/dx .+ (τxyv[2:end-1,2:end] - τxyv[2:end-1,1:end-1])./dy
            @. Ry     = -(Pt1[:,2:end] - Pt1[:,1:end-1])/dy + (τyy[:,2:end] - τyy[:,1:end-1])/dy .+ (τxyv[2:end,2:end-1] - τxyv[1:end-1,2:end-1])./dx# .+ av_ya(Rog)
            # Updates rates
            @. dVxdt         = dVxdt*(1-Vdmp/nx) + Rx
            @. dVydt         = dVydt*(1-Vdmp/ny) + Ry
            @. dPfdt         = dPfdt*(1-Vdmp/ny) + RPf
            @. dPtdt         = RPt # no damping-pong on Pt
            @. dϕdt          = Rϕ  # no damping-pong on phi
            # Updates solutions
            @. Vx[2:end-1,:] = Vx[2:end-1,:] + dtVx*dVxdt / 20   
            @. Vy[:,2:end-1] = Vy[:,2:end-1] + dtVy*dVydt / 20  
            @. Pt            = Pt            + dtPt*dPtdt / 1
            @. Pf            = Pf            + dtPf*dPfdt / 1e11
            @. ln1mϕ         = ln1mϕ         + dtϕ *dϕdt  / 100
            @. ϕ             = ϕ             - dtϕ *Rϕ2   / 100
            # @. ϕ  = 1.0 - exp.(ln1mϕ)

            # convergence check
            if mod(iter, nout)==0 || iter==1
                norm_Rx = norm(Rx)/sqrt(length(Rx)); norm_Ry = norm(Ry)/sqrt(length(Ry)); norm_RPt = norm(RPt)/sqrt(length(RPt)); norm_RPt1 = norm(RPt1)/sqrt(length(RPt)); norm_RPf = norm(RPf)/sqrt(length(RPf)); norm_RPf1 = norm(RPf1)/sqrt(length(RPf)); norm_Rϕ = norm(Rϕ)/sqrt(length(Rϕ))
                norm_Rϕ1 = norm(Rϕ1)/sqrt(length(Rϕ))
                norm_Rϕ2 = norm(Rϕ2)/sqrt(length(Rϕ))
                err = maximum([norm_Rx, norm_Ry, norm_RPt, norm_RPf, norm_Rϕ])
                # push!(err_evo1, err); push!(err_evo2, itg)
                if isnan(err) error("Nans!") end
                @printf("iter = %05d, err = %1.2e norm[Rx=%1.2e, Ry=%1.2e, RPt=%1.2e, RPt1=%1.2e, RPf=%1.2e, RPf1=%1.2e, Rϕ=%1.2e, Rϕ1=%1.2e, Rϕ2=%1.2e] (Fchk=%1.2e) \n", iter, err, norm_Rx, norm_Ry, norm_RPt, norm_RPt1, norm_RPf, norm_RPf1, norm_Rϕ, norm_Rϕ1, norm_Rϕ2, maximum(Fchk))
            end
            iter+=1; #itg=iter
        end
        @show mem

        # RPt    .= .-( ∇V  .+ (Pt.-Pf)./ηϕ./(1.0.-ϕ) .+ 1.0./Kd.*(  (Pt.-Pt0)./dt .- α     .*(Pf.-Pf0)./dt ) )
        # RPf    .= .-( ∇qD .- (Pt.-Pf)./ηϕ./(1.0.-ϕ) .- α  ./Kd.*(  (Pt.-Pt0)./dt .- 1.0./B.*(Pf.-Pf0)./dt ) )

        # @info "Checks!"
        # # Test trial pressure
        # dPtdPt0    = zero(Pt)
        # dPtdPf0    = zero(Pt)
        # dPtddivVs  = zero(Pt)
        # dPtddivqD  = zero(Pt)
        # Pt_trial   = zero(Pt)
        # Pt_trial2  = zero(Pt)
        # Pt_corr    = zero(Pt)
        # CoeffPtTrial(dPtdPt0, dPtdPf0, dPtddivVs, dPtddivqD, dt, B, Kd, α, ηϕ, ϕ)
        # @. Pt_trial = dPtdPt0*Pt0 +  dPtdPf0*Pf0 + dPtddivVs*∇V + dPtddivqD*∇qD
        # TrialPt(Pt_trial2, ∇V, ∇qD, Pt0, Pf0, dt, B, Kd, α, ηϕ, ϕ)
        # CorrPt(Pt_corr, ∇V, ∇qD, Pt0, Pf0, dt, B, Kd, α, ηϕ, ϕ, λ, sinψ)
        # println( norm(Pt_trial .- Pt)/length(Pt1) )
        # println( norm(Pt_corr .- Pt1)/length(Pt1) )
        # println( (Pt[1] - Pt_trial[1])/Pt[1]*100, ' ', Pt[1], ' ', Pt_trial[1], ' ', Pt_trial[2] )

        # dPfdPt0    = zero(Pt)
        # dPfdPf0    = zero(Pt)
        # dPfddivVs  = zero(Pt)
        # dPfddivqD  = zero(Pt)
        # Pf_trial   = zero(Pt)
        # Pf_trial2  = zero(Pt)
        # Pf_corr    = zero(Pt)
        # CoeffPfTrial(dPfdPt0, dPfdPf0, dPfddivVs, dPfddivqD, dt, B, Kd, α, ηϕ, ϕ)
        # @. Pf_trial = dPfdPt0*Pt0 +  dPfdPf0*Pf0 + dPfddivVs*∇V + dPfddivqD*∇qD
        # TrialPf(Pf_trial2, ∇V, ∇qD, Pt0, Pf0, dt, B, Kd, α, ηϕ, ϕ)
        # CorrPf(Pf_corr, ∇V, ∇qD, Pt0, Pf0, dt, B, Kd, α, ηϕ, ϕ, λ, sinψ)
        # println( norm(Pf_trial .- Pf)/length(Pf) )
        # println( norm(Pf_corr .- Pf1)/length(Pf1) )
        # println( (Pf[1] - Pf_trial[1])/Pf[1]*100, ' ', Pf[1], ' ', Pf_trial[1], ' ', Pf_trial2[1] )

        @. ε̇II = sqrt( 0.5*Exx.^2 + 0.5*Eyy.^2 + 0.5*(Exx+Eyy)^2 + Exyc.^2)
        t = t + dt
        push!(evo_t, t); push!(evo_τxx, maximum(τxx))
        # Plotting
        p1 = heatmap(ustrip.(dimensionalize(xc, m, CharDim))./1e3, ustrip.(dimensionalize(yc, m, CharDim))./1e3, ustrip.(dimensionalize(Pt', Pa, CharDim))./1e6, aspect_ratio=1, c=:inferno, title="Pt", xlim=(0,10))
        p2 = heatmap(ustrip.(dimensionalize(xc, m, CharDim))./1e3, ustrip.(dimensionalize(yc, m, CharDim))./1e3, ustrip.(dimensionalize(Pf', Pa, CharDim))./1e6, aspect_ratio=1, c=:inferno, title="Pf", xlim=(0,10))
        p3 = heatmap(ustrip.(dimensionalize(xc, m, CharDim))./1e3, ustrip.(dimensionalize(yc, m, CharDim))./1e3, ustrip.(dimensionalize(Pe', Pa, CharDim))./1e6, aspect_ratio=1, c=:inferno, title="Pe", xlim=(0,10))
        # p4 = heatmap(ustrip.(dimensionalize(xc, m, CharDim))./1e3, ustrip.(dimensionalize(yc, m, CharDim))./1e3,                                           α', aspect_ratio=1, c=:inferno, title="α")
        # p4 = heatmap(xc, yc, Tii' , aspect_ratio=1, xlims=(dx/2, Lx-dx/2), ylims=(0, Ly), c=:inferno, title="τii")
        p4 = heatmap(ustrip.(dimensionalize(xc, m, CharDim))./1e3, ustrip.(dimensionalize(yc, m, CharDim))./1e3,                                           Pla', aspect_ratio=1, c=:inferno, title="Plastic", xlim=(0,10))
        p5 = heatmap(ustrip.(dimensionalize(xc, m, CharDim))./1e3, ustrip.(dimensionalize(yc, m, CharDim))./1e3,                                           ϕ',  aspect_ratio=1, c=:inferno, title="ϕ", xlim=(0,10))
        p6 = heatmap(ustrip.(dimensionalize(xc, m, CharDim))./1e3, ustrip.(dimensionalize(yc, m, CharDim))./1e3, log10.(ustrip.(dimensionalize(ε̇II', s^-1, CharDim))), aspect_ratio=1, c=:inferno, title="ε̇II", xlim=(0,10))

        display(plot(p1, p2, p3, p4, p5, p6, layout=(3,2)) )

        Pt .= Pt1
        Pf .= Pf1
        # ϕ  .= ϕ1
        @show norm(mt.-mf)/length(Pt)
        @show norm( (Pt1 .- Pf1) .- ( Pe .+ λ.*sinψ.*(mt.-mf) ) )/length(Pt)
    end
    return
end

Stokes2D_vep()
