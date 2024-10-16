using GeoParams, Plots, Printf, MathTeXEngine, BenchmarkTools, Statistics
import LinearAlgebra:norm
using IncompleteLU, ILUZero, SparseArrays, LinearAlgebra

include("AssembleKuu.jl")

# Makie.update_theme!(fonts = (regular = texfont(), bold = texfont(:bold), italic = texfont(:italic)))

const cmy = 356.25*3600*24*100
const ky  = 356.25*3600*24*1e3

upper(ilu) = SparseMatrixCSC(ilu.m, ilu.n,ilu.u_colptr, ilu.u_rowval, ilu.u_nzval)
lower(ilu) = SparseMatrixCSC(ilu.m, ilu.n,ilu.l_colptr, ilu.l_rowval, ilu.l_nzval)

```Function from T. James - https://apps.dtic.mil/sti/trecms/pdf/AD1184946.pdf```
function run_incomplete_cholesky3(A, n = size(A, 1)) 
    Anz = nonzeros(A)
    #incomplete cholesky
    @inbounds for k = 1:n
    # Get the row indices
    r1 = Int(SparseArrays.getcolptr(A)[k])
    r2 = Int(SparseArrays.getcolptr(A)[k+1]-1)
    r1 = searchsortedfirst(rowvals(A), k, r1, r2, Base.Order.Forward) # @assert r2 ≥ r1
    Anz[r1] = sqrt(Anz[r1])
    # Loop through non-zero elements and update the kth column
    for r = r1+1:r2
        i = rowvals(A)[r]
        A[i,k] = A[i,k]/A[k,k]
        A[k,i] = 0
    end
    # Loop through the remaining columns, k:n,
    #    and update them as needed
    # (that if [i,k] is non zero AND [i, j] is nonzero then we need to # update the value of [i, ]
    for r = r1+1:r2
        j = rowvals(A)[r]
        s1 = Int(SparseArrays.getcolptr(A)[j])
        s2 = Int(SparseArrays.getcolptr(A)[j+1]-1)
        s1 = searchsortedfirst(rowvals(A), j, s1, s2,
            Base.Order.Forward)
        # @assert s2 ≥ s1 
        for s = s1:s2
            i = rowvals(A)[s]
            A[i,j] = A[i,j] - A[i,k]*A[j,k] 
            if i != j
                A[j,i] = 0 end
            end 
        end
    end
    return A 
end

function DiagMechanics2Dx!( Dx, ηc, kc, ηv, Δx, Δy, b, Ncx, Ncy )
    ηW = zeros(Ncx+1, Ncy); ηW[2:end-0,:] .= ηc; ηW[1,:]   = ηW[2,:]
    ηE = zeros(Ncx+1, Ncy); ηE[1:end-1,:] .= ηc; ηE[end,:] = ηE[end-1,:]
    kW = zeros(Ncx+1, Ncy); kW[2:end-0,:] .= kc; ηW[1,:]   = kW[2,:]
    kE = zeros(Ncx+1, Ncy); kE[1:end-1,:] .= kc; ηE[end,:] = kE[end-1,:]
    ηS = zeros(Ncx+1, Ncy); ηS            .= ηv[:,1:end-1]
    ηN = zeros(Ncx+1, Ncy); ηN            .= ηv[:,2:end-0]
    @. Dx[:,2:end-1]  = (4//3*ηE/Δx^2 + 4//3*ηW/Δx^2 + ηS/Δy^2 + ηN/Δy^2 + kW/Δx^2 + kE/Δx^2)
    # @. Dx[:,2:end-1] +=  (-2//3*ηE+ ηN + kE)/Δx/Δy  + (-2//3*ηE + ηS + kE)/Δx/Δy + (ηN - 2//3*ηW + kW)/Δy/Δx + (ηS- 2//3*ηW + kW)/Δy/Δx  
end

function DiagMechanics2Dy!( Dy, ηc, kc, ηv, Δx, Δy, b, Ncx, Ncy )
    ηS = zeros(Ncx, Ncy+1); ηS[:,2:end-0] .= ηc
    ηN = zeros(Ncx, Ncy+1); ηN[:,1:end-1] .= ηc
    kS = zeros(Ncx, Ncy+1); kS[:,2:end-0] .= kc
    kN = zeros(Ncx, Ncy+1); kN[:,1:end-1] .= kc
    ηW = zeros(Ncx, Ncy+1); ηW            .= ηv[1:end-1,:]
    ηE = zeros(Ncx, Ncy+1); ηE            .= ηv[2:end-0,:]
    @. Dy[2:end-1,:]  = (4//3*ηS/Δy^2 + 4//3*ηN/Δy^2 + ηW/Δx^2 + ηE/Δx^2 + kS/Δy^2 + kN/Δy^2)
    # @. Dy[2:end-1,:] += (-2//3*ηN+ ηE + kN)/Δx/Δy  + (-2//3*ηN + ηW + kN)/Δx/Δy + (ηE - 2//3*ηS + kS)/Δy/Δx + (ηW - 2//3*ηS + kS)/Δy/Δx  
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

@views function MainStokes2D()

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
    # Set sharp viscosity on vertices
    for inc in eachindex(η_inc)
        ηv[(xv.-x_inc[inc]).^2 .+ (yv'.-y_inc[inc]).^2 .< r_inc[inc]^2 ] .= η_inc[inc]
    end
    # Set smooth viscosity on centroids
    ηc                  .= 0.25.*(ηv[1:end-1,1:end-1] .+ ηv[2:end-0,1:end-1] .+ ηv[1:end-1,2:end-0] .+ ηv[2:end-0,2:end-0])
    # Set sharp viscosity on centroid positions (temp array)
    ηc2 = η_mat*ones(Ncx+2, Ncy+2)
    for inc in eachindex(η_inc)
        ηc2[(xc.-x_inc[inc]).^2 .+ (yc'.-y_inc[inc]).^2 .< r_inc[inc]^2 ] .= η_inc[inc]
    end
    # Set smooth viscosity on vertices
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
    niter  = 1e5
    nout   = 200
    ϵ      = 1e-5
    CFL    = 0.999
    cfact  = 0.95

    PC       = :isai  # best :ilu0 and :ichol ---> same performance
    if  PC === :diag
        CFL    = 0.999
        ismaxloc = true
    elseif PC == :ichol
        CFL    = 0.999*9e3*n
    elseif PC == :ilu
        CFL    = 0.999*1e4*n
    elseif PC == :ilu0
        CFL    = 0.999*9000*n
    elseif PC == :parilu0 # works but requires too many iterations to apply PC
        CFL    = 0.999*9000*n
        nit    = 100
        tolILU = 1e-9
        rel    = 1.0
        tol_isai = 1e-4
    elseif PC == :isai 
        ismaxloc = false
        CFL    = 0.999*9000*n
        CFL    = 0.999*5000 # for ISAI lev 0
        # CFL    = 0.999*5200 # for ISAI lev 1
        # CFL    = 0.999*4950 # for ISAI diag
        # CFL    = 0.999*6800 # for Jacobi

        CFL    = 0.999*5000/2*n# for ISAI lev 0
        # CFL    = 0.999*5200/1.5*2 # for ISAI lev 1
        nit    = 50
        tolILU = 1e-9
        tol_isai = 1e-4
    end

   
    if PC == :diag
        if ismaxloc
            DiagMechanics2Dx!( Dx, ηc_maxloc, ηb, ηv_maxloc, Δx, Δy, bx, Ncx, Ncy )
            DiagMechanics2Dy!( Dy, ηc_maxloc, ηb, ηv_maxloc, Δx, Δy, by, Ncx, Ncy )
        else
            DiagMechanics2Dx!( Dx, ηc, ηb, ηv, Δx, Δy, bx, Ncx, Ncy )
            DiagMechanics2Dy!( Dy, ηc, ηb, ηv, Δx, Δy, by, Ncx, Ncy )
        end
    end

    # Direct solution
    VxDir     = zeros(Ncx+1, Ncy+2); VxDir .= ε̇bg.*xv .+   0*yc'
    VyDir     = zeros(Ncx+2, Ncy+1); VyDir .=    0*xc .- ε̇bg*yv'
    NumVx     = reshape(1:(Ncx+1)*Ncy, Ncx+1, Ncy)
    NumVy     = reshape(NumVx[end].+ (1:(Ncy+1)*Ncx), Ncx, Ncy+1)
    
    # Assemble Kuu
    Kuu, kxx, kyy = KuuBlock(ηc, ηb, ηv, Δx, Δy, NumVx, NumVy)
    Kuu_maxloc, kxx_maxloc, kyy_maxloc = KuuBlock(ηc_maxloc, ηb, ηv_maxloc, Δx, Δy, NumVx, NumVy)

    
    # spy(Kuu)
    Kc = cholesky(Kuu)

    ResidualMomentumX!(Rx, VxDir, VyDir, Pt, bx, ε̇xx, ε̇xy, ∇v, ηc, ηb, ηv, ones(size(Dx)), τxx, τxy, Δx, Δy, 1.0)
    ResidualMomentumY!(Ry, VxDir, VyDir, Pt, by, ε̇yy, ε̇xy, ∇v, ηc, ηb, ηv, ones(size(Dy)), τyy, τxy, Δx, Δy, 1.0)
    errVx = norm(Rx)/(length(Rx))
    errVy = norm(Ry)/(length(Ry))
    @printf("Rx = %2.4e\n", errVx)
    @printf("Ry = %2.4e\n", errVy)

    b = [Rx[:,2:end-1][:]; Ry[2:end-1,:][:]]

    V = Kc\b

    VxDir[:,2:end-1] .+= V[NumVx]
    VyDir[2:end-1,:] .+= V[NumVy]

    ResidualMomentumX!(Rx, VxDir, VyDir, Pt, bx, ε̇xx, ε̇xy, ∇v, ηc, ηb, ηv, ones(size(Dx)), τxx, τxy, Δx, Δy, 1.0)
    ResidualMomentumY!(Ry, VxDir, VyDir, Pt, by, ε̇yy, ε̇xy, ∇v, ηc, ηb, ηv, ones(size(Dy)), τyy, τxy, Δx, Δy, 1.0)
    errVx = norm(Rx.*Dx)/(length(Rx))
    errVy = norm(Ry.*Dy)/(length(Ry))
    @printf("Rx = %2.4e\n", errVx)
    @printf("Ry = %2.4e\n", errVy)

    # Cholesky PC
    L = copy(Kuu)
    D = diag(L)
    @show minimum(D)
    @show maximum(D)

    # @time ichol!(L) # way to slow!
    if PC == :ichol
        α = 0*9e5
        n = size(Kuu,1)
        Kuu_shift = Kuu .+ α.*I(n)
        @time L = run_incomplete_cholesky3( Kuu_shift, size(Kuu,1))
        L .-= α.*I(n)
    elseif PC==:ilu
        @time LU = ilu(Kuu, τ = 5000.0)
        @show nnz(Kuu)
        @show nnz(LU)
    elseif PC==:ilu0
        @time LU = ilu0(Kuu)
        @show typeof(Kuu)
        @show nnz(Kuu)
        @show nnz(LU)
    elseif PC==:parilu0
        pxx    = ILU_v3(kxx, 500, tolILU)
        pyy    = ILU_v3(kyy, 500, tolILU)
        # @time LU = ilu0(Kuu)
        # L = lower(LU)
        # U = upper(LU)
        # Luu, Uuu = AssembleLU(pxx, pyy, NumVx, NumVy)

        # @show typeof(L)
        # @show typeof(Luu)
        # @show norm(Luu-L)/norm(Luu)

        # @show typeof(U)
        # @show typeof(Uuu)
        # @show norm(Uuu-U)/norm(Uuu)
    elseif PC==:isai
        pxx    = ILU_v3(kxx, 50, tolILU)
        pyy    = ILU_v3(kyy, 50, tolILU)
        ipxx   = ISAI_Poisson2D(pxx)
        ipyy   = ISAI_Poisson2D(pyy)


        # ipxx.cC .= 1 ./ reshape(D[NumVx[:]], Ncx+1, Ncy)
        # ipxx.cS .=0 
        # ipxx.cN .=0 
        # ipxx.cW .=0 
        # ipxx.cE .=0 
        # ipyy.cC .= 1 ./ reshape(D[NumVy[:]], Ncx, Ncy+1)
        # ipyy.cS .=0 
        # ipyy.cN .=0 
        # ipyy.cW .=0 
        # ipyy.cE .=0 

        # # Assemble factors
        # @time LU = ilu0(Kuu)
        # L = lower(LU)
        # U = upper(LU)
        # Luu, Uuu = AssembleLU(pxx, pyy, NumVx, NumVy)
        # @show typeof(L)
        # @show typeof(Luu)
        # @show norm(Luu-L)/norm(Luu)
        # @show typeof(U)
        # @show typeof(Uuu)
        # @show norm(Uuu-U)/norm(Uuu)

        # # Assemble inverses
        # Luui, Uuui = AssembleLU(ipxx, ipyy, NumVx, NumVy)
        # Li_full = SparseMatrixCSC( inv(Matrix(Luu+I(size(L,1)))) )
        # Li = SparseMatrixCSC( Li_full .* (Matrix(Luu+I(size(L,1))).!=0) )
        # @show norm(Matrix(Luui) .- Li .* (Matrix(Luu).!=0))/norm(Luu)
        # Ui_full = SparseMatrixCSC( inv(Matrix(Uuu)) )
        # Ui = SparseMatrixCSC( Ui_full              .* (Matrix(Uuu).!=0) )
        # @show norm(Matrix(Uuui) .- Ui .* (Matrix(Uuu).!=0))/norm(Uuu)

        # Luu1, Uuu1 =  AssembleLU_lev1(pxx, pyy, NumVx, NumVy)

        # Li = SparseMatrixCSC( Li_full .* (Matrix(Luu1+I(size(L,1))).!=0) )
        # Ui = SparseMatrixCSC( Ui_full              .* (Matrix(Uuu1).!=0) )

        # Li = spdiagm(diag(Li))
        # Ui = spdiagm(diag(Ui))

        # # original Jacobi
        # DiagMechanics2Dx!( Dx, ηc, ηb, ηv, Δx, Δy, bx, Ncx, Ncy )
        # DiagMechanics2Dy!( Dy, ηc, ηb, ηv, Δx, Δy, by, Ncx, Ncy )
        # Ui = spdiagm([1 ./Dx[:,2:end-1][:]; 1 ./Dy[2:end-1,:][:]])
        # Dx .= 1
        # Dy .= 1

        # b .= [Vx[:,2:end-1][:]; Vy[2:end-1,:][:]]
        # x = Ui*(Li*b)
        # @show norm(x)
        # b .= [Vx[:,2:end-1][:]; Vy[2:end-1,:][:]]
        # x = Uuui*((Luui+I(size(L,1)))*b)
        # @show norm(x)
        # ApplyISAI!(Rx[:,2:end-1], Vx[:,2:end-1], ipxx)
        # ApplyISAI!(Ry[2:end-1,:], Vy[2:end-1,:], ipyy)
        # x .= [Rx[:,2:end-1][:]; Ry[2:end-1,:][:]]
        # @show norm(x)
    end

    # if PC == :diag
    #     nit    = 50
    #     tolILU = 1e-9
    #     pxx    = ILU_v3(kxx, nit, tolILU)
    #     pyy    = ILU_v3(kyy, nit, tolILU)
    #     Dx[:,2:end-1] .= pxx.cC
    #     Dy[2:end-1,:] .= pyy.cC
    # end

    # # ldiv!(V, LU, b)
    # # V = L\(L'\b)
    # # VxDir[:,2:end-1] .+= V[NumVx]
    # # VyDir[2:end-1,:] .+= V[NumVy]
    # x = zero(V)
    # # spy(Kuu)
    # PC       = false 
    ismaxloc = false

    tempx = zero(Rx)
    tempy = zero(Ry)
    Sx    = zero(Rx)
    Sy   = zero(Ry)
    Sx0  = zero(Rx)
    Sy0  = zero(Ry)
    Rx0  = zero(Rx)
    Ry0  = zero(Ry)

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
            
            if PC == :ichol
                # Apply PC
                b .= [Rx[:,2:end-1][:]; Ry[2:end-1,:][:]]
                x = L'\(L\b)
                Rx[:,2:end-1] .= reshape(x[1:NumVx[end]], Ncx+1, Ncy )
                Ry[2:end-1,:] .= reshape(x[NumVx[end]+1:end], Ncx, Ncy+1 )
            elseif PC == :ilu || PC==:ilu0
                b .= [Rx[:,2:end-1][:]; Ry[2:end-1,:][:]]
                # ldiv!(x, LU, b)
                x = LU\b
                Rx[:,2:end-1] .= reshape(x[1:NumVx[end]], Ncx+1, Ncy )
                Ry[2:end-1,:] .= reshape(x[NumVx[end]+1:end], Ncx, Ncy+1 )
            elseif PC == :parilu0
                tempx .= Rx
                tempy .= Ry
                Rx .= 0.
                Ry .= 0.
                ForwardBackwardSolve!(Rx[:,2:end-1], tempx[:,2:end-1], pxx, nit, tol_isai, rel)
                ForwardBackwardSolve!(Ry[2:end-1,:], tempy[2:end-1,:], pyy, nit, tol_isai, rel)
                # LaggedSubstitutions!(Rx[:,2:end-1], Sx[:,2:end-1], Rx0[:,2:end-1], Sx0[:,2:end-1], pxx, tempx[:,2:end-1])
                # LaggedSubstitutions!(Ry[2:end-1,:], Sy[2:end-1,:], Ry0[2:end-1,:], Sy0[2:end-1,:], pyy, tempy[2:end-1,:])
            elseif PC == :isai
                # b .= [Rx[:,2:end-1][:]; Ry[2:end-1,:][:]]
                # x = Ui*(Li*b)
                # Rx[:,2:end-1] .= reshape(x[1:NumVx[end]], Ncx+1, Ncy )
                # Ry[2:end-1,:] .= reshape(x[NumVx[end]+1:end], Ncx, Ncy+1 )

                tempx .= Rx
                tempy .= Ry
                # Rx    .= 0.
                # Ry    .= 0.
                ApplyISAI!(Rx[:,2:end-1], tempx[:,2:end-1], ipxx)
                ApplyISAI!(Ry[2:end-1,:], tempy[2:end-1,:], ipyy)
                # # # ForwardBackwardSolveISAI!(Rx[:,2:end-1], tempx[:,2:end-1], pxx, ipxx, nit, tol_isai)
                # # # ForwardBackwardSolveISAI!(Ry[2:end-1,:], tempy[2:end-1,:], pyy, ipyy, nit, tol_isai)
            end

            @. ∂Vx∂τ                 = (2-c*hVx)/(2+c*hVx)*∂Vx∂τ + 2*hVx/(2+c*hVx).*Rx
            @. δVx                   = hVx*∂Vx∂τ
            @. Vx[2:end-1,2:end-1]  += δVx[2:end-1,2:end-1]

            @. ∂Vy∂τ                 = (2-c*hVy)/(2+c*hVy)*∂Vy∂τ + 2*hVy/(2+c*hVy).*Ry
            @. δVy                   = hVy*∂Vy∂τ
            @. Vy[2:end-1,2:end-1]  += δVy[2:end-1,2:end-1]
            
            if mod(iter, nout) == 0 || iter==1

                ResidualMomentumX!(KδVx1, Vx,      Vy,      Pt, bx, ε̇xx, ε̇xy, ∇v, ηc, ηb, ηv, Dx, τxx, τxy, Δx, Δy, 0.0) # this allocates because of Vx.-δVx
                ResidualMomentumX!(KδVx,  Vx.-δVx, Vy.-δVy, Pt, bx, ε̇xx, ε̇xy, ∇v, ηc, ηb, ηv, Dx, τxx, τxy, Δx, Δy, 0.0)
                ResidualMomentumY!(KδVy1, Vx,      Vy,      Pt, by, ε̇yy, ε̇xy, ∇v, ηc, ηb, ηv, Dy, τyy, τxy, Δx, Δy, 0.0)
                ResidualMomentumY!(KδVy,  Vx.-δVx, Vy.-δVy, Pt, by, ε̇yy, ε̇xy, ∇v, ηc, ηb, ηv, Dy, τyy, τxy, Δx, Δy, 0.0)
                
                # GershgorinMechanics2Dx_Local!( λmaxlocVx, ηc, ηb, ηv, Dx, Δx, Δy, Ncx, Ncy )
                # hVx[:,2:end-1] .= 2.0./sqrt.(λmaxlocVx)*CFL
                # GershgorinMechanics2Dy_Local!( λmaxlocVy, ηc, ηb, ηv, Dy, Δx, Δy, Ncx, Ncy )
                # hVy[2:end-1,:] .= 2.0./sqrt.(λmaxlocVy)*CFL

                if PC == :ichol
                    b .= [KδVx1[:,2:end-1][:]; KδVy1[2:end-1,:][:]]
                    x = L'\(L\b)
                    KδVx1[:,2:end-1] .= reshape(x[1:NumVx[end]], Ncx+1, Ncy )
                    KδVy1[2:end-1,:] .= reshape(x[NumVx[end]+1:end], Ncx, Ncy+1 )

                    b .= [KδVx[:,2:end-1][:]; KδVy[2:end-1,:][:]]
                    x = L'\(L\b)
                    KδVx[:,2:end-1] .= reshape(x[1:NumVx[end]], Ncx+1, Ncy )
                    KδVy[2:end-1,:] .= reshape(x[NumVx[end]+1:end], Ncx, Ncy+1 )
                elseif PC == :ilu || PC==:ilu0
                    b .= [KδVx1[:,2:end-1][:]; KδVy1[2:end-1,:][:]]
                    # ldiv!(x, LU, b)
                    x = LU\b
                    KδVx1[:,2:end-1] .= reshape(x[1:NumVx[end]], Ncx+1, Ncy )
                    KδVy1[2:end-1,:] .= reshape(x[NumVx[end]+1:end], Ncx, Ncy+1 )

                    b .= [KδVx[:,2:end-1][:]; KδVy[2:end-1,:][:]]
                    # ldiv!(x, LU, b)
                    x = LU\b
                    KδVx[:,2:end-1] .= reshape(x[1:NumVx[end]], Ncx+1, Ncy )
                    KδVy[2:end-1,:] .= reshape(x[NumVx[end]+1:end], Ncx, Ncy+1 )
                elseif PC == :parilu0
                    tempx .= KδVx1
                    tempy .= KδVy1
                    ForwardBackwardSolve!(KδVx1[:,2:end-1], tempx[:,2:end-1], pxx, nit, tol_isai, rel)
                    ForwardBackwardSolve!(KδVy1[2:end-1,:], tempy[2:end-1,:], pyy, nit, tol_isai, rel)
                    tempx .= KδVx
                    tempy .= KδVy
                    ForwardBackwardSolve!(KδVx[:,2:end-1], tempx[:,2:end-1], pxx, nit, tol_isai, rel)
                    ForwardBackwardSolve!(KδVy[2:end-1,:], tempy[2:end-1,:], pyy, nit, tol_isai, rel)
                elseif PC == :isai
                    # b .= [KδVx1[:,2:end-1][:]; KδVy1[2:end-1,:][:]]
                    # x = Ui*(Li*b)
                    # KδVx1[:,2:end-1] .= reshape(x[1:NumVx[end]], Ncx+1, Ncy )
                    # KδVy1[2:end-1,:] .= reshape(x[NumVx[end]+1:end], Ncx, Ncy+1 )

                    # b .= [KδVx[:,2:end-1][:]; KδVy[2:end-1,:][:]]
                    # x = Ui*(Li*b)
                    # KδVx[:,2:end-1] .= reshape(x[1:NumVx[end]], Ncx+1, Ncy )
                    # KδVy[2:end-1,:] .= reshape(x[NumVx[end]+1:end], Ncx, Ncy+1 )

                    tempx .= KδVx1
                    tempy .= KδVy1
                    ApplyISAI!(KδVx1[:,2:end-1], tempx[:,2:end-1], ipxx)
                    ApplyISAI!(KδVy1[2:end-1,:], tempy[2:end-1,:], ipyy)
                    # ForwardBackwardSolveISAI!(KδVx1[:,2:end-1], tempx[:,2:end-1], pxx, ipxx, nit, tol_isai)
                    # ForwardBackwardSolveISAI!(KδVy1[2:end-1,:], tempy[2:end-1,:], pyy, ipyy, nit, tol_isai)
                    
                    tempx .= KδVx
                    tempy .= KδVy
                    ApplyISAI!(KδVx[:,2:end-1], tempx[:,2:end-1], ipxx)
                    ApplyISAI!(KδVy[2:end-1,:], tempy[2:end-1,:], ipyy)
                    # # ForwardBackwardSolveISAI!(KδVx[:,2:end-1], tempx[:,2:end-1], pxx, ipxx, nit, tol_isai)
                    # # ForwardBackwardSolveISAI!(KδVy[2:end-1,:], tempy[2:end-1,:], pyy, ipyy, nit, tol_isai)
                end

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
                if ( isnan(errVx) || errVx>1e6 ) error() end
                ( errVx < ϵ ) && break
            end
        end
        probes.iters[it] = iters

        @show (minimum(λmaxlocVx), maximum(λmaxlocVx))

        # Visualisation
        if mod(it, 10)==0 || it==1
            p1=heatmap(ustrip.(dimensionalize(xc[2:end-1], m, CharDim)./1e3), ustrip.(dimensionalize(yc[2:end-1], m, CharDim)./1e3), ustrip.(dimensionalize(Pt, Pa, CharDim))', title = "$(mean(probes.iters[1:it])) iterations", aspect_ratio=1.0, clims=(-3,3) )
            p2=heatmap(ustrip.(dimensionalize(xv, m, CharDim)./1e3), ustrip.(dimensionalize(yv, m, CharDim)./1e3), log10.(ustrip.(dimensionalize(ηv, Pas, CharDim)))', title = "$(mean(probes.iters[1:it])) iterations", aspect_ratio=1.0 )
            # p3=heatmap(ustrip.(dimensionalize(xv, m, CharDim)./1e3), ustrip.(dimensionalize(yc, m, CharDim)./1e3), ustrip.(dimensionalize(Vx, m*s^-1, CharDim))', title = "$(mean(probes.iters[1:it])) iterations", aspect_ratio=1.0 )
            p3=heatmap(ustrip.(dimensionalize(xc, m, CharDim)./1e3), ustrip.(dimensionalize(yv, m, CharDim)./1e3), ustrip.(dimensionalize(Vy, m*s^-1, CharDim))', title = "$(mean(probes.iters[1:it])) iterations", aspect_ratio=1.0 )
            p4=heatmap(ustrip.(dimensionalize(xc, m, CharDim)./1e3), ustrip.(dimensionalize(yv, m, CharDim)./1e3), ustrip.(dimensionalize(VyDir, m*s^-1, CharDim))', title = "$(mean(probes.iters[1:it])) iterations", aspect_ratio=1.0 )
            display(plot(p1,p2,p3,p4))
        end
    end
end

MainStokes2D()