using LinearAlgebra, SparseArrays, ILUZero

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

function BackwardSubstitution!(x, U, b)
    n = size(x,1)
    x[n] = b[n]/U[n,n] 
    for i=(n−1):−1:1
        x[i] = (b[i] − U[i,i+1:n]' * x[i+1:n]) / U[i,i] 
    end
end

function ForwardSubstitution!(x, L, b)
    n = size(x,1)
    x[1] = b[1]/L[1,1]
    for i=2:n
        x[i] = (b[i] − L[i,1:i−1]'*x[1:i−1]) / L[i,i]
    end
end

function BackwardSubstitutionLagged!(x, x0, U, b)
    n = size(x,1)
    x[n] = b[n]/U[n,n] 
    for i=(n−1):−1:1
        x[i] = (b[i] − U[i,i+1:n]'*x0[i+1:n]) / U[i,i] 
    end
end

function ForwardSubstitutionLagged!(x, x0, L, b)
    n = size(x,1)
    x[1] = b[1]/L[1,1]
    for i=2:n
        x[i] = (b[i] − L[i,1:i−1]'*x0[1:i−1]) / L[i,i]
    end
end

function ConstructM(k, ρ, Cp, Δx, Δy, Δt, transient, b, T_South, T_North)
    Ncx, Ncy = size(b, 1), size(b, 2)
    cC = transient ./ Δt .+ k.y[:,1:end-1]/ρ/Cp/Δy^2 + k.y[:,2:end-0]/ρ/Cp/Δy^2 .+ k.x[1:end-1,:]/ρ/Cp/Δx^2 .+ k.x[2:end-0,:]/ρ/Cp/Δx^2
    cS = -k.y[:,1:end-1]/ρ/Cp/Δy^2; cC[:,  1] -= cS[:,  1]; b[:,  1] .-= 2*T_South.*cS[:,  1]; cS[:,  1] .= 0.
    cN = -k.y[:,2:end-0]/ρ/Cp/Δy^2; cC[:,end] -= cN[:,end]; b[:,end] .-= 2*T_North.*cN[:,end]; cN[:,end] .= 0. 
    cW = -k.x[1:end-1,:]/ρ/Cp/Δx^2; cC[  1,:] += cW[  1,:]; cW[  1,:] .= 0. 
    cE = -k.x[2:end-0,:]/ρ/Cp/Δx^2; cC[end,:] += cE[end,:]; cE[end,:] .= 0.
    num = reshape(1:Ncx*Ncy, Ncx, Ncy)
    iC  = num
    iW  = ones(Ncx, Ncy); iW[2:end-0,:] .= num[1:end-1,:]
    iE  = ones(Ncx, Ncy); iE[1:end-1,:] .= num[2:end-0,:]
    iS  = ones(Ncx, Ncy); iS[:,2:end-0] .= num[:,1:end-1]
    iN  = ones(Ncx, Ncy); iN[:,1:end-1] .= num[:,2:end-0]
    VM   = [cC[:]; cW[:]; cE[:]; cS[:]; cN[:]]
    IM   = [iC[:]; iC[:]; iC[:]; iC[:]; iC[:]]
    JM   = [iC[:]; iW[:]; iE[:]; iS[:]; iN[:]]
    return droptol!(sparse(IM, JM, VM), 1e-13), (cC=cC, cS=cS, cW=cW, cE=cE, cN=cN)
end

function ILU_v2(coeff, nit)
    # M2Di style storage
    pC = copy(coeff.cC); pC0 = copy(coeff.cC)
    pS = copy(coeff.cS); pS0 = copy(coeff.cS)
    pN = copy(coeff.cN); pN0 = copy(coeff.cN)
    pW = copy(coeff.cW); pW0 = copy(coeff.cW)  
    pE = copy(coeff.cE); pE0 = copy(coeff.cE)

    # Iterations over coefficients
    for iter=1:nit

        # println("ILU iter $iter --- v2")
        pC0 .= pC
        pS0 .= pS
        pN0 .= pN
        pW0 .= pW
        pE0 .= pE
        
        pS[:,2:end-0] .=  1.0./pC0[:,1:end-1] .* (coeff.cS[:,2:end-0])
        pN[:,1:end-1] .=                         (coeff.cN[:,1:end-1])

        pW[2:end-0,:] .=  1.0./pC0[1:end-1,:] .* (coeff.cW[2:end-0,:])
        pE[1:end-1,:] .=                         (coeff.cE[1:end-1,:])
        
        pC .=  coeff.cC
        # Central coefficient E/W
        pC[2:end-0,:] .-= pW0[2:end-0,:].*pE0[1:end-1,:]
        # Central coefficient N/S
        pC[:,2:end-0] .-= pS0[:,2:end-0].*pN0[:,1:end-1]

        if norm(pC.-pC0)/length(pC) <1e-8
            @info "ILU iter v2 converged in $iter sweeps"
            break
        end 
    end
    return (cS=pS, cW=pW, cC=pC, cE=pE, cN=pN)
end

function ILU_v3(coeff, nit)
    # M2Di style storage
    pC = copy(coeff.cC); pC0 = copy(coeff.cC)
    pS = copy(coeff.cS);
    pN = copy(coeff.cN);
    pW = copy(coeff.cW);  
    pE = copy(coeff.cE);

    # Iterations over coefficients
    for iter=1:nit

        # println("ILU iter $iter --- v3")
        pC0 .= pC
        
        pS[:,2:end-0] .=  1.0./pC[:,1:end-1] .* (coeff.cS[:,2:end-0])
        pN[:,1:end-1] .=                         (coeff.cN[:,1:end-1])

        pW[2:end-0,:] .=  1.0./pC[1:end-1,:] .* (coeff.cW[2:end-0,:])
        pE[1:end-1,:] .=                         (coeff.cE[1:end-1,:])
        
        pC .=  coeff.cC
        # Central coefficient E/W
        pC[2:end-0,:] .-= pW[2:end-0,:].*pE[1:end-1,:]
        # Central coefficient N/S
        pC[:,2:end-0] .-= pS[:,2:end-0].*pN[:,1:end-1]

        if norm(pC.-pC0)/length(pC) <1e-8
            @info "ILU iter v3 converged in $iter sweeps"
            break
        end 
    end
    return (cS=pS, cW=pW, cC=pC, cE=pE, cN=pN)
end

function ForwardBackwardSolve!(T2D, b, pc, nit, tol, Ncx=size(T2D,1), Ncy=size(T2D,2))
    r2D = zeros(size(T2D)) # temp array
    for iter=1:nit
        for j in axes(T2D, 2), i in axes(T2D, 1) 
            if i>1   rW = r2D[i-1,j] else rW = 0. end
            if i<Ncx TE = T2D[i+1,j] else TE = 0. end
            if j>1   rS = r2D[i,j-1] else rS = 0. end
            if j<Ncy TN = T2D[i,j+1] else TN = 0. end
            r2D[i,j] +=                      -pc.cW[i,j]*rW - pc.cS[i,j]*rS +   b[i,j]  - r2D[i,j] 
            T2D[i,j] +=  (1.0/pc.cC[i,j]) * (-pc.cE[i,j]*TE - pc.cN[i,j]*TN  + r2D[i,j]) - T2D[i,j]
        end
    end
end

@views function maxloc!(A2, A)
    A2[2:end-1,2:end-1] .= max.(max.(max.(A[1:end-2,2:end-1], A[3:end,2:end-1]), A[2:end-1,2:end-1]), max.(A[2:end-1,1:end-2], A[2:end-1,3:end]))
    A2[[1,end],:] .= A2[[2,end-1],:]; A2[:,[1,end]] .= A2[:,[2,end-1]]
end

@views function minloc!(A2, A)
    A2[2:end-1,2:end-1] .= min.(min.(min.(A[1:end-2,2:end-1], A[3:end,2:end-1]), A[2:end-1,2:end-1]), min.(A[2:end-1,1:end-2], A[2:end-1,3:end]))
    A2[[1,end],:] .= A2[[2,end-1],:]; A2[:,[1,end]] .= A2[:,[2,end-1]]
end

#####################################################################################
#####################################################################################
#####################################################################################

function main()

    hetero  = true

    # ############################
    
    # Unit system
    CharDim    = SI_units(length=1000m, temperature=1000C, stress=1e7Pa, viscosity=1e20Pas)

    # Physical parameters
    Lx         = nondimensionalize(2e4m, CharDim)
    Ly         = nondimensionalize(2e4m, CharDim)
    T0         = nondimensionalize(500C, CharDim)
    ρ          = nondimensionalize(3000kg/m^3, CharDim)
    Cp         = nondimensionalize(1050J/kg/K, CharDim)
    k0         = nondimensionalize(1.5J/s/m/K, CharDim)
    Qr0        = nondimensionalize(1.e-3J/s/m^3, CharDim)
    ΔT         = nondimensionalize(20K, CharDim)
    σ          = Ly/40
    t          = 0.

    # Transient problem?
    transient  = 0

    # BCs
    # T_North    = nondimensionalize(500C, CharDim)
    # T_South    = nondimensionalize(520C, CharDim)
    T_North    = nondimensionalize(0C, CharDim)
    T_South    = nondimensionalize(0C, CharDim)

    # Numerical parameters
    Ncx        = 4
    Ncy        = 5
    Nt         = 1
    Δx         = Lx/Ncx
    Δy         = Ly/Ncy
    xc         = LinRange(-Lx/2-Δx/2, Lx/2+Δx/2, Ncx+2)
    yc         = LinRange(-Ly/2-Δy/2, Ly/2+Δy/2, Ncy+2)
    xv         = LinRange(-Lx/2, Lx/2, Ncx+1)
    yv         = LinRange(-Ly/2, Ly/2, Ncy+1)
    Δt         = nondimensionalize(2e11s, CharDim)

    # Allocate arrays
    Tc         = T0 .+ ΔT.*exp.(-(yc').^2/(2σ^2) .- xc.^2/(2σ^2) )  
    Tc0        = copy(Tc) 
    ∂T∂x       =   zeros(Ncx+1, Ncy+0)
    qTx        = T0*ones(Ncx+1, Ncy+0)
    ∂T∂y       =   zeros(Ncx+0, Ncy+1)
    qTy        = T0*ones(Ncx+0, Ncy+1)
    RT         =   zeros(Ncx+2, Ncy+2)
    Qr         =   zeros(Ncx+2, Ncy+2) .+ Qr0.*exp.(-((yc').-2.2).^2/2σ^2 .- xc.^2/(2σ^2)) 
    ∂T∂τ       =   zeros(Ncx+2, Ncy+2)
    hT         =   zeros(Ncx+2, Ncy+2) # Time step is local
    KδT1       =   zeros(Ncx+2, Ncy+2)
    δT         =   zeros(Ncx+2, Ncy+2)
    KδT        =   zeros(Ncx+2, Ncy+2)

    DM         =   ones(Ncx+2, Ncy+2)

    kv         = k0.*ones(Ncx+1, Ncy+1)
    ηi    = (s=1e4, w=1e-4) 
    x_inc = [0.0   0.2 -0.3 -0.4  0.0 -0.3 0.4  0.3  0.35 -0.1]*20 
    y_inc = [0.0   0.4  0.4 -0.3 -0.2  0.2 -0.2 -0.4 0.2  -0.4]*20
    r_inc = [0.08 0.09 0.05 0.08 0.08  0.1 0.07 0.08 0.07 0.07]*20 
    η_inc = [ηi.s ηi.w ηi.w ηi.s ηi.w ηi.s ηi.w ηi.s ηi.s ηi.w]
    for inc in eachindex(η_inc)
        kv[(xv.-x_inc[inc]).^2 .+ (yv'.-y_inc[inc]).^2 .< r_inc[inc]^2 ] .= η_inc[inc]
    end
    # 1 step smoothing
    kc                   = k0.*ones(Ncx+0, Ncy+0)
    kc                  .= 0.25.*(kv[1:end-1,1:end-1] .+ kv[2:end-0,1:end-1] .+ kv[1:end-1,2:end-0] .+ kv[2:end-0,2:end-0])
    kv[2:end-1,2:end-1] .= 0.25.*(kc[1:end-1,1:end-1] .+ kc[2:end-0,1:end-1] .+ kc[1:end-1,2:end-0] .+ kc[2:end-0,2:end-0])

    k          = (x=0.5.*(kv[:,1:end-1] .+ kv[:,2:end-0]), y=0.5.*(kv[1:end-1,:] .+ kv[2:end-0,:]))
        
    @info "Coeff min/max"
    @show (minimum(k.x), maximum(k.x))
    @show (minimum(k.y), maximum(k.y))
    # Local max preconditioning
    # PC         = :ichol
    PC         = :ilu0
    # PC         = :diag
    ismaxloc   = false
    λmaxlocT   = zeros(Ncx+0, Ncy+0)
    kv_maxloc  = zeros(Ncx+1, Ncy+1)
    maxloc!(kv_maxloc, kv)
    k_maxloc   = (x=0.5.*(kv_maxloc[:,1:end-1] .+ kv_maxloc[:,2:end-0]), y=0.5.*(kv_maxloc[1:end-1,:] .+ kv_maxloc[2:end-0,:]))
    
    kv_minloc  = zeros(Ncx+1, Ncy+1)
    minloc!(kv_minloc, kv)
    k_minloc   = (x=0.5.*(kv_minloc[:,1:end-1] .+ kv_minloc[:,2:end-0]), y=0.5.*(kv_minloc[1:end-1,:] .+ kv_minloc[2:end-0,:]))
    
    # Monitoring
    probes    = (iters = zeros(Nt), t = zeros(Nt), Ẇ0 = zeros(Nt), τxyi = zeros(Nt), Vx0 = zeros(Nt), maxT = zeros(Nt))
    
    # PT solver
    niter  = 25000
    nout   = 10
    ϵ      = 1e-11
    CFL    = 0.99
    cfact  = 0.9

    # Direct solution
    b    = zeros(Ncx,Ncy)
    b   .= Qr[2:end-1,2:end-1]/ρ/Cp
    M, coeff = ConstructM(k, ρ, Cp, Δx, Δy, Δt, transient, b, T_South, T_North)
    # Tdir = (M\b[:])

    display(M)

    @info "-------- Try ILU decompositions --------"

    # Incomplete LU zero
    # SparseMatrixCSC{Float64, Int64}
    # LU
    @time LU = ilu0(M)    

    U_aim = upper(LU)
    L_aim = lower(LU)

    display(U_aim)
    display(L_aim)

    # Incomplete LU zero - Chow & Patel, 2015 - VERSION 1
    L = spzeros(size(M))
    U = spzeros(size(M))

    # Initial guess
    for j=1:size(M,1)
        for i=1:size(M,2)
            if abs(M[i,j])>1e-13 # restriction to sparsity pattern!!!
                if i>j
                    L[i,j] = M[i,j]
                else
                    U[i,j] = M[i,j]
                end
            end
        end
    end

    # Iterations over sparse matrix
    nit = 30

    L0, U0 = copy(L), copy(U)
    for iter=1:nit
        # println("ILU iter $iter --- v1")
        U0 .= U
        L0 .= L
        for i=1:size(M,2)
            for j=1:size(M,1)
                if abs(M[i,j])>1e-13 # restriction to sparsity pattern!!!
                    if i>j 
                        L[i,j] = 1.0/U0[j,j] * (M[i,j] - L0[i,1:j-1]' * U0[1:j-1,j] )
                    else
                        U[i,j] =              (M[i,j] -  L0[i,1:i-1]' * U0[1:i-1,j] )
                    end
                end
            end
        end
    end

    # Incomplete LU zero - Chow & Patel, 2015 - VERSION 2
    pc = ILU_v2(coeff, nit)

    pc = ILU_v3(coeff, nit)

    # Construct the matix (just for checks)
    num = reshape(1:Ncx*Ncy, Ncx, Ncy)
    iC  = num
    iW  = ones(Ncx, Ncy); iW[2:end-0,:] .= num[1:end-1,:]
    iE  = ones(Ncx, Ncy); iE[1:end-1,:] .= num[2:end-0,:]
    iS  = ones(Ncx, Ncy); iS[:,2:end-0] .= num[:,1:end-1]
    iN  = ones(Ncx, Ncy); iN[:,1:end-1] .= num[:,2:end-0]
    IM   = [iC[:]; iC[:]; iC[:]; iC[:]; iC[:]]
    JM   = [iC[:]; iW[:]; iE[:]; iS[:]; iN[:]]

    # Lower
    VM   = [pc.cC[:]; 0*pc.cW[:]; pc.cE[:]; 0*pc.cS[:]; pc.cN[:]]
    U_PC   =  droptol!(sparse(IM, JM, VM), 1e-13)

    # Upper
    VM   = [0*pc.cC[:]; pc.cW[:]; 0*pc.cE[:]; pc.cS[:]; 0*pc.cN[:]]
    L_PC   =  droptol!(sparse(IM, JM, VM), 1e-13)

    # Whole (sum of L and U)
    VM   = [pc.cC[:]; pc.cW[:]; pc.cE[:]; pc.cS[:]; pc.cN[:]]
    PC   =  droptol!(sparse(IM, JM, VM), 1e-13)

    @info "Check version 1: difference between v1 and ILUZero" 
    # @show norm(U_aim - U)
    # @show norm(L_aim - L)
    @show norm(diag(U_aim) .- diag(U))
    @show norm(diag(U_aim,1) .- diag(U,1))
    @show norm(diag(L_aim,-1) .- diag(L,-1))
    @show norm(diag(L_aim,-Ncx) .- diag(L,-Ncx))

    @info "Check version 2: difference between v1 and v2"
    # @show norm(U_aim - U_PC)
    # @show norm(L_aim - L_PC)
    @show norm(diag(U) .- diag(PC))
    @show norm(diag(U,1) .- diag(PC,1))
    @show norm(diag(L,-1) .- diag(PC,-1))
    @show norm(diag(L,-Ncx) .- diag(PC,-Ncx))
    @show norm(diag(U,Ncx) .- diag(PC,Ncx))

    @info "-------- Try approximate solve with ILUZero --------"
    @info "Step 0 --- exact complete solve"
    Tdir = (M\b[:])
    @info "Step 1 --- exact ILUZero"
    Tilu0 = LU\b[:]
    @show norm(Tilu0 .- Tdir)/length(Tdir)
    @info "Step 2 --- iterative 2-step solve from Chow & Patel, 2015"
    nit   = 50
    tol   = 1e-5
    # First L\b
    IL    = I(size(L,1))
    DLinv = IL
    GL    = droptol!(IL - DLinv*(L+IL), 1e-10) # clean: no diagonal entries
    a     = zero(b[:])
    for iter=1:nit
        a .+= GL*a .+  DLinv*b[:] - a
        if norm(GL*a .+  DLinv*b[:] - a)/length(b) < tol
            @show "$iter" norm(GL*a .+  DLinv*b[:] - a)/length(b)
            break
        end
    end
    # Second L\b
    DUinv =  spdiagm(1.0./diag(U))
    GU    = droptol!(IL - DUinv*U, 1e-10) # clean: no diagonal entries
    Tilu    = zero(b[:])
    for iter=1:nit
        Tilu .+= GU*Tilu .+  DUinv*a - Tilu
        if norm(GU*Tilu .+  DUinv*a - Tilu)/length(b) < tol 
            @show "$iter" norm(GU*Tilu .+  DUinv*a - Tilu)/length(b)
            break
        end
    end
    @show norm(Tilu .- Tdir)/length(Tdir)
    @info "Step 2 --- iterative U⁻¹(L⁻¹b) solve from Chow & Patel, 2015"
    IL    = I(size(L,1))
    DLinv = IL
    GL    = droptol!(IL - DLinv*(L+IL), 1e-10) # clean: no diagonal entries
    a     = zero(b[:])
    DUinv =  spdiagm(1.0./diag(U))
    GU    = droptol!(IL - DUinv*U, 1e-10) # clean: no diagonal entries
    Tilu    = zero(b[:])
    for iter=1:nit
        a    .+= GL*a    .+  DLinv*b[:] - a
        Tilu .+= GU*Tilu .+  DUinv*a    - Tilu
        if norm(GL*a .+  DLinv*b[:] - a)/length(b) < tol  &&  norm(GU*Tilu .+  DUinv*a - Tilu)/length(b) < tol 
            @show "$iter" norm(GL*a .+  DLinv*b[:] - a)/length(b)
            break
        end
    end
    @show norm(Tilu .- Tdir)/length(Tdir)
    @info "Step 2 --- iterative L\b solve using coefficients from Chow & Patel, 2015"
    T2D = zeros(Ncx, Ncy)

    ForwardBackwardSolve!(T2D, reshape(b, Ncx, Ncy), pc, nit, tol)
    @show norm(T2D[:] .- Tdir)/length(Tdir)
    @info "End"
    
    # b0 = zero(b)
    # b .= 0
    # for iter=1:nit
    #     b0 .= b
    #     BackwardSubstitutionLagged!(b, b0, U, a)
    #     @show norm(b - Tilu0)/length(b)
    #     if norm(b - Tilu0)/length(b)<1e-7 
    #         @show "$iter" norm(b - Tilu0)/length(b)
    #         break
    #     end
    # end
    # display(Matrix(U_aim))
    # L = run_incomplete_cholesky3(M)
    # display(Matrix(L'))


end

main()