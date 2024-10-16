@views function FillCoefficients!(I::Vector{Int64}, J::Vector{Int64}, V::Vector{Float64}, istart::Int64, icoef, jcoef, vcoef)
    for i in eachindex(icoef)
        I[istart+i] = icoef[i]
        J[istart+i] = jcoef[i]
        V[istart+i] = vcoef[i]
    end
end

@views function KuuBlock(ηc, ηb, ηv, dx, dy, NumVx, NumVy)
    DiagCoeff = 1e7
    Ncx = size(ηc, 1); Ncy = size(ηc, 2)
    # Vx
    eW    = zeros(Ncx+1, Ncy);  eW[2:end-0,:] .= ηc
    eE    = zeros(Ncx+1, Ncy);  eE[1:end-1,:] .= ηc
    ebW   = zeros(Ncx+1, Ncy);  ebW[2:end-0,:] .= ηb
    ebE   = zeros(Ncx+1, Ncy);  ebE[1:end-1,:] .= ηb
    eS    = zeros(Ncx+1, Ncy);  eS           .= ηv[:,1:end-1]
    eN    = zeros(Ncx+1, Ncy);  eN           .= ηv[:,2:end-0]
    cVxC  = -(-eN ./ dy - eS ./ dy) ./ dy + ((4 // 3) * eE ./ dx + (4 // 3) * eW ./ dx) ./ dx + (ebE ./ dx + ebW ./ dx) ./ dx
    cVxW  = -4 // 3 * eW ./ dx .^ 2 - ebW ./ dx .^ 2
    cVxE  = -4 // 3 * eE ./ dx .^ 2 - ebE ./ dx .^ 2
    cVxS  = -eS ./ dy .^ 2
    cVxN  = -eN ./ dy .^ 2
    cVySW = -eS ./ (dx .* dy) + (2 // 3) * eW ./ (dx .* dy) - ebW ./ (dx .* dy)
    cVySE = -2 // 3 * eE ./ (dx .* dy) + eS ./ (dx .* dy) + ebE ./ (dx .* dy)
    cVyNW = eN ./ (dx .* dy) - 2 // 3 * eW ./ (dx .* dy) + ebW ./ (dx .* dy)
    cVyNE = (2 // 3) * eE ./ (dx .* dy) - eN ./ (dx .* dy) - ebE ./ (dx .* dy)
    iVxC  = NumVx
    iVxW      =  ones(Int64, size(NumVx)); iVxW[2:end-0,: ] = NumVx[1:end-1,:]
    iVxE      =  ones(Int64, size(NumVx)); iVxE[1:end-1,: ] = NumVx[2:end-0,:]        
    iVxS      =  ones(Int64, size(NumVx)); iVxS[: ,2:end-0] = NumVx[:,1:end-1]
    iVxN      =  ones(Int64, size(NumVx)); iVxN[: ,1:end-1] = NumVx[:,2:end-0]
    iVySW     =  ones(Int64, size(NumVx)); iVySW[2:end-0,:] = NumVy[:,1:end-1]
    iVySE     =  ones(Int64, size(NumVx)); iVySE[1:end-1,:] = NumVy[:,1:end-1]
    iVyNW     =  ones(Int64, size(NumVx)); iVyNW[2:end-0,:] = NumVy[:,2:end-0]
    iVyNE     =  ones(Int64, size(NumVx)); iVyNE[1:end-1,:] = NumVy[:,2:end-0]
    # N-S Dirichlet nodes
    cVxS[:,  1] .= 0.0; 
    cVxN[:,end] .= 0.0; 
    cVxC[1,:]  .= DiagCoeff; cVxC[end,:]  .= DiagCoeff
    cVxW[1,:]  .= 0.0; cVxW[end,:]  .= 0.0
    cVxE[1,:]  .= 0.0; cVxE[end,:]  .= 0.0
    cVxS[1,:]  .= 0.0; cVxS[end,:]  .= 0.0
    cVxN[1,:]  .= 0.0; cVxN[end,:]  .= 0.0
    cVySW[1,:] .= 0.0; cVySW[end,:] .= 0.0
    cVyNW[1,:] .= 0.0; cVyNW[end,:] .= 0.0
    cVySE[1,:] .= 0.0; cVySE[end,:] .= 0.0
    cVyNE[1,:] .= 0.0; cVyNE[end,:] .= 0.0
    # Symmetry - kill Dirichlet connections
    cVxW[    2,:] .= 0.0
    cVxE[end-1,:] .= 0.0
    # Symmetry - kill Dirichlet connections
    cVySW[  :,  1] .= 0.0
    cVySE[  :,  1] .= 0.0
    cVyNW[  :,end] .= 0.0
    cVyNE[  :,end] .= 0.0
    # Vy
    eS    = zeros(Ncx, Ncy+1);  eS[:,2:end-0] .= ηc
    eN    = zeros(Ncx, Ncy+1);  eN[:,1:end-1] .= ηc
    ebS   = zeros(Ncx, Ncy+1);  ebS[:,2:end-0] .= ηb
    ebN   = zeros(Ncx, Ncy+1);  ebN[:,1:end-1] .= ηb
    eW    = zeros(Ncx, Ncy+1);  eW           .= ηv[1:end-1,:]
    eE    = zeros(Ncx, Ncy+1);  eE           .= ηv[2:end-0,:]
    cVyC  = ((4 // 3) * eN ./ dy + (4 // 3) * eS ./ dy) ./ dy + (ebN ./ dy + ebS ./ dy) ./ dy - (-eE ./ dx - eW ./ dx) ./ dx
    cVyW  = -eW ./ dx .^ 2
    cVyE  = -eE ./ dx .^ 2
    cVyS  = -4 // 3 * eS ./ dy .^ 2 - ebS ./ dy .^ 2
    cVyN  = -4 // 3 * eN ./ dy .^ 2 - ebN ./ dy .^ 2
    cVxSW = (2 // 3) * eS ./ (dx .* dy) - eW ./ (dx .* dy) - ebS ./ (dx .* dy)
    cVxSE = eE ./ (dx .* dy) - 2 // 3 * eS ./ (dx .* dy) + ebS ./ (dx .* dy)
    cVxNW = -2 // 3 * eN ./ (dx .* dy) + eW ./ (dx .* dy) + ebN ./ (dx .* dy)
    cVxNE = -eE ./ (dx .* dy) + (2 // 3) * eN ./ (dx .* dy) - ebN ./ (dx .* dy)
    iVyC      = NumVy
    iVyW      =  ones(Int64, size(NumVy)); iVyW[2:end-0,: ] = NumVy[1:end-1,:]
    iVyE      =  ones(Int64, size(NumVy)); iVyE[1:end-1,: ] = NumVy[2:end-0,:]
    iVyS      =  ones(Int64, size(NumVy)); iVyS[: ,2:end-0] = NumVy[:,1:end-1]
    iVyN      =  ones(Int64, size(NumVy)); iVyN[: ,1:end-1] = NumVy[:,2:end-0]
    iVxSW     =  ones(Int64, size(NumVy)); iVxSW[:,2:end-0] = NumVx[1:end-1,:]
    iVxSE     =  ones(Int64, size(NumVy)); iVxSE[:,2:end-0] = NumVx[2:end-0,:]
    iVxNW     =  ones(Int64, size(NumVy)); iVxNW[:,1:end-1] = NumVx[1:end-1,:]
    iVxNE     =  ones(Int64, size(NumVy)); iVxNE[:,1:end-1] = NumVx[2:end-0,:]
    # N-S Dirichlet nodes
    cVyW[  1,:] .= 0.0
    cVyE[end,:] .= 0.0
    cVyC[:,1]  .= DiagCoeff; cVyC[:,end]  .= DiagCoeff
    cVyW[:,1]  .= 0.0; cVyW[:,end]  .= 0.0
    cVyE[:,1]  .= 0.0; cVyE[:,end]  .= 0.0
    cVyS[:,1]  .= 0.0; cVyS[:,end]  .= 0.0
    cVyN[:,1]  .= 0.0; cVyN[:,end]  .= 0.0
    cVxSW[:,1] .= 0.0; cVxSW[:,end] .= 0.0
    cVxNW[:,1] .= 0.0; cVxNW[:,end] .= 0.0
    cVxSE[:,1] .= 0.0; cVxSE[:,end] .= 0.0
    cVxNE[:,1] .= 0.0; cVxNE[:,end] .= 0.0
    # Symmetry - kill Dirichlet connections
    cVyS[:,     2] .= 0.0
    cVyN[:, end-1] .= 0.0
    # Symmetry - kill Dirichlet connections
    cVxSW[  1,  :] .= 0.0
    cVxSE[end,  :] .= 0.0
    cVxNW[  1,  :] .= 0.0
    cVxNE[end,  :] .= 0.0

    # Sparse matrix Kuu
    nVx = size(cVxC,1)*size(cVxC,2)
    nVy = size(cVyC,1)*size(cVyC,2)
    Iuu   = zeros(  Int64, 9*(nVx + nVy) )
    Juu   = zeros(  Int64, 9*(nVx + nVy) )
    Vuu   = zeros(Float64, 9*(nVx + nVy) )

    #------------------- Vx
    FillCoefficients!(Iuu, Juu, Vuu, 0*nVx, iVxC[:], iVxC[:], cVxC[:])
    FillCoefficients!(Iuu, Juu, Vuu, 1*nVx, iVxC[:], iVxW[:], cVxW[:])
    FillCoefficients!(Iuu, Juu, Vuu, 2*nVx, iVxC[:], iVxE[:], cVxE[:])
    FillCoefficients!(Iuu, Juu, Vuu, 3*nVx, iVxC[:], iVxS[:], cVxS[:])
    FillCoefficients!(Iuu, Juu, Vuu, 4*nVx, iVxC[:], iVxN[:], cVxN[:])
    FillCoefficients!(Iuu, Juu, Vuu, 5*nVx, iVxC[:], iVySW[:], 0*cVySW[:])
    FillCoefficients!(Iuu, Juu, Vuu, 6*nVx, iVxC[:], iVySE[:], 0*cVySE[:])
    FillCoefficients!(Iuu, Juu, Vuu, 7*nVx, iVxC[:], iVyNW[:], 0*cVyNW[:])
    FillCoefficients!(Iuu, Juu, Vuu, 8*nVx, iVxC[:], iVyNE[:], 0*cVyNE[:])
    #------------------- Vy
    FillCoefficients!(Iuu, Juu, Vuu, 9*nVx+0*nVy, iVyC[:], iVyC[:], cVyC[:])
    FillCoefficients!(Iuu, Juu, Vuu, 9*nVx+1*nVy, iVyC[:], iVyW[:], cVyW[:])
    FillCoefficients!(Iuu, Juu, Vuu, 9*nVx+2*nVy, iVyC[:], iVyE[:], cVyE[:])
    FillCoefficients!(Iuu, Juu, Vuu, 9*nVx+3*nVy, iVyC[:], iVyS[:], cVyS[:])
    FillCoefficients!(Iuu, Juu, Vuu, 9*nVx+4*nVy, iVyC[:], iVyN[:], cVyN[:])
    FillCoefficients!(Iuu, Juu, Vuu, 9*nVx+5*nVy, iVyC[:], iVxSW[:], 0*cVxSW[:])
    FillCoefficients!(Iuu, Juu, Vuu, 9*nVx+6*nVy, iVyC[:], iVxSE[:], 0*cVxSE[:])
    FillCoefficients!(Iuu, Juu, Vuu, 9*nVx+7*nVy, iVyC[:], iVxNW[:], 0*cVxNW[:])
    FillCoefficients!(Iuu, Juu, Vuu, 9*nVx+8*nVy, iVyC[:], iVxNE[:], 0*cVxNE[:])
    #------------------- Assemble
    Kuu = sparse( Iuu, Juu, Vuu)
    droptol!(Kuu, 1e-13)
    return Kuu, (cS=cVxS, cW=cVxW, cC=cVxC, cE=cVxE, cN=cVxN), (cS=cVyS, cW=cVyW, cC=cVyC, cE=cVyE, cN=cVyN)
end

@views function ILU_v3(coeff, nit, tol)
    # M2Di style storage
    pC = copy(coeff.cC); pC0 = copy(coeff.cC)
    pS = copy(coeff.cS);
    pN = copy(coeff.cN);
    pW = copy(coeff.cW);  
    pE = copy(coeff.cE);

    # Iterations over coefficients
    for iter=1:nit

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

        err = norm(pC.-pC0)/length(pC)
        # println("parILU v3 iter $iter --- r = $err")

        if err <tol
            @info "ILU iter v3 converged in $iter sweeps"
            break
        end 
    end
    return (cS=pS, cW=pW, cC=pC, cE=pE, cN=pN)
end

@views function AssembleLU(pxx, pyy, NumVx, NumVy)
    # Construct the matrices (just for checks)
    iVxC      =  NumVx
    iVxW      =  ones(Int64, size(NumVx)); iVxW[2:end-0,: ] = NumVx[1:end-1,:]
    iVxE      =  ones(Int64, size(NumVx)); iVxE[1:end-1,: ] = NumVx[2:end-0,:]        
    iVxS      =  ones(Int64, size(NumVx)); iVxS[: ,2:end-0] = NumVx[:,1:end-1]
    iVxN      =  ones(Int64, size(NumVx)); iVxN[: ,1:end-1] = NumVx[:,2:end-0]
    iVyC      =  NumVy
    iVyW      =  ones(Int64, size(NumVy)); iVyW[2:end-0,: ] = NumVy[1:end-1,:]
    iVyE      =  ones(Int64, size(NumVy)); iVyE[1:end-1,: ] = NumVy[2:end-0,:]
    iVyS      =  ones(Int64, size(NumVy)); iVyS[: ,2:end-0] = NumVy[:,1:end-1]
    iVyN      =  ones(Int64, size(NumVy)); iVyN[: ,1:end-1] = NumVy[:,2:end-0]
    # Sparse matrix Kuu
    nVx = size(iVxC,1)*size(iVxC,2)
    nVy = size(iVyC,1)*size(iVyC,2)
    Iuu   = zeros(  Int64, 5*(nVx + nVy) )
    Juu   = zeros(  Int64, 5*(nVx + nVy) )
    Vuu   = zeros(Float64, 5*(nVx + nVy) )
    #------------------- Vx
    FillCoefficients!(Iuu, Juu, Vuu, 0*nVx, iVxC[:], iVxC[:], pxx.cC[:]*0)
    FillCoefficients!(Iuu, Juu, Vuu, 1*nVx, iVxC[:], iVxW[:], pxx.cW[:])
    FillCoefficients!(Iuu, Juu, Vuu, 2*nVx, iVxC[:], iVxE[:], pxx.cE[:]*0)
    FillCoefficients!(Iuu, Juu, Vuu, 3*nVx, iVxC[:], iVxS[:], pxx.cS[:])
    FillCoefficients!(Iuu, Juu, Vuu, 4*nVx, iVxC[:], iVxN[:], pxx.cN[:]*0)
    #------------------- Vy
    # FillCoefficients!(Iuu, Juu, Vuu, 5*nVx+0*nVy, iVyC[:], iVyC[:], pyy.cC[:]*0)
    FillCoefficients!(Iuu, Juu, Vuu, 5*nVx+0*nVy, iVyC[:], iVyC[:], 0*ones(size(pyy.cC[:]))) # set 1 on diagonal
    FillCoefficients!(Iuu, Juu, Vuu, 5*nVx+1*nVy, iVyC[:], iVyW[:], pyy.cW[:])
    FillCoefficients!(Iuu, Juu, Vuu, 5*nVx+2*nVy, iVyC[:], iVyE[:], pyy.cE[:]*0)
    FillCoefficients!(Iuu, Juu, Vuu, 5*nVx+3*nVy, iVyC[:], iVyS[:], pyy.cS[:])
    FillCoefficients!(Iuu, Juu, Vuu, 5*nVx+4*nVy, iVyC[:], iVyN[:], pyy.cN[:]*0)
    Luu = droptol!(sparse( Iuu, Juu, Vuu), 1e-10)
    #------------------- Vx
    FillCoefficients!(Iuu, Juu, Vuu, 0*nVx, iVxC[:], iVxC[:], pxx.cC[:])
    FillCoefficients!(Iuu, Juu, Vuu, 1*nVx, iVxC[:], iVxW[:], pxx.cW[:]*0)
    FillCoefficients!(Iuu, Juu, Vuu, 2*nVx, iVxC[:], iVxE[:], pxx.cE[:])
    FillCoefficients!(Iuu, Juu, Vuu, 3*nVx, iVxC[:], iVxS[:], pxx.cS[:]*0)
    FillCoefficients!(Iuu, Juu, Vuu, 4*nVx, iVxC[:], iVxN[:], pxx.cN[:])
    #------------------- Vy
    FillCoefficients!(Iuu, Juu, Vuu, 5*nVx+0*nVy, iVyC[:], iVyC[:], pyy.cC[:])
    FillCoefficients!(Iuu, Juu, Vuu, 5*nVx+1*nVy, iVyC[:], iVyW[:], pyy.cW[:]*0)
    FillCoefficients!(Iuu, Juu, Vuu, 5*nVx+2*nVy, iVyC[:], iVyE[:], pyy.cE[:])
    FillCoefficients!(Iuu, Juu, Vuu, 5*nVx+3*nVy, iVyC[:], iVyS[:], pyy.cS[:]*0)
    FillCoefficients!(Iuu, Juu, Vuu, 5*nVx+4*nVy, iVyC[:], iVyN[:], pyy.cN[:])
    Uuu = droptol!(sparse( Iuu, Juu, Vuu), 1e-10)
    #-------------------
    return Luu, Uuu 
end

@views function LaggedSubstitutions!(x2, y2, x20, y20, pc, b, nit=1, Ncx=size(x2,1), Ncy=size(x2,2) )

    for it=1:nit
        x20 .= x2
        y20 .= y2
        for j in axes(x20, 2), i in axes(x20, 1) 
            if i>1   yW = y20[i-1,j] else yW = 0. end
            if i<Ncx xE = x20[i+1,j] else xE = 0. end
            if j>1   yS = y20[i,j-1] else yS = 0. end
            if j<Ncy xN = x20[i,j+1] else xN = 0. end

            y2[i,j] =                      -pc.cW[i,j]*yW - pc.cS[i,j]*yS +    b[i,j]  
        end 
        for j in axes(x20, 2), i in axes(x20, 1) 
            if i>1   yW = y20[i-1,j] else yW = 0. end
            if i<Ncx xE = x20[i+1,j] else xE = 0. end
            if j>1   yS = y20[i,j-1] else yS = 0. end
            if j<Ncy xN = x20[i,j+1] else xN = 0. end
            x2[i,j] =  (1.0/pc.cC[i,j]) * (-pc.cE[i,j]*xE - pc.cN[i,j]*xN  + y2[i,j])
        end 
    end
end

function ISAI_Poisson2D(pc)
    LC   = ones(size(pc.cC))
    LiW  = copy(pc.cW)
    LiS  = copy(pc.cS)
    LiC  = ones(size(pc.cC))

    UC   = copy(pc.cC) 
    UiC  = copy(pc.cC) 
    UiE  = copy(pc.cE)
    UiN  = copy(pc.cN)

    # Iterations over coefficients
    for iter=1:1
        # Lower part
        LiW[2:end,1:end-0]   .=  -1.0./LC[1:end-1,1:end-0] .* (LiW[2:end,1:end-0].*LC[1:end-1,1:end-0])
        LiS[1:end-0,2:end]   .=  -1.0./LC[1:end-0,1:end-1] .* (LiS[1:end-0,2:end].*LC[1:end-0,1:end-1])
        LiC[1:end-0,1:end-0] .=   1.0./LC[1:end-0,1:end-0] 
        # Upper part
        UiC[1:end-0,1:end-0] .=  1.0./UC[1:end-0,1:end-0] 
        UiE[1:end-1,1:end-0] .= -1.0./UC[1:end-1,1:end-0] .* (UiE[1:end-1,1:end-0]./UC[2:end-0,1:end-0])
        UiN[1:end-0,1:end-1] .= -1.0./UC[1:end-0,1:end-1] .* (UiN[1:end-0,1:end-1]./UC[1:end-0,2:end-0])
    end
    return (cS=LiS, cW=LiW, cC=UiC, cE=UiE, cN=UiN)
end

function ApplyISAI!(F_PC, F, ipc, Ncx=size(F,1), Ncy=size(F,2))
    # Linv * F
    for j in axes(F, 2), i in axes(F, 1) 
        if i>1   fW = F[i-1,j] else fW = 0. end
        if j>1   fS = F[i,j-1] else fS = 0. end
        F_PC[i,j] =             2F[i,j] + ipc.cW[i,j]*fW + ipc.cS[i,j]*fS
    end
    F .= F_PC
    # Uinv * F
    for j in axes(F, 2), i in axes(F, 1) 
        if i<Ncx fE = F[i+1,j] else fE = 0. end
        if j<Ncy fN = F[i,j+1] else fN = 0. end
        F_PC[i,j] = 2ipc.cC[i,j]*F[i,j] + ipc.cE[i,j]*fE + ipc.cN[i,j]*fN
    end
    
end

@views function ForwardBackwardSolveISAI!(T2D, b, pc, ipc, nit, tol, Ncx=size(T2D,1), Ncy=size(T2D,2))
    # println("ForwardBackwardSolveISAI!")
    r2D    = zeros(size(T2D)) # temp array
    F2D    = zeros(size(T2D)) # temp array
    F2D0   = zeros(size(T2D)) # temp array
    F2D_PC = zeros(size(T2D)) # temp array
    for iter=1:nit
        for j in axes(T2D, 2), i in axes(T2D, 1) 
            if i>1   rW = r2D[i-1,j] else rW = 0. end
            if j>1   rS = r2D[i,j-1] else rS = 0. end
            F2D[i,j] =  -pc.cW[i,j]*rW - pc.cS[i,j]*rS +  b[i,j] - r2D[i,j] 
        end
        # apply PC
        for j in axes(T2D, 2), i in axes(T2D, 1) 
            if i>1   rW = F2D[i-1,j] else rW = 0. end
            if j>1   rS = F2D[i,j-1] else rS = 0. end
            F2D_PC[i,j] = ipc.cW[i,j]*rW + ipc.cS[i,j]*rS  + F2D[i,j]
            # F2D_PC[i,j] = F2D[i,j]
        end
        for j in axes(T2D, 2), i in axes(T2D, 1) 
            r2D[i,j] += F2D_PC[i,j]
        end

        for j in axes(T2D, 2), i in axes(T2D, 1) 
            if i<Ncx fE = T2D[i+1,j] else fE = 0. end
            if j<Ncy fN = T2D[i,j+1] else fN = 0. end
            F2D[i,j] =  (-pc.cE[i,j]*fE - pc.cN[i,j]*fN + r2D[i,j]) - T2D[i,j]*pc.cC[i,j]
        end
        if iter==1 F2D0.=F2D end
        # apply PC
        for j in axes(T2D, 2), i in axes(T2D, 1) 
            if i<Ncx fE = F2D[i+1,j] else fE = 0. end
            if j<Ncy fN = F2D[i,j+1] else fN = 0. end
            F2D_PC[i,j] = F2D[i,j].*ipc.cC[i,j] + fE.*ipc.cE[i,j] + fN.*ipc.cN[i,j]
            # F2D_PC[i,j] = F2D[i,j]./pc.cC[i,j]
        end
        for j in axes(T2D, 2), i in axes(T2D, 1) 
            T2D[i,j] += F2D_PC[i,j]
        end

        
        if iter==1 F2D0.=F2D end
        if norm(F2D[:])./norm(F2D0[:]) < tol
            # @show "$iter" norm(F2D)/length(b)
            break
        end
    end
end

@views function ForwardBackwardSolve!(x, b, pc, nit, tol, rel, Ncx=size(x,1), Ncy=size(x,2))
    # println("ForwardBackwardSolve!")
    y  = zeros(size(x)) # temp array
    F  = zeros(size(x))
    F0 = zeros(size(x))
    for iter=1:nit
        for j in axes(x, 2), i in axes(x, 1) 
            if i>1   yW = y[i-1,j] else yW = 0. end
            if i<Ncx xE = x[i+1,j] else xE = 0. end
            if j>1   yS = y[i,j-1] else yS = 0. end
            if j<Ncy xN = x[i,j+1] else xN = 0. end
            y[i,j] +=                     rel* (-pc.cW[i,j]*yW - pc.cS[i,j]*yS + b[i,j]  - y[i,j])
            F[i,j]  =                     rel* (-pc.cE[i,j]*xE - pc.cN[i,j]*xN + y[i,j]) - x[i,j]*pc.cC[i,j]
            x[i,j] += rel*( (1.0/pc.cC[i,j]) * (-pc.cE[i,j]*xE - pc.cN[i,j]*xN + y[i,j]) - x[i,j])
        end
        if iter==1 F0 .= F end
        if norm(F)./norm(F0) < tol
            # @show "$iter" norm(F)/length(b)
            break
        end
    end
end

@views function AssembleLU_lev1(pxx, pyy, NumVx, NumVy)

    # Construct the matrices (just for checks)
    iVxC      =  NumVx
    iVxW      =  ones(Int64, size(NumVx)); iVxW[2:end-0,: ] = NumVx[1:end-1,:]
    iVxWW     =  ones(Int64, size(NumVx)); iVxWW[3:end-0,: ] = NumVx[1:end-2,:]
    iVxE      =  ones(Int64, size(NumVx)); iVxE[1:end-1,: ] = NumVx[2:end-0,:] 
    iVxEE     =  ones(Int64, size(NumVx)); iVxEE[1:end-2,: ] = NumVx[3:end-0,:]        
    iVxS      =  ones(Int64, size(NumVx)); iVxS[: ,2:end-0] = NumVx[:,1:end-1]
    iVxSS     =  ones(Int64, size(NumVx)); iVxSS[: ,3:end-0] = NumVx[:,1:end-2]
    iVxN      =  ones(Int64, size(NumVx)); iVxN[: ,1:end-1] = NumVx[:,2:end-0]
    iVxNN     =  ones(Int64, size(NumVx)); iVxNN[: ,1:end-2] = NumVx[:,3:end-0]
    iVyC      =  NumVy
    iVyW      =  ones(Int64, size(NumVy)); iVyW[2:end-0,: ] = NumVy[1:end-1,:]
    iVyWW     =  ones(Int64, size(NumVy)); iVyWW[3:end-0,: ] = NumVy[1:end-2,:]
    iVyE      =  ones(Int64, size(NumVy)); iVyE[1:end-1,: ] = NumVy[2:end-0,:]
    iVyEE     =  ones(Int64, size(NumVy)); iVyEE[1:end-2,: ] = NumVy[3:end-0,:]
    iVyS      =  ones(Int64, size(NumVy)); iVyS[: ,2:end-0] = NumVy[:,1:end-1]
    iVySS     =  ones(Int64, size(NumVy)); iVySS[: ,3:end-0] = NumVy[:,1:end-2]
    iVyN      =  ones(Int64, size(NumVy)); iVyN[: ,1:end-1] = NumVy[:,2:end-0]
    iVyNN     =  ones(Int64, size(NumVy)); iVyNN[: ,1:end-2] = NumVy[:,3:end-0]
    # Sparse matrix Kuu
    nVx = size(iVxC,1)*size(iVxC,2)
    nVy = size(iVyC,1)*size(iVyC,2)
    Iuu   = zeros(  Int64, 9*(nVx + nVy) )
    Juu   = zeros(  Int64, 9*(nVx + nVy) )
    Vuu   = zeros(Float64, 9*(nVx + nVy) )
    #------------------- Vx
    cWW = ones(size(NumVx)); cWW[[1     2 3 end], :] .= 0.; 
    cEE = ones(size(NumVx)); cEE[[1 end-2 end-1 end], :] .= 0.;
    cSS = ones(size(NumVx)); cSS[:, [1     2  3 end]] .= 0.; 
    cNN = ones(size(NumVx)); cNN[:, [1 end-2 end-1 end]] .= 0.;
    FillCoefficients!(Iuu, Juu, Vuu, 0*nVx, iVxC[:], iVxC[:], pxx.cC[:]*0)
    FillCoefficients!(Iuu, Juu, Vuu, 1*nVx, iVxC[:], iVxW[:], pxx.cW[:])
    FillCoefficients!(Iuu, Juu, Vuu, 2*nVx, iVxC[:], iVxE[:], pxx.cE[:]*0)
    FillCoefficients!(Iuu, Juu, Vuu, 3*nVx, iVxC[:], iVxS[:], pxx.cS[:])
    FillCoefficients!(Iuu, Juu, Vuu, 4*nVx, iVxC[:], iVxN[:], pxx.cN[:]*0)
    FillCoefficients!(Iuu, Juu, Vuu, 5*nVx, iVxC[:], iVxWW[:], cWW[:])
    FillCoefficients!(Iuu, Juu, Vuu, 6*nVx, iVxC[:], iVxEE[:], cEE[:]*0)
    FillCoefficients!(Iuu, Juu, Vuu, 7*nVx, iVxC[:], iVxSS[:], cSS[:])
    FillCoefficients!(Iuu, Juu, Vuu, 8*nVx, iVxC[:], iVxNN[:], cNN[:]*0)
    #------------------- Vy
    cWW = ones(size(NumVy)); cWW[[1     2 3 end], :] .= 0.; 
    cEE = ones(size(NumVy)); cEE[[1 end-2 end-1 end], :] .= 0.;
    cSS = ones(size(NumVy)); cSS[:, [1     2 3 end]] .= 0.; 
    cNN = ones(size(NumVy)); cNN[:, [1 end-2 end-1 end]] .= 0.;
    FillCoefficients!(Iuu, Juu, Vuu, 9*nVx+0*nVy, iVyC[:], iVyC[:], 0*ones(size(pyy.cC[:]))) # set 1 on diagonal
    FillCoefficients!(Iuu, Juu, Vuu, 9*nVx+1*nVy, iVyC[:], iVyW[:], pyy.cW[:])
    FillCoefficients!(Iuu, Juu, Vuu, 9*nVx+2*nVy, iVyC[:], iVyE[:], pyy.cE[:]*0)
    FillCoefficients!(Iuu, Juu, Vuu, 9*nVx+3*nVy, iVyC[:], iVyS[:], pyy.cS[:])
    FillCoefficients!(Iuu, Juu, Vuu, 9*nVx+4*nVy, iVyC[:], iVyN[:], pyy.cN[:]*0)
    FillCoefficients!(Iuu, Juu, Vuu, 9*nVx+5*nVy, iVyC[:], iVyW[:], cWW[:])
    FillCoefficients!(Iuu, Juu, Vuu, 9*nVx+6*nVy, iVyC[:], iVyE[:], cEE[:]*0)
    FillCoefficients!(Iuu, Juu, Vuu, 9*nVx+7*nVy, iVyC[:], iVyS[:], cSS[:])
    FillCoefficients!(Iuu, Juu, Vuu, 9*nVx+8*nVy, iVyC[:], iVyN[:], cNN[:]*0)
    Luu = droptol!(sparse( Iuu, Juu, Vuu), 1e-10)
    #------------------- Vx
    cWW = ones(size(NumVx)); cWW[[1     2 3 end], :] .= 0.; 
    cEE = ones(size(NumVx)); cEE[[1 end-2 end-1 end], :] .= 0.;
    cSS = ones(size(NumVx)); cSS[:, [1     2  3 end]] .= 0.; 
    cNN = ones(size(NumVx)); cNN[:, [1 end-2 end-1 end]] .= 0.;
    FillCoefficients!(Iuu, Juu, Vuu, 0*nVx, iVxC[:], iVxC[:], pxx.cC[:])
    FillCoefficients!(Iuu, Juu, Vuu, 1*nVx, iVxC[:], iVxW[:], pxx.cW[:]*0)
    FillCoefficients!(Iuu, Juu, Vuu, 2*nVx, iVxC[:], iVxE[:], pxx.cE[:])
    FillCoefficients!(Iuu, Juu, Vuu, 3*nVx, iVxC[:], iVxS[:], pxx.cS[:]*0)
    FillCoefficients!(Iuu, Juu, Vuu, 4*nVx, iVxC[:], iVxN[:], pxx.cN[:])
    FillCoefficients!(Iuu, Juu, Vuu, 5*nVx, iVxC[:], iVxWW[:], cWW[:]*0)
    FillCoefficients!(Iuu, Juu, Vuu, 6*nVx, iVxC[:], iVxEE[:], cEE[:])
    FillCoefficients!(Iuu, Juu, Vuu, 7*nVx, iVxC[:], iVxSS[:], cSS[:]*0)
    FillCoefficients!(Iuu, Juu, Vuu, 8*nVx, iVxC[:], iVxNN[:], cNN[:])
    #------------------- Vy
    cWW = ones(size(NumVy)); cWW[[1     2 3 end], :] .= 0.; 
    cEE = ones(size(NumVy)); cEE[[1 end-2 end-1 end], :] .= 0.;
    cSS = ones(size(NumVy)); cSS[:, [1     2 3 end]] .= 0.; 
    cNN = ones(size(NumVy)); cNN[:, [1 end-2 end-1 end]] .= 0.;
    FillCoefficients!(Iuu, Juu, Vuu, 9*nVx+0*nVy, iVyC[:], iVyC[:], pyy.cC[:]) # set 1 on diagonal
    FillCoefficients!(Iuu, Juu, Vuu, 9*nVx+1*nVy, iVyC[:], iVyW[:], pyy.cW[:]*0)
    FillCoefficients!(Iuu, Juu, Vuu, 9*nVx+2*nVy, iVyC[:], iVyE[:], pyy.cE[:])
    FillCoefficients!(Iuu, Juu, Vuu, 9*nVx+3*nVy, iVyC[:], iVyS[:], pyy.cS[:]*0)
    FillCoefficients!(Iuu, Juu, Vuu, 9*nVx+4*nVy, iVyC[:], iVyN[:], pyy.cN[:])
    FillCoefficients!(Iuu, Juu, Vuu, 9*nVx+5*nVy, iVyC[:], iVyW[:], cWW[:]*0)
    FillCoefficients!(Iuu, Juu, Vuu, 9*nVx+6*nVy, iVyC[:], iVyE[:], cEE[:])
    FillCoefficients!(Iuu, Juu, Vuu, 9*nVx+7*nVy, iVyC[:], iVyS[:], cSS[:]*0)
    FillCoefficients!(Iuu, Juu, Vuu, 9*nVx+8*nVy, iVyC[:], iVyN[:], cNN[:])
    Uuu = droptol!(sparse( Iuu, Juu, Vuu), 1e-10)
    #-------------------
    return Luu, Uuu 
end