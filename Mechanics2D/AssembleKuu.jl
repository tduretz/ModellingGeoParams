function FillCoefficients!(I::Vector{Int64}, J::Vector{Int64}, V::Vector{Float64}, istart::Int64, icoef, jcoef, vcoef)
    for i=1:length(icoef)
        I[istart+i] = icoef[i]
        J[istart+i] = jcoef[i]
        V[istart+i] = vcoef[i]
    end
end

function KuuBlock(ηc, ηb, ηv, dx, dy, NumVx, NumVy)
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
    return Kuu
end