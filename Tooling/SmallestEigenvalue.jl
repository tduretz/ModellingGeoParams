using LinearAlgebra, Arpack
let
    # Minimum eigenvalue 
    A = [7. -1. -2.; -1. 7. -3.; -2. -3. 3.]
    n = size(A, 1)
    for it=1:4
        p = 2^(it-1)
        @show (p, n)
        @show ((n-1)^(2p-1) / ((n-1)^(2p-1) + 1.0))
        @show term = tr( (A .- tr(A)./n * I(n))^(2p) )
        @show αp   = ( ((n-1)^(2p-1) / ((n-1)^(2p-1) + 1.0)) * term )^(1/2/p)
        @show tr(A)/(n) - αp
    end
    eigs(A,  which=:SM, nev=1)
    @show (tr(A^2) - (tr(A)^2)/n)
    @show tr( (A .- tr(A)./n * I(n))^2 )

    # Matrix free
    cC  = [0; 7; 7; 3; 0]
    cWW = [0; 0; -0; -2; 0]
    cW  = [0;  0; -1; -3; 0]
    cE  = [0; -1; -3; 0; 0]
    cEE = [0; -2;  0; 0; 0]
    p = 1

    @show tr(A)./n
    @show sum(cC)/n

    @show tr((A)^2)
    @show sum(cC.^2) + sum(cWW.^2) + sum(cW.^2) .+ sum(cE.^2) .+ sum(cEE.^2) 

    # # @show tr((A1)^p)
    trA_n = sum(cC)/n
    # @show trA_n = sum(cC[2:end-1].-trA_n)/n
    A1 = A .- tr(A)./n * I(n)
    # Matrix free evaluation of the first iteration only (p = 1)
    term = sum((cC[2:end-1].-trA_n).^2) + sum(cWW.^2) + sum(cW.^2) .+ sum(cE.^2) .+ sum(cEE.^2) 
    αp   = ( ((n-1)^(2p-1) / ((n-1)^(2p-1) + 1.0)) * term )^(1/2/p)
    @show trA_n - αp

end