using LinearAlgebra, SparseArrays, ILUZero

upper(ilu) = SparseMatrixCSC(ilu.m, ilu.n, ilu.u_colptr, ilu.u_rowval, ilu.u_nzval)
lower(ilu) = SparseMatrixCSC(ilu.m, ilu.n, ilu.l_colptr, ilu.l_rowval, ilu.l_nzval)

let 
    # Generate matrix and rhs
    n        = 100
    cW       = -1.0*ones(n-1)
    cC       =  2.0*ones(n  ) ; cC[1] += 1; cC[end] += 1
    M        = SparseMatrixCSC(Tridiagonal(cW[:], cC[:], cW[:]))
    b        = 3.0*ones(n  )
    LU       = ilu0(M)
    # Standard solve 
    x1       = LU\b
    # Split matrix in L and U parts
    U        = upper(LU)
    L        = lower(LU) + I(n)
    x2       = U\(L\b)
    # Compare
    @show norm(x1-x2)
end