using LinearAlgebra, SparseArrays, ILUZero

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

function Gershgorin(M)
    λmax  = 0.
    for i in axes(M,1)
        if sum(abs.(M[i,:])) >  λmax
            λmax = sum(abs.(M[i,:]))
        end
    end
    return λmax
end

function GershgorinLocal(M)
    λmax  = zeros(size(M,1))
    for i in axes(M,1)
        λmax[i] = sum(abs.(M[i,:]))
    end
    return λmax
end

function ichol!(L)
	n = size(L,1)

	for k = 1:n
		L[k,k] = sqrt(L[k,k])
		for i = (k+1):n
		    if (L[i,k] != 0)
		        L[i,k] = L[i,k]/L[k,k]            
		    end
		end
		for j = (k+1):n
		    for i = j:n
		        if (L[i,j] != 0)
		            L[i,j] = L[i,j] - L[i,k]*L[j,k] 
		        end
		    end
		end
	end

    for i = 1:n
        for j = i+1:n
            L[i,j] = 0
        end
    end           
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

#####################################################################################
#####################################################################################
#####################################################################################

# Residual function
r(M,x,b) = b -  M*x

function res!(R, x, η, Δ, b)
    V = zeros(length(x)+2)
    q = zeros(length(x)+1)
    V[2:end-1] .= x
    V[1]       = -V[2]
    V[end]     = -V[end-1]
    @. q = -η * (V[2:end] - V[1:end-1]) / Δ 
    @. R = b - (q[2:end] - q[1:end-1]) / Δ
end

function main()

    hetero  = true

    # ############################
    
    # Domain
    L   = 1.0
    ncx = 5
    xv  = LinRange(-L/2, L/2, ncx+1)
    Δ   = L/ncx
    # Spatially varying coefficient
    η                         = ones(ncx+1)
    if hetero
        η[xv.<-0.4]              .= 1e3
        η[xv.>-0.2 .&& xv.<-0.0] .= 1e-3
        η[xv.>0.1 .&& xv.<-0.3]  .= 1e4 
    end
    # 1 step smoothing
    ηc                        = 0.5.*(η[1:end-1] .+ η[2:end])
    η[2:end-1]                = 0.5.*(ηc[1:end-1] .+ ηc[2:end])
    # Max loc business
    η_maxloc = copy(η)
    η_maxloc[2:end-1] .= max.(η_maxloc[2:end-1], max.(η_maxloc[3:end-0], η_maxloc[1:end-2]))
    # Sparse
    cC       = (η[1:end-1] + η[2:end-0])/Δ^2
    cC[1]   += η[1]./Δ^2
    cC[end] += η[end]/Δ^2
    cW       = -η[2:end-1]/Δ^2
    M        = SparseMatrixCSC(Tridiagonal(cW[:], cC[:], cW[:]))
    display(M)

    R  = ones(size(M,1))
    x  = 10*rand(size(M,1))
    b  = ones(size(M,1))
    
    # Test residuals
    res!(R, x, η, Δ, b) 
    @show R
    R .= r(M, x, b)
    @show R

    # Test diagonal PC
    dM    = spdiagm(diag(M))
    R    .= dM\r(M, x, b) 
    @show R
    res!(R, x, η, Δ, b) 
    R   ./= diag(M)
    @show R

    # Incomplete LU zero
    # SparseMatrixCSC{Float64, Int64}
    # LU
    @time LU = ilu0(M)    
    display(LU)

    # Incomplete LU zero - Chow & Patel, 2015 - VERSION 1
    L = zeros(size(M))
    U = zeros(size(M))

    # Initial guess
    for j=1:size(M,1)
        for i=1:size(M,2)
            if i>j 
                L[i,j] = M[i,j]
            else
                U[i,j] = M[i,j]
            end
        end
    end

    # Iterations over sparse matrix
    nit = 10

    L0, U0 = copy(L), copy(U)
    for iter=1:nit
        println("ILU iter $iter")
        U0 .= U
        L0 .= L
        for i=1:size(M,2)
            for j=1:size(M,1)
                if i>j 
                    # @show M[i,j]
                    L[i,j] = 1.0/U0[j,j] * (M[i,j] - 0*L0[i,1:j-1]' * U0[1:j-1,j] )
                    # @show (L0[i,1:j-1]', U0[1:j-1,j] )
                    # @show L0[i,1:j-1]' * U0[1:j-1,j]
                else
                    U[i,j] =              (M[i,j] -  L0[i,1:i-1]' * U0[1:i-1,j] )
                    # if i==j
                    #     @show (L0[i,1:i-1]', U0[1:i-1,j])
                    #     @show L0[i,1:i-1]' * U0[1:i-1,j]
                    #     # @show M[i,j]
                    # end
                end
            end
        end
    # @show norm(L)
    end
    display(U)
    display(L)

    # Incomplete LU zero - Chow & Patel, 2015 - VERSION 2

    # M2Di style storage
    MC = zeros(size(M,1)+2, 1);     MC[2:end-1] .= diag(M)
    MW = zeros(size(M,1)+2, 1);     MW[3:end-1] .= diag(M,-1)
    ME = zeros(size(M,1)+2, 1);     ME[2:end-2] .= diag(M,+1)
    PC = zeros(size(M,1)+2, 1);     PC[2:end-1] .= diag(M)
    PW = zeros(size(M,1)+2, 1);     PW[3:end-1] .= diag(M,-1)
    PE = zeros(size(M,1)+2, 1);     PE[2:end-2] .= diag(M,+1)

    # Iterations over coefficients
    for iter=1:nit
        PW[3:end]  .=   1.0./PC[2:end-1] .* (MW[3:end])
        PE[1:end-2] .=  1.0./1.0         .* (ME[1:end-2])
        PC[3:end-1] .=  1.0./1.0         .* (MC[3:end-1] .- PW[3:end-1].*PE[2:end-2]  )
    end
    @show PC
    @show PW[3:end-0]
    @show PE[2:end-1]

    # CHECK
    @show norm(PC[2:end-1] - diag(U))
    @show norm(PE[2:end-2] - diag(U,+1))
    @show norm(PW[3:end-1] - diag(L,-1))

    upper(ilu) = SparseMatrixCSC(ilu.m, ilu.n,ilu.u_colptr, ilu.u_rowval, ilu.u_nzval)
    lower(ilu) = SparseMatrixCSC(ilu.m, ilu.n,ilu.l_colptr, ilu.l_rowval, ilu.l_nzval)
    U_aim = upper(LU)
    L_aim = lower(LU)
    @show norm(U_aim - U)
    @show norm(L_aim - L)

    U2       = SparseMatrixCSC(Tridiagonal(0.0.*PW[3:end-1], PC[2:end-1], PE[2:end-2]))
    L2       = SparseMatrixCSC(Tridiagonal(PW[3:end-1], 0.0.*PC[2:end-1], 0.0.*PE[2:end-2]))
    @show norm(U_aim - U2)
    @show norm(L_aim - L2)
        
end


main()