using LinearAlgebra, SparseArrays, Preconditioners

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
		            L[i,j] = L[i,j] - L[i,k]*L[j,k]  # ?? in place
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

function main()

    auto_PT = true
    hetero  = true

    # Domain
    L   = 1.0
    ncx = 200
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
    # @show M[2,:]
    # Sparse
    cC       = (η_maxloc[1:end-1] + η_maxloc[2:end-0])/Δ^2
    cC[1]   += η_maxloc[1]./Δ^2
    cC[end] += η_maxloc[end]/Δ^2
    cW       = -η_maxloc[2:end-1]/Δ^2
    M_maxloc = SparseMatrixCSC(Tridiagonal(cW[:], cC[:], cW[:]))

    L = spzeros(size(M)) 
    ichol!(L, M)

    # Let's make a solve
    b  = ones(size(M,1))
    x0 = M\b

    # Brutal
    x1 = L\(L'\b)

    # Backward-forward
    x2 = zero(x1)
    y  = zero(x1)
    BackwardSubstitution!(y, L', b)
    ForwardSubstitution!(x2, L, y )

    # L = run_incomplete_cholesky3(SparseMatrixCSC(M), size(M,1))

    ##########################################################
    
    # Iterative parameters
    niter = 200000
    λmin0 = 16.
    ϵ     = 1e-12 
    cfact = 0.95

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

    # Arrays for PT
    x3    = zero(x0)
    δx    = zero(x0)
    dxdτ  = zero(x0)
    R     = zero(x0)
    Mδx   = zero(x0)
    Mδx1  = zero(x0)

    # PT solve
    println("Plain PT")
    λmin  = λmin0
    # λmax  = maximum(GershgorinLocal(M))
    λmax  = GershgorinLocal(M_maxloc)
    @show (λmin, maximum(λmax))
    h     = 2.0./sqrt.(λmax)*0.999
    c     = 2.0*sqrt(λmin)
    for iter=1:2niter
        R      .= r(M, x3, b)
        @. dxdτ =  (2-c*h)/(2+c*h) * dxdτ + 2*h/(2+c*h)*R
        @. δx   = h*dxdτ
        @. x3  += δx
        if mod(iter, 10)==0
            if auto_PT 
                Mδx1 .= r(M, x3, b)
                Mδx  .= r(M, x3.-δx, b)
                λmin = abs( sum(δx.*(Mδx.-Mδx1)) / sum(δx.*δx) ) 
                c    = 2.0*sqrt(λmin) * cfact
            end
            if norm(R)/length(R)<ϵ
                @show (iter, norm(r(M, x3, b))/length(R))
                break
            end
        end
    end

    # Preconditionning a PT solve - diagonal PC
    println("PT + diag. PC")
    # @show dp = DiagonalPreconditioner(M)
    # @show dp
    x3   .= 0.0
    dxdτ .= 0.0
    dM    = I(size(M,1))
    dM    = spdiagm(diag(M))
    λmin  = 1e-1
    # res!(Mδx1, ones(size(x3)), η, Δ, b)
    λmax  = norm(2*(dM\Mδx1)) 
    # λmax  = GershgorinLocal(dM\M) * 1.6
    λmax  = 2 .* (η[1:end-1] .+ η[2:end-0]) ./ Δ^2 ./ diag(M) *1.5
    # @show λmax  = 2 .* diag(M)  ./ diag(M)
    h     = 2.0./sqrt.(λmax)*0.999
    c     = 2.0*sqrt(λmin)
    for iter=1:niter
        R      .= dM\r(M, x3, b) 
        # res!(R, x3, η, Δ, b) 
        # R ./= diag(M)
        @. dxdτ =  (2-c*h)/(2+c*h) * dxdτ + 2*h/(2+c*h)*R
        @. δx   = h*dxdτ
        @. x3  += δx
        if mod(iter, 100)==0
            if auto_PT 
                Mδx1 .= dM\r(M, x3,     b) 
                Mδx  .= dM\r(M, x3.-δx, b) 
                # res!(Mδx1, x3, η, Δ, b)
                # Mδx1 ./= diag(M)
                # res!(Mδx, x3.-δx, η, Δ, b)
                # Mδx ./= diag(M)
                λmin  = abs( sum(.-δx.*(Mδx1.-Mδx) ) / sum(δx.*δx) ) 
                c     = 2.0*sqrt(λmin) * cfact
            end
            if norm(r(M, x3, b))/length(R)<1e-12 
                @show (iter, norm(r(M, x3, b))/length(R))
                break 
            end
        end
    end

    # Preconditionning a PT solve - incomplete Cholesky
    println("PT + ILU PC")
    x3   .= 0.0
    x3   .= 0.0
    dxdτ .= 0.0
    dM    = I(size(M,1))
    dM    = spdiagm(diag(M))
    λmin  = 1e-1
    # res!(Mδx1, ones(size(x3)), η, Δ, b)
    λmax  = norm(2*(dM\Mδx1)) 
    # λmax  = GershgorinLocal(dM\M) * 1.6
    λmax  = 2 .* (η[1:end-1] .+ η[2:end-0]) ./ Δ^2 ./ diag(M) *1.5
    # @show λmax  = 2 .* diag(M)  ./ diag(M)
    h     = 2.0./sqrt.(λmax)*0.999
    c     = 2.0*sqrt(λmin)
    for iter=1:niter
        # R      .= dM\r(M, x3, b) 
        R      .= L\(L'\r(M, x3, b))
        @. dxdτ =  (2-c*h)/(2+c*h) * dxdτ + 2*h/(2+c*h)*R
        @. δx   = h*dxdτ
        @. x3  += δx
        if mod(iter, 100)==0
            if auto_PT 
                Mδx1 .= L\(L'\r(M, x3,     b)) 
                Mδx  .= L\(L'\r(M, x3.-δx, b)) 
                # Mδx1 .= dM\r(M, x3,     b) 
                # Mδx  .= dM\r(M, x3.-δx, b) 
                λmin  = abs( sum(.-δx.*(Mδx1.-Mδx) ) / sum(δx.*δx) ) 
                c     = 2.0*sqrt(λmin) * cfact
            end
            if norm(r(M, x3, b))/length(R)<1e-12 
                @show (iter, norm(r(M, x3, b))/length(R))
                break 
            end
        end
    end
    # dxdτ .= 0.0
    # λmin  = 1e-1
    # # λmax  = GershgorinLocal((L\(L'\M))) * 1.6
    # λmax  = 2 .* (η[1:end-1] .+ η[2:end-0]) ./ Δ^2 ./ diag(M) *1.5
    # h     = 2.0./sqrt.(λmax)*0.999
    # c     = 2.0*sqrt(λmin)
    # # @show L
    # # @show CholeskyPreconditioner(SparseMatrixCSC(M), 1)
    # for iter=1:niter
    #     # R      .= L\(L'\r(M, x3, b)) 
    #     # R .= r(M, x3, b)
    #     # BackwardSubstitution!(y, L', R)
    #     # ForwardSubstitution!(R, L, y )
    #     R      .= dM\r(M, x3, b)
    #     @. dxdτ =  (2-c*h)/(2+c*h) * dxdτ + 2*h/(2+c*h)*R
    #     @. δx   = h*dxdτ
    #     @. x3  += δx
    #     if mod(iter, 10)==0
    #         if auto_PT 
                # Mδx1 .= L\(L'\r(M, x3,     b)) 
                # Mδx  .= L\(L'\r(M, x3.-δx, b)) 
    #             Mδx1 .= dM\r(M, x3,     b) 
    #             Mδx  .= dM\r(M, x3.-δx, b) 
    #             λmin  = abs( sum(.-δx.*(Mδx1.-Mδx) ) / sum(δx.*δx) ) 
    #             c     = 2.0*sqrt(λmin) * cfact
    #         end
    #         if norm(r(M, x3, b))/length(R)<1e-12 
    #             @show (iter, norm(r(M, x3, b))/length(R))
    #             break 
    #         end
    #     end
    # end

    # Preconditionning a PT solve - lagged incomplete Cholesky
    # println("PT + ILU PC")
    # x3   .= 0.0
    # dxdτ .= 0.0
    # λmin  = 1e-1
    # λmax  = Gershgorin((L*L')\M) * 2.6
    # h     = 2.0/sqrt(λmax)*0.999
    # c     = 2.0*sqrt(λmin)
    # R0    = zero(R)
    # y0    = zero(y)
    # Mδx10 = zero(Mδx1)
    # yy0   = zero(Mδx1)
    # yy    = zero(Mδx1)
    # Mδx0  = zero(Mδx)
    # xx0   = zero(Mδx1)
    # xx    = zero(Mδx1)
    # for iter=1:niter
    #     R0 .= R
    #     y0 .= y0
    #     # R      .= L\(L'\r(M, x3, b)) 
    #     R .= r(M, x3, b)
    #     BackwardSubstitutionLagged!(y, y0, L', R)
    #     ForwardSubstitutionLagged!(R, R0, L, y )
    #     @. dxdτ =  (2-c*h)/(2+c*h) * dxdτ + 2*h/(2+c*h)*R
    #     @. δx   = h*dxdτ
    #     @. x3  += δx
    #     # @show (iter, norm(r(M, x3, b))/length(R))
    #     if mod(iter, 10)==0
    #         if auto_PT 

    #             # Mδx1 .= L\(L'\r(M, x3,     b)) 
    #             # Mδx  .= L\(L'\r(M, x3.-δx, b)) 

    #             # BackwardSubstitution!(y, L', r(M, x3,     b))
    #             # ForwardSubstitution!(Mδx1, L, y )
    #             # BackwardSubstitution!(y, L', r(M, x3.-δx,     b))
    #             # ForwardSubstitution!(Mδx, L, y )

    #             # # Matrix free
    #             # Mδx10  .= Mδx1
    #             # Mδx0   .= Mδx
    #             # yy0    .= yy
    #             # xx0    .= xx

    #             BackwardSubstitutionLagged!(yy, yy0, L', r(M, x3,     b))
    #             ForwardSubstitutionLagged!(Mδx1, Mδx10, L, yy )

    #             BackwardSubstitutionLagged!(xx, xx0, L', r(M, x3.-δx,     b))
    #             ForwardSubstitutionLagged!(Mδx, Mδx0, L, xx )

    #             λmin  = abs( sum(.-δx.*(Mδx1.-Mδx) ) / sum(δx.*δx) ) 
    #             c     = 2.0*sqrt(λmin) * cfact /1.5
    #         end
    #         if norm(r(M, x3, b))/length(R)<1e-12 
    #             @show (iter, norm(r(M, x3, b))/length(R))
    #             break 
    #         end
    #     end
    # end

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
    # display(M)

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

    # # Incomplete cholesky
    # L = copy(M)
    # ichol!(L)
    # display(L)

    # L1 = run_incomplete_cholesky3(SparseMatrixCSC(M), size(M,1))
    # display(L1)

    # MC = zeros(size(M,1)+2, 1);     MC[2:end-1] .= diag(M)
    # MW = zeros(size(M,1)+2, 1);     MW[3:end-1] .= diag(M,-1)
    # ME = zeros(size(M,1)+2, 1);     ME[2:end-2] .= diag(M,+1)
    
end


main()