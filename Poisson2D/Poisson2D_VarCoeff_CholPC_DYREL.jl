using GeoParams, Plots, Printf, MathTeXEngine, BenchmarkTools, Statistics
import LinearAlgebra:norm
using IncompleteLU, ILUZero
# Makie.update_theme!(fonts = (regular = texfont(), bold = texfont(:bold), italic = texfont(:italic)))

const cmy = 356.25*3600*24*100
const ky  = 356.25*3600*24*1e3

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

function GershgorinThermics2D( k, ρ, Cp, Δx, Δy, Δt, DM, transient )
    return maximum( ((transient ./ Δt .+ k.y[:,1:end-1]/ρ/Cp/Δy^2 .+ k.y[:,2:end-0]/ρ/Cp/Δy^2 .+ k.x[1:end-1,:]/ρ/Cp/Δx^2 .+ k.x[2:end-0,:]/ρ/Cp/Δx^2) + k.y[:,1:end-1]/ρ/Cp/Δy^2 + k.y[:,2:end-0]/ρ/Cp/Δy^2 + k.x[1:end-1,:]/ρ/Cp/Δx^2 + k.x[2:end-0,:]/ρ/Cp/Δx^2)./DM[2:end-1,2:end-1] )
end

function DiagThermics2D!( D, k, ρ, Cp, Δx, Δy, Δt, transient )
    D[2:end-1,2:end-1] .= (transient ./ Δt .+ k.y[:,1:end-1]/ρ/Cp/Δy^2 .+ k.y[:,2:end-0]/ρ/Cp/Δy^2 .+ k.x[1:end-1,:]/ρ/Cp/Δx^2 .+ k.x[2:end-0,:]/ρ/Cp/Δx^2) 
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
    return droptol!(sparse(IM, JM, VM), 1e-13)
end

function TraceThermics2D( k, ρ, Cp, Δx, Δy, Δt, transient, A )
    return sum( (transient ./ Δt .+ k.y[:,1:end-1]/ρ/Cp/Δy^2 .+ k.y[:,2:end-0]/ρ/Cp/Δy^2 .+ k.x[1:end-1,:]/ρ/Cp/Δx^2 .+ k.x[2:end-0,:]/ρ/Cp/Δx^2) .- A )
end

function GershgorinThermics2D_Local!( λmaxlocT, k, ρ, Cp, Δx, Δy, Δt, DM, transient )
    λmaxlocT .= ( (transient ./ Δt .+ k.y[:,1:end-1]/ρ/Cp/Δy^2 .+ k.y[:,2:end-0]/ρ/Cp/Δy^2 .+ k.x[1:end-1,:]/ρ/Cp/Δx^2 .+ k.x[2:end-0,:]/ρ/Cp/Δx^2) + k.y[:,1:end-1]/ρ/Cp/Δy^2 + k.y[:,2:end-0]/ρ/Cp/Δy^2 + k.x[1:end-1,:]/ρ/Cp/Δx^2 + k.x[2:end-0,:]/ρ/Cp/Δx^2)
    λmaxlocT ./= DM[2:end-1,2:end-1]
end

@views function maxloc!(A2, A)
    A2[2:end-1,2:end-1] .= max.(max.(max.(A[1:end-2,2:end-1], A[3:end,2:end-1]), A[2:end-1,2:end-1]), max.(A[2:end-1,1:end-2], A[2:end-1,3:end]))
    A2[[1,end],:] .= A2[[2,end-1],:]; A2[:,[1,end]] .= A2[:,[2,end-1]]
end

@views function minloc!(A2, A)
    A2[2:end-1,2:end-1] .= min.(min.(min.(A[1:end-2,2:end-1], A[3:end,2:end-1]), A[2:end-1,2:end-1]), min.(A[2:end-1,1:end-2], A[2:end-1,3:end]))
    A2[[1,end],:] .= A2[[2,end-1],:]; A2[:,[1,end]] .= A2[:,[2,end-1]]
end

@views function ResidualThermics2D!(RT, Tc, Tc0, Qr, Δt, ρ, Cp, k, Δx, Δy, ∂T∂x, ∂T∂y, qTx, qTy, rhs, T_South, T_North, DM, transient )
    
    Tc[:,1]   .= -Tc[:,2]     .+ 2*T_South*rhs
    Tc[:,end] .= -Tc[:,end-1] .+ 2*T_North*rhs
    # Tc[:,1]   .= Tc[:,2]     
    # Tc[:,end] .= Tc[:,end-1] 
    Tc[1,:]   .= Tc[2,:]
    Tc[end,:] .= Tc[end-1,:]

    @. ∂T∂x        = (Tc[2:end,2:end-1] - Tc[1:end-1,2:end-1])/Δx
    @. qTx         = -k.x .* ∂T∂x
    @. ∂T∂y        = (Tc[2:end-1,2:end] - Tc[2:end-1,1:end-1])/Δy
    @. qTy         = -k.y * ∂T∂y
    @. RT[2:end-1,2:end-1] = -transient*(Tc[2:end-1,2:end-1] - rhs*Tc0[2:end-1,2:end-1]) / Δt - 1.0/(ρ*Cp) * (qTx[2:end,:] - qTx[1:end-1,:])/Δx - 1.0/(ρ*Cp) * (qTy[:,2:end] - qTy[:,1:end-1])/Δy + rhs/(ρ*Cp) *  Qr[2:end-1,2:end-1]
    @. RT                ./= DM
end

@views function MainPoisson2D(n, auto)

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
    Ncx        = n*20
    Ncy        = n*20
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
    # r1         = nondimensionalize(2e3m, CharDim)
    # kv[(xv.+r1).^2 .+ (yv'.-r1).^2 .< r1^2] .= 1000.
    # r2         = nondimensionalize(2.55e3m, CharDim)
    # kv[(xv.+2.6r1).^2 .+ (yv'.+1.7r2).^2 .< r2^2] .= 1e-3
    
    # 1 step smoothing
    kc                   = k0.*ones(Ncx+0, Ncy+0)
    kc                  .= 0.25.*(kv[1:end-1,1:end-1] .+ kv[2:end-0,1:end-1] .+ kv[1:end-1,2:end-0] .+ kv[2:end-0,2:end-0])
    kv[2:end-1,2:end-1] .= 0.25.*(kc[1:end-1,1:end-1] .+ kc[2:end-0,1:end-1] .+ kc[1:end-1,2:end-0] .+ kc[2:end-0,2:end-0])

    k          = (x=0.5.*(kv[:,1:end-1] .+ kv[:,2:end-0]), y=0.5.*(kv[1:end-1,:] .+ kv[2:end-0,:]))
        
    # Local max preconditioning
    PC         = :ichol
    # PC         = :ilu0
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
    M    = ConstructM(k, ρ, Cp, Δx, Δy, Δt, transient, b, T_South, T_North)
    Tdir = (M\b[:])

    if PC == :ichol# Cholesky PC
        L = copy(M)
        # @time ichol!(L) # way to slow!
        @time L = run_incomplete_cholesky3(M, size(M,1))
    elseif PC== :ilu
        # @time LU = ilu(M, τ = 0.1)
        @time LU = ilu0(M)
    elseif PC == :ilu0
        @time LU = ilu0(M)
    end

    iters = 0
    for it=1:Nt
        # History
        @. Tc0  = Tc
        
        # PT steps
        t       += Δt 

        if PC == :diag 
            DiagThermics2D!( DM, k, ρ, Cp, Δx, Δy, Δt, transient ) 
            # DM[2:end-1,2:end-1] .= reshape(diag(M), Ncx, Ncy)
        end

        # DYREL
        if PC == :ichol
            λmaxT = GershgorinThermics2D( k, ρ, Cp, Δx, Δy, Δt, DM, transient )*0.39e-3  
        elseif PC==:ilu
            λmaxT = GershgorinThermics2D( k, ρ, Cp, Δx, Δy, Δt, DM, transient )*0.65e-3  
        elseif PC==:ilu0
            λmaxT = GershgorinThermics2D( k, ρ, Cp, Δx, Δy, Δt, DM, transient )*0.38e-3 
        else
            λmaxT = GershgorinThermics2D( k, ρ, Cp, Δx, Δy, Δt, DM, transient )
        end         
        λminT = 1.0

        hT   .= 2.0./sqrt.(λmaxT)*CFL
        cT    = 2.0*sqrt(λminT)*cfact

        if ismaxloc
            GershgorinThermics2D_Local!( λmaxlocT, k_maxloc, ρ, Cp, Δx, Δy, Δt, DM, transient ) 
            hT[2:end-1,2:end-1] .= 2.0./sqrt.(λmaxlocT)*CFL 
        end

        @views for iter=1:niter
            iters  += 1

            # Residuals
            ResidualThermics2D!(RT, Tc, Tc0, Qr, Δt, ρ, Cp, k, Δx, Δy, ∂T∂x, ∂T∂y, qTx, qTy, 1.0, T_South, T_North, DM, transient )

            if PC == :ichol
                # Apply PC
                RT[2:end-1,2:end-1] = reshape(L'\(L\RT[2:end-1,2:end-1][:]), Ncx, Ncy )
            elseif PC== :ilu || PC== :ilu0
                RT[2:end-1,2:end-1] = reshape(LU\RT[2:end-1,2:end-1][:], Ncx, Ncy )
            end

            @. ∂T∂τ                  = (2-cT*hT)/(2+cT*hT)*∂T∂τ + 2*hT/(2+cT*hT)*RT
            @. δT                    = hT*∂T∂τ
            @. Tc[2:end-1,2:end-1] .+= δT[2:end-1,2:end-1]

            if mod(iter, nout) == 0 
                ResidualThermics2D!(KδT1, Tc    , Tc0, Qr, Δt, ρ, Cp, k, Δx, Δy, ∂T∂x, ∂T∂y, qTx, qTy, 0.0, T_South, T_North, DM, transient )
                if PC == :ichol
                    KδT1[2:end-1,2:end-1] = reshape(L'\(L\KδT1[2:end-1,2:end-1][:]), Ncx, Ncy )
                elseif PC==:ilu || PC== :ilu0
                    KδT1[2:end-1,2:end-1] = reshape(LU\KδT1[2:end-1,2:end-1][:], Ncx, Ncy )
                end
                ResidualThermics2D!(KδT,  Tc.-δT, Tc0, Qr, Δt, ρ, Cp, k, Δx, Δy, ∂T∂x, ∂T∂y, qTx, qTy, 0.0, T_South, T_North, DM, transient )
                if PC == :ichol
                    KδT[2:end-1,2:end-1] = reshape(L'\(L\KδT[2:end-1,2:end-1][:]), Ncx, Ncy )
                elseif PC==:ilu || PC== :ilu0
                    KδT[2:end-1,2:end-1] = reshape(LU\KδT[2:end-1,2:end-1][:], Ncx, Ncy )
                end
                λminT =  abs(sum(.-δT.*(KδT1.-KδT))/sum(δT.*δT) / 1.0 )
                # λmaxT = GershgorinThermics2D( k, ρ, Cp, Δx, Δy, Δt, DM, transient )          
                if auto
                    # hT   .= 2/sqrt(λmaxT)*CFL 
                    cT    = 2*sqrt(λminT)*cfact
                end
                
                if ismaxloc
                    GershgorinThermics2D_Local!( λmaxlocT, k_maxloc, ρ, Cp, Δx, Δy, Δt, DM, transient )
                    hT[2:end-1,2:end-1]  .= 2.0./sqrt.(λmaxlocT)*CFL
                end
            
                ResidualThermics2D!(RT, Tc, Tc0, Qr, Δt, ρ, Cp, k, Δx, Δy, ∂T∂x, ∂T∂y, qTx, qTy, 1.0, T_South, T_North, DM, transient )             

                errT = norm(RT.*DM)/sqrt(length(RT))
                @printf("Iteration %05d --- Time step %4d --- Δt = %2.2e \n", iter, it, ustrip(dimensionalize(Δt, s, CharDim)))
                @printf("fT = %2.4e\n", errT)
                @show (λminT, λmaxT)
                if ( isnan(errT) ) error() end
                ( errT < ϵ  ) && break
            end
        end
        probes.t[it]     = t
        probes.maxT[it]  = maximum(Tc)
        probes.iters[it] = iters

        # Visualisation
        if mod(it, 10)==0 || it==1

            @show mean(Tdir)
            p0=heatmap(ustrip.(dimensionalize(xc[2:end-1], m, CharDim)./1e3), ustrip.(dimensionalize(yc[2:end-1], m, CharDim)./1e3), ustrip.(dimensionalize(reshape(Tdir, Ncx, Ncy), C, CharDim))', title = "Direct solution", aspect_ratio=1.0, color=:turbo )
            p1=heatmap(ustrip.(dimensionalize(xc, m, CharDim)./1e3), ustrip.(dimensionalize(yc, m, CharDim)./1e3), ustrip.(dimensionalize(Tc, C, CharDim))', title = "$(mean(probes.iters[1:it])) iterations", aspect_ratio=1.0, color=:turbo )
            p2=heatmap(ustrip.(dimensionalize(xv, m, CharDim)./1e3), ustrip.(dimensionalize(yv, m, CharDim)./1e3), log10.(ustrip.(dimensionalize(kv, J/s/m/K, CharDim)))', title = "$(mean(probes.iters[1:it])) iterations", aspect_ratio=1.0, color=:turbo )
            display(plot(p0,p1,p2))
        end
    end
    # return iters
end

# MainPoisson2D(1, true)
# MainPoisson2D(2, true)
# MainPoisson2D(4, true)
# MainPoisson2D(8, true)
# MainPoisson2D(16, true)|
# [6340 9120 10700 12070 14830]

MainPoisson2D(4, true)