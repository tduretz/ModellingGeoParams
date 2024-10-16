using GeoParams, Plots, Printf, MathTeXEngine, BenchmarkTools, Statistics, SparseArrays
import LinearAlgebra: norm, dot
using ILUZero

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

function ApplyPC!(R_PC, R, L, LU, DM, PC)
    Ncx, Ncy = size(R,1)-2, size(R,2)-2
    if PC == :ichol
        # Apply PC
        R_PC[2:end-1,2:end-1] = reshape(L'\(L\R[2:end-1,2:end-1][:]), Ncx, Ncy )
    elseif PC== :ilu 
        R_PC[2:end-1,2:end-1] = reshape(LU\R[2:end-1,2:end-1][:], Ncx, Ncy )
    elseif PC==:diag
        R_PC[2:end-1,2:end-1] = R[2:end-1,2:end-1]./DM[2:end-1,2:end-1]
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

@views function ResidualThermics2D!(RT, Tc, Tc0, Qr, Δt, ρ, Cp, k, Δx, Δy, ∂T∂x, ∂T∂y, qTx, qTy, rhs, T_South, T_North, DM, transient )
    
    Tc[:,1]   .= -Tc[:,2]     .+ 2*T_South*rhs
    Tc[:,end] .= -Tc[:,end-1] .+ 2*T_North*rhs 
    Tc[1,:]   .= Tc[2,:]
    Tc[end,:] .= Tc[end-1,:]

    @. ∂T∂x        = (Tc[2:end,2:end-1] - Tc[1:end-1,2:end-1])/Δx
    @. qTx         = -k.x .* ∂T∂x
    @. ∂T∂y        = (Tc[2:end-1,2:end] - Tc[2:end-1,1:end-1])/Δy
    @. qTy         = -k.y * ∂T∂y
    @. RT[2:end-1,2:end-1] = -transient*(Tc[2:end-1,2:end-1] - rhs*Tc0[2:end-1,2:end-1]) / Δt - 1.0/(ρ*Cp) * (qTx[2:end,:] - qTx[1:end-1,:])/Δx - 1.0/(ρ*Cp) * (qTy[:,2:end] - qTy[:,1:end-1])/Δy + rhs/(ρ*Cp) *  Qr[2:end-1,2:end-1]
end

@views function MainPoisson2D(n, auto, solver, ϵ2PCG, PC, nout)

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
    RT_PC      =   zero(RT)
    RT_PC0     =   zero(RT)
    Qr         =   zeros(Ncx+2, Ncy+2) .+ Qr0.*exp.(-((yc').-2.2).^2/2σ^2 .- xc.^2/(2σ^2)) 
    ∂T∂τ       =   zeros(Ncx+2, Ncy+2)
    DM         =   ones(Ncx+2, Ncy+2)
    p          = zero(RT)
    Ap         = zero(RT)
    L, LU      = 0., 0.                # dummies
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
        
    # Monitoring
    probes    = (iters = zeros(Nt), t = zeros(Nt), Ẇ0 = zeros(Nt), τxyi = zeros(Nt), Vx0 = zeros(Nt), maxT = zeros(Nt))

    # PT solver
    niter  = 25000
    ϵ      = 1e-11
    CFL    = 0.99
    cfact  = 0.9
    err    = zeros(niter)

    # Direct solution
    b    = zeros(Ncx,Ncy)
    b   .= Qr[2:end-1,2:end-1]/ρ/Cp
    M    = ConstructM(k, ρ, Cp, Δx, Δy, Δt, transient, b, T_South, T_North)
    Tdir = (M\b[:])

    # Construct preconditioner
    if PC == :ichol
        L = copy(M)
        @time L = run_incomplete_cholesky3(M, size(M,1))
    elseif PC== :ilu
        @time LU = ilu0(M)
    elseif PC== :diag
        DiagThermics2D!( DM, k, ρ, Cp, Δx, Δy, Δt, transient ) 
    end

    iters = 0
    for it=1:Nt
        # History
        @. Tc0  = Tc
        
        # PT steps
        t       += Δt 

        # DYREL
        PC_fact = 1.0
        PC == :ichol ? PC_fact = 0.39e-3 : nothing
        PC == :ilu   ? PC_fact = 0.39e-3 : nothing
        λmaxT = GershgorinThermics2D( k, ρ, Cp, Δx, Δy, Δt, DM, transient )*PC_fact          
        λminT = 1.0
        hT    = 2.0./sqrt(maximum(λmaxT))*CFL
        cT    = 2.0*sqrt(λminT)*cfact

        # Preconditioned residual
        ResidualThermics2D!(RT, Tc, Tc0, Qr, Δt, ρ, Cp, k, Δx, Δy, ∂T∂x, ∂T∂y, qTx, qTy, 1.0, T_South, T_North, DM, transient )
        ApplyPC!(RT_PC, RT, L, LU, DM, PC)
        ∂T∂τ .= RT_PC
        
        a = (2-cT*hT)/(2+cT*hT)
        b = 2*hT/(2+cT*hT)
        hb = b*hT

        errT0       = 1.0
        switched2CG = false

        ################################### Iterative solver ###################################
        @time @views for iter=1:niter
            iters  += 1

            RT_PC0 .= RT_PC

            if solver==:PCG 
                # Ap
                ResidualThermics2D!(Ap, ∂T∂τ, Tc0, Qr, Δt, ρ, Cp, k, Δx, Δy, ∂T∂x, ∂T∂y, qTx, qTy, 0.0, T_South, T_North, DM, false )
                Ap *= -1.
                # α
                dot0 = dot(RT[2:end-1,2:end-1], RT_PC[2:end-1,2:end-1])
                α_CG = dot0 / dot(∂T∂τ[2:end-1,2:end-1], Ap[2:end-1,2:end-1]) 
                hb   = α_CG
            end
                    
            # Updates 1
            Tc[2:end-1,2:end-1] .+= hb .* ∂T∂τ[2:end-1,2:end-1] 
            ResidualThermics2D!(RT, Tc, Tc0, Qr, Δt, ρ, Cp, k, Δx, Δy, ∂T∂x, ∂T∂y, qTx, qTy, 1.0, T_South, T_North, DM, transient )

            # New preconditioned residual
            ApplyPC!(RT_PC, RT, L, LU, DM, PC)
            
            # β
            if solver==:PCG 
                β_CG = dot(RT[2:end-1,2:end-1], RT_PC[2:end-1,2:end-1]) / dot0
                a    = β_CG
            end

            # Set DYREL parameters
            if  solver==:DYREL && (mod(iter, nout) == 0 || iter==1)
                λminT =  abs(sum(.-hb*∂T∂τ.*(RT_PC .- RT_PC0))/sum((hb*∂T∂τ).^2)*0.8 )
                auto ?  cT    = 2*sqrt(λminT)*cfact : nothing
                a  = (2-cT*hT)/(2+cT*hT)
                b  = 2*hT/(2+cT*hT)
                hb = b*hT
            end

            # Update 2
            ∂T∂τ[2:end-1,2:end-1]  .= RT_PC[2:end-1,2:end-1] .+ a.*∂T∂τ[2:end-1,2:end-1]
            (iters ==374) && @show norm(∂T∂τ)
            
            # Check (always)
            errT = norm(RT)/sqrt(length(RT))
            iter==1 ? errT0 = errT : nothing
            (mod(iter, nout) == 0 || iter==1) ? @printf("Iteration %05d --- Time step %4d --- Δt = %2.2e \n", iter, it, ustrip(dimensionalize(Δt, s, CharDim)))  : nothing
            (mod(iter, nout) == 0 || iter==1) ? @printf("fT = %2.4e\n", errT/errT0) : nothing
            err[iter] = errT/errT0
            if ( isnan(errT/errT0) ) error() end
            ( errT/errT0 < ϵ  ) && break
            if ( solver==:DYREL && errT/errT0 < ϵ2PCG && switched2CG == false ) 
                solver      = :PCG 
                ApplyPC!(RT_PC, RT, L, LU, DM, PC)
                ∂T∂τ       .= RT_PC
                switched2CG = true
            end
        end

        probes.t[it]     = t
        probes.maxT[it]  = maximum(Tc)
        probes.iters[it] = iters

        # Visualisation
        if mod(it, 10)==0 || it==1
            p0=heatmap(ustrip.(dimensionalize(xc[2:end-1], m, CharDim)./1e3), ustrip.(dimensionalize(yc[2:end-1], m, CharDim)./1e3), ustrip.(dimensionalize(reshape(Tdir, Ncx, Ncy), C, CharDim))', title = "Direct solution", aspect_ratio=1.0, color=:turbo )
            p1=heatmap(ustrip.(dimensionalize(xc, m, CharDim)./1e3), ustrip.(dimensionalize(yc, m, CharDim)./1e3), ustrip.(dimensionalize(Tc, C, CharDim))', title = "$(mean(probes.iters[1:it])) iterations", aspect_ratio=1.0, color=:turbo )
            p2=heatmap(ustrip.(dimensionalize(xv, m, CharDim)./1e3), ustrip.(dimensionalize(yv, m, CharDim)./1e3), log10.(ustrip.(dimensionalize(kv, J/s/m/K, CharDim)))', title = "$(mean(probes.iters[1:it])) iterations", aspect_ratio=1.0, color=:turbo )
            p4=plot(1:iters,log10.(err[1:iters]))
            display(plot(p0,p1,p2,p4))
        end
    end
    return iters
end

@info "DYREL"
MainPoisson2D(4, true, :DYREL, 1e-30, :diag,  100)
MainPoisson2D(4, true, :DYREL, 1e-30, :ilu,   100)
MainPoisson2D(4, true, :DYREL, 1e-30, :ichol, 100)

ϵ2PCG = 1e-6
@info "DYREL 2 CG if error below $(ϵ2PCG)"
MainPoisson2D(4, true, :DYREL, ϵ2PCG, :diag,  100)
MainPoisson2D(4, true, :DYREL, ϵ2PCG, :ilu,   100)
MainPoisson2D(4, true, :DYREL, ϵ2PCG, :ichol, 100)

@info "PCG"
MainPoisson2D(4, true, :PCG, 1e-30, :diag,  100)
MainPoisson2D(4, true, :PCG, 1e-30, :ilu,   100)
MainPoisson2D(4, true, :PCG, 1e-30, :ichol, 100)