using Plots, ForwardDiff
Lode(τII, J3) = -3.0*sqrt(3.0)/2.0*J3/τII^3

function πplanePlots()

    # von Mises
    τ0  = 1.0
    θ   = -((1/6)*π):0.001:((1/6)*π)
    # θ   = -π/2:0.001:π/2
    @show length(θ)
    τII = τ0.*ones(size(θ))
    p = plot( θ, sqrt(2).*τII, proj=:polar)

    # Mohr-Coulomb Abbo
    ϕ  = 30. *π/180
    σm = 0.0
    c  = 1.0
    θt = 25. *π/180

    k  =  (cos.(θ) .- 1 ./sqrt(3) .* sin(ϕ).*sin.(θ))
    for i in eachindex(θ)
        sgnθ = sign.(θ[i])
        A = 1/3*cos(θt)*(3+tan(θt)*tan(3*θt) + 1/sqrt(3)*sgnθ*(tan(3*θt)-3*tan(θt))*sin(ϕ))
        B = 1/(3*cos(3*θt))*(sgnθ*sin(θt) + 1/sqrt(3)*sin(ϕ)*cos(θt))
        if abs(θ[i])>θt
            k[i] = A - B*sin(3*θ[i])
        end
    end
    τII = (c*cos(ϕ) .- σm*sin(ϕ))./k
    
    p = plot!( θ, sqrt(2).* (τII), proj=:polar)
    display(p)
end


function Yield_MCAS95(τ, σm, ϕ, c, θt )
    τII = sqrt(0.5*(τ[1]^2 + τ[2]^2 + τ[3]^2) + τ[4]^2)
    J3  = τ[1]*τ[2]*τ[3] + τ[3]*τ[4]^2 + 2*τ[4]*τ[5]*τ[6] + τ[1]*τ[6]^2 + τ[2]*τ[5]^2
    L   = Lode(τII,J3)
    L> 1.0 ? L= 1.0 : nothing
    L<-1.0 ? L=-1.0 : nothing
    θ   =  1.0/3.0*asin(L)
    if abs(θ)>θt
        sgnθ = sign(θ)
        A = 1/3*cos(θt)*(3+tan(θt)*tan(3*θt) + 1/sqrt(3)*sgnθ*(tan(3*θt)-3*tan(θt))*sin(ϕ))
        B = 1/(3*cos(3*θt))*(sgnθ*sin(θt) + 1/sqrt(3)*sin(ϕ)*cos(θt))
        k = A - B*sin(3*θ)
    else
        k   = cos(θ) - 1/sqrt(3)*sin(ϕ)*sin(θ)
    end
    F   = σm*sin(ϕ) + k*τII - c*cos(ϕ)
    return F
end

function Fλ(λ̇, ∂Q∂τ, ϕ, ψ, c, θt, G, K, Δt, τxx0, τyy0, τzz0, τxy0, P0, ε̇xx, ε̇yy, ε̇zz, ε̇xy, ∇v)
    τxx  = τxx0 + 2*G*Δt*(ε̇xx - λ̇*∂Q∂τ[1])
    τyy  = τyy0 + 2*G*Δt*(ε̇yy - λ̇*∂Q∂τ[2])
    τzz  = τzz0 + 2*G*Δt*(ε̇zz - λ̇*∂Q∂τ[3])
    τxy  = τxy0 + 2*G*Δt*(ε̇xy - λ̇*∂Q∂τ[4])
    τxz  = 0.
    τyz  = 0.
    P    = P0   -   K*Δt*(∇v  - λ̇*sin(ψ) )
    σm   = -P
    F    = Yield_MCAS95([τxx; τyy; τzz; τxy; τxz; τyz], σm, ϕ, c, θt )
    return F
end


function main_MCAS95()

    K   = 3e10
    G   = 1e10
    c   = 1e7

    ϕ   = 35. * π/180
    ψ   = 10. * π/180
    ηvp = 0.
    θt  = 27. * π/180

    nt  = 100
    Δt  = 1e11

    ∂Vx∂x = -1.0e-15
    ∂Vy∂y =  0.9e-15
    ∂Vz∂z =  0.  
    ∂Vx∂y =  1e-15
    ∂Vy∂x =  ∂Vx∂y
    ∇v   =  (∂Vx∂x + ∂Vy∂y + ∂Vz∂z)

    ε̇xx  = ∂Vx∂x - 1.0/3.0*∇v
    ε̇yy  = ∂Vy∂y - 1.0/3.0*∇v
    ε̇zz  = ∂Vz∂z - 1.0/3.0*∇v
    ε̇xy  = 0.5*(∂Vx∂y + ∂Vy∂x)

    τxx = 0.
    τyy = 0.
    τzz = 0.
    τxy = 0.
    τxz = 0.
    τyz = 0.
    P   = 0.

    hist = (τII=zeros(nt), P=zeros(nt), σ1=zeros(nt), σ3=zeros(nt), θ=zeros(nt), iter=zeros(nt))

    for it=1:nt
        @info "Step $it $nt"
        τxx0 = τxx
        τyy0 = τyy
        τzz0 = τzz
        τxy0 = τxy
        P0   = P

        τxx  = τxx0 + 2*G*Δt*ε̇xx
        τyy  = τyy0 + 2*G*Δt*ε̇yy
        τzz  = τzz0 + 2*G*Δt*ε̇zz
        τxy  = τxy0 + 2*G*Δt*ε̇xy
        P    = P0   -   K*Δt*∇v

        σm  = -P
        τII = sqrt(0.5*(τxx^2 + τyy^2 + τzz^2) + τxy^2)
        J3  = τxx*τyy*τzz + 2*τxy*τxz*τyz + τxx*τyz^2 + τyy*τxz^2 + τzz*τxy^2
        L   = Lode(τII,J3)
        L> 1.0 ? L= 1.0 : nothing
        L<-1.0 ? L=-1.0 : nothing
        θ   =  1.0/3.0*asin(L)
        -30>θ*180/π>30 ? error("θ is out of limit") : nothing

        σ1 = 2/sqrt(3)*τII*sin(θ + 2/3*π) + σm
        σ2 = 2/sqrt(3)*τII*sin(θ)         + σm
        σ3 = 2/sqrt(3)*τII*sin(θ - 2/3*π) + σm

        F  = Yield_MCAS95([τxx; τyy; τzz; τxy; τxz; τyz], σm, ϕ, c, θt )
        𝐹τ = τ -> Yield_MCAS95(τ, σm, ϕ, c, θt )

        iter = 0
        if F>0
            @info "Yield at step $it"
            α     = LinRange(0.1, 1.0, 5)
            Fmin  = zero(α)
            λ̇, F0 = 0.0, 0.0
            ∂F∂τ  = ForwardDiff.gradient( 𝐹τ, [τxx; τyy; τzz; τxy; τxz; τyz])
            
            for _=1:200
                iter +=1
                F     = Fλ(λ̇, ∂F∂τ, ϕ, ψ, c, θt, G, K, Δt, τxx0, τyy0, τzz0, τxy0, P0, ε̇xx, ε̇yy, ε̇zz, ε̇xy, ∇v)
                iter==1 ? F0 = F :  nothing
                # @show F/F0
                abs(F/F0) < 1e-10 ? break : nothing
                𝐹𝜆    = λ̇ -> Fλ(λ̇, ∂F∂τ, ϕ, ψ, c, θt, G, K, Δt, τxx0, τyy0, τzz0, τxy0, P0, ε̇xx, ε̇yy, ε̇zz, ε̇xy, ∇v)
                ∂F∂λ̇  = ForwardDiff.derivative(𝐹𝜆, λ̇)
                Δλ̇    = F/∂F∂λ̇
                Fmin .= 𝐹𝜆.(λ̇ .- α*Δλ̇)
                _,imin = findmin(Fmin)
                λ̇    -= α[imin]*Δλ̇
            end
            τxx  = τxx0 + 2*G*Δt*(ε̇xx - λ̇*∂F∂τ[1])
            τyy  = τyy0 + 2*G*Δt*(ε̇yy - λ̇*∂F∂τ[2])
            τzz  = τzz0 + 2*G*Δt*(ε̇zz - λ̇*∂F∂τ[3])
            τxy  = τxy0 + 2*G*Δt*(ε̇xy - λ̇*∂F∂τ[4])
            τxz  = 0.
            τyz  = 0.
            P    = P0   -   K*Δt*(∇v  - λ̇*sin(ϕ) )
            σm  = -P
            τII = sqrt(0.5*(τxx^2 + τyy^2 + τzz^2) + τxy^2)
            J3  = τxx*τyy*τzz + 2*τxy*τxz*τyz + τxx*τyz^2 + τyy*τxz^2 + τzz*τxy^2
            L   = Lode(τII,J3)
            L> 1.0 ? L= 1.0 : nothing
            L<-1.0 ? L=-1.0 : nothing
            θ   =  1.0/3.0*asin(L)
            @show abs(θ*180/π)
            -30>θ*180/π>30 ? error("θ is out of limit") : nothing
        end

        σ1 = 2/sqrt(3)*τII*sin(θ + 2/3*π) + σm
        σ2 = 2/sqrt(3)*τII*sin(θ)         + σm
        σ3 = 2/sqrt(3)*τII*sin(θ - 2/3*π) + σm

        hist.τII[it]  = τII
        hist.P[it]    = P
        hist.σ1[it]   = σ1
        hist.σ3[it]   = σ3
        hist.θ[it]    = θ
        hist.iter[it] = iter
        @show (σ1, σ2, σ3)
        σ2 > σ1 ? error("σ2 > σ1") : nothing

        if mod(it, 1)==0

            θ   = -((1/6)*π):0.001:((1/6)*π)
            k  =  (cos.(θ) .- 1 ./sqrt(3) .* sin(ϕ).*sin.(θ))
            for i in eachindex(θ)
                sgnθ = sign.(θ[i])
                A = 1/3*cos(θt)*(3+tan(θt)*tan(3*θt) + 1/sqrt(3)*sgnθ*(tan(3*θt)-3*tan(θt))*sin(ϕ))
                B = 1/(3*cos(3*θt))*(sgnθ*sin(θt) + 1/sqrt(3)*sin(ϕ)*cos(θt))
                if abs(θ[i])>θt
                    k[i] = A - B*sin(3*θ[i])
                end
            end
            τII = (c*cos(ϕ) .- σm*sin(ϕ))./k
            p4 = plot( θ,  (τII), proj=:polar)
            p4 = scatter!( hist.θ[1:it], hist.τII[1:it], proj=:polar)

            p1 = plot( hist.P[1:it], hist.τII[1:it], xlabel="P", ylabel="τII")
            p2 = plot( hist.σ1[1:it], hist.σ3[1:it], xlabel="σ1", ylabel="σ3")
            p3 = scatter( 1:it, hist.iter[1:it]) 
            display(plot(p1, p2, p3, p4))
            sleep(0.1)
        end
    end
    # s3 = LinRange(-3e8, 0., 100)
    # s1 = (2*c*cos(ϕ) .+ (1-sin(ϕ)).*s3) ./ (1+sin(ϕ))

    # p1 = plot(  1:nt, hist.τII, xlabel="step", ylabel="τII")
    # p2 = plot(  1:nt, hist.P, xlabel="step", ylabel="P")
    # p3 = plot(s3, s1)
    # # p3 = scatter!( hist.σ3, hist.σ1, xlabel="σ3", ylabel="σ1")
    # p4 = scatter( 1:nt, hist.iter, xlabel="step", ylabel="iter") 
    # display(plot(p1, p2, p3, p4))
end

main_MCAS95()
# πplanePlots()