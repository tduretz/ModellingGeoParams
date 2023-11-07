# using JLD2, CairoMakie

# let
#     data = load("example.jld2")
#     yc   = data["y"].c
#     yv   = data["y"].v
#     Vx   = data["vx"]
#     Pt   = data["pt"]
#     τxy  = data["τxy"]
#     θ    = data["θ"]
#     it   = data["it"]

#     f = Figure( fontsize=25 )
#     ax1 = Axis( f[1, 1] )
#     lines!(ax1, Pt./1e3, yv./1e3 )

#     ax2 = Axis( f[1, 2] )
#     lines!(ax2, Vx, yc./1e3 )

#     @show minimum(θ)
#     @show maximum(θ)

#     θ1 = 62.143438338297955*ones(size(yv)) .+ rand(length(yv)).*1e-9

#     @show minimum(θ1)
#     @show maximum(θ1)
    
#     # θ1 .= θ
#     @show sum(isnan.(θ))
#     @show sum(isinf.(θ))

    

#     ax3 = Axis( f[2, 1] )
#     lines!(ax3, θ1, yv./1e3 )

#     ax4 = Axis( f[2, 2] )
#     lines!(ax4, 1:it, τxy[1:it]./1e3 )

#     display(f)

# end

using CairoMakie
let 
    n = 10
    θ = 62.143438338297955*ones(n) .+ rand(n).*1e-9
    f   = Figure()
    ax2 = Axis( f[1, 1] )
    lines!(ax2, θ, 1:n)
    display(f)
end

# using Plots
# let 
#     n = 10
#     θ = 62.143438338297955*ones(n) .+ rand(n).*1e-9
#     plot(θ, 1:n)
# end