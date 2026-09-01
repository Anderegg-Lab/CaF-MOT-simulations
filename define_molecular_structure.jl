using Serialization

states = deserialize("states_cooling_repump_sim.jl")
ground_states_v0 = states[1:12]
ground_states_v1 = states[13:24]
A_states = states[25:28]
ground_states = states[1:24]
excited_states = A_states
d = zeros(ComplexF64, length(states), length(states), 3)

f00_A = 0.978
f01_A = 1-f00_A

d[1:12, 25:28, :] .= sqrt(f00_A).*tdms_between_states(ground_states_v0, A_states)
d[25:28, 1:12, :] .= sqrt(f00_A).*tdms_between_states(A_states, ground_states_v0)
d[13:24, 25:28, :] .= sqrt(f01_A).*tdms_between_states(ground_states_v1, A_states)
d[25:28, 13:24, :] .= sqrt(f01_A).*tdms_between_states(A_states, ground_states_v1)


# Define constants for the laser cooling transition
@everywhere begin
    @consts begin
        λ = 606e-9
		Γ = 2π * 8.3e6
		m = UnitsToValue.@with_unit 59 "u"
		k = 2π / λ
    end
end

# Zeeman matrices precomputed by precompute_zeeman.jl (unitless, normalized by Γ)
(Zeeman_x_mat, Zeeman_y_mat, Zeeman_z_mat) = deserialize("Zeeman_matrices.jl")

@everywhere import LoopVectorization: @turbo
@everywhere function add_terms_dψ!(dψ, ψ, p, r, t, gaussian_trap_scalar=0)
    # Apply AC Stark shifts to ground and excited states wavefunction
    if p.sim_params.trap_scalar !=0
        @turbo for i ∈ 1:p.n_states
            dψ_i_re = zero(eltype(dψ.re))
            dψ_i_im = zero(eltype(dψ.im))
            for j ∈ 1:p.n_states
                ψ_i_re = ψ.re[j]
                ψ_i_im = ψ.im[j]
                
                H_re = gaussian_trap_scalar*p.sim_params.trap_scalar * p.sim_params.H_ODT_matrix[i,j]
                
                dψ_i_re += ψ_i_re * H_re
                dψ_i_im += ψ_i_im * H_re
                
            end
            dψ.re[i] += dψ_i_im
            dψ.im[i] -= dψ_i_re
        end
    end
    # Apply Zeeman shifts to all ground states (v=0 and v=1)
    if p.sim_params.Bz_gradient !=0
        Bx = r[1]*p.sim_params.Bz_gradient*1e2 / k / 2 #unitsless, Bz_gradient should be in G/cm
        By = r[2]*p.sim_params.Bz_gradient*1e2 / k / 2
        Bz = -1*r[3]*p.sim_params.Bz_gradient*1e2 / k
        # v=0 ground states (1:12)
        @turbo for i ∈ 1:12
            dψ_i_re = zero(eltype(dψ.re))
            dψ_i_im = zero(eltype(dψ.im))
            for j ∈ 1:12
                ψ_i_re = ψ.re[j]
                ψ_i_im = ψ.im[j]

                H_re = Bx * p.sim_params.Zeeman_Hx[i, j] + Bz * p.sim_params.Zeeman_Hz[i, j]
                H_im = By * p.sim_params.Zeeman_Hy[i, j]

                dψ_i_re += ψ_i_re * H_re - ψ_i_im * H_im
                dψ_i_im += ψ_i_re * H_im + ψ_i_im * H_re
            end
            dψ.re[i] += dψ_i_im
            dψ.im[i] -= dψ_i_re
        end
        # v=1 ground states (13:24) — same Zeeman matrices, offset indices
        @turbo for i ∈ 1:12
            dψ_i_re = zero(eltype(dψ.re))
            dψ_i_im = zero(eltype(dψ.im))
            for j ∈ 1:12
                ψ_i_re = ψ.re[j+12]
                ψ_i_im = ψ.im[j+12]

                H_re = Bx * p.sim_params.Zeeman_Hx[i, j] + Bz * p.sim_params.Zeeman_Hz[i, j]
                H_im = By * p.sim_params.Zeeman_Hy[i, j]

                dψ_i_re += ψ_i_re * H_re - ψ_i_im * H_im
                dψ_i_im += ψ_i_re * H_im + ψ_i_im * H_re
            end
            dψ.re[i+12] += dψ_i_im
            dψ.im[i+12] -= dψ_i_re
        end
    end
    return nothing
end
