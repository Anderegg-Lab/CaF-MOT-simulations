# PROBLEM TO CALCULATE TRAJECTORIES #
include("cooling_params.jl")

t_start = 0.0
t_end   = 40e-3
t_span  = (t_start, t_end) ./ (1/Γ)

# p = initialize_prob(sim_type, energies, freqs, sats, pols, beam_radius, d, m/(ħ*k^2/Γ), Γ, k, sim_params, update_p!, add_terms_dψ!; freq_groups=freq_groups)
p = initialize_prob(sim_type, energies, freqs, sats, pols, beam_radius, d, m/(ħ*k^2/Γ), Γ, k, sim_params, update_p!, add_terms_dψ!; freq_groups=freq_groups, iris_factor=iris_factor)

cb1 = DiscreteCallback(condition_discrete, stochastic_collapse_new!, save_positions=(false,false))
cb2 = DiscreteCallback(terminate_condition_MOT_capture, terminate!)
cbs = CallbackSet(cb1, cb2)

kwargs = (alg=DP5(), reltol=1e-3, abstol=3e-5, saveat=1000, maxiters=1000000000, callback=cbs)

prob = ODEProblem(ψ_fast_grouped!, p.u0, sim_type.(t_span), p; kwargs...)
prob.p.add_spontaneous_decay_kick = true
