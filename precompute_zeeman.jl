
using QuantumStates, UnitsToValue, Serialization

states = deserialize("states_cooling_repump_sim.jl")
ground_states_v0 = states[1:12]

Γ = 2π * 8.3e6

Zeeman_x(state, state′) = (QuantumStates.Zeeman(state, state′, -1) - QuantumStates.Zeeman(state, state′, 1)) / √2
Zeeman_y(state, state′) = im * (QuantumStates.Zeeman(state, state′, -1) + QuantumStates.Zeeman(state, state′, 1)) / √2
Zeeman_z(state, state′) = QuantumStates.Zeeman(state, state′, 0)

scale = 1e-4 * gS * μB * (2π / Γ) / h

Zeeman_x_mat = real.(operator_to_matrix(Zeeman_x, ground_states_v0) .* scale)
Zeeman_y_mat = imag.(operator_to_matrix(Zeeman_y, ground_states_v0) .* scale)
Zeeman_z_mat = real.(operator_to_matrix(Zeeman_z, ground_states_v0) .* scale)

serialize("Zeeman_matrices.jl", (Zeeman_x_mat, Zeeman_y_mat, Zeeman_z_mat))

println("Saved Zeeman_matrices.jl ($(size(Zeeman_x_mat)) matrices)")
