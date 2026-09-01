# DEFINE STATES #
energy_offset = (2π / Γ) * energy(states[13])

energies = energy.(states) .* (2π / Γ)

# DEFINE FREQUENCIES #
detuning = 0 # in MHz
Δ = detuning *1e6 #in Hz
pol_F1⁻ = σ⁻
pol_F0  = σ⁻
pol_F1⁺ = σ⁻
pol_F2  = σ⁺

A_energy = energy(states[25])
f_F1⁻ = A_energy - energy(states[1]) + Δ
f_F0  = A_energy - energy(states[4]) + Δ
f_F1⁺ = A_energy - energy(states[5]) + Δ
f_F2  = A_energy - energy(states[8]) + Δ

f_v1_F1⁻ = A_energy - energy(states[13]) + Δ
f_v1_F0  = A_energy - energy(states[16]) + Δ
f_v1_F1⁺ = A_energy - energy(states[17]) + Δ
f_v1_F2  = A_energy - energy(states[20]) + Δ

f_v1_v0 = energy(states[16]) - energy(states[4])

freqs = [f_F1⁻, f_F0, f_F1⁺, f_F2, f_v1_F1⁻, f_v1_F0, f_v1_F1⁺, f_v1_F2] .* (2π / Γ) # normalized by Γ

# RWA frequency groups: (laser freq indices, target ground state indices)
# Group 1: v=0 lasers (freqs 1-4) → v=0 ground states (1-12)
# Group 2: v=1 lasers (freqs 5-8) → v=1 ground states (13-24)
freq_groups = [(1:4, 1:12), (5:8, 13:24)]


# DEFINE SATURATION INTENSITIES #
iris_factor = 1.5  # hard cutoff at iris_factor × beam_radius; Inf = no cutoff
beam_radius = 1e-2
Isat = π*h*c*Γ/(3λ^3)
P = 5e-3
I = 2P / (π * beam_radius^2)

s = I / Isat

sats = [s, s, s, s, s, s, s, s]

# DEFINE POLARIZATIONS #
pols = [pol_F1⁻, pol_F0, pol_F1⁺, pol_F2, pol_F1⁻, pol_F0, pol_F1⁺, pol_F2]

# DEFINE FUNCTION TO UPDATE PARAMETERS DURING SIMULATION #
@everywhere function update_p!(p, r, t)
    return nothing
end

@everywhere function update_p_diffusion!(p, r, t)
    return nothing
end
