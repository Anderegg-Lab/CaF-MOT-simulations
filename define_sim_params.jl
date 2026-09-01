### DEFINE OTHER PARAMETERS FOR THE SIMULATION ###

sim_type = Float64

σx_initial = 1e-6
σy_initial = 1e-6
σz_initial = 1e-6
Tx_initial = 50e-6
Ty_initial = 50e-6
Tz_initial = 50e-6


using Serialization

import MutableNamedTuples: MutableNamedTuple
sim_params = MutableNamedTuple(
    #ODT params
    trap_scalar = 0.0,
    blue=false,
    # H_ODT_matrix = MMatrix{size(H_ODT_matrix)...}(sim_type.(H_ODT_matrix)),
    ODT_idxs = Vector{Tuple{Int,Int}}(),
    ODT_size = (1.8e-6, 9.6e-6, 1.8e-6), #ODT (30e-6, 2e-3, 30e-6), 
    ODT_position = [0.,0.],

    #magnetic field params
    Bz_gradient = 0.0,
    Zeeman_Hx = QuantumStates.MMatrix{size(Zeeman_x_mat)...}(sim_type.(Zeeman_x_mat)),
	Zeeman_Hy = QuantumStates.MMatrix{size(Zeeman_y_mat)...}(sim_type.(Zeeman_y_mat)),
	Zeeman_Hz = QuantumStates.MMatrix{size(Zeeman_z_mat)...}(sim_type.(Zeeman_z_mat)),

    #initial cloud params
    # x_dist = Normal(0, σx_initial),
    # y_dist = Normal(0, σy_initial),
    # z_dist = Normal(0, σz_initial),
    
    vx_dist = Normal(0, sqrt(kB*Tx_initial/2m)),
    vy_dist = Normal(0, sqrt(kB*Ty_initial/2m)),
    vz_dist = Normal(0, sqrt(kB*Tz_initial/2m)),
    
    x_dist = Uniform(0, σx_initial),
    y_dist = Uniform(0, σy_initial),
    z_dist = Uniform(0, σz_initial),
    
    # vx_dist = Uniform(0, sqrt(kB*Tx_initial/2m)),
    # vy_dist = Uniform(0, sqrt(kB*Ty_initial/2m)),
    # vz_dist = Uniform(0, sqrt(kB*Tz_initial/2m)),

    # f_z = StructArray(zeros(Complex{sim_type}, 32, 32)),

    dt_diffusion = 1e-7 / (2π/Γ)
)