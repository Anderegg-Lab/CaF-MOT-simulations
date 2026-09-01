@everywhere begin
    using LinearAlgebra, QuantumStates, OpticalBlochEquations, DifferentialEquations, UnitsToValue, StructArrays, StaticArrays, Parameters, ProgressMeter, Plots, Serialization
    import Distributions: Normal, Geometric, Exponential, Uniform
    include("helper_functions.jl")
    include("define_molecular_structure.jl")
    include("define_sim_params.jl")
    include("define_prob.jl")
    include("compute_size_temperature.jl")
end

function flip(ϵ)
    return SVector{3,ComplexF64}(ϵ[3],-ϵ[2],ϵ[1])
end

@everywhere function update_red_MOT_params!(prob, detuning, P, beam_radius, Bz_gradient, s_ratios, pol_config=1)
    prob.p.denom = sim_type((beam_radius*k)^2/2)

    prob.p.sim_params.Bz_gradient = Bz_gradient

    pol_F1⁻ = σ⁻
    pol_F0  = σ⁻
    pol_F1⁺ = σ⁻
    pol_F2  = σ⁺
    if pol_config == 2
        pol_F1⁻ = σ⁺
        pol_F0  = σ⁺
        pol_F1⁺ = σ⁺
        pol_F2  = σ⁻
    end
    pols = [pol_F1⁻, pol_F0, pol_F1⁺, pol_F2]
    n_freqs = length(prob.p.ωs)
    ϵs = zeros(Complex{sim_type},6,n_freqs,3)
    for i ∈ eachindex(pols)
        pol = pols[i]
        ϵs[1,i,:] .= rotate_pol(pol, x̂)
        ϵs[2,i,:] .= rotate_pol(pol, ŷ)
        ϵs[3,i,:] .= rotate_pol(flip(pol), ẑ)
        ϵs[4,i,:] .= rotate_pol(pol, -x̂)
        ϵs[5,i,:] .= rotate_pol(pol, -ŷ)
        ϵs[6,i,:] .= rotate_pol(flip(pol), -ẑ)
    end
    ϵs = StructArray(MArray{Tuple{6,n_freqs,3}}(ϵs))
    prob.p.ϵs = ϵs

    Δ = detuning *1e6 #in Hz
    A_energy = energy(states[13])
    f_F1⁻ = A_energy - energy(states[1]) + Δ
    f_F0  = A_energy - energy(states[4]) + Δ
    f_F1⁺ = A_energy - energy(states[5]) + Δ
    f_F2  = A_energy - energy(states[8]) + Δ
    freqs = [f_F1⁻, f_F0, f_F1⁺, f_F2] .* (2π / Γ) # angular units, normalized by Γ
    for i ∈ eachindex(freqs)
        prob.p.ωs[i] = freqs[i]
    end

    Isat = π * h * c * Γ / (3 * λ^3)
    I = 2 * P / (π * beam_radius^2)
    total_sat = I / Isat
    s_ratios_denom = sum(s_ratios)
    for i ∈ eachindex(s_ratios)
        prob.p.sats[i] = s_ratios[i]/s_ratios_denom*total_sat
    end

    return nothing
end
@everywhere function update_red_MOT_with_repump_params!(prob, detuning, detuning_repump, P, P_repump, beam_radius, Bz_gradient, s_ratios, s_ratios_repump, pol_config=1, pol_config_repump=1; iris_factor=prob.p.iris_factor)
    prob.p.denom = sim_type((beam_radius*k)^2/2)
    prob.p.iris_factor = sim_type(iris_factor)

    prob.p.sim_params.Bz_gradient = Bz_gradient

    pol_F1⁻ = σ⁻
    pol_F0  = σ⁻
    pol_F1⁺ = σ⁻
    pol_F2  = σ⁺

    pol_v1_F1⁻ = σ⁻
    pol_v1_F0  = σ⁻
    pol_v1_F1⁺ = σ⁻
    pol_v1_F2  = σ⁺

    if pol_config == 2
        pol_F1⁻ = σ⁺
        pol_F0  = σ⁺
        pol_F1⁺ = σ⁺
        pol_F2  = σ⁻
    end

    if pol_config_repump == 2
        pol_v1_F1⁻ = σ⁺
        pol_v1_F0  = σ⁺
        pol_v1_F1⁺ = σ⁺
        pol_v1_F2  = σ⁻
    end
    pols = [pol_F1⁻, pol_F0, pol_F1⁺, pol_F2, pol_v1_F1⁻, pol_v1_F0, pol_v1_F1⁺, pol_v1_F2]
    n_freqs = length(prob.p.ωs)
    ϵs = zeros(Complex{sim_type},6,n_freqs,3)
    for i ∈ eachindex(pols)
        pol = pols[i]
        ϵs[1,i,:] .= rotate_pol(pol, x̂)
        ϵs[2,i,:] .= rotate_pol(pol, ŷ)
        ϵs[3,i,:] .= rotate_pol(flip(pol), ẑ)
        ϵs[4,i,:] .= rotate_pol(pol, -x̂)
        ϵs[5,i,:] .= rotate_pol(pol, -ŷ)
        ϵs[6,i,:] .= rotate_pol(flip(pol), -ẑ)
    end
    ϵs = StructArray(MArray{Tuple{6,n_freqs,3}}(ϵs))
    prob.p.ϵs = ϵs

    Δ = detuning *1e6 #in Hz
    Δ_v1 = detuning_repump*1e6
    A_energy = energy(states[25])
    f_F1⁻ = A_energy - energy(states[1]) + Δ
    f_F0  = A_energy - energy(states[4]) + Δ
    f_F1⁺ = A_energy - energy(states[5]) + Δ
    f_F2  = A_energy - energy(states[8]) + Δ

    f_v1_F1⁻ = A_energy - energy(states[13]) + Δ_v1
    f_v1_F0  = A_energy - energy(states[16]) + Δ_v1
    f_v1_F1⁺ = A_energy - energy(states[17]) + Δ_v1
    f_v1_F2  = A_energy - energy(states[20]) + Δ_v1

    freqs = [f_F1⁻, f_F0, f_F1⁺, f_F2, f_v1_F1⁻, f_v1_F0, f_v1_F1⁺, f_v1_F2] .* (2π / Γ) # angular units, normalized by Γ
    for i ∈ eachindex(freqs)
        prob.p.ωs[i] = freqs[i]
    end
    Isat = π * h * c * Γ / (3 * λ^3)
    λ_repump = 628.55e-9
    Isat_repump = π * h * c * Γ / (3 * λ_repump^3)
    I = 2 * P / (π * beam_radius^2)
    I_repump = 2 * P_repump / (π * beam_radius^2)
    total_sat = I / Isat
    total_sat_repump = I_repump / Isat_repump
    s_ratios_denom = sum(s_ratios)
    s_ratios_denom_repump = sum(s_ratios_repump)
    for i ∈ eachindex(s_ratios)
        prob.p.sats[i] = s_ratios[i]/s_ratios_denom*total_sat
    end
    for i ∈ eachindex(s_ratios_repump)
        prob.p.sats[length(s_ratios)+i] = s_ratios_repump[i]/s_ratios_denom_repump*total_sat_repump
    end

    return nothing
end

function MOT_captured(all_sols)
    num = 0
    idxs = []
    if isfinite(prob.p.iris_factor)
        _factor = prob.p.iris_factor
    else
        _factor = 1.5
    end
    for (idx, sol) ∈ enumerate(all_sols)
        if sol.retcode == ReturnCode.Success
            _x, _y, _z = x(sol.u[end]), y(sol.u[end]), z(sol.u[end])
            if (abs(_x) <= _factor*sqrt(2*prob.p.denom)/k) &&
                (abs(_y) <= _factor*sqrt(2*prob.p.denom)/k) &&
                (abs(_z) <= _factor*sqrt(2*prob.p.denom)/k)
                num += 1
                push!(idxs, idx)
            end
        end
    end
    return num, idxs
end

function MOT_captured_arb_time(all_sols, time_idx)
    num = 0
    idxs = []
    if isfinite(prob.p.iris_factor)
        _factor = prob.p.iris_factor
    else
        _factor = 1.5
    end
    for (idx, sol) ∈ enumerate(all_sols)
        if sol.retcode == ReturnCode.Success
            _x, _y, _z = x(sol.u[time_idx]), y(sol.u[time_idx]), z(sol.u[time_idx])
            if (abs(_x) <= _factor*sqrt(2*prob.p.denom)/k) &&
                (abs(_y) <= _factor*sqrt(2*prob.p.denom)/k) &&
                (abs(_z) <= _factor*sqrt(2*prob.p.denom)/k)
                num += 1
                push!(idxs, idx)
            end
        end
    end
    return num, idxs
end

function MOT_captured_vs_time(all_sols)
    _, max_time_idx = findmax(sol.t for sol ∈ all_sols)
    times = all_sols[max_time_idx].t
    captured_nums = zeros(length(times))
    captured_idxs = []
    for i ∈ eachindex(times)
        captured_nums[i], idxs = MOT_captured_arb_time(all_sols, i)
        push!(captured_idxs, idxs)
    end
    return captured_nums, captured_idxs
end

function get_data(sols, hist_binning=0.001, temp_fit_guess = [1, 10e-6], MOT_temp_binning = 0.01, MOT_temp_fit_guess = [1, 1e-3], max_v=7.5, σ_min = -1e-2, σ_binning=1e-4, σ_max = 1e-2, MOT_size_fit_guess = [1e-3, 0., 1000])
    data = MutableNamedTuple(
        xs = [],
        ys = [],
        zs = [],
        vxs = [],
        vys = [],
        vzs = [],
        Ts = [],
        captured_nums = [],
        captured_idxs = [],
        T_final = 0.0,
        captured_num_final = 0,
        captured_idxs_final = [],
        MOT_captured_nums = [],
        MOT_captured_idxs = [],
        MOT_captured_num_final = 0,
        MOT_captured_idxs_final = [],
        retcodes = [],
        MOT_Ts = [],
        MOT_T_final = 0.0,
        σs = [],
        σ_final = 0.0,
    )
    data.xs = [x.(sol.u) for sol ∈ sols]
    data.ys = [y.(sol.u) for sol ∈ sols]
    data.zs = [z.(sol.u) for sol ∈ sols]
    data.vxs = [vx.(sol.u) for sol ∈ sols]
    data.vys = [vy.(sol.u) for sol ∈ sols]
    data.vzs = [vz.(sol.u) for sol ∈ sols]
    data.captured_num_final, data.captured_idxs_final = captured(sols)
    data.captured_nums, data.captured_idxs = captured_vs_time(sols)
    data.MOT_captured_num_final, data.MOT_captured_idxs_final = MOT_captured(sols)
    data.MOT_captured_nums, data.MOT_captured_idxs = MOT_captured_vs_time(sols)
    data.retcodes = [sol.retcode == ReturnCode.Success ? 1 : 2 for sol in sols]
    try
        data.Ts = T_vs_time(sols, hist_binning, temp_fit_guess)
    catch DomainError
        data.Ts = []
    end
    try
        data.T_final = T_ensemble_sol(sols, hist_binning, temp_fit_guess)
    catch DomainError
        data.T_final = 0.0
    end
    try
        data.MOT_Ts = T_vs_time_MOT(sols, MOT_temp_binning, MOT_temp_fit_guess, max_v)
    catch
        data.MOT_Ts = []
    end
    try
        data.MOT_T_final = T_ensemble_sol_MOT(sols, MOT_temp_binning, MOT_temp_fit_guess, max_v)
    catch
        data.MOT_T_final = 0.0
    end
    try
        data.σ_final = σ_geom_ensemble_sol_MOT(sols, σ_min, σ_binning, σ_max, MOT_size_fit_guess)
    catch
        data.σ_final = 0.0
    end
    try
        data.σs = σ_vs_time_MOT(sols, σ_min, σ_binning, σ_max, MOT_size_fit_guess)
    catch
        data.σs = []
    end
    return data
end

function save_data(data, prob; directory="", filename="")
    serialize("$directory/data_$filename.jl", data)
    serialize("$directory/prob_$filename.jl", prob)
    return nothing
end

function save_all(prob, data, script_path, base_dir = "D:/paper_data/"; fig_name="misc")
    timestamp = Dates.format(now(), "yyyy-mm-dd_HHMMSS")
    dest_dir = "$base_dir/$fig_name/$timestamp"
    mkpath(dest_dir)

    src = script_path
    cp(src, joinpath(dest_dir, "script_$fig_name.jl"); force=true)

    open(joinpath(dest_dir, "pkg_status_$fig_name.txt"), "w") do io
        Pkg.status(io=io)
    end

    for f in ["Project.toml", "Manifest.toml"]
        if isfile(f)
            cp(f, joinpath(dest_dir, "$(splitext(f)[1])_$fig_name.toml"); force=true)
        end
    end

    git_file = joinpath(dest_dir, "git_info_$fig_name.txt")

    open(git_file, "w") do io
        try
            inside = readchomp(`git rev-parse --is-inside-work-tree`)
            if inside == "true"
                commit = readchomp(`git rev-parse HEAD`)
                branch = readchomp(`git rev-parse --abbrev-ref HEAD`)
                println(io, "branch: ", branch)
                println(io, "commit: ", commit)
            else
                println(io, "Not inside a Git repository.")
            end
        catch e
            println(io, "Git info unavailable: ", e)
        end
    end

    save_data(prob, data; directory=dest_dir, filename=fig_name)

    println("Saved data, prob, script, environment, and git info to: $dest_dir")

    return dest_dir
end


function set_initial_cloud!(prob, σ = (30e-6, 30e-6, 30e-6), T=(50e-6, 50e-6, 50e-6), uniform = false)
    σx_initial, σy_initial, σz_initial = σ
    Tx_initial, Ty_initial, Tz_initial = T
    

    if uniform
        prob.p.sim_params.vx_dist = Uniform(-sqrt(kB*Tx_initial/2m), sqrt(kB*Tx_initial/2m))
        prob.p.sim_params.vy_dist = Uniform(-sqrt(kB*Ty_initial/2m), sqrt(kB*Ty_initial/2m))
        prob.p.sim_params.vz_dist = Uniform(-sqrt(kB*Tz_initial/2m), sqrt(kB*Tz_initial/2m))
        prob.p.sim_params.x_dist = Uniform(-σx_initial, σx_initial)
        prob.p.sim_params.y_dist = Uniform(-σy_initial, σy_initial)
        prob.p.sim_params.z_dist = Uniform(-σz_initial, σz_initial)
    else
        prob.p.sim_params.vx_dist = Normal(0, sqrt(kB*Tx_initial/2m))
        prob.p.sim_params.vy_dist = Normal(0, sqrt(kB*Ty_initial/2m))
        prob.p.sim_params.vz_dist = Normal(0, sqrt(kB*Tz_initial/2m))
        prob.p.sim_params.x_dist = Normal(0, σx_initial)
        prob.p.sim_params.y_dist = Normal(0, σy_initial)
        prob.p.sim_params.z_dist = Normal(0, σz_initial)
    end
end

function set_initial_beam!(prob, σ = (30e-6, 30e-6, 30e-6), σ0 = (0,0,0), v=(50, 50, 50), v_sigma = (10,10,10))
    σx_initial, σy_initial, σz_initial = σ
    σx0_initial, σy0_initial, σz0_initial = σ0
    vx_initial, vy_initial, vz_initial = v
    vx_sigma, vy_sigma, vz_sigma = v_sigma

    prob.p.sim_params.vx_dist = Normal(vx_initial, vx_sigma)
    prob.p.sim_params.vy_dist = Normal(vy_initial, vy_sigma)
    prob.p.sim_params.vz_dist = Normal(vz_initial, vz_sigma)
    prob.p.sim_params.x_dist = Normal(σx0_initial, σx_initial)
    prob.p.sim_params.y_dist = Normal(σy0_initial, σy_initial)
    prob.p.sim_params.z_dist = Normal(σz0_initial, σz_initial)

end

function set_initial_beam_optionally_uniform!(prob, σ = (30e-6, 30e-6, 30e-6), σ0 = (0,0,0), v=(50, 50, 50), v_sigma = (10,10,10), uniform=[false,false,false,false,false,false])
    """
        set initial molecular beam parameters:
        the list uniform contains 6 booleans that correspond to whether x, y, z, vx, vy, vz (in that order) should be a uniform (true) or normal (false) distribution
        parameters:
            σ (3-vector) = sigma of the position distribution if uniform = false, +/- limit of the uniform distribution from the midpoint if uniform = true; default = (30e-6, 30e-6, 30e-6)
            σ0 (3-vector) = mean of normal or midpoint of uniform position distribution; default = (0,0,0)
            v (3-vector) = mean of normal or midpoint of uniform velocity distribution; default = (50,50,50)
            v_sigma (3-vector) = sigma of velocity distribution if uniform = false, +/- limit of the uniform distribution from the midpoint if uniform = true; default = (10, 10, 10) 
            uniform (boolean list of 6 elements) = whether x, y, z, vx, vy, vz (in that order) should be a uniform (true) or normal (false) distribution; default=[false]*6
        
    """

    # expand lists and vectors to correspoding scalers
    σx_initial, σy_initial, σz_initial = σ
    σx0_initial, σy0_initial, σz0_initial = σ0
    vx_initial, vy_initial, vz_initial = v
    vx_sigma, vy_sigma, vz_sigma = v_sigma
    
    # x distribution
    if uniform[1]
        prob.p.sim_params.x_dist = Uniform(σx0_initial-σx_initial, σx0_initial+σx_initial)
    else
        prob.p.sim_params.x_dist = Normal(σx0_initial, σx_initial)
    end

    # y distribution
    if uniform[2]
        prob.p.sim_params.y_dist = Uniform(σy0_initial-σy_initial, σy0_initial+σy_initial)
    else
        prob.p.sim_params.y_dist = Normal(σy0_initial, σy_initial)
    end

    # z distribution
    if uniform[3]
        prob.p.sim_params.z_dist = Uniform(σz0_initial-σz_initial, σz0_initial+σz_initial)
    else
        prob.p.sim_params.z_dist = Normal(σz0_initial, σz_initial)
    end

    # vx distribution
    if uniform[4]
        prob.p.sim_params.vx_dist = Uniform(vx_initial-vx_sigma, vx_initial+vx_sigma)
    else
        prob.p.sim_params.vx_dist = Normal(vx_initial, vx_sigma)
    end
    
    # vy distribution
    if uniform[5]
        prob.p.sim_params.vy_dist = Uniform(vy_initial-vy_sigma, vy_initial+vy_sigma)
    else
        prob.p.sim_params.vy_dist = Normal(vy_initial, vy_sigma)
    end

    # vz distribution
    if uniform[6]
        prob.p.sim_params.vz_dist = Uniform(vz_initial-vz_sigma, vz_initial+vz_sigma)
    else
        prob.p.sim_params.vz_dist = Normal(vz_initial, vz_sigma)
    end


end

;