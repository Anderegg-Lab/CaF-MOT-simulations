## IMPORTS
using Distributed, Serialization, Pkg, Dates
procs_to_use = 15
if nprocs() <= procs_to_use
    addprocs(abs(procs_to_use-nprocs()+1))
end
include("const_red_MOT.jl")

;

## DEFINE CONSTANT PARAMETERS
w0 = 8e-3 #beam 1/e2 radius, meters
P0 = 40e-3 #total power per MOT beam, W
s_ratios = [3,1,3,5] #HF components power ratios
pol_config = 1 # polarization configuration
beam_start_position_xprime = -3e-2 #-3cm along molecular beam travel direction from center of MOT
n_trajectories = 1 #number of particles per thread
n_total = n_trajectories*procs_to_use #total number of particles
prob.p.add_spontaneous_decay_kick = true
Bz = -14
Delta = -10
# uniform_halfwidths = (0.5e-3, 0.5e-3, 0.5e-3) #halfwidth of initial cloud uniform distribution
iris_factor = 1.5 #size of iris'ed MOT beams as a multiple of beam radius
# I0 = 2*P0/(pi*(8e-3)^2) #for same intensity scan
;

## DEFINE SCAN PARAMETERS
P_ratios = [0.5, 1, 2, 4] #Power scaling factor
ws = [0.5*w0, (1/sqrt(2))*w0, w0, sqrt(2)*w0, 2*w0] #MOT beam 1/e2 radius
vs = 2:2:20 #molecular beam forward velocity in m/s

## SCAN
Dates.format(now(), "yyyy-mm-dd_HHMMSS") |> display
all_data = []
for P_ratio in P_ratios
    power_data = []
    for w in ws
        size_data = []
        update_red_MOT_with_repump_params!(prob, Delta, 0, P_ratio*P0, P_ratio*P0, w, Bz, s_ratios, s_ratios, pol_config, pol_config; iris_factor=iris_factor)
        for v in vs
            set_initial_beam_optionally_uniform!(prob, (2.5e-3/sqrt(2), 2.5e-3/sqrt(2), 2.5e-3), (beam_start_position_xprime/sqrt(2), beam_start_position_xprime/sqrt(2), 0), (v/sqrt(2), v/sqrt(2), 0), (0.1/sqrt(2), 0.1/sqrt(2), 0), [true, true, true, false, false, false])
            sols = distributed_solve(n_trajectories, prob, prob_func!, scan_nothing, [0])
            data = get_data(sols[1])
            push!(size_data, data)
        end
        push!(power_data, size_data)
    end
    push!(all_data, power_data)
end
;
## SAVE DATA
src = abspath(@__FILE__)
dest_dir = save_all(all_data, prob, src, "D:/paper_data/"; fig_name="2b")

;
## PREPARE DATA FOR PLOTTING

captured_nums = zeros((length(all_data), length(all_data[1]), length(all_data[1][1])))
for i in eachindex(all_data)
    for j in eachindex(all_data[1])
        for k in eachindex(all_data[1][1])
            captured_nums[i,j,k] = all_data[i][j][k].MOT_captured_num_final
        end
    end
end

captured_nums_normed = captured_nums/n_total
;

## PLOT DATA
p1 = heatmap(vs, ws*1e3, captured_nums_normed[1, :, :], ylabel="beam size (mm)", title="P_0_1 = 0.5")
p2 = heatmap(vs, ws*1e3, captured_nums_normed[2, :, :], title="P_0_1 = 1")
p3 = heatmap(vs, ws*1e3, captured_nums_normed[3, :, :], title="P_0_1 = 2")
p4 = heatmap(vs, ws*1e3, captured_nums_normed[4, :, :], title="P_0_1 = 4")
p5 = plot(vs, [captured_nums_normed[1, i, :] for i in eachindex(all_data[1])], markers=:circ, legends=false, ylabel="Fraction captured")
p6 = plot(vs, [captured_nums_normed[2, i, :] for i in eachindex(all_data[1])], markers=:circ, legends=false)
p7 = plot(vs, [captured_nums_normed[3, i, :] for i in eachindex(all_data[1])], markers=:circ, legends=false)
p8 = plot(vs, [captured_nums_normed[4, i, :] for i in eachindex(all_data[1])], markers=:circ, labels = ["4mm" "5.7mm" "8mm" "11.3mm" "16mm"])

fig = plot(p1, p2, p3, p4, p5, p6, p7, p8,
     layout=(2, 4),
     size=(2400, 800), plot_title="Delta=-10MHz, dBz=14G/cm, varying P_0 and P_1; 1=40mW")

## SAVE FIGURE

savefig(fig, "$dest_dir/2b.png")
;