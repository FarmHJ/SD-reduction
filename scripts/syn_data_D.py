import itertools
import matplotlib.pyplot as plt
import myokit
import numpy as np
import os

import modelling


######################
# Parameter simulation
######################
model = 'Li-D'
protocol = 'Milnes'
prot_mode = 'partial'
# param_names = ['Kmax', 'Ku', 'Vhalf']
param_names = ['Kb', 'Ku']

# Define parameter range
grid = 10
Kb_range = 10**np.linspace(-5, np.log10(3e4), grid)
Ku_range = 10**np.linspace(np.log10(2e-5), np.log10(2e2), grid)

dimless_conc = np.array([10 ** i for i in np.linspace(-8, np.log10(0.65), 4)])

# Set up simulation
sim = modelling.ModelSimController(model, protocol)
sim.load_control_state()

plt.rcParams.update({'font.size': 8})

##############
# Ideal set up
##############
fig = plt.figure(figsize=(grid * 2, grid * 2))
gs = fig.add_gridspec(grid, grid, wspace=0.1)
axs = [[fig.add_subplot(gs[i, j]) for j in range(grid)] for i in range(grid)]

count = 0
for Kb, Ku in itertools.product(Kb_range, Ku_range):

    param_values = [Kb, Ku]
    param_dict = {param_names[i]: param_values[i] for i in range(2)}
    r, c = int(count / grid), int(count % grid)

    sim.update_initial_state(paces=0)
    sim.set_parameters(param_dict, set_Kt=False)

    for d in dimless_conc:
        sim.set_dimless_conc(d)
        log = sim.simulate()

        axs[r][c].plot(log.time() / 1e3, log[sim.ikr_key])
        axs[r][c].tick_params(labelbottom=False, labelleft=False)
    count += 1

# Save protocol figure
fig.savefig(os.path.join(modelling.FIG_DIR, 'syn_data', model, 'syn-data.pdf'),
            bbox_inches='tight', dpi=500)
plt.close()

#####################
# Experimental set up
#####################
# sweep_num = 1
# _, prot_length, _ = modelling.simulation.protocol_period(protocol, mode='full')
# prot_start, prot_period, _ = modelling.simulation.protocol_period(protocol,
#                                                                   mode=prot_mode)
# general_win = np.arange(prot_start, prot_start + prot_period, 10)
# log_times = []
# for i in range(sweep_num):
#     log_times.extend(general_win + prot_length * i)

# control_log_file = os.path.join(modelling.PARAM_DIR, 'control_states',
#                                 model, f'control_log_{protocol}_dt10.csv')
# control_log = myokit.DataLog.load_csv(control_log_file)
# control_log_win = control_log.trim(prot_start, prot_start + prot_period)

# fig = plt.figure(figsize=(7, 6))
# gs = fig.add_gridspec(4, 2, wspace=0.15, hspace=0.1)
# axs = [[fig.add_subplot(gs[i, j]) for j in range(2)] for i in range(4)]

# # Randomly plot 8 synthetic data
# np.random.seed(0)
# idx = np.random.choice(np.arange(5), size=8, replace=True)
# idy = np.random.choice(np.arange(5), size=8, replace=True)
# idz = np.random.choice(np.arange(5), size=8, replace=True)

# # for n in range(8):
# #     param_values = [Kmax_range[idx[n]], Ku_range[idy[n]], Vhalf_range[idz[n]]]
# #     param_dict = {param_names[i]: param_values[i] for i in range(3)}
# #     # print(param_dict)

# #     r, c = int(n / 2), n % 2

# #     sim.set_parameters(param_dict)

# #     for d in dimless_conc:
# #         sim.update_initial_state(paces=0)
# #         sim.set_dimless_conc(d)
# #         log = sim.simulate(prepace=0, save_signal=sweep_num,
# #                            log_var=[sim.time_key, sim.ikr_key],
# #                            log_times=log_times, reset=False)

# #         plot_log = []
# #         for s in range(sweep_num):
# #             plot_log.extend(list(log[sim.ikr_key, s] /
# #                                  control_log_win[sim.ikr_key]))
# #         axs[r][c].plot(plot_log)
# #         # axs[r][c].set_ylim(-0.001, 0.01)
# #     axs[r][c].text(0.01, 0.8, r"$V_\mathrm{half-trap} = $"f'{int(param_values[2])}',
# #                    transform=axs[r][c].transAxes)

# fig_dir = os.path.join(modelling.FIG_DIR, 'syn_data', model, protocol)
# # fig.savefig(os.path.join(fig_dir, f'syn-data-{prot_mode}.pdf'),
# #             bbox_inches='tight')

# # fig = plt.figure(figsize=(5, 3))
# # # n = 0
# # # param_values = [Kmax_range[idx[n]], Ku_range[idy[n]], Vhalf_range[idz[n]]]
# # # param_dict = {param_names[i]: param_values[i] for i in range(3)}

# # sim.set_drug_parameters('dofetilide')
# # # sim.set_parameters(param_dict)
# # dimless_conc = [0]
# # sweep_num = 2

# # for d in dimless_conc:
# #     sim.update_initial_state(paces=0)
# #     # sim.set_dimless_conc(d)
# #     sim.set_conc(0)
# #     log = sim.simulate(prepace=0, save_signal=sweep_num,
# #                        log_var=[sim.time_key, sim.ikr_key],
# #                     #    log_times=log_times,
# #                        reset=False)

# #     plot_log = []
# #     for s in range(sweep_num):
# #         plot_log.extend(list(log[sim.ikr_key, s] /
# #                              control_log_win[sim.ikr_key]))
# #     plt.plot(plot_log)
# # fig.savefig(os.path.join(fig_dir, 'syn-data-full-test.pdf'),
# #             bbox_inches='tight')

# fig = plt.figure(figsize=(7, 3))
# n = 0
# param_values = [Kmax_range[idx[n]], Ku_range[idy[n]], Vhalf_range[idz[n]]]
# param_dict = {param_names[i]: param_values[i] for i in range(3)}

# sim.set_parameters(param_dict)

# for d in dimless_conc:
#     sim.update_initial_state(paces=0)
#     sim.set_dimless_conc(d)
#     log = sim.simulate(prepace=0, save_signal=sweep_num,
#                         log_var=[sim.time_key, sim.ikr_key],
#                         log_times=log_times,
#                         reset=False)

#     # plot_log = []
#     # for s in range(sweep_num):
#     #     plot_log.extend(list(log[sim.ikr_key, s]))  # /
#                                 # control_log_win[sim.ikr_key]))
#     # print(min(plot_log), max(plot_log))
#     plt.plot(log[sim.ikr_key])
#     # plt.plot()
# plt.ylim((0, 0.05))
# fig.savefig(os.path.join(fig_dir, 'syn-data-partial.pdf'),
#             bbox_inches='tight')
