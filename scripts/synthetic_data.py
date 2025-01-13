import itertools
import matplotlib.pyplot as plt
import myokit
import numpy as np
import os
import random

import modelling


######################
# Parameter simulation
######################
model = 'Li-SD'
protocol = 'Milnes'

# Define parameter range
param_names = ['Kmax', 'Ku', 'Vhalf']
grid = 5
Kmax_range = 10**np.linspace(np.log10(1), np.log10(1e8), grid)
Ku_range = 10**np.linspace(np.log10(1.8e-5), np.log10(1), grid)
Vhalf_range = np.linspace(-200, -1, grid)

dimless_conc = np.array([10 ** i for i in np.linspace(-8, np.log10(0.65), 4)])

# Set up simulation
sim = modelling.ModelSimController(model, protocol)
sim.load_control_state()

plt.rcParams.update({'font.size': 8})
fig = plt.figure(figsize=(11 * 2, 12 * 2))
gs = fig.add_gridspec(12, 11, wspace=0.1)
axs = [[fig.add_subplot(gs[i, j]) for j in range(11)] for i in range(12)]

##############
# Ideal set up
##############
# count = 0
# for Kmax, Ku, Vhalf in itertools.product(
#         Kmax_range, Ku_range, Vhalf_range):

#     param_values = [Kmax, Ku, Vhalf]
#     param_dict = {param_names[i]: param_values[i] for i in range(3)}
#     r, c = int(count / 11), int(count % 11)

#     sim.update_initial_state(paces=0)
#     sim.set_parameters(param_dict)

#     for d in dimless_conc:
#         sim.set_dimless_conc(d)
#         log = sim.simulate()

#         axs[r][c].plot(log.time() / 1e3, log[sim.ikr_key])
#         axs[r][c].tick_params(labelbottom=False, labelleft=False)
#     count += 1

# # Save protocol figure
# fig.savefig(os.path.join(modelling.FIG_DIR, 'syn-data.pdf'),
#             bbox_inches='tight', dpi=500)
# plt.close()

#####################
# Experimental set up
#####################
sweep_num = 10
prot_start, prot_period, prot_plot_delay = \
    modelling.simulation.protocol_period(protocol)

general_win = np.arange(prot_start,
                        prot_start + prot_period, 10)
log_times = []
for s in range(sweep_num):
    log_times.extend(general_win + 25e3 * s)

control_log_file = os.path.join(modelling.PARAM_DIR, 'control_states',
                                model, f'control_log_{protocol}_dt10.csv')
control_log = myokit.DataLog.load_csv(control_log_file)
control_log_win = control_log.trim(prot_start,
                                   prot_start + prot_period)

fig = plt.figure(figsize=(7, 6))
gs = fig.add_gridspec(4, 2, wspace=0.15, hspace=0.1)
axs = [[fig.add_subplot(gs[i, j]) for j in range(2)] for i in range(4)]

# Randomly plot 8 synthetic data
np.random.seed(0)
idx = np.random.choice(np.arange(5), size=8, replace=True)
idy = np.random.choice(np.arange(5), size=8, replace=True)
idz = np.random.choice(np.arange(5), size=8, replace=True)

for n in range(8):
    param_values = [Kmax_range[idx[n]], Ku_range[idy[n]], Vhalf_range[idz[n]]]
    param_dict = {param_names[i]: param_values[i] for i in range(3)}

    r, c = int(n / 2), n % 2

    sim.set_parameters(param_dict)

    for d in dimless_conc:
        sim.update_initial_state(paces=0)
        sim.set_dimless_conc(d)
        log = sim.simulate(prepace=0, save_signal=sweep_num,
                           log_var=[sim.time_key, sim.ikr_key],
                           log_times=log_times, reset=False)

        plot_log = []
        for s in range(sweep_num):
            plot_log.extend(list(log[sim.ikr_key, s] /
                                 control_log_win[sim.ikr_key]))
        axs[r][c].plot(plot_log)
        axs[r][c].set_ylim(-0.05, 1.05)

fig.savefig(os.path.join(modelling.FIG_DIR, 'syn-data-exp.pdf'),
            bbox_inches='tight', dpi=500)
