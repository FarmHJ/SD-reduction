import matplotlib.pyplot as plt
import myokit
import numpy as np
import os
import pandas as pd

import modelling

model = 'Li-SD'
protocol_list = ['Milnes', 'ramp', 'step', 'staircase']

plt.rcParams.update({'font.size': 8})
fig = plt.figure(figsize=(8, 4.5))
gs = fig.add_gridspec(2, 3, wspace=0.25, hspace=0.45, height_ratios=[2, 3])

subgridspecs = (2, 1)
subgs = []
for i in range(3):
    subgs.append(gs[0, i].subgridspec(*subgridspecs, wspace=0.08,
                                      hspace=0.08))
subgs.append(gs[1, :].subgridspec(*subgridspecs, wspace=0.08,
                                  hspace=0.08))
axs = [[[fig.add_subplot(subgs[k][i, j]) for j in range(
         subgridspecs[1])] for i in range(subgridspecs[0])]
       for k in range(4)]

data_dir = os.path.join(modelling.PARAM_DIR, 'control_states', model)

for p, prot in enumerate(protocol_list):
    sim = modelling.ModelSimController(model, prot)

    log = myokit.DataLog.load_csv(
        os.path.join(data_dir, f'control_log_{prot}.csv'))

    # Plot control condition
    if p == 0:
        axs[p][0][0].plot(np.array(log.time()) / 1e3, log[sim.Vm_key], 'k',
                          zorder=5)
        axs[p][1][0].plot(np.array(log.time()) / 1e3, log[sim.ikr_key], 'k',
                          zorder=5)
    else:
        axs[p][0][0].plot(log.time(), log[sim.Vm_key], 'k', zorder=5)
        axs[p][1][0].plot(log.time(), log[sim.ikr_key], 'k', zorder=5)

    # Simulate drug condition
    sim.load_control_state()
    sim.set_drug_parameters('dofetilide')
    sim.set_conc(10)
    sim.update_initial_state(paces=0)
    log = sim.simulate()

    # Plot drug condition
    if p == 0:
        axs[p][1][0].plot(log.time() / 1e3, log[sim.ikr_key], 'r--', zorder=5)
        axs[p][1][0].set_xlabel('Time (s)')
        axs[p][0][0].set_xlim(0, max(log.time() / 1e3))
    else:
        axs[p][1][0].plot(log.time(), log[sim.ikr_key], 'r--', zorder=5)
        axs[p][1][0].set_xlabel('Time (ms)')
        axs[p][0][0].set_xlim(0, max(log.time()))
    axs[p][1][0].sharex(axs[p][0][0])
    axs[p][0][0].tick_params(labelbottom=False)
    axs[p][0][0].set_title(prot)
    axs[p][0][0].set_ylim(-125, 50)
    if p != 3:
        axs[p][1][0].set_ylim(-0.03, 1)
        axs[p][1][0].set_yticks([0, 0.5, 1])
    axs[p][0][0].grid(linestyle='--')
    axs[p][1][0].grid(linestyle='--')
    axs[p][0][0].set_yticks([-100, -50, 0, 50])

# Save protocol figure
fig.savefig(os.path.join(modelling.FIG_DIR, 'sim-protocol-drug.png'),
            bbox_inches='tight', dpi=500)
plt.close()

##########################
# Drug concentration range
##########################
# Load drug parameters - training and validation
param_names = modelling.model_details.param_names[model]

params = pd.read_csv(os.path.join(modelling.PARAM_DIR, f'{model}.csv'),
                     index_col=0)
params_validate = pd.read_csv(os.path.join(modelling.PARAM_DIR, f'{model}-validation.csv'),
                              index_col=0)
params = pd.concat([params, params_validate])

plt.rcParams.update({'font.size': 8})
fig = plt.figure(figsize=(7, 3))
gs = fig.add_gridspec(1, 5, wspace=0.78)
axs = [fig.add_subplot(gs[0, j]) for j in range(5)]

for p, name in enumerate(param_names):
    axs[p].violinplot(params[name], showmeans=False, showmedians=True)
    axs[p].set_title(name)
    if name in ['Kmax', 'Ku', 'halfmax']:
        axs[p].set_yscale('log')
        axs[p].set_ylabel('(log)')
    axs[p].tick_params(labelbottom=False)

    print(name, min(params[name]), max(params[name]))
fig.savefig(os.path.join(modelling.FIG_DIR, 'drug-param-dist.pdf'),
            bbox_inches='tight', dpi=500)
plt.close()
