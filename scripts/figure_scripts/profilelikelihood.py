import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

import modelling


# Define model and protocol
model = 'Li-SD'
protocol = 'Milnes'

# Define directories to load data and to save figures
results_dir = os.path.join(modelling.PARAM_DIR, model, protocol)
fig_dir = os.path.join(modelling.FIG_DIR, 'syn_data', model, protocol)
if not os.path.isdir(fig_dir):
    os.makedirs(fig_dir)

# Set up figure structure
plt.rcParams.update({'font.size': 8})
fig = plt.figure(figsize=(7, 6))
gs = fig.add_gridspec(3, 1, hspace=0.05)
subgs = []
subgridspecs = (1, 3)
for i in range(3):
    subgs.append(gs[i].subgridspec(*subgridspecs, wspace=0.25))
axs = [[fig.add_subplot(subgs[k][0, j]) for j in range(
    subgridspecs[1])] for k in range(len(subgs))]

# Define list of drugs (for ground truth parameter), parameter and their
# labels
drug_list = ['dofetilide', 'cisapride', 'verapamil']
param_names = ['Kmax', 'Ku', 'Vhalf']
param_label = [r"$K_\mathrm{max}$", r"$K_u$", r"$V_\mathrm{half-trap}$"]

# Define variable for y-axis limits
bottom_lim = [0] * len(param_names)
top_lim = [0] * len(param_names)
for d, drug in enumerate(drug_list):
    # Load previous computed profile
    error_list = pd.read_csv(os.path.join(
        results_dir, f'profilelikelihood2_{drug}_Milnes_exp.csv'),
        index_col=0)

    # Load parameter combination of CiPA drugs
    params = pd.read_csv(os.path.join(modelling.PARAM_DIR,
                                      f'{model}.csv'), index_col=0)
    drug_params = params.loc[[drug]]

    for p, param in enumerate(param_names):
        # Plot error or likelihood profile
        axs[d][p].plot(error_list[param], error_list[f'likelihood_{param}'],
                       'o')
        if param == 'Kt':
            actual_value = 3.5e-5
        else:
            actual_value = drug_params.loc[drug, param]

        # Plot actual parameter value
        axs[d][p].axvline(actual_value, 0, 1, color='grey', ls='--',
                          zorder=-1)
        if d == 0:
            axs[d][p].set_title(param_label[p])
        ymin, ymax = axs[d][p].get_ylim()
        bottom_lim[p] = np.min((bottom_lim[p], ymin))
        top_lim[p] = np.max((top_lim[p], ymax))

        if d != len(drug_list) - 1:
            axs[d][p].tick_params(labelbottom=False)
        if param != 'Vhalf':
            axs[d][p].set_xscale('log')
        axs[d][0].set_ylabel(drug, fontsize=9)

for d in range(len(drug_list)):
    for p in range(len(param_names)):
        axs[d][p].set_ylim(bottom=bottom_lim[p], top=top_lim[p])

# Save figure
fig.savefig(os.path.join(fig_dir, 'profilelikelihood2_Milnes_exp.pdf'),
            bbox_inches='tight')
plt.close()
