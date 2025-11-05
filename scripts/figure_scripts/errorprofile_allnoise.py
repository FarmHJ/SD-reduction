import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
import os
import pandas as pd

import modelling


# Define model and protocol
model = 'Li-SD'
protocol = 'Milnes'
plot_data = 'errorprofile'
prot_mode = 'partial'

tag = 'error'
folder = 'error_profile'

# Combination of all noise models
noise_level = [0, 1, 1]
scale_level = [1, 1, 0.5]

# Define directories to load data and to save figures
results_dir = os.path.join(modelling.PARAM_DIR, model, protocol, folder)
fig_dir = os.path.join(modelling.FIG_DIR, 'syn_data', model)
if not os.path.isdir(fig_dir):
    os.makedirs(fig_dir)

# Define list of drugs (for ground truth parameter) and the parameters of
# interest
drug_list = ['dofetilide', 'cisapride', 'verapamil']
param_names = ['Kmax', 'Ku', 'Vhalf']

# Define figure details: label of parameters, colour and marker for the
# various noise levels
param_label = modelling.model_details.param_label_units
color = ['blue', 'orange', 'green']
marker = ['o', '^', '*']
noise_title = ['w/o noise', 'w/ noise', 'w/ noise, scaled']

# Set up figure structure
plt.rcParams.update({'font.size': 8})
fig = plt.figure(figsize=(2 * len(param_names) + 1, 2 * len(drug_list)))
gs = fig.add_gridspec(len(drug_list), 1, hspace=0.265)
subgs = []
subgridspecs = (1, len(param_names))
for i in range(len(drug_list)):
    subgs.append(gs[i].subgridspec(*subgridspecs, wspace=0.25))
axs = [[fig.add_subplot(subgs[k][0, j]) for j in range(
    subgridspecs[1])] for k in range(len(subgs))]

# Define variable to capture y-axis limits
bottom_lim = [0.1] * len(param_names)
top_lim = [0] * len(param_names)

# Plot error profile for all noise levels
for n in range(3):
    noise = noise_level[n]
    scale = scale_level[n]
    noise_tag = f'noise{int(noise)}_scale{int(scale * 10)}'
    # Plot also for all drugs of interest
    for d, drug in enumerate(drug_list):
        # Load results of error profile
        error_list = pd.read_csv(os.path.join(
            results_dir,
            f'{plot_data}_{drug}_{protocol}_{prot_mode}_{noise_tag}.csv'),
            index_col=0)

        # Load parameter combination of the given drug
        params = pd.read_csv(os.path.join(modelling.PARAM_DIR,
                                          f'{model}.csv'), index_col=0)
        drug_params = params.loc[[drug]]

        # Plot for all parameters of interest
        for p, param in enumerate(param_names):
            # Plot error profile
            axs[d][p].scatter(error_list[param], error_list[f'{tag}_{param}'],
                              color=color[n], marker=marker[n], s=20,
                              label=noise_title[n])

            # Plot actual parameter value
            actual_value = drug_params.loc[drug, param]
            axs[d][p].axvline(actual_value, 0, 1, color='grey', ls='--',
                              zorder=-1)

            # Plot a zoom-in view of for parameter Vhalf for drug dofetilide
            # to show the minimum RMSD
            if drug == 'dofetilide' and param == 'Vhalf' and n == 0:
                chosen_list = error_list.loc[error_list['Vhalf'] > -2]
                y_in_min = min(chosen_list[f'{tag}_{param}']) * 0.95
                y_in_max = max(chosen_list[f'{tag}_{param}']) * 1.05
                ax_in = axs[d][p].inset_axes([0.55, 0.55, 0.33, 0.33],
                                             xlim=(-1.5, 0),
                                             ylim=(y_in_min, y_in_max))
                ax_in.scatter(error_list[param], error_list[f'{tag}_{param}'],
                              color=color[n], marker=marker[n], s=15)
                ax_in.axvline(actual_value, 0, 1, color='grey', ls='--',
                              zorder=-1)
                axs[d][p].indicate_inset_zoom(ax_in, edgecolor="black",
                                              alpha=0.7)

            # Label the parameter of interest for each column
            if d == 0:
                axs[d][p].set_title(param_label[param])

            # Get the limits of y-axis for each parameter of interest
            ymin, ymax = axs[d][p].get_ylim()
            bottom_lim[p] = np.min((bottom_lim[p], ymin))
            top_lim[p] = np.max((top_lim[p], ymax))

            # Set all y-axes to logarithmic scale except for the parameter
            # Vhalf
            if param != 'Vhalf':
                axs[d][p].set_xscale('log')


class ScalarFormatterClass(ScalarFormatter):
    def _set_format(self):
        self.format = "%1.2f"


yScalarFormatter = ScalarFormatterClass(useMathText=True)
yScalarFormatter.set_powerlimits((0, 0))

for d in range(len(drug_list)):
    for p in range(len(param_names)):
        # Set the limits of y-axes for each parameter of interest
        # for ease of comparison
        axs[d][p].set_ylim(bottom=bottom_lim[p] * 0.9, top=top_lim[p])
    # Set the label of y-axes to two floating points
    axs[d][2].yaxis.set_major_formatter(yScalarFormatter)
    axs[d][0].set_ylabel('RMSD')
    # Indicate the drugs of interest of each row
    axs[d][0].text(-0.3, 0.5, drug_list[d], weight='bold',
                   horizontalalignment='right', verticalalignment='center',
                   rotation='vertical', transform=axs[d][0].transAxes)
axs[1][0].legend(handletextpad=0.6, borderpad=0.35, borderaxespad=0.3,
                 handlelength=1)

# Adjust the position of the ticks of the inset
ax_in.xaxis.set_tick_params(labelsize=7.5, pad=0.02)
ax_in.yaxis.set_tick_params(labelsize=7.5, pad=0.02)

# Save figure
fig.savefig(os.path.join(
    fig_dir, f'{plot_data}_{protocol}_{prot_mode}.pdf'),
    bbox_inches='tight')
plt.close()
