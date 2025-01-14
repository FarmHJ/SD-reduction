import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

import modelling


# Define model and protocol
model = 'Li-SD'
protocol = 'Milnes'
plot_data = 'profilelikelihood'  # 'profilelikelihood' or 'errorprofile

# Define directories to load data and to save figures
results_dir = os.path.join(modelling.PARAM_DIR, model, protocol,
                           'profile_likelihood')
fig_dir = os.path.join(modelling.FIG_DIR, 'syn_data', model, protocol)
if not os.path.isdir(fig_dir):
    os.makedirs(fig_dir)

if plot_data == 'profilelikelihood':
    tag = 'likelihood'
else:
    tag = 'error'

# ####################
# # Multiple parameter
# ####################
# # Define list of drugs (for ground truth parameter), parameter and their
# # labels
# drug_list = ['dofetilide', 'cisapride', 'verapamil']
# # drug_list = ['dofetilide', 'cisapride']
# param_names = ['Kmax', 'Ku', 'Vhalf', 'Kt']
# param_names = [param_names[-1]]
# param_label = [r"$K_\mathrm{max}$", r"$K_u$", r"$V_\mathrm{half-trap}$",
#                r"$K_t$"]
# param_label = [param_label[-1]]

# # Set up figure structure
# plt.rcParams.update({'font.size': 8})
# fig = plt.figure(figsize=(2 * len(param_names) + 1, 2 * len(drug_list)))
# gs = fig.add_gridspec(len(drug_list), 1, hspace=0.2)
# subgs = []
# subgridspecs = (1, len(param_names))
# for i in range(len(drug_list)):
#     subgs.append(gs[i].subgridspec(*subgridspecs, wspace=0.25))
# axs = [[fig.add_subplot(subgs[k][0, j]) for j in range(
#     subgridspecs[1])] for k in range(len(subgs))]

# # Define variable for y-axis limits
# bottom_lim = [0] * len(param_names)
# top_lim = [0] * len(param_names)
# for d, drug in enumerate(drug_list):
#     # Load previous computed profile
#     error_list = pd.read_csv(os.path.join(
#         results_dir, f'{plot_data}_{drug}_Milnes_exp_Kt.csv'),
#         index_col=0)
#     # error_list2 = pd.read_csv(os.path.join(
#     #     results_dir, f'{plot_data}_{drug}_Milnes_exp_ext.csv'),
#     #     index_col=0)
#     # error_list = pd.concat([error_list1, error_list2])
#     print(error_list)

#     # Load parameter combination of CiPA drugs
#     params = pd.read_csv(os.path.join(modelling.PARAM_DIR,
#                                       f'{model}.csv'), index_col=0)
#     drug_params = params.loc[[drug]]

#     for p, param in enumerate(param_names):
#         # Plot error or likelihood profile
#         axs[d][p].plot(error_list[param], error_list[f'{tag}_{param}'],
#                        'o')
#         if param == 'Kt':
#             actual_value = 3.5e-5
#         else:
#             actual_value = drug_params.loc[drug, param]

#         # Plot actual parameter value
#         axs[d][p].axvline(actual_value, 0, 1, color='grey', ls='--',
#                           zorder=-1)
#         if d == 0:
#             axs[d][p].set_title(param_label[p])

#         ymin, ymax = axs[d][p].get_ylim()
#         bottom_lim[p] = np.min((bottom_lim[p], ymin))
#         top_lim[p] = np.max((top_lim[p], ymax))

#         # if d != len(drug_list) - 1:
#         #     axs[d][p].tick_params(labelbottom=False)
#         if param != 'Vhalf':
#             axs[d][p].set_xscale('log')
#         axs[d][0].set_ylabel(drug, fontsize=9)

# for d in range(len(drug_list)):
#     for p in range(len(param_names)):
#         axs[d][p].set_ylim(bottom=bottom_lim[p], top=top_lim[p])

# # Save figure
# fig.savefig(os.path.join(fig_dir, f'{plot_data}_Milnes_exp_Kt.pdf'),
#             bbox_inches='tight')
# plt.close()

##################
# Single parameter
##################
# Define list of drugs (for ground truth parameter), parameter and their
# labels
drug_list = ['dofetilide', 'cisapride', 'verapamil']
param_name = 'Kt'
param_label = [r"$K_\mathrm{max}$", r"$K_u$", r"$V_\mathrm{half-trap}$",
               r"$K_t$"]
param_label = param_label[-1]
if plot_data == 'profilelikelihood':
    tag = 'likelihood'
else:
    tag = 'error'

# Set up figure structure
plt.rcParams.update({'font.size': 8})

fig = plt.figure(figsize=(2 * len(drug_list), 2))
gs = fig.add_gridspec(1, len(drug_list), wspace=0.1)
axs = [fig.add_subplot(gs[0, j]) for j in range(len(drug_list))]

# Define variable for y-axis limits
bottom_lim = 0
top_lim = 0
for d, drug in enumerate(drug_list):
    # Load previous computed profile
    error_list = pd.read_csv(os.path.join(
        results_dir, f'{plot_data}_{drug}_Milnes_exp_Kt.csv'),
        index_col=0)
    # error_list2 = pd.read_csv(os.path.join(
    #     results_dir, f'{plot_data}_{drug}_Milnes_exp_ext.csv'),
    #     index_col=0)
    # error_list = pd.concat([error_list1, error_list2])

    # Load parameter combination of CiPA drugs
    params = pd.read_csv(os.path.join(modelling.PARAM_DIR,
                                      f'{model}.csv'), index_col=0)
    drug_params = params.loc[[drug]]

    # Plot error or likelihood profile
    axs[d].plot(error_list[param_name], error_list[f'{tag}_{param_name}'],
                'o')
    if param_name == 'Kt':
        actual_value = 3.5e-5
    else:
        actual_value = drug_params.loc[drug, param_name]

    # Plot actual parameter value
    axs[d].axvline(actual_value, 0, 1, color='grey', ls='--',
                   zorder=-1)
    axs[d].set_title(drug)
    ymin, ymax = axs[d].get_ylim()
    bottom_lim = np.min((bottom_lim, ymin))
    top_lim = np.max((top_lim, ymax))

    if param_name != 'Vhalf':
        axs[d].set_xscale('log')
    if d == 0:
        axs[d].set_ylabel(param_label, fontsize=9)
    else:
        axs[d].tick_params(labelleft=False)

for d in range(len(drug_list)):
    axs[d].set_ylim(bottom=bottom_lim, top=top_lim)

# Save figure
fig.savefig(os.path.join(fig_dir, f'{plot_data}_Milnes_exp_{param_name}.pdf'),
            bbox_inches='tight')
plt.close()
