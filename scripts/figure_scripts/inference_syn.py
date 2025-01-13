import glob
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import myokit
import numpy as np
import os
import pandas as pd
import random

import modelling


model = 'Li-SD'
protocol = 'Milnes'
param_names = ['Kmax', 'Ku', 'Vhalf']
param_label = [r"$K_\mathrm{max}$", r"$K_u$", r"$V_\mathrm{half-trap}$"]
setup = 'exp'

results_dir = os.path.join(modelling.PARAM_DIR, model, protocol)
opt_params = pd.read_csv(os.path.join(results_dir,
                                      f'inference_{protocol}_{setup}_noise.csv'),
                         index_col=0)

Kmax_range = 10**np.linspace(np.log10(9), np.log10(1e8), 5)
Ku_range = 10**np.linspace(np.log10(1.8e-5), np.log10(0.3), 5)
Vhalf_range = np.linspace(-200, -1, 5)


def log_tick_formatter(val, pos=None):
    return f"$10^{{{int(val)}}}$"


# Set up figure structure
plt.rcParams.update({'font.size': 8})
cmap = plt.get_cmap('viridis')

param_interest = 'error'

fig_dir = os.path.join(modelling.FIG_DIR, 'syn_data', model, protocol)
if not os.path.isdir(fig_dir):
    os.makedirs(fig_dir)

###########
# 3D layers
###########
# TODO: grid position is wrong, adjust grid and tick label position
# fit_params = [p for p in opt_params.columns if 'fit_' in p]
# param_plot_list = ['error'] + fit_params
# param_plot_list = ['error']
# for param_plot in param_plot_list:

#     fig = plt.figure()
#     ax = fig.add_subplot(projection='3d')

#     opt_param_list = sorted(opt_params[param_plot].unique())

#     if 'fit' in param_plot:
#         opt_params['error'] = np.sqrt(np.square(np.subtract(
#             opt_params[f'ref_{param_plot[4:]}'], opt_params[param_plot])))
#     else:
#         opt_params['error'] = opt_params[param_plot]

#     if param_plot != 'fit_Vhalf':
#         vmin, vmax = min(opt_params['error']), max(opt_params['error'])
#         cmap_norm = matplotlib.colors.LogNorm(vmin, vmax)
#     else:
#         vmin, vmax = 0, 200
#         cmap_norm = matplotlib.colors.Normalize(vmin, vmax)
#     scale_map = matplotlib.cm.ScalarMappable(norm=cmap_norm, cmap=cmap)

#     third_dim = sorted(opt_params['ref_Vhalf'].unique())
#     for i, Ku in enumerate(third_dim):

#         layer_df = opt_params[opt_params['ref_Vhalf'] == Ku]

#         # x_grid = np.log10(layer_df['ref_Kmax'].values).reshape((5, 5))
#         # y_grid = layer_df['ref_Vhalf'].values.reshape((5, 5))
#         # z_grid = np.log10(layer_df['ref_Ku'].values).reshape((5, 5))
#         diff_arr = np.asarray(layer_df['error'].values).reshape((5, 5))[::-1, :]

#         # xmin, xmax = np.min(x_grid), np.max(x_grid)
#         # ymin, ymax = np.min(y_grid), np.max(y_grid)
#         # extent = [xmin, xmax, ymin, ymax]

#         nx, nz = diff_arr.shape
#         xi, zi = np.mgrid[0:nx + 1, 0:nz + 1]
#         yi = np.full_like(zi, i)

#         colors = cmap(cmap_norm(diff_arr))
#         ax.plot_surface(xi, yi, zi, rstride=1, cstride=1,
#                         facecolors=colors, shade=False)

#     tick_pos = ax.get_xticks()[1:-1]
#     tick_loc = [(tick_pos[i] - tick_pos[i - 1]) / 2 + tick_pos[i - 1]
#                 for i in range(1, len(tick_pos))]
#     xlabels = np.unique(np.log10(opt_params['ref_Kmax'].values))
#     xlabels = [log_tick_formatter(x) for x in xlabels]
#     ax.set_xticks(tick_loc, labels=xlabels)

#     tick_pos = ax.get_zticks()[1:-1]
#     tick_loc = [(tick_pos[i] - tick_pos[i - 1]) / 2 + tick_pos[i - 1]
#                 for i in range(1, len(tick_pos))]
#     zlabels = np.unique(np.log10(opt_params['ref_Ku'].values))
#     zlabels = [log_tick_formatter(x) for x in zlabels]
#     ax.set_zticks(tick_loc, labels=zlabels)

#     tick_pos = ax.get_yticks()[1::2]
#     ylabels = np.unique(opt_params['ref_Vhalf'].values)
#     ax.set_yticks(tick_pos, labels=ylabels)

#     # ax.xaxis.set_major_formatter(
#     #     mticker.FuncFormatter(log_tick_formatter))
#     # ax.zaxis.set_major_formatter(
#     #     mticker.FuncFormatter(log_tick_formatter))

#     ax.view_init(30, -40)
#     ax.set_xlabel(r"$K_\mathrm{max}$")
#     ax.set_ylabel(r"$V_\mathrm{half-trap}$")
#     ax.set_zlabel(r"$K_u$")
#     cax = ax.inset_axes([0.08, -0.11, 0.8, 0.03])
#     fig.colorbar(scale_map, orientation='horizontal', ax=ax, cax=cax,
#                  label=r'$\Delta \widetilde{\mathrm{APD}}_{90}$')

#     fname = os.path.join(fig_dir, 'inference_error_noise_3dlayer.png')
#     fig.savefig(fname, bbox_inches='tight')

# ####################
# # Relationship plots
# ####################

panels = 3
fig = plt.figure(figsize=(2 * panels + 1, 2))
gs = fig.add_gridspec(1, panels, wspace=0.4)
axs = [fig.add_subplot(gs[0, j]) for j in range(panels)]

for p, param in enumerate(param_names):
    ref_values = opt_params[f'ref_{param}'].values
    fit_values = opt_params[f'fit_{param}'].values
    if param == 'Vhalf':
        axs[p].scatter(ref_values, fit_values, marker='o', zorder=-10,
                       alpha=0.5)
    else:
        axs[p].scatter(np.log10(ref_values), np.log10(fit_values),
                       marker='o', zorder=-10, alpha=0.5)

    axs[p].set_title(param_label[p], fontdict={'fontsize': 9})
    axs[p].set_ylabel('fitted ' + param_label[p])
    axs[p].set_xlabel('actual ' + param_label[p])

axs[2].axhline(-65, c='r', ls='--')
axs[2].axhline(-85, c='g', ls='--')
axs[2].axvline(-65, c='r', ls='--')
axs[2].axvline(-85, c='g', ls='--')

for i in range(panels):
    lims = [
        np.min([axs[i].get_xlim(), axs[i].get_ylim()]),  # min of both axes
        np.max([axs[i].get_xlim(), axs[i].get_ylim()]),  # max of both axes
    ]

    # now plot both limits against eachother
    axs[i].plot(lims, lims, '--', color='grey', alpha=0.7, zorder=0)
    axs[i].spines[['right', 'top']].set_visible(False)

    if i != 2:
        axs[i].xaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
        axs[i].yaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))

axs[2].add_patch(matplotlib.patches.Rectangle((-65, -65), 80, 80,
                                              facecolor='red',
                                              alpha=0.5))
axs[2].add_patch(matplotlib.patches.Rectangle((-210, -210), 125, 125,
                                              facecolor='green',
                                              alpha=0.5))

fig.savefig(os.path.join(fig_dir, 'param_relationship_noise.pdf'),
            bbox_inches='tight')

# #############################
# # Fractional block comparison
# #############################
# fig = plt.figure(figsize=(7, 6))
# gs = fig.add_gridspec(4, 2, wspace=0.15, hspace=0.1)
# axs = [[fig.add_subplot(gs[i, j]) for j in range(2)] for i in range(4)]

# opt_params = opt_params.sort_values('error')
# dimless_conc = np.array([10 ** i for i in np.linspace(-8, np.log10(0.65), 4)])

# sweep_num = 10
# prot_start, prot_period, _ = \
#     modelling.simulation.protocol_period(protocol)

# general_win = np.arange(prot_start, prot_start + prot_period,
#                         10)
# log_times = []
# for s in range(sweep_num):
#     log_times.extend(general_win + 25e3 * s)

# sim = modelling.ModelSimController(model, protocol)
# sim.load_control_state()

# control_log_file = os.path.join(modelling.PARAM_DIR, 'control_states',
#                                 model, f'control_log_{protocol}_dt10.csv')
# control_log = myokit.DataLog.load_csv(control_log_file)
# control_log_win = control_log.trim(prot_start,
#                                    prot_start + prot_period)

# linestyle = {'ref': 'k', 'fit': 'r--'}

# factor = np.floor(opt_params.shape[0] / 8)
# for p in range(8):
#     param_set = opt_params.iloc[int(factor * p), :]
#     r, c = int(p / 2), p % 2

#     for i in ['ref', 'fit']:
#         ref_param = [p for p in opt_params.columns if f'{i}_' in p]
#         param = param_set[ref_param]
#         param = param.rename(index={p: p[4:] for p in ref_param})
#         sim.set_parameters(param)

#         for conc in dimless_conc:

#             sim.update_initial_state(paces=0)
#             sim.set_dimless_conc(conc)
#             log = sim.simulate(prepace=0, save_signal=sweep_num,
#                                log_var=[sim.time_key, sim.ikr_key],
#                                log_times=log_times, reset=False)

#             plot_log = []
#             for s in range(sweep_num):
#                 plot_log.extend(list(log[sim.ikr_key, s] /
#                                      control_log_win[sim.ikr_key]))

#             # Li_plot_log = [i * -1 + 1 for i in Li_plot_log]
#             axs[r][c].plot(plot_log, linestyle[i], label=i)
#             axs[r][c].set_ylim(-0.05, 1.05)

#     if r != 3:
#         axs[r][c].tick_params(labelbottom=False)
# handles, labels = axs[0][0].get_legend_handles_labels()
# unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if
#           l not in labels[:i]]
# axs[0][0].legend(*zip(*unique), handlelength=1.5)
# for i in range(2):
#     axs[3][i].set_xlabel('Time (ms)')
# for i in range(4):
#     axs[i][0].set_ylabel('Fractional block')
# # fig.sharex(['Time (ms)'], [(0, len(frac_block) * 10)])
# fig.savefig(os.path.join(fig_dir, 'inference_simulations_noise.pdf'),
#             bbox_inches='tight')

##########################################
# Fractional block comparison - with noise
##########################################

# fig = plt.figure(figsize=(7, 6))
# gs = fig.add_gridspec(4, 2, wspace=0.15, hspace=0.1)
# axs = [[fig.add_subplot(gs[i, j]) for j in range(2)] for i in range(4)]

# # opt_params = opt_params.sort_values('error')
# dimless_conc = np.array([10 ** i for i in np.linspace(-8, np.log10(0.65), 4)])

# sweep_num = 10
# prot_start, prot_period, _ = \
#     modelling.simulation.protocol_period(protocol)

# general_win = np.arange(prot_start, prot_start + prot_period,
#                         10)
# log_times = []
# for s in range(sweep_num):
#     log_times.extend(general_win + 25e3 * s)

# sim = modelling.ModelSimController(model, protocol)
# sim.load_control_state()

# control_log_file = os.path.join(modelling.PARAM_DIR, 'control_states',
#                                 model, f'control_log_{protocol}_dt10.csv')
# control_log = myokit.DataLog.load_csv(control_log_file)
# control_log_win = control_log.trim(prot_start,
#                                    prot_start + prot_period)

# control_log_win_noise = myokit.DataLog.load_csv(os.path.join(
#     results_dir, f'control_log_{protocol}_noise_trimmed.csv'))

# linestyle = {'ref': 'k', 'fit': 'r--'}

# file_prefix = f'log_{protocol}_noise_'
# result_files = glob.glob(os.path.join(results_dir,
#                                       f'{file_prefix}*.csv'))

# id_list = []
# for paths in result_files:
#     id_num = int(os.path.basename(paths)[len(file_prefix):-10])
#     id_list.append(id_num)
# id_list = sorted(set(id_list))



# for p, id in enumerate(id_list):
#     param_set = opt_params.iloc[int(id), :]
#     row, col = int(p / 2), p % 2

#     ref_param = [p for p in opt_params.columns if 'ref_' in p]
#     param = param_set[ref_param]
#     param = param.rename(index={p: p[4:] for p in ref_param})
#     print(param)
#     sim.set_parameters(param)

#     for c, conc in enumerate(dimless_conc):

#         data = myokit.DataLog.load_csv(os.path.join(
#             results_dir, f'log_{protocol}_noise_{id}_conc{c}.csv'))
#         SD_frac_block = []
#         for s in range(sweep_num):
#             SD_frac_block.extend(list(np.array(data[sim.ikr_key, s]) /
#                                       np.array(control_log_win_noise[sim.ikr_key])))
#         axs[row][col].plot(SD_frac_block, 'k', label='data')

#         sim.update_initial_state(paces=0)
#         sim.set_dimless_conc(conc)
#         log = sim.simulate(prepace=0, save_signal=sweep_num,
#                            log_var=[sim.time_key, sim.ikr_key],
#                            log_times=log_times, reset=False)

#         plot_log = []
#         for s in range(sweep_num):
#             plot_log.extend(list(log[sim.ikr_key, s] /
#                                  control_log_win[sim.ikr_key]))

#         # Li_plot_log = [i * -1 + 1 for i in Li_plot_log]
#         axs[row][col].plot(plot_log, 'r--', label='fitted')
#         axs[row][col].set_ylim(-0.05, 1.05)

#     if row != 3:
#         axs[row][col].tick_params(labelbottom=False)
# handles, labels = axs[0][0].get_legend_handles_labels()
# unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if
#           l not in labels[:i]]
# axs[0][0].legend(*zip(*unique), handlelength=1.5)
# for i in range(2):
#     axs[3][i].set_xlabel('Time (ms)')
# for i in range(4):
#     axs[i][0].set_ylabel('Fractional block')
# # fig.sharex(['Time (ms)'], [(0, len(frac_block) * 10)])
# fig.savefig(os.path.join(fig_dir, 'inference_simulations_noise.pdf'),
#             bbox_inches='tight')
