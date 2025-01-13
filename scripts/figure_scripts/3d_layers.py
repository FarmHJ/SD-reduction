import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import os
import pandas as pd

import modelling


model = 'Li-SD'
protocol = 'Milnes'
param_names = ['Kmax', 'Ku', 'Vhalf']
setup = 'exp'

results_dir = os.path.join(modelling.PARAM_DIR, model, protocol)
opt_params = pd.read_csv(os.path.join(results_dir,
                                      f'inference_{protocol}_{setup}.csv'),
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

#     fname = os.path.join(fig_dir, 'inference_error_3dlayer.png')
#     fig.savefig(fname, bbox_inches='tight')

############
# 3D scatter
############

fig = plt.figure(figsize=(4.5, 4))
gs = fig.add_gridspec(1, 1, wspace=0.35)
axs = fig.add_subplot(gs[0, 0], projection='3d')

# opt_params['diff'] = opt_params['error']

vmin, vmax = min(opt_params['error']), max(opt_params['error'])
cmap_norm = matplotlib.colors.LogNorm(vmin, vmax)
scale_map = matplotlib.cm.ScalarMappable(norm=cmap_norm, cmap=cmap)

# Plot the surface
third_dim = sorted(opt_params['ref_Vhalf'].unique())
for i, value in enumerate(third_dim):
    layer_df = opt_params[opt_params['ref_Vhalf'] == value]

    x_grid = layer_df['ref_Vhalf'].values.reshape((5, 5))
    y_grid = np.log10(layer_df['ref_Kmax'].values).reshape((5, 5))
    z_grid = np.log10(layer_df['ref_Ku'].values).reshape((5, 5))

    axs.plot_surface(x_grid, y_grid, z_grid, edgecolor='grey',
                     color='0.7', lw=0.5, alpha=0.5)

axs.scatter(opt_params['ref_Vhalf'], np.log10(opt_params['ref_Kmax']),
            np.log10(opt_params['ref_Ku']),
            c=scale_map.to_rgba(opt_params['error']),
            marker='o', zorder=-10, alpha=1, s=40)
# axs.view_init(32, 55)
axs.view_init(25, 40)
axs.set_rasterization_zorder(0)

# Adjust figure
axs.set_xlabel(r"$V_\mathrm{half-trap}$")
axs.set_ylabel(r"$K_\mathrm{max}$")
axs.set_zlabel(r"$K_u$")

axs.set_xlim(min(opt_params['ref_Vhalf']),
             max(opt_params['ref_Vhalf']))
axs.set_ylim(min(np.log10(opt_params['ref_Kmax'])),
             max(np.log10(opt_params['ref_Kmax'])))
axs.set_zlim(min(np.log10(opt_params['ref_Ku'])),
             max(np.log10(opt_params['ref_Ku'])))

axs.xaxis.set_major_locator(mticker.MaxNLocator(nbins=6))
axs.yaxis.set_major_formatter(
    mticker.FuncFormatter(log_tick_formatter))
axs.yaxis.set_major_locator(
    mticker.MaxNLocator(nbins=6, integer=True))
axs.zaxis.set_major_formatter(
    mticker.FuncFormatter(log_tick_formatter))
axs.zaxis.set_major_locator(mticker.MaxNLocator(integer=True))

cax = axs.inset_axes([0.08, -0.11, 0.8, 0.03])
fig.colorbar(scale_map, orientation='horizontal', ax=axs, cax=cax,
             label=r'$\Delta \widetilde{\mathrm{APD}}_{90}$')

# Save figure
fname = os.path.join(fig_dir, 'inference_error_3dscatter.png')
plt.subplots_adjust(hspace=0.05)
fig.savefig(fname, bbox_inches='tight')

####################
# multiple 2D layers
####################
# fit_params = [p for p in opt_params.columns if 'fit_' in p]
# param_plot_list = ['error'] + fit_params
param_plot_list = ['error']
for param_plot in param_plot_list:
    fig = plt.figure(figsize=(9, 2))
    gs = fig.add_gridspec(1, 5, wspace=0.35)
    axs = [[fig.add_subplot(gs[0, j]) for j in range(5)]]

    opt_param_list = sorted(opt_params[param_plot].unique())

    if 'fit' in param_plot:
        opt_params['diff'] = np.sqrt(np.square(np.subtract(
            opt_params[f'ref_{param_plot[4:]}'], opt_params[param_plot])))
    else:
        opt_params['diff'] = opt_params[param_plot]

    if param_plot != 'fit_Vhalf':
        vmin, vmax = min(opt_params['diff']), max(opt_params['diff'])
        cmap_norm = matplotlib.colors.LogNorm(vmin, vmax)
    else:
        vmin, vmax = 0, 200
        cmap_norm = matplotlib.colors.Normalize(vmin, vmax)
    scale_map = matplotlib.cm.ScalarMappable(norm=cmap_norm, cmap=cmap)

    third_dim = sorted(opt_params['ref_Ku'].unique())
    for i, Ku in enumerate(third_dim):

        layer_df = opt_params[opt_params['ref_Ku'] == Ku]

        x_grid = layer_df['ref_Vhalf'].values
        y_grid = np.log10(layer_df['ref_Kmax'].values)
        diff_arr = np.asarray(layer_df['diff'].values).reshape((5, 5))[::-1, :]

        xmin, xmax = np.min(x_grid), np.max(x_grid)
        ymin, ymax = np.min(y_grid), np.max(y_grid)
        extent = [xmin, xmax, ymin, ymax]

        im = axs[0][i].imshow(diff_arr, extent=extent, aspect='auto',
                              norm=cmap_norm)
        tick_pos = axs[0][i].get_yticks()
        ylabels = np.unique(np.log10(opt_params['ref_Kmax'].values))
        ylabels = [log_tick_formatter(x) for x in np.flip(ylabels)]
        axs[0][i].set_yticks(tick_pos, labels=ylabels)

        axs[0][i].set_title('{:.2e}'.format(Ku), fontdict={'fontsize': 9})
        if i == 0:
            axs[0][0].set_ylabel(r"$K_\mathrm{max}$")
            axs[0][0].set_title(r"$K_u = $"'{:.2e}'.format(Ku),
                                fontdict={'fontsize': 9})

        axs[0][i].set_xlabel(r"$V_\mathrm{half-trap}$")

    cax = axs[0][4].inset_axes([1.05, 0, 0.07, 1])
    fig.colorbar(im, ax=axs, cax=cax)

    fname = os.path.join(fig_dir, f'inference_{param_plot}_2d.pdf')
    fig.savefig(fname, bbox_inches='tight')
