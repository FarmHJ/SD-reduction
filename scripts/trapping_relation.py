import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import os

import modelling


def Kn(Kt, Vhalf, V):
    return Kt / (1 + np.exp(-(V - Vhalf) / 6.789))


def log_tick_formatter(val, pos=None):
    return f"$10^{{{int(val)}}}$"


plt.rcParams.update({'font.size': 9})
cmap = plt.get_cmap('viridis')

fig = plt.figure(figsize=(9, 5))
gs = fig.add_gridspec(2, 5, wspace=0.15, hspace=0.4)
axs = [[fig.add_subplot(gs[i, j]) for j in range(5)] for i in range(2)]

prot_step = np.linspace(-120, 40, 20)
Kt_range = 10**np.linspace(np.log10(1e-5), np.log10(1e8), 20)
Vhalf_range = np.linspace(-200, -1, 20)

Vhalf_arr, V_arr, Kt_arr = np.meshgrid(Vhalf_range, prot_step, Kt_range,
                                       indexing='ij')
Kn_grid = Kn(Vhalf_arr, V_arr, Kt_arr)

Vhalf_list = Vhalf_arr.flatten()
V_list = V_arr.flatten()
Kt_list = Kt_arr.flatten()
Kn_list = Kn_grid.flatten()

vmin, vmax = np.min(Kn_grid), np.max(Kn_grid)
cmap_norm = matplotlib.colors.Normalize(vmin, vmax)
scale_map = matplotlib.cm.ScalarMappable(norm=cmap_norm, cmap=cmap)

############
# 3d scatter
############
# fig = plt.figure(figsize=(4.5, 4))
# gs = fig.add_gridspec(1, 1)
# axs = fig.add_subplot(gs[0, 0], projection='3d')

# axs.scatter(Vhalf_list, V_list, np.log10(Kt_list), c=scale_map.to_rgba(Kn_list),
#             marker='o', zorder=-10, alpha=1, s=40)
# # axs.view_init(32, 55)
# axs.view_init(25, 40)
# axs.set_rasterization_zorder(0)

# # Adjust figure
# axs.set_xlabel(r"$V_\mathrm{half-trap}$")
# axs.set_ylabel(r"$V$")
# axs.set_zlabel(r"$K_t$")
# axs.zaxis.set_major_formatter(
#     mticker.FuncFormatter(log_tick_formatter))

# cax = axs.inset_axes([0.08, -0.11, 0.8, 0.03])
# fig.colorbar(scale_map, orientation='horizontal', ax=axs, cax=cax,
#              label=r'$K_n$')

# # Save figure
# plt.savefig(os.path.join(modelling.FIG_DIR, 'trapping_param.pdf'),
#             bbox_inches='tight')

####################
# Multiple 2d layers
####################
fig = plt.figure(figsize=(9, 9))
gs = fig.add_gridspec(4, 5, hspace=0.2, wspace=0.17)
axs = [[fig.add_subplot(gs[i, j]) for j in range(5)] for i in range(4)]

for i in range(20):

    r, c = int(i / 5), i % 5

    Vhalf_layer = Vhalf_arr[i, :, :]
    V_layer = V_arr[i, :, :]
    Kt_layer = Kt_arr[i, :, :]
    Kn_color = Kn_grid[i, :, :]

    xmin, xmax = np.min(Kt_layer), np.max(Kt_layer)
    ymin, ymax = np.min(V_layer), np.max(V_layer)
    extent = [xmin, xmax, ymin, ymax]

    im = axs[r][c].imshow(Kn_color, extent=extent, aspect='auto',
                          norm=cmap_norm, origin='lower')

    if c == 0:
        axs[r][c].set_ylabel(r"$V$ (mV)")
        axs[r][c].set_title(r"$V_\mathrm{half} = $" +
                            '{:.1f}'.format(Vhalf_arr[i, 0, 0]),
                            fontdict={'fontsize': 9})
    else:
        axs[r][c].tick_params(labelleft=False)
        axs[r][c].set_title('{:.1f}'.format(Vhalf_arr[i, 0, 0]),
                            fontdict={'fontsize': 9})

    if r == 3:
        axs[r][c].set_xlabel(r"$K_t$")
        tick_pos = axs[r][c].get_xticks()
        # print(tick_pos)
        xlabels = [int(tick_pos[0])] + \
            [log_tick_formatter(np.log10(x)) for x in tick_pos[1:]]
        axs[r][c].set_xticks(tick_pos, labels=xlabels)
    else:
        axs[r][c].tick_params(labelbottom=False)

cax = axs[0][4].inset_axes([1.05, 0, 0.07, 1])
fig.colorbar(im, ax=axs, cax=cax, label=r'$K_n$')

plt.savefig(os.path.join(modelling.FIG_DIR, 'trapping_param_2d.pdf'),
            bbox_inches='tight')
