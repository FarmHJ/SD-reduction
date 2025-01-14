import matplotlib.pyplot as plt
import myokit
import numpy as np
import os

import modelling


# Define parameter range
Kt_range = 10**np.linspace(-5, 8, 5)

color_seq = ['#7e7e7e', '#986464', '#989864', '#986496', '#988364',
             '#64986a', '#74a9cf', '#045a8d', '#2b8cbe']
li_states = ['IC1', 'IC2', 'C1', 'C2', 'O', 'IO',
             'IObound', 'Obound', 'Cbound']
param_name = ['Kt', 'Kmax', 'Ku', 'Vhalf']

drug_list = ['dofetilide', 'verapamil', 'cisapride']
drug_conc = [10, 300, 100]
model = 'Li-SD'
protocol = 'Milnes'

fig = plt.figure(figsize=(9, 6))
gs = fig.add_gridspec(len(drug_list), 5, wspace=0.1)
axs = [[fig.add_subplot(gs[i, j]) for j in range(5)] for i in range(len(drug_list))]

for d, drug in enumerate(drug_list):
    fig_dir = os.path.join(modelling.FIG_DIR, 'syn_data', model, 'Kt')
    if not os.path.isdir(fig_dir):
        os.makedirs(fig_dir)

    sim = modelling.ModelSimController(model, protocol)
    sim.load_control_state()

    sim.set_drug_parameters(drug)

    for n, p in enumerate(Kt_range):
        sim.set_parameters({'Kt': p}, set_Kt=False)
        print(sim.get_parameters())

        sim.update_initial_state(paces=0)
        sim.set_conc(drug_conc[d])
        log = sim.simulate(log_var='all', reset=False)

        # for i in range(10):
        axs[d][n].stackplot(log.time() / 1e3,
                            *[log[f'ikr.{s}'] for s in li_states],
                            labels=li_states, colors=color_seq, zorder=-10)
        axs[d][n].set_ylim(0, 1)
        axs[d][n].set_xlim(0, max(log.time()) / 1e3)
        axs[d][n].set_rasterization_zorder(0)

        axs_ikr = axs[d][n].twinx()

        color = 'tab:red'
        # axs_ikr.set_ylabel('IKr', color=color)
        axs_ikr.plot(log.time() / 1e3, log['ikr.IKr'], color=color)
        axs_ikr.set_ylim(0, 0.85)
        if n != 4:
            axs_ikr.tick_params(labelright=False)
        else:
            axs_ikr.tick_params(axis='y', labelcolor=color)

        if d != 2:
            axs[d][n].sharex(axs[2][n])
            axs[d][n].tick_params(labelbottom=False)
        else:
            axs[d][n].set_xlabel('Time (s)')

    axs[d][0].set_ylabel(drug)
    for c in range(1, 5):
        axs[d][c].sharey(axs[d][0])
        axs[d][c].tick_params(labelleft=False)

fig.savefig(os.path.join(fig_dir, f'steady_state_occupancy.pdf'),
            bbox_inches='tight')
# fig_t.savefig(os.path.join(fig_dir, f'{drug}_transient_phase.pdf'),
#               bbox_inches='tight')
