import matplotlib.pyplot as plt
import myokit
import os

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

results_dir = os.path.join(modelling.PARAM_DIR, 'control_states', model)
if not os.path.isdir(results_dir):
    os.makedirs(results_dir)

for p, prot in enumerate(protocol_list):
    sim = modelling.ModelSimController(model, prot)
    log = sim.simulate()

    control_state = sim.sim.state()
    log.save_csv(os.path.join(results_dir, f'control_log_{prot}.csv'))
    myokit.save_state(os.path.join(results_dir, f'control_state_{prot}.csv'),
                      control_state)

    # Create figure for stimulus protocols only
    if p == 0:
        axs[p][0][0].plot(log.time() / 1e3, log[sim.Vm_key], 'k', zorder=5)
        axs[p][1][0].plot(log.time() / 1e3, log[sim.ikr_key], 'k', zorder=5)
        axs[p][1][0].set_xlabel('Time (s)')
        axs[p][0][0].set_xlim(0, max(log.time() / 1e3))
    else:
        axs[p][0][0].plot(log.time(), log[sim.Vm_key], 'k', zorder=5)
        axs[p][1][0].plot(log.time(), log[sim.ikr_key], 'k', zorder=5)
        axs[p][1][0].set_xlabel('Time (ms)')
        axs[p][0][0].set_xlim(0, max(log.time()))
    # panel_protocol.set_ylabel('Voltage (mV)')
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
fig.savefig(os.path.join(modelling.FIG_DIR, 'sim-protocol.png'),
            bbox_inches='tight', dpi=500)
plt.close()
