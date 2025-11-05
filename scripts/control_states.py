import matplotlib.pyplot as plt
import myokit
import numpy as np
import os

import modelling

model = 'Li-SD'
protocol_list = ['Milnes', 'ramp', 'step', 'staircase']

plt.rcParams.update({'font.size': 8})
fig = plt.figure(figsize=(7, 4.5))
gs = fig.add_gridspec(2, 3, wspace=0.05, hspace=0.45, height_ratios=[2, 3])

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

    if p == 0:
        axs[p][0][0].plot(log.time() / 1e3, log[sim.Vm_key], 'k', zorder=5)
        axs[p][1][0].plot(log.time() / 1e3, log[sim.ikr_key], 'k', zorder=5)
    else:
        axs[p][0][0].plot(log.time(), log[sim.Vm_key], 'k', zorder=5)
        axs[p][1][0].plot(log.time(), log[sim.ikr_key], 'k', zorder=5)

    log = sim.simulate(timestep=10)
    log.save_csv(os.path.join(results_dir, f'control_log_{prot}_dt10.csv'))

    # Create figure for stimulus protocols only
    if p == 0:
        # axs[p][1][0].plot(log.time() / 1e3, log[sim.ikr_key], 'r--', zorder=5)
        axs[p][1][0].set_xlabel('Time (s)')
        axs[p][0][0].set_xlim(0, max(log.time() / 1e3))
        axs[p][0][0].set_ylabel('Voltage (mV)')
        axs[p][1][0].set_ylabel('Current (A/F)')
    else:
        # axs[p][1][0].plot(log.time(), log[sim.ikr_key], 'r--', zorder=5)
        axs[p][1][0].set_xlabel('Time (ms)')
        axs[p][0][0].set_xlim(0, max(log.time()))
    axs[p][1][0].sharex(axs[p][0][0])
    axs[p][0][0].tick_params(labelbottom=False)
    axs[p][0][0].set_title(prot)
    axs[p][0][0].set_ylim(-125, 50)
    if p != 3:
        axs[p][1][0].set_ylim(-0.03, 1)
        axs[p][1][0].set_yticks([0, 0.5, 1])
        if p != 0:
            axs[p][0][0].tick_params(labelleft=False)
            axs[p][1][0].tick_params(labelleft=False)
    axs[p][0][0].grid(linestyle='--')
    axs[p][1][0].grid(linestyle='--')
    axs[p][0][0].set_yticks([-100, -50, 0, 50])

axs[3][0][0].set_ylabel('Voltage (mV)')
axs[3][1][0].set_ylabel('Current (A/F)')

# Save protocol figure
fig.savefig(os.path.join(modelling.FIG_DIR, f'sim-protocol-{model[3:]}.png'),
            bbox_inches='tight', dpi=500)
plt.close()

################################################
# Generate noisy signal under control conditions
# for period of interest of all protocols
################################################
for p, prot in enumerate(protocol_list):
    sweep_num = 3 if prot == 'staircase' else 10
    prepace = 0

    _, prot_length, _ = modelling.simulation.protocol_period(prot, mode='full')
    prot_start, prot_period, _ = modelling.simulation.protocol_period(
        prot, mode='partial')
    general_win = np.arange(prot_start, prot_start + prot_period, 10)
    log_times = []
    for i in range(sweep_num):
        log_times.extend(general_win + prot_length * i)

    control_log_file = os.path.join(modelling.PARAM_DIR, 'control_states',
                                    model, f'control_log_{prot}_dt10.csv')
    control_log = myokit.DataLog.load_csv(control_log_file)
    control_log_win = control_log.trim(prot_start, prot_start + prot_period)

    # Set up simulator to generate synthetic data
    sim = modelling.ModelSimController(model, prot)
    sim.load_control_state()

    results_dir = os.path.join(modelling.PARAM_DIR, model, prot)
    noise_level = [0, 1, 1]
    scale_level = [1, 1, 0.5]
    for n in range(3):
        noise = noise_level[n]
        scale = scale_level[n]
        noise_tag = f'noise{int(noise)}_scale{int(scale * 10)}'
        np.random.seed(0)
        control_log_win_noise = sim.add_noise(
            control_log_win, modelling.noise_level * 1e3 * noise,
            scale=scale)
        # 1e3 to adjust for order of magnitude
        control_log_win_noise.save_csv(os.path.join(
            results_dir, f'control_log_{prot}_partial_{noise_tag}.csv'))
