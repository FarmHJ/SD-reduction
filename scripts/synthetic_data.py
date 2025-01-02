import itertools
import matplotlib.pyplot as plt
import numpy as np
import os

import modelling


######################
# Parameter simulation
######################
model = 'Li-SD'
protocol = 'Milnes'

plt.rcParams.update({'font.size': 8})
fig = plt.figure(figsize=(11 * 2, 12 * 2))
gs = fig.add_gridspec(12, 11, wspace=0.1)
axs = [[fig.add_subplot(gs[i, j]) for j in range(11)] for i in range(12)]

# Set up simulation
sim = modelling.ModelSimController(model, protocol)
sim.load_control_state()

dimless_conc = np.array([10 ** i for i in np.linspace(-8, np.log10(0.65), 4)])

# Define parameter range
param_names = ['Kmax', 'Ku', 'Vhalf']
Kmax_range = 10**np.linspace(np.log10(1), np.log10(1e8), 5)
Ku_range = 10**np.linspace(np.log10(1.8e-5), np.log10(1), 5)
Vhalf_range = np.linspace(-200, -1, 5)

count = 0
for Kmax, Ku, Vhalf in itertools.product(
        Kmax_range, Ku_range, Vhalf_range):

    param_values = [Kmax, Ku, Vhalf]
    param_dict = {param_names[i]: param_values[i] for i in range(3)}
    r, c = int(count / 11), int(count % 11)

    sim.update_initial_state(paces=0)
    sim.set_parameters(param_dict)

    for d in dimless_conc:
        sim.set_dimless_conc(d)
        log = sim.simulate()

        axs[r][c].plot(log.time() / 1e3, log[sim.ikr_key])
        axs[r][c].tick_params(labelbottom=False, labelleft=False)
    count += 1

# Save protocol figure
fig.savefig(os.path.join(modelling.FIG_DIR, 'syn-data.pdf'),
            bbox_inches='tight', dpi=500)
plt.close()
