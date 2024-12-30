import myokit
import os

import modelling

model = 'Li-SD'
protocol_list = ['Milnes']

results_dir = os.path.join(modelling.PARAM_DIR, 'control_states', model)
if not os.path.isdir(results_dir):
    os.makedirs(results_dir)

for prot in protocol_list:
    sim = modelling.ModelSimController(model, prot)
    # sim.set_conc(0)
    control_log = sim.simulate(prepace=999,
                               log_var=[sim.time_key, sim.ikr_key])

    control_state = sim.sim.state()
    control_log.save_csv(os.path.join(results_dir, f'control_log_{prot}.csv'))
    myokit.save_state(os.path.join(results_dir, f'control_state_{prot}.csv'),
                      control_state)
