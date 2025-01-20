import myokit
import numpy as np
import os
import pandas as pd

import modelling


# Define model and protocol
model = 'Li-SD'
protocol = 'staircase'
prot_mode = 'partial'
grid = 20

# Define list of drugs (for ground truth parameter), parameter and their
# exploration range
drug_list = ['dofetilide', 'cisapride', 'verapamil']
# drug_list = ['quinidine']
param_names = ['Kmax', 'Ku', 'Vhalf', 'Kt']
ranges = {
    'Kmax': 10**np.linspace(0, 10, grid + 8),
    'Ku': 10**np.linspace(np.log10(1.8e-6), 0, grid + 8),
    'Vhalf': np.concatenate((np.linspace(-240, -1, grid + 5),
                             np.arange(-0.75, 0, 0.25))),
    'Kt': 10**np.linspace(-5, 8, grid + 8)}

dimless_conc = np.array([10 ** i for i in np.linspace(-8, np.log10(0.65), 4)])

results_dir = os.path.join(modelling.PARAM_DIR, model, protocol,
                           'error_profile')
if not os.path.isdir(results_dir):
    os.makedirs(results_dir)

#####################
# Experimental set up
#####################
# Experimental set up is the first 10 pulses after drug addition, 10ms time
# step and Milnes protocol at 0mV

# Define time period of interest within the protocol (for 10 pulses)
sweep_num = 5
_, prot_length, _ = modelling.simulation.protocol_period(protocol, mode='full')

prot_start, prot_period, _ = modelling.simulation.protocol_period(protocol,
                                                                  mode=prot_mode)
general_win = np.arange(prot_start, prot_start + prot_period, 10)
log_times = []
for i in range(sweep_num):
    log_times.extend(general_win + prot_length * i)

sim = modelling.ModelSimController(model, protocol)
sim.load_control_state()

# Load control data with 10ms time step
control_log_file = os.path.join(modelling.PARAM_DIR, 'control_states',
                                model, f'control_log_{protocol}_dt10.csv')
control_log = myokit.DataLog.load_csv(control_log_file)
control_log_win = control_log.trim(prot_start, prot_start + prot_period)

for drug in drug_list:
    print('Computing error profile for', drug)

    # Simulate fractional block for optimised parameter combination (CiPA drug)

    sim.set_drug_parameters(drug)
    print(sim.get_parameters())
    ref_log_conc = []
    for c in dimless_conc:
        sim.update_initial_state(paces=0)
        sim.set_dimless_conc(c)
        ref_log = sim.simulate(prepace=0, save_signal=sweep_num,
                               log_times=log_times,
                               log_var=[sim.time_key, sim.ikr_key],
                               reset=False)

        if sweep_num == 1:
            ref_frac_block = ref_log[sim.ikr_key] / control_log_win[sim.ikr_key]
        else:
            ref_frac_block = []
            for s in range(sweep_num):
                ref_frac_block.extend(list(ref_log[sim.ikr_key, s] /
                                        control_log_win[sim.ikr_key]))
        ref_log_conc.append(ref_frac_block)

    # Compute error profile for parameters of interest
    error_list = {}
    for n, p in enumerate(param_names):
        # Reset parameters within simulator
        sim.reset_parameters()
        sim.set_drug_parameters(drug)

        # Set up dictionary to save error values
        error_list[p] = ranges[p]
        error_p_list = []

        for i in ranges[p]:

            # Simulate signal
            sim.set_parameters({p: i}, set_Kt=False)
            print(sim.get_parameters())

            error_conc_list = []
            for c, conc in enumerate(dimless_conc):
                sim.update_initial_state(paces=0)
                sim.set_dimless_conc(conc)
                log = sim.simulate(prepace=0, save_signal=sweep_num,
                                   log_times=log_times,
                                   log_var=[sim.time_key, sim.ikr_key],
                                   reset=False)

                # Compute and compile fractional block data
                if sweep_num == 1:
                    frac_block = log[sim.ikr_key] / control_log_win[sim.ikr_key]
                else:
                    frac_block = []
                    for s in range(sweep_num):
                        frac_block.extend(list(log[sim.ikr_key, s] /
                                               control_log_win[sim.ikr_key]))

                # Compute mean squared difference between CiPA and one
                # parameter fixed
                r = np.array(frac_block) - np.array(ref_log_conc[c])
                e = np.sqrt(np.sum(r**2, axis=0) / len(log))
                error_conc_list.append(e)

            error_p_list.append(np.mean(error_conc_list))

        error_list[f'error_{p}'] = error_p_list

    # Save computed error profile
    pd.DataFrame.from_dict(error_list).to_csv(
        os.path.join(results_dir,
                     f'errorprofile_{drug}_{protocol}_{prot_mode}_pulses5_xramp.csv'))
