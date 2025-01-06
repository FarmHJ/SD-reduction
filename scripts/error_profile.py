import myokit
import numpy as np
import os
import pandas as pd

import modelling


# Define model and protocol
model = 'Li-SD'
protocol = 'Milnes'

# Define list of drugs (for ground truth parameter), parameter and their
# exploration range
drug_list = ['dofetilide', 'cisapride', 'verapamil']
param_names = ['Kmax', 'Ku', 'Vhalf', 'Kt']
ranges = {
    'Kmax': 10**np.linspace(0, 8, 20),
    'Ku': 10**np.linspace(np.log10(1.8e-5), 0, 20),
    'Vhalf': np.linspace(-200, -1, 20),
    'Kt': 10**np.linspace(-5, 8, 20)}

##############
# Ideal set up
##############
# Ideal set up is till steady state, 0.1 ms time step and full Milnes protocol
print('Difference profile for ideal set up')

# Define recording timestep and drug concentration
log_times = np.arange(0, 25e3, 0.1)
dimless_conc = np.array([10 ** i for i in np.linspace(-8, np.log10(0.65), 4)])

# Load control data for computation of fractional block
results_dir = os.path.join(modelling.PARAM_DIR, model, protocol)
control_log_file = os.path.join(modelling.PARAM_DIR, 'control_states',
                                model, f'control_log_{protocol}.csv')
control_log = myokit.DataLog.load_csv(control_log_file)

for drug in drug_list:
    print('Computing error profile for ', drug)

    # Simulate fractional block for optimised parameter combination (CiPA drug)
    sim = modelling.ModelSimController(model, protocol)
    sim.load_control_state()
    sim.set_drug_parameters(drug)
    ref_log_conc = []
    for c in dimless_conc:
        sim.update_initial_state(paces=0)
        sim.set_dimless_conc(c)
        ref_log = sim.simulate(log_times=log_times,
                               log_var=[sim.time_key, sim.ikr_key],
                               reset=False)
        ref_frac_block = list(ref_log[sim.ikr_key] /
                              control_log[sim.ikr_key])
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

            error_conc_list = []
            for c, conc in enumerate(dimless_conc):
                sim.update_initial_state(paces=0)
                sim.set_dimless_conc(conc)
                log = sim.simulate(log_times=log_times,
                                   log_var=[sim.time_key, sim.ikr_key],
                                   reset=False)

                # Compute fractional block
                frac_block = log[sim.ikr_key] / \
                    control_log[sim.ikr_key]

                # Compute mean squared difference in fractional block between
                # optimised parameter combination and with one parameter fixed.
                r = np.array(frac_block) - np.array(ref_log_conc[c])
                e = np.sqrt(np.sum(r**2, axis=0) / len(log))
                error_conc_list.append(e)

            error_p_list.append(np.mean(error_conc_list))

        error_list[f'error_{p}'] = error_p_list

    # Save computed error profile
    pd.DataFrame.from_dict(error_list).to_csv(
        os.path.join(results_dir,
                     f'errorprofile_{drug}_Milnesfull.csv'))

#####################
# Experimental set up
#####################
# Experimental set up is the first 10 pulses after drug addition, 10ms time
# step and Milnes protocol at 0mV

# Define time period of interest within the protocol (for 10 pulses)
prot_start, prot_period, _ = modelling.simulation.protocol_period(protocol)
general_win = np.arange(prot_start, prot_start + prot_period, 10)
log_times = []
for i in range(10):
    log_times.extend(general_win + 25e3 * i)

# Load control data with 10ms time step
control_log_file = os.path.join(modelling.PARAM_DIR, 'control_states',
                                model, f'control_log_{protocol}_dt10.csv')
control_log = myokit.DataLog.load_csv(control_log_file)
control_log_win = control_log.trim(prot_start, prot_start + prot_period)
sweep_num = 10

for drug in drug_list:
    print('Computing error profile for ', drug)

    # Simulate fractional block for optimised parameter combination (CiPA drug)
    sim = modelling.ModelSimController(model, protocol)
    sim.load_control_state()
    sim.set_drug_parameters(drug)
    ref_log_conc = []
    for c in dimless_conc:
        sim.update_initial_state(paces=0)
        sim.set_dimless_conc(c)
        ref_log = sim.simulate(prepace=0, save_signal=sweep_num,
                               log_times=log_times,
                               log_var=[sim.time_key, sim.ikr_key],
                               reset=False)

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

            error_conc_list = []
            for c, conc in enumerate(dimless_conc):
                sim.update_initial_state(paces=0)
                sim.set_dimless_conc(conc)
                log = sim.simulate(prepace=0, save_signal=sweep_num,
                                   log_times=log_times,
                                   log_var=[sim.time_key, sim.ikr_key],
                                   reset=False)

                # Compute and compile fractional block data
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
                     f'errorprofile_{drug}_Milnes_exp.csv'))
