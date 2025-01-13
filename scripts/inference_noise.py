import itertools
import matplotlib.pyplot as plt
import myokit
import numpy as np
import os
import pandas as pd
import pints
import random

import modelling


model = 'Li-SD'
protocol = 'Milnes'
param_names = ['Kmax', 'Ku', 'Vhalf']
setup = 'exp'  # 'exp' for experimental setup and 'ideal' for ideal setup

grid = 5

#############################
# Inference on synthetic data
#############################
"""
Ideal setup - drug simulated to steady state, time step at 0.1 ms, full trace
of data
Experimental setup - first 10 pulses after addition of drug, time step at 10
ms, only 0 mV step of the Miles protocol is used
"""
results_dir = os.path.join(modelling.PARAM_DIR, model, protocol)
if not os.path.isdir(results_dir):
    os.makedirs(results_dir)


# PINTS model
class InfModel(pints.ForwardModel):
    def __init__(self, sim):
        """
        Set up model to be fitted
        """
        self.sim = sim
        self.param_names = param_names

    def n_outputs(self):
        return 1

    def n_parameters(self):
        return len(self.param_names)

    def set_conc(self, conc):
        """
        Set dimensionless concentration
        """
        self.conc = conc

    def simulate(self, parameters, times):
        """
        Generate model simulation
        """

        self.sim.update_initial_state(paces=0)
        param_dict = {self.param_names[i]: parameters[i] for i in range(3)}
        self.sim.set_parameters(param_dict)
        self.sim.set_dimless_conc(self.conc)

        log = self.sim.simulate(prepace=prepace, save_signal=sweep_num,
                                log_times=times,
                                log_var=[self.sim.time_key, self.sim.ikr_key],
                                reset=False)

        if setup == 'exp':
            output = []
            for s in range(sweep_num):
                output += list(log[self.sim.ikr_key, s] /
                               control_log_win[self.sim.ikr_key])
        elif setup == 'ideal':
            output = log[self.sim.ikr_key] / control_log[self.sim.ikr_key]

        return output


####################
# Experimental setup
####################
if setup == 'exp':
    sweep_num = 10
    prepace = 0

    prot_start, prot_period, _ = modelling.simulation.protocol_period(protocol)
    general_win = np.arange(prot_start, prot_start + prot_period, 10)
    log_times = []
    for i in range(sweep_num):
        log_times.extend(general_win + 25e3 * i)

    control_log_file = os.path.join(modelling.PARAM_DIR, 'control_states',
                                    model, f'control_log_{protocol}_dt10.csv')
    control_log = myokit.DataLog.load_csv(control_log_file)
    control_log_win = control_log.trim(prot_start, prot_start + prot_period)

#############
# Ideal setup
#############
elif setup == 'ideal':
    sweep_num = 1
    prepace = 99

    # Define time array
    log_times = np.arange(0, 25e3, 0.1)

    # Load control data - to compute fractional black
    control_log_file = os.path.join(modelling.PARAM_DIR, 'control_states',
                                    model, f'control_log_{protocol}.csv')
    control_log = myokit.DataLog.load_csv(control_log_file)

# Set up simulator to generate synthetic data
sim = modelling.ModelSimController(model, protocol)
sim.load_control_state()

np.random.seed(0)
control_log_win_noise = sim.add_noise(control_log_win, modelling.noise_level * 1e3)
# 1e3 to adjust for order of magnitude
control_log_win_noise.save_csv(os.path.join(
    results_dir, f'control_log_{protocol}_noise_trimmed.csv'))

# plt.figure(figsize=(7, 3))
# plt.plot(control_log_win_noise.time(), control_log_win_noise[sim.ikr_key])
# plt.savefig('test.pdf')

# Define drug concentration range for synthetic data
dimless_conc = np.array([10 ** i for i in np.linspace(-8, np.log10(0.65), 4)])

# Define parameter space to be explored, or set of parameter combinations
# For synthetic data
Kmax_range = 10**np.linspace(np.log10(1), np.log10(1e8), grid)
Ku_range = 10**np.linspace(np.log10(1.8e-5), np.log10(1), grid)
Vhalf_range = np.linspace(-200, -1, grid)

param_dict_list = []
for Kmax, Ku, Vhalf in itertools.product(
        Kmax_range, Ku_range, Vhalf_range):

    param_values = [Kmax, Ku, Vhalf]
    param_dict = {param_names[i]: param_values[i] for i in range(3)}
    param_dict_list.append(param_dict)

# Set up inference problem
# Define prior and boundaries
prior_list = [pints.UniformLogPrior(1, 1e8),
              pints.UniformLogPrior(1e-5, 1)]
# Prior for Vhalf: uniform prior(-150, 0)
boundaries = pints.RectangularBoundaries([1e-1, 1e-8, -200],
                                         [1e10, 1e1, 0])

# # Set up structure to plot synthetic data with noise
# fig = plt.figure(figsize=(7, 6))
# gs = fig.add_gridspec(4, 2, wspace=0.15, hspace=0.1)
# axs = [[fig.add_subplot(gs[i, j]) for j in range(2)] for i in range(4)]

random.seed(0)
sample = random.sample(range(0, len(param_dict_list)), 8)
print(sample)
count = 0

# For each parameter combination
for p_num, param in enumerate(param_dict_list):
    sim.set_parameters(param, set_Kt=False)
    if p_num in sample:
        print(param)

    errors = []
    for c, conc in enumerate(dimless_conc):
        # Prepare synthetic data
        sim.update_initial_state(paces=0)
        sim.set_dimless_conc(conc)

        log = sim.simulate(prepace=prepace, save_signal=sweep_num,
                           log_times=log_times,
                           log_var=[sim.time_key, sim.ikr_key],
                           reset=False)
        log = sim.add_noise(log, modelling.noise_level * 1e3)  # --> uA
        # 1e3 to adjust for order of magnitude

        if setup == 'exp':
            SD_frac_block = []
            for s in range(sweep_num):
                SD_frac_block.extend(list(log[sim.ikr_key, s] /
                                          control_log_win_noise[sim.ikr_key]))
        elif setup == 'ideal':
            SD_frac_block = log[sim.ikr_key] / control_log[sim.ikr_key]

        # Plot noisy synthetic data
        if p_num in sample:
            log.save_csv(os.path.join(
                results_dir, f'log_{protocol}_noise_{int(p_num)}_conc{c}.csv'))
            # print(p_num, count)
            # r, c = int(count / 2), count % 2
            # axs[r][c].plot(SD_frac_block)
            # axs[r][c].set_ylim(-0.05, 1.05)

    if p_num in sample:
        count += 1

#         # Instantiate forward model
#         inf_model = InfModel(sim)
#         inf_model.set_conc(c)
#         # Create single output problem
#         problem = pints.SingleOutputProblem(inf_model, log_times,
#                                             SD_frac_block)

#         # Error function
#         errors.append(pints.RootMeanSquaredError(problem))

#     error_fn = pints.SumOfErrors(errors,
#                                  [1 / len(dimless_conc)] * len(dimless_conc))
#     # Transform parameters
#     transform = pints.ComposedTransformation(
#         pints.LogTransformation(error_fn.n_parameters() - 1),
#         pints.IdentityTransformation(1))

#     # Perform inference
#     reps = 5
#     param_scores = []
#     for i in range(reps):
#         # Generate initial guess from prior distribution
#         np.random.seed((i + 1) * 100)
#         guess = [p.sample()[0][0] for p in prior_list]
#         # Prior for Vhalf
#         guess.append(np.random.uniform(-150, 0))
#         print('initial guess: ', guess)

#         # Run optimisation
#         opt = pints.OptimisationController(error_fn, guess,
#                                            boundaries=boundaries,
#                                            transformation=transform,
#                                            method=pints.CMAES)
#         # opt.set_log_to_screen(False)
#         opt.set_parallel(True)
#         opt_param, s = opt.run()

#         # Save outcome
#         param_scores.append(list(opt_param) + [s])

#     # Organise and compile optimisation outcome
#     score_col = [f'fit_{i}' for i in param_names] + ['error']
#     scores_dict = pd.DataFrame(param_scores, columns=score_col)
#     scores_dict = scores_dict.sort_values(by=['error'])

#     # ref_col = [f'ref_{i}' for i in param_names]
#     param_dict = pd.DataFrame(param, index=[0])
#     col_rename = {i: f'ref_{i}' for i in param_names}
#     param_dict = param_dict.rename(columns=col_rename)

#     param_scores = pd.concat([scores_dict.iloc[[0], :].reset_index(drop=True),
#                               param_dict], axis=1)
#     if p_num == 0:
#         combined_scores = param_scores
#     else:
#         combined_scores = pd.concat([combined_scores,
#                                      param_scores])
#     print(combined_scores)

# # Save all inference results
# combined_scores.to_csv(os.path.join(results_dir,
#                                     f'inference_Milnes_{setup}_noise.csv'))
# fig_dir = os.path.join(modelling.FIG_DIR, 'syn_data', model, protocol)
# fig.savefig(os.path.join(fig_dir, 'syn-data-noisy.pdf'),
#             bbox_inches='tight')
