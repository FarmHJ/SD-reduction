import myokit
import numpy as np
import os
import pandas as pd
import pints

import modelling


# Define model and protocol
model = 'Li-SD'
protocol = 'Milnes'
prot_mode = 'partial'

# Define noise level
noise_level = 1
scale_level = 1
noise_tag = f'noise{int(noise_level)}_scale{int(scale_level * 10)}'
results_dir = os.path.join(modelling.PARAM_DIR, model, protocol,
                           'profile_likelihood')

# Define details for simulation including
# list of drugs (for ground truth parameter), parameters of interest,
# their exploration range and dimensionless concentration range
drug_list = ['dofetilide', 'cisapride', 'verapamil']
param_names = ['Kmax', 'Ku', 'Vhalf']
param_interest = ['Kmax', 'Ku', 'Vhalf']

grid = 20
ranges = {
    'Kmax': 10**np.linspace(0, 10, grid + 8),
    'Ku': 10**np.linspace(np.log10(1.8e-6), 0, grid + 8),
    'Vhalf': np.concatenate((np.linspace(-240, -1, grid + 5),
                             np.arange(-0.75, 0, 0.25))),
    'Kt': 10**np.linspace(-5, 8, grid + 8)}
dimless_conc = np.array([10 ** i for i in np.linspace(-8, np.log10(0.65), 4)])


# Define PINTS model to run optimisation for profile likelihood
class InfModel(pints.ForwardModel):
    def __init__(self, sim, param_names):
        """
        Set up model to be fitted
        """
        self.sim = sim
        self.param_names = param_names

    def n_outputs(self):
        """
        Number of outputs
        """
        return 1

    def n_parameters(self):
        """
        Number of parameters
        """
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

        # Setup simulation
        # Reset model
        self.sim.update_initial_state(paces=0)

        # Set parameter values
        param_dict = {self.param_names[i]: parameters[i] for i in
                      range(len(self.param_names))}
        self.sim.set_parameters(param_dict)

        # Set concentration
        self.sim.set_dimless_conc(self.conc)

        # Simulate current signal
        log = self.sim.simulate(prepace=0, save_signal=sweep_num,
                                log_times=times,
                                log_var=[self.sim.time_key, self.sim.ikr_key],
                                reset=False)

        # Compute and compile fractional block
        output = []
        for s in range(sweep_num):
            output += list(log[self.sim.ikr_key, s] /
                           control_log_win[self.sim.ikr_key])

        return output


# Define boundaries for parameters and parameter transformation for
# optimisation
boundaries_dict = {'Kmax': (1e-1, 1e10),
                   'Ku': (1e-8, 1e1),
                   'Vhalf': (-300, 0),
                   'Kt': (1e-7, 1e10)}
transform_dict = {'Kmax': pints.LogTransformation(1),
                  'Ku': pints.LogTransformation(1),
                  'Vhalf': pints.IdentityTransformation(1),
                  'Kt': pints.LogTransformation(1)}

###################################################
# Simulation set up to follow experimental settings
###################################################
# Experimental recordings are taken for the first 10 pulses after drug
# addition at 10ms time step using the modified Milnes' protocol at 0mV.

# Define the time period of interest within the protocol (for 10 pulses)
# to record the corresponding data points.
sweep_num = 10
_, prot_length, _ = modelling.simulation.protocol_period(protocol, mode='full')

prot_start, prot_period, _ = modelling.simulation.protocol_period(
    protocol, mode=prot_mode)
general_win = np.arange(prot_start, prot_start + prot_period, 10)
log_times = []
for i in range(sweep_num):
    log_times.extend(general_win + prot_length * i)

# Load current signals under control condition for the denominator of the
# fractional block.
# Performed separately to reduce repeated simulations.
control_log_file = os.path.join(modelling.PARAM_DIR, 'control_states',
                                model, f'control_log_{protocol}_dt10.csv')
control_log = myokit.DataLog.load_csv(control_log_file)
control_log_win = control_log.trim(prot_start, prot_start + prot_period)
control_log_win_noise = myokit.DataLog.load_csv(os.path.join(
    results_dir, '..', 'inference',
    f'control_log_{protocol}_{noise_tag}_trimmed.csv'))

# Set up a simulation controller
sim = modelling.ModelSimController(model, protocol)
# Get the model to steady state.
sim.load_control_state()

# Load parameters of the given drug-binding formulation for all CiPA training
# drugs.
params = pd.read_csv(os.path.join(modelling.PARAM_DIR, f'{model}.csv'),
                     index_col=0)
np.random.seed(0)
for drug in drug_list:
    print('Performing error profile for ', drug)

    # Get parameters of the given drug
    param_drug = params.loc[[drug]]
    param_drug = param_drug.to_dict(orient='index')[drug]
    # Ensure that the trapping rate Kt is fixed at its defined value.
    param_drug.update({'Kt': 3.5e-5})

    # Instantiate dictionary to save error profile outcome
    error_profile = {}

    # Generate fractional block without noise
    # Input drug by setting parameter values of given drug
    sim.set_drug_parameters(drug)
    ref_log_conc = []
    for c in dimless_conc:
        # Reset simulation model to steady state
        sim.update_initial_state(paces=0)
        # Set given concentration
        sim.set_dimless_conc(c)
        # Simulate current signals
        ref_log = sim.simulate(prepace=0, save_signal=sweep_num,
                               log_times=log_times,
                               log_var=[sim.time_key, sim.ikr_key],
                               reset=False)
        # Add noise
        ref_log = sim.add_noise(ref_log,
                                modelling.noise_level * 1e3 * noise_level,
                                scale=scale_level)
        # modelling.noise_level is given by 6.6e-6uA
        # multiplication of 1e3 is performed to adjust for the order of
        # magnitude.

        # Compute and compile fractional block data
        ref_frac_block = []
        for s in range(sweep_num):
            ref_frac_block.extend(list(ref_log[sim.ikr_key, s] /
                                       control_log_win_noise[sim.ikr_key]))
        ref_log_conc.append(ref_frac_block)

    # Perform error profile for each parameter of interest
    for n, p in enumerate(param_interest):
        # Set the initial point at a given drug (parameter combination)
        sim.reset_parameters()
        sim.set_drug_parameters(drug)

        # Get parameters that are to be inferred
        # All parameters except the parameter of interest
        inferring_params = [i for i in param_names if i != p]

        # Define inference problem
        errors = []
        for c, conc in enumerate(dimless_conc):
            # Set up the inference model using PINTS
            inf_model = InfModel(sim, inferring_params)
            inf_model.set_conc(conc)
            problem = pints.SingleOutputProblem(inf_model, log_times,
                                                ref_log_conc[c])
            errors.append(pints.RootMeanSquaredError(problem))
        
        # Summation of RMSD across all drug concentrations
        error_fn = pints.SumOfErrors(
            errors, [1 / len(dimless_conc)] * len(dimless_conc))

        # Parameter transformation
        transform_list = [transform_dict[name] for name in inferring_params]
        transform = pints.ComposedTransformation(*transform_list)

        # Generate initial guess as the true parameter combination of the given drug
        guess = [param_drug[name] for name in inferring_params]
        print('initial guess: ', guess)

        # Define parameter boundaries
        low_bound = [boundaries_dict[name][0] for name in inferring_params]
        up_bound = [boundaries_dict[name][1] for name in inferring_params]
        boundaries = pints.RectangularBoundaries(lower=low_bound,
                                                 upper=up_bound)

        # Run optimisation using CMAES
        opt = pints.OptimisationController(error_fn, guess,
                                           boundaries=boundaries,
                                           transformation=transform,
                                           method=pints.CMAES)
        opt.set_log_to_screen(False)
        # opt.set_max_iterations(5)
        opt.set_parallel(True)
        opt_param_main, s = opt.run()
        print('optimised: ', opt_param_main)

        error_list = []
        param_list = []
        # Save outcome
        error_list.append(s)
        param_list.append(param_drug[p])

        # Rearrange the parameter values
        # Explore the parameter values starting from the reference point
        # (the true parameter combination)
        split_idx = np.where(ranges[p] > param_drug[p])[0]
        split_idx = split_idx[0] if split_idx.size > 0 else -1
        low_ranges = ranges[p][:split_idx]
        up_ranges = ranges[p][split_idx:]

        # Run the optimisation for fixed values larger than the true
        # parameter value
        # Initial guess is set as the value optimised from previous fixed point
        for count, i in enumerate(up_ranges):
            # Set the parameter of interest to the given value
            sim.set_parameters({p: i}, set_Kt=False)

            errors = []
            # For a series of (four) drug concentrations, optimise for the
            # remaining parameters
            for c, conc in enumerate(dimless_conc):
                # Set up problem and define RMSD
                inf_model = InfModel(sim, inferring_params)
                inf_model.set_conc(conc)
                problem = pints.SingleOutputProblem(inf_model, log_times,
                                                    ref_log_conc[c])
                errors.append(pints.RootMeanSquaredError(problem))
            error_fn = pints.SumOfErrors(
                errors, [1 / len(dimless_conc)] * len(dimless_conc))

            # Parameter transformation
            transform_list = [transform_dict[name] for name in
                              inferring_params]
            transform = pints.ComposedTransformation(*transform_list)

            # Take initial guess as optimised parameter value of previous run
            guess = opt_param_main if count == 0 else opt_param
            print('initial guess: ', guess)

            # Define parameter boundaries
            low_bound = [boundaries_dict[name][0] for name in inferring_params]
            up_bound = [boundaries_dict[name][1] for name in inferring_params]
            boundaries = pints.RectangularBoundaries(lower=low_bound,
                                                     upper=up_bound)

            # Run optimisation
            opt = pints.OptimisationController(error_fn, guess,
                                               boundaries=boundaries,
                                               transformation=transform,
                                               method=pints.CMAES)
            opt.set_log_to_screen(False)
            # opt.set_max_iterations(5)
            opt.set_parallel(True)
            opt_param, s = opt.run()
            print('optimised ', opt_param)

            # Save outcome
            error_list.append(s)
            param_list.append(i)

        # Run the optimisation for fixed values smaller than the true
        # parameter value
        count = 0
        for i in reversed(low_ranges):
            # Set the parameter of interest to the given value
            sim.set_parameters({p: i}, set_Kt=False)

            errors = []
            for c, conc in enumerate(dimless_conc):
                # Set up problem and define RMSD
                inf_model = InfModel(sim, inferring_params)
                inf_model.set_conc(conc)
                problem = pints.SingleOutputProblem(inf_model, log_times,
                                                    ref_log_conc[c])
                errors.append(pints.RootMeanSquaredError(problem))
            error_fn = pints.SumOfErrors(
                errors, [1 / len(dimless_conc)] * len(dimless_conc))

            # Parameter transformation
            transform_list = [transform_dict[name] for name
                              in inferring_params]
            transform = pints.ComposedTransformation(*transform_list)

            # Take initial guess as optimised parameter value of previous run
            guess = opt_param_main if count == 0 else opt_param
            print('initial guess: ', guess)
            low_bound = [boundaries_dict[name][0] for name in inferring_params]
            up_bound = [boundaries_dict[name][1] for name in inferring_params]
            boundaries = pints.RectangularBoundaries(lower=low_bound,
                                                     upper=up_bound)

            # Run optimisation
            opt = pints.OptimisationController(error_fn, guess,
                                               boundaries=boundaries,
                                               transformation=transform,
                                               method=pints.CMAES)
            opt.set_log_to_screen(False)
            # opt.set_max_iterations(5)
            opt.set_parallel(True)
            opt_param, s = opt.run()
            print('optimised: ', opt_param)

            # Save outcome
            error_list.append(s)
            param_list.append(i)
            count += 1

        # Organise outputs of all parameters of interest
        error_profile[p] = param_list
        error_profile[f'likelihood_{p}'] = error_list

    # Save error profile
    pd.DataFrame.from_dict(error_profile).to_csv(
        os.path.join(results_dir, f'profilelikelihood_{drug}_{protocol}_'
                     f'{prot_mode}_{noise_tag}.csv'))
