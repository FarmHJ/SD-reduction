# Difference in simulation results is due to the formulation of drug
# concentration.
import myokit
import numpy as np
import os
import pandas as pd

import modelling


def IKrmodel_mmt(model):
    """
    Take model name and returns mmt file directory
    """
    IKrmodel_dir = os.path.join(modelling.MAIN_DIR, 'models', 'IKr_models')
    filenames = {
        'Li-SD': 'Li-SD.mmt'}

    return os.path.join(IKrmodel_dir, filenames[model])


def get_protocol(prot_name):
    """
    Returns the protocol in myokit format
    """
    if prot_name not in ['ramp', 'step', 'staircase', 'Milnes']:
        raise ValueError(
            "Choice of protocol must be either 'ramp', 'step', 'staircase' "
            "or 'Milnes'.")
    prot_dir = os.path.join(modelling.MAIN_DIR, 'models', 'protocols')
    return myokit.load_protocol(os.path.join(prot_dir, f'{prot_name}.mmt'))


def protocol_period(prot_name):
    # (protocol start time, protocol period of interest,
    # protocol time delay for plot)
    prot_time = {
        'Milnes': (1005 + 100, 10000 - 100, 60)}  # remove peak

    return prot_time[prot_name]


class ModelSimController(object):
    """
    To control the simulations of models
    """

    def __init__(self, model_name, protocol_name):
        super(ModelSimController, self).__init__()

        self.model_name = model_name
        self.protocol_name = protocol_name
        # Load model
        self.model = myokit.load_model(IKrmodel_mmt(model_name))
        if model_name == 'Li-SD':
            self.param_names = ['Kmax', 'Ku', 'halfmax', 'n', 'Vhalf']

        # parameters
        self._parameters = {}

        if protocol_name in ['ramp', 'staircase']:
            self.set_ramp_protocol(protocol_name)

        self.sim = myokit.Simulation(self.model)
        self.sim.set_tolerance(abs_tol=modelling.ABS_TOL,
                               rel_tol=modelling.REL_TOL)
        self.initial_state = self.sim.state()

        protocol = get_protocol(protocol_name)
        self._cl = protocol.characteristic_time()
        self.sim.set_protocol(protocol)

        del protocol

        self.time_key = 'engine.time'
        self.ikr_key = 'ikr.IKr'
        self.Vm_key = 'membrane.V'
        self.ikr_key_head = 'ikr'
        self.ikr_component = self.model.get(self.ikr_key_head)

    def set_ramp_protocol(self, prot_name):
        c = self.model.get('membrane')
        v = c.get('V')
        v.set_binding(None)

        # Add a p variable
        if c.has_variable('vp'):
            vp = c.get('vp')
        else:
            vp = c.add_variable('vp')
            vp.set_binding('pace')
        vp.set_rhs(0)
        # vp.set_binding('pace')

        if prot_name == 'ramp':
            # v1 = c.add_variable('v1')
            v1 = c.get('v1') if c.has_variable('v1') else c.add_variable('v1')
            v1.set_rhs('727.5 - 1.25 * engine.time')

            # Set a new right-hand side equation for V
            v.set_rhs("""
                piecewise(
                    (engine.time >= 550 and engine.time < 646), v1,
                    vp)
            """)
        elif prot_name == 'staircase':
            # Add a v1 variable
            v1 = c.get('v1') if c.has_variable('v1') else c.add_variable('v1')
            v1.set_rhs('-150 + 0.1 * engine.time')

            # Add a v2 variable
            v2 = c.get('v2') if c.has_variable('v2') else c.add_variable('v2')
            v2.set_rhs('5694 - 0.4 * engine.time')

            # Set a new right-hand side equation for V
            v.set_rhs("""
                piecewise(
                    (engine.time > 300 and engine.time <= 700), v1,
                    (engine.time >=14410 and engine.time < 14510), v2,
                    vp)
            """)

    def load_control_state(self):
        data_dir = os.path.join(modelling.PARAM_DIR, 'control_states',
                                self.model_name)

        control_state = myokit.load_state(
            os.path.join(data_dir, f'control_state_{self.protocol_name}.csv'))
        self.initial_state = control_state

    def set_drug_parameters(self, drug, binding_model='SD'):

        params = pd.read_csv(os.path.join(modelling.PARAM_DIR,
                                          f'Li-{binding_model}.csv'),
                             index_col=0)
        SD_params = params.loc[[drug]]

        if 'Cmax' in SD_params.columns:
            SD_params = SD_params.drop(columns=['Cmax'])
        elif 'error' in SD_params.columns:
            SD_params = SD_params.drop(columns=['error'])
        SD_params = SD_params.to_dict(orient='index')[drug]

        self.set_parameters(SD_params)

    def set_parameters(self, param_input, set_Kt=True):

        if set_Kt:
            param_input['Kt'] = 3.5e-5

        dict_contents = list(param_input.items())
        param_dict = {f"{self.ikr_key_head}.{k}": content
                      for k, content in dict_contents}

        self._parameters.update(param_dict)

    def reset_parameters(self):
        param_dict = {}
        for k in self.param_names + ['Kt', 'gKr']:
            param_dict[self.ikr_key_head + '.' + k] = \
                self.model.get(self.ikr_component.var(k)).eval()

        self._parameters.update(param_dict)
        del param_dict

    def get_parameters(self):
        return self._parameters

    def set_conc(self, concentration):
        if concentration < 0:
            ValueError('Drug concentration is lower than 0.')
        self._conc = float(concentration)

        param_dict = {self.ikr_key_head + '.D': self._conc}
        self._parameters.update(param_dict)

    def convert_dimless_conc(self, halfmax, Hill):
        Dn = np.power(self._conc, Hill)
        return Dn / (Dn + halfmax)

    def set_dimless_conc(self, dimless_conc):
        if dimless_conc < 0 or dimless_conc >= 1:
            ValueError('Dimensionless concentration is lower than 0 or higher'
                       ' than 1.')
        self._dimless_conc = dimless_conc
        self._conc = self._dimless_conc / (1 - self._dimless_conc)

        param_dict = {f'{self.ikr_key_head}.D': self._conc,
                      f'{self.ikr_key_head}.halfmax': 1,
                      f'{self.ikr_key_head}.n': 1}
        self._parameters.update(param_dict)

    def _set_parameters(self):
        """
        Set the parameter values
        """
        for p in self._parameters.keys():
            self.sim.set_constant(p, self._parameters[p])

    def set_initial_state(self, states):
        self.initial_state = states

    def update_initial_state(self, paces=1000):
        """
        Mainly to simulate till steady state for drug-free conditions
        """

        # Update fixed parameters
        self._set_parameters()

        # Ensure initial condition
        self.sim.reset()
        self.sim.set_state(self.initial_state)

        for _ in range(paces):
            self.sim.pre(self._cl)
        self.initial_state = self.sim.state()

    def simulate(self, prepace=1000, save_signal=1, timestep=0.1,
                 log_var=None, reset=True, log_times=None):

        self._set_parameters()
        self.prepace = prepace

        self.sim.set_time(0)
        if reset:
            self.sim.reset()
            self.sim.set_state(self.initial_state)

        if log_var is None:
            log_var = [self.time_key, self.Vm_key, self.ikr_key]
        elif log_var == 'all':
            log_var = None

        if log_times is not None:
            timestep = None

        for _ in range(self.prepace):
            self.sim.pre(self._cl)

        log = self.sim.run(self._cl, log=log_var,
                           log_interval=timestep, log_times=log_times)

        for i in range(1, save_signal):
            self.sim.set_time(0)
            log2 = self.sim.run(self._cl, log=log_var,
                                log_interval=timestep,
                                log_times=log_times).npview()
            log2[self.time_key] += self._cl * i
            log = log.extend(log2)
        log = log.npview()

        if save_signal > 1:
            log = log.fold(self._cl)

        return log
