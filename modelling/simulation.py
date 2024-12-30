# Difference in simulation results is due to the formulation of drug
# concentration.
import myokit
import os

import modelling


def IKrmodel_mmt(model):
    """
    Take model name and returns mmt file directory
    """
    IKrmodel_dir = os.path.join(modelling.MAIN_DIR, 'models', 'IKr_models')
    filenames = {
        'Li-SD': 'li2016-SD.mmt'}

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

        # Load model
        self.model = myokit.load_model(IKrmodel_mmt(model_name))

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

        self.model_name = model_name
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

    def set_initial_state(self, states):
        self.initial_state = states

    def simulate(self, prepace=1000, save_signal=1, timestep=0.1,
                 log_var=None, reset=True, log_times=None):

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
