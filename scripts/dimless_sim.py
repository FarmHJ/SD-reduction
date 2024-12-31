import matplotlib.pyplot as plt
import myokit
import numpy as np
import os

import modelling
import modelling.model_details


model = 'Li-SD'
protocol = 'Milnes'
drug = 'dofetilide'

plt.rcParams.update({'font.size': 8})
fig = plt.figure(figsize=(4, 4))

data_dir = os.path.join(modelling.PARAM_DIR, 'control_states', model)

# Set up simulation
sim = modelling.ModelSimController(model, protocol)
control_state = myokit.load_state(
    os.path.join(data_dir, f'control_state_{protocol}.csv'))
sim.initial_state = control_state
sim.set_drug_parameters(drug)

# dimless_conc = np.array([10 ** i for i in np.linspace(-8, np.log10(0.65), 10)])
# for d in dimless_conc:

#     sim.set_dimless_conc(d)
#     sim.update_initial_state(paces=0)
#     log = sim.simulate()

#     # Plot drug condition
#     plt.plot(log.time() / 1e3, log[sim.ikr_key])
# plt.xlabel('Time (s)')
# plt.xlim(0, max(log.time() / 1e3))

# # Save protocol figure
# fig.savefig(os.path.join(modelling.FIG_DIR, 'sim-dimless-conc.pdf'),
#             bbox_inches='tight', dpi=500)
# plt.close()

####################################################
# Comparison of conversion to original concentration
####################################################
# drug_param = sim.get_parameters()
# Hill = drug_param['ikr.n']
# halfmax = drug_param['ikr.halfmax']
# conc_list = [2, 10, 100, 1000]
# for d in conc_list:
#     sim.set_drug_parameters(drug)
#     sim.set_conc(d)
#     sim.update_initial_state(paces=0)
#     log_conc = sim.simulate()
#     log_conc_win = log_conc.trim(1005 + 100, 1005 + 10000)

#     plt.plot(log_conc.time() / 1e3, log_conc[sim.ikr_key])

#     dimless_conc = sim.convert_dimless_conc(halfmax, Hill)
#     sim.set_dimless_conc(dimless_conc)
#     sim.update_initial_state(paces=0)
#     log_dimless = sim.simulate()
#     log_dimless_win = log_dimless.trim(1005 + 100, 1005 + 10000)

#     plt.plot(log_dimless.time() / 1e3, log_dimless[sim.ikr_key], '--')

#     np.testing.assert_almost_equal(log_conc[sim.ikr_key],
#                                    log_dimless[sim.ikr_key])

# fig.savefig(os.path.join(modelling.FIG_DIR, 'dimless-conc-conversion.pdf'),
#             bbox_inches='tight', dpi=500)
# plt.close()

###################################
# Dimensionless concentration range
###################################
dimless_conc_list = []
for drug in modelling.model_details.drug_list:
    sim.set_drug_parameters(drug)
    drug_param = sim.get_parameters()
    Hill = drug_param['ikr.n']
    halfmax = drug_param['ikr.halfmax']

    for conc in modelling.model_details.lit_drug_conc[drug]:
        sim.set_conc(conc)
        dimless_conc = sim.convert_dimless_conc(halfmax, Hill)
        dimless_conc_list.append(dimless_conc)

quantiles = [5, 95]
data_quantiles = [np.percentile(dimless_conc_list, q) for q in quantiles]
print(data_quantiles)

fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(3, 3))
axs.violinplot(dimless_conc_list, showmeans=False, showmedians=True,
               quantiles=[0.05, 0.95])
axs.set_yscale('log')
axs.set_ylabel('Dimensionless concentration (log)')
fig.savefig(os.path.join(modelling.FIG_DIR, 'dimless-conc-dist.pdf'),
            bbox_inches='tight', dpi=500)
plt.close()
