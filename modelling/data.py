import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

import modelling

drug_names = ['dofetilide', 'bepridil', 'terfenadine',
              'cisapride', 'verapamil', 'ranolazine',
              'quinidine', 'sotalol', 'chlorpromazine',
              'ondansetron', 'diltiazem', 'mexiletine']


class DataLibrary(object):
    """
    A data library class that reads the experimental data
    """
    def __init__(self):
        super(DataLibrary, self).__init__()

    def set_drug(self, drug):
        self.drug = drug

        self.data = pd.read_csv(os.path.join(modelling.DATA_DIR,
                                             self.drug + '.csv'))
        self.n_exp = max(self.data['exp'])
        self.n_sweeps = 10

        self.concs = np.sort(pd.unique(self.data['conc'].values))

    def get_conc_reps(self):
        conc_reps = {}
        for c in self.concs:
            reps = max(self.data.loc[self.data['conc'] == c, 'exp'])
            conc_reps.update({c: reps})

        return conc_reps

    def set_conc(self, conc):
        if conc not in self.concs:
            NameError("Choose a concentration within the \
                      list in `concs`")
        self.conc_data = self.data.loc[self.data['conc'] == conc]

    def plot_signal(self):

        conc_reps = self.get_conc_reps()

        cmap = matplotlib.cm.get_cmap('viridis')
        norm = matplotlib.colors.Normalize(0, self.n_sweeps)

        fig = plt.figure(figsize=(2 * self.n_exp, len(self.concs)))
        gs = fig.add_gridspec(len(self.concs), self.n_exp,
                              hspace=0.15, wspace=0.07)
        axs = [[fig.add_subplot(gs[i, j]) for j in
                range(conc_reps[self.concs[i]])] for i in
               range(len(self.concs))]

        # Remove artificial spike
        # self.data = self.data.loc[self.data['time'] >= 1005]

        for c, conc in enumerate(self.concs):
            for n in range(conc_reps[conc]):
                subset_data = self.data.loc[(self.data['conc'] == conc)
                                            & (self.data['exp'] == n + 1)]
                for s in range(self.n_sweeps):
                    subset_data_sweep = subset_data.loc[
                        subset_data['sweep'] == s]
                    axs[c][n - 1].plot(subset_data_sweep['time'] * 1e-3,
                                       subset_data_sweep['frac'],
                                       color=cmap(norm(s)))

        adjust_labels(axs, self.concs, 'normalised \n current')
        fig.savefig(os.path.join(modelling.FIG_DIR, 'exp_data',
                                 f'{self.drug}.pdf'),
                    bbox_inches='tight')
        plt.close()

    def get_mean_signal(self, cache=False):

        data_file = os.path.join(modelling.DATA_DIR,
                                 f'{self.drug}_mean.csv')
        if cache:
            mean_signal = pd.read_csv(data_file)
        else:
            conc_reps = self.get_conc_reps()
            mean_signal = pd.DataFrame({})

            for c in self.concs:
                signal_sweeps = pd.DataFrame({})
                for s in range(self.n_sweeps):
                    full_exps = 0
                    frac_block = [0] * 1000
                    block_stack = []

                    for exp in range(conc_reps[c]):
                        signal = self.data.loc[(self.data['conc'] == c) &
                                               (self.data['exp'] == exp + 1) &
                                               (self.data['sweep'] == s + 1)]

                        if signal.shape[0] == 1000:
                            frac_block += signal['frac'].values
                            block_stack.append(signal['frac'].values)
                            full_exps += 1

                    mean_block = frac_block / full_exps
                    std_block = np.std(block_stack, axis=0)
                    df = pd.DataFrame({'time': signal['time'],
                                       'frac': mean_block,
                                       'frac_std': std_block,
                                       'sweep': [s + 1] * signal.shape[0],
                                       'conc': [c] * signal.shape[0]})
                    signal_sweeps = pd.concat([signal_sweeps, df])

                mean_signal = pd.concat([mean_signal, signal_sweeps])

            mean_signal.to_csv(data_file, index=False)

        return mean_signal

    def plot_mean_signal(self):

        data = self.get_mean_signal()

        fig = plt.figure(figsize=(3, 3))
        ax = fig.add_subplot(111)
        colors = matplotlib.cm.get_cmap('tab10')

        # Remove artificial spike
        nospike_data = data.loc[self.data['time'] >= 1005]

        for c, conc in enumerate(self.concs):
            data_sweep = nospike_data.loc[nospike_data['conc'] == conc]
            for s in range(1, self.n_sweeps + 1):
                subset_data_sweep = data_sweep.loc[
                    data_sweep['sweep'] == s]
                time = subset_data_sweep['time']
                mean = subset_data_sweep['frac']
                std = subset_data_sweep['frac_std']
                ax.plot((time + s * max(time)) * 1e-3, mean, color=colors(c),
                        label=str(conc) + ' nM', zorder=1)
                ax.fill_between((time + s * max(time)) * 1e-3, mean + std,
                                mean - std, color=colors(c), alpha=0.4,
                                zorder=-1)

        handles = legend_without_duplicate_labels(ax)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Normalised current')
        ax.legend(*zip(*handles), handlelength=1, ncol=2, columnspacing=1)
        ax.set_ylim(0, 1.25)
        ax.grid(alpha=0.4, zorder=-10)
        ax.set_title(self.drug)
        fig.savefig(os.path.join(modelling.FIG_DIR, 'exp_data',
                                 f'{self.drug}_mean.pdf'),
                    bbox_inches='tight')
        plt.close()


def legend_without_duplicate_labels(ax):
    """
    Remove duplicated labels in legends
    """
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if
              l not in labels[:i]]

    return unique


def adjust_labels(axs, drug_conc_list, ylabel, xlabel='Time (s)',
                  ylim_init=(0, 1)):

    y_lb, y_ub = ylim_init
    x_lb, x_ub = 0, 10

    for row in range(len(axs)):
        axs[row][0].set_ylabel(ylabel)
        drug_conc_str = "{0:.0e}".format(drug_conc_list[row])
        base, power = drug_conc_str.split("e")
        power = int(power)

        # Indicate drug concentration
        axs[row][0].text(
            0.05, 0.95,
            base + r"$\times 10^{{{:d}}} $".format(power) + 'nM',
            fontsize=8, ha='left', va='top',
            transform=axs[row][0].transAxes)

        for col in range(len(axs[row])):
            y_bottom, y_top = axs[row][col].get_ylim()
            y_lb = min(y_lb, y_bottom)
            y_ub = max(y_ub, y_top)
            x_left, x_right = axs[row][col].get_xlim()
            x_lb = min(x_lb, x_left)
            x_ub = max(x_ub, x_right)

            axs[row][col].set_rasterization_zorder(0)

    for row in range(len(axs)):
        for col in range(len(axs[row])):
            axs[row][col].set_ylim(y_lb, y_ub)
            axs[row][col].set_xlim(x_lb, x_ub)

            if col != 0:
                axs[row][col].sharex(axs[row][0])
                axs[row][col].sharey(axs[row][0])
                axs[row][col].tick_params(labelleft=False)

            if row != len(axs) - 1:
                axs[row][col].tick_params(labelbottom=False)
            else:
                axs[row][col].set_xlabel(xlabel)
