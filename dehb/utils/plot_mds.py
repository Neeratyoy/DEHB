import os
import json
import pickle
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from sklearn.manifold import MDS
from sklearn.decomposition import PCA

import ConfigSpace
from ConfigSpace import ConfigurationSpace


def vector_to_configspace(cs, vector):
    '''Converts numpy array to ConfigSpace object

    Works when cs is a ConfigSpace object and the input vector is in the domain [0, 1].
    '''
    new_config = cs.sample_configuration()
    for i, hyper in enumerate(cs.get_hyperparameters()):
        if type(hyper) == ConfigSpace.OrdinalHyperparameter:
            ranges = np.arange(start=0, stop=1, step=1/len(hyper.sequence))
            param_value = hyper.sequence[np.where((vector[i] < ranges) == False)[0][-1]]
        elif type(hyper) == ConfigSpace.CategoricalHyperparameter:
            ranges = np.arange(start=0, stop=1, step=1/len(hyper.choices))
            param_value = hyper.choices[np.where((vector[i] < ranges) == False)[0][-1]]
        else:  # handles UniformFloatHyperparameter & UniformIntegerHyperparameter
            # rescaling continuous values
            param_value = hyper.lower + (hyper.upper - hyper.lower) * vector[i]
            if type(hyper) == ConfigSpace.UniformIntegerHyperparameter:
                param_value = np.round(param_value).astype(int)   # converting to discrete (int)
        new_config[hyper.name] = param_value
    return new_config

def load_json(filename):
    with open(filename, 'r') as f:
        res = json.load(f)
    return res

def load_pkl(filename):
    with open(filename, 'rb') as f:
        res = pickle.load(f)
    return res

def process_history(res, cs=None, per_budget=False):
    assert all(i in res.keys() for i in ['runtime', 'regret_validation', 'history'])

    trajectory = np.array([np.array(l[0]) for l in res['history']])
    fitness = np.array([l[1] for l in res['history']])
    budgets = np.array([l[2] for l in res['history']])

    trajectory_cs = None
    if cs is not None and isinstance(cs, ConfigurationSpace):
        trajectory_cs = np.array([vector_to_configspace(cs, vec).get_array() for vec in trajectory])

    fidelities = None
    if per_budget:
        fidelities = {}
        for budget in np.unique(budgets):
            fidelities[budget] = np.where(budgets == budget)[0]

    history = {}
    history['trajectory'] = trajectory
    history['fitness'] = fitness
    history['budgets'] = budgets
    history['fidelities'] = fidelities
    history['trajectory_cs'] = trajectory_cs
    return history

def get_mds(X):
    embedding = MDS(n_components=2)
    return embedding.fit_transform(X)

def get_pca(X):
    pca = PCA(n_components=2)
    return pca.fit_transform(X)

def generate_colors(i, budgets=None):
    alphas = np.linspace(0.1, 1, i)
    rgba_colors = np.zeros((i, 4))
    rgba_colors[:, 3] = alphas
    if budgets is None:
        light, dark = color_pairs(2)
        rgba_colors[:, 0] = light[0]
        rgba_colors[:, 1] = light[1]
        rgba_colors[:, 2] = light[2]
        if i > 1:
            rgba_colors[i-1, 0] = dark[0]
            rgba_colors[i-1, 1] = dark[1]
            rgba_colors[i-1, 2] = dark[2]
        return rgba_colors
    budgets = budgets[:i]
    budget_set = np.unique(budgets)
    for j, b in enumerate(budget_set):
        idxs = np.where(budgets == b)[0]
        light, dark = color_pairs(j)
        rgba_colors[idxs, 0] = light[0]
        rgba_colors[idxs, 1] = light[1]
        rgba_colors[idxs, 2] = light[2]
    return rgba_colors

def color_pairs(i=0):
    colors = [((0.60784314, 0.54901961, 0.83137255),  #9B8CD4 -- BLUE BELL
               (0.18039216, 0.21960784, 0.18039216)), #2E382E -- JET
              ((0.45882353, 0.29803922, 0.16078431),  #754C29 -- DONKEY BROWN
               (0.15294118, 0.09019608, 0.05490196)), #27170E -- ZINNWALDITE BROWN
              ((0.83921569, 0.70196078, 0.02352941),  #D6B306 -- VIVID AMBER
               (0.83921569, 0.34901961, 0.00000000)), #D65900 -- TENNE
              ((0.56470588, 0.80784314, 0.45098039),  #90CE73 -- PISTACHIO
               (0.27843137, 0.60784314, 0.49803922)), #479B7F -- WINTERGREEN DREAM
              ((0.75294118, 0.37647059, 0.36078431),  #C0605C -- INDIAN RED
               (0.43137255, 0.01176471, 0.12941176))  #6E0321 -- BURGUNDY
    ]
    return colors[i]


class AnimateRun():
    def __init__(self, path, filename, per_budget=False, output_path=None):
        self.path = path
        self.filename = os.path.join(path, filename)
        self.cs = None
        self.per_budget = per_budget
        self.output_path = output_path
        if self.output_path is None:
            self.output_path = self.path

        if os.path.isfile(os.path.join(path, 'configspace.pkl')):
            self.cs = load_pkl(os.path.join(path, 'configspace.pkl'))
        self.res = load_json(self.filename)
        self.incumbent = self.res['regret_validation']
        self.runtimes = self.res['runtime']
        self.history = process_history(self.res, self.cs, self.per_budget)

    def build_config_frequency(self):
        '''Applicable only for ConfigSpace
        '''
        config_dict = {}
        for i, config in enumerate(self.history['trajectory_cs']):
            if str(config) not in config_dict:
                config_dict[str(config)] = [self.history['budgets'][i]]
            else:
                config_dict[str(config)].append(self.history['budgets'][i])
        return config_dict

    def plot_gif(self, filename=None, per_budget=False, delay=150, repeat=True, type='raw'):
        '''Plots a gif over the optimisation trajectory for a run
        '''
        if filename is not None:
            filename = os.path.join(self.output_path, filename)

        if type == "cs":
            X = get_mds(self.history['trajectory_cs'])
        else:
            X = get_mds(self.history['trajectory'])
        xlim = (np.min(X[:,0])-0.5, np.max(X[:,0])+0.5)
        ylim = (np.min(X[:,1])-0.5, np.max(X[:,1])+0.5)
        plt.clf()
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set(xlim=xlim, ylim=ylim, xlabel="$MDS-X$", ylabel="$MDS-Y$")
        scat = ax.scatter(X[0,0], X[0,1])

        if per_budget:
            budgets = self.history['budgets']
        else:
            budgets = None
        def animate(i):
            rgba_colors = generate_colors(i, budgets)
            scat.set_offsets(X[:i])
            scat.set_facecolor(rgba_colors[:i])
            ax.set_title('# function evaluations: {}'.format(i))

        anim = FuncAnimation(fig, animate, interval=delay, frames=range(X.shape[0]),
                             repeat=repeat, repeat_delay=5000)
        if filename is not None:
            anim.save('{}.gif'.format(filename), writer='imagemagick')
        else:
            plt.show()

    def plot_gif_compare(self, filename=None, per_budget=False, delay=150, repeat=True):
        '''Plots a gif over the optimisation trajectory for a run comparing parameter spaces
        '''
        if filename is not None:
            filename = os.path.join(self.output_path, filename)

        X = get_mds(self.history['trajectory'])
        X_cs = get_mds(self.history['trajectory_cs'])
        plt.clf()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        xlim = (np.min(X[:,0])-0.5, np.max(X[:,0])+0.5)
        ylim = (np.min(X[:,1])-0.5, np.max(X[:,1])+0.5)
        ax1.set(xlim=xlim, ylim=ylim)
        scat1 = ax1.scatter(X[0,0], X[0,1])
        xlim_cs = (np.min(X_cs[:,0])-0.5, np.max(X_cs[:,0])+0.5)
        ylim_cs = (np.min(X_cs[:,1])-0.5, np.max(X_cs[:,1])+0.5)
        ax2.set(xlim=xlim_cs, ylim=ylim_cs)
        scat2 = ax2.scatter(X_cs[0,0], X_cs[0,1])

        if per_budget:
            budgets = self.history['budgets']
        else:
            budgets = None
        def animate(i):
            rgba_colors = generate_colors(i, budgets)
            scat1.set_offsets(X[:i])
            scat1.set_facecolor(rgba_colors[:i])
            scat2.set_offsets(X_cs[:i])
            scat2.set_facecolor(rgba_colors[:i])
            ax1.set_title('[Uniform] # function evaluations: {}'.format(i))
            ax2.set_title('[ConfigSpace] # function evaluations: {}'.format(i))

        anim = FuncAnimation(fig, animate, interval=delay, frames=range(X.shape[0]),
                             repeat=repeat, repeat_delay=5000)
        if filename is not None:
            anim.save('{}.gif'.format(filename), writer='imagemagick')
        else:
            plt.show()

    def plot_budget_histogram(self, filename=None):
        '''Plots a histogram for the no. of function evaluations per budget in a run
        '''
        if filename is not None:
            filename = os.path.join(self.output_path, filename)
        plot_dict = {}
        for key in self.history['fidelities']:
            plot_dict[str(key)] = len(self.history['fidelities'][key])
        plt.clf()
        plt.bar(list(plot_dict.keys()), list(plot_dict.values()))
        plt.title("# function evaluations per budget")
        plt.ylabel("# of function evaluations")
        plt.xlabel("Budgets")
        if filename is not None:
            plt.savefig('{}.png'.format(filename), dpi=300)
        else:
            plt.show()

    def plot_final(self, filename=None, per_budget=False, type='raw'):
        '''Plots a scatter plot showing the optimisation trajectory for a run
        '''
        if filename is not None:
            filename = os.path.join(self.output_path, filename)
        if type == 'cs':
            X = get_mds(self.history['trajectory_cs'])
        else:
            X = get_mds(self.history['trajectory'])
        xlim = (np.min(X[:,0])-0.5, np.max(X[:,0])+0.5)
        ylim = (np.min(X[:,1])-0.5, np.max(X[:,1])+0.5)
        plt.clf()
        if per_budget:
            rgba_colors = generate_colors(X.shape[0], self.history['budgets'])
            for i, key in enumerate(self.history['fidelities'].keys()):
                budget = self.history['fidelities'][key]
                # reversing the arrays to get the high alpha value label in legend
                plt.scatter(X[budget,0][::-1], X[budget,1][::-1],
                            color=rgba_colors[budget][::-1], label=key)
            plt.legend()
        else:
            rgba_colors = generate_colors(X.shape[0], None)
            plt.scatter(X[:,0], X[:,1], color=rgba_colors)
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.xlabel("$MDS-X$")
        plt.ylabel("$MDS-Y$")
        plt.title("Trajectory of function evaluations")
        if filename is not None:
            plt.savefig('{}.png'.format(filename), dpi=300)
        else:
            plt.show()

    def plot_final_compare(self, filename=None, per_budget=False):
        '''Plots a scatter plot comparing parameter spaces based on their trajectories in a run
        '''
        if filename is not None:
            filename = os.path.join(self.output_path, filename)
        X_cs = get_mds(self.history['trajectory_cs'])
        xlim_cs = (np.min(X_cs[:,0])-0.5, np.max(X_cs[:,0])+0.5)
        ylim_cs = (np.min(X_cs[:,1])-0.5, np.max(X_cs[:,1])+0.5)
        X = get_mds(self.history['trajectory'])
        xlim = (np.min(X[:,0])-0.5, np.max(X[:,0])+0.5)
        ylim = (np.min(X[:,1])-0.5, np.max(X[:,1])+0.5)
        plt.clf()
        fig, (ax1, ax2) = plt.subplots(1, 2)
        if per_budget:
            rgba_colors = generate_colors(X.shape[0], self.history['budgets'])
            for key in self.history['fidelities'].keys():
                budget = self.history['fidelities'][key]
                # reversing the arrays to get the high alpha value label in legend
                ax1.scatter(X[budget,0][::-1], X[budget,1][::-1],
                            color=rgba_colors[budget][::-1], label=key)
                ax2.scatter(X_cs[budget,0][::-1], X_cs[budget,1][::-1],
                            color=rgba_colors[budget][::-1], label=key)
            ax1.legend()
            ax2.legend()
        else:
            rgba_colors = generate_colors(X.shape[0], None)
            ax1.scatter(X[:,0], X[:,1], color=rgba_colors)
            ax2.scatter(X_cs[:,0], X_cs[:,1], color=rgba_colors)
        ax1.set(xlim=xlim, ylim=ylim, xlabel="$MDS-X$", ylabel="$MDS-Y$")
        ax2.set(xlim=xlim_cs, ylim=ylim_cs, xlabel="$MDS-X$", ylabel="$MDS-Y$")
        plt.suptitle("Trajectory of function evaluations")
        ax1.set_title('Uniform parameter space')
        ax2.set_title('ConfigSpace parameter space')
        if filename is not None:
            plt.savefig('{}.png'.format(filename), dpi=300)
        else:
            plt.show()
