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
    if i > 1:
        rgba_colors[i-1, 0] = 1
        rgba_colors[i-1, 1] = 1
        rgba_colors[i-1, 2] = 1
    return rgba_colors

def plot_gif(X, delay=100, repeat=True, filename=None, xlim=None, ylim=None, budgets=None):
    plt.clf()
    fig, ax = plt.subplots(figsize=(5, 5))
    if xlim is None:
        xlim = (np.min(X[:,0])-1, np.max(X[:,0])+1)
    if ylim is None:
        ylim = (np.min(X[:,1])-1, np.max(X[:,1])+1)
    ax.set(xlim=xlim, ylim=ylim)
    scat = ax.scatter(X[0,0], X[0,1])

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

def color_pairs(i=0):
    colors = [((0.60784314, 0.54901961, 0.83137255), (0.18039216, 0.21960784, 0.18039216)), # (9B8CD4, 2E382E) -- (BLUE BELL, JET)
              ((0.45882353, 0.29803922, 0.16078431), (0.15294118, 0.09019608, 0.05490196)), # (754C29, 27170E) -- (DONKEY BROWN, ZINNWALDITE BROWN)
              ((0.83921569, 0.70196078, 0.02352941), (0.83921569, 0.34901961, 0.00000000)), # (D6B306, D65900) -- (VIVID AMBER, TENNE)
              ((0.56470588, 0.80784314, 0.45098039), (0.27843137, 0.60784314, 0.49803922)), # (90CE73, 479B7F) -- (PISTACHIO, WINTERGREEN DREAM)
              ((0.75294118, 0.37647059, 0.36078431), (0.43137255, 0.01176471, 0.12941176))  # (C0605C, 6E0321) -- (INDIAN RED, BURGUNDY)
    ]
    return colors[i]


class AnimateRun():
    def __init__(self, path, filename, per_budget=False, output_path=None):
        self.path = path
        self.filename = os.path.join(path, filename)
        self.cs = None
        self.per_budget = per_budget
        self.output_path = output_path

        if os.path.isfile(os.path.join(path, 'configspace.pkl')):
            self.cs = load_pkl(os.path.join(path, 'configspace.pkl'))
        self.res = load_json(self.filename)
        self.incumbent = self.res['regret_validation']
        self.runtimes = self.res['runtime']
        self.history = process_history(self.res, self.cs, self.per_budget)

    def plot_gif(self, delay=100, repeat=True, colors=2, filename=None, type='raw'):
        if filename is not None:
            if self.output_path is None:
                filename = os.path.join(self.path, filename)
            else:
                filename = os.path.join(self.output_path, filename)
            filename = "{}_{}".format(filename, type)
        if type == 'cs':
            X = get_mds(self.history['trajectory'])
        else:
            X = get_mds(self.history['trajectory_cs'])
        plot_gif(X, delay, repeat, filename, colors)

    def plot_each_budget_gif(self, filename, delay=100, repeat=True, type='raw'):
        filename = "{}_{}".format(filename, type)
        if self.output_path is None:
            filename = os.path.join(self.path, filename)
        else:
            filename = os.path.join(self.output_path, filename)
        if type == 'cs':
            X = get_mds(self.history['trajectory'])
        else:
            X = get_mds(self.history['trajectory_cs'])
        xlim = (np.min(X[:,0])-1, np.max(X[:,0])+1)
        ylim = (np.min(X[:,1])-1, np.max(X[:,1])+1)
        length = len(run.history['fidelities'].keys())
        for i, budget in enumerate(run.history['fidelities'].keys()):
            X_budget = X[run.history['fidelities'][budget]]
            plot_gif(X_budget, delay, repeat, "{}_{}".format(filename, budget), xlim, ylim)

    def plot_all_budget_gif(self, filename, delay=100, repeat=True, type='raw'):
        filename = "{}_{}".format(filename, type)
        if self.output_path is None:
            filename = os.path.join(self.path, filename)
        else:
            filename = os.path.join(self.output_path, filename)
        if type == 'cs':
            X = get_mds(self.history['trajectory'])
        else:
            X = get_mds(self.history['trajectory_cs'])
        xlim = (np.min(X[:,0])-1, np.max(X[:,0])+1)
        ylim = (np.min(X[:,1])-1, np.max(X[:,1])+1)
        plot_gif(X=X, delay=delay, repeat=repeat, filename=filename, budgets=self.history['budgets'])
