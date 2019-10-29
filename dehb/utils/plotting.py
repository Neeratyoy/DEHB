import os
import json
import sys
import argparse
import collections
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import torch
from scipy import stats
seaborn.set_style("ticks")

from matplotlib import rcParams
rcParams["font.size"] = "30"
rcParams['text.usetex'] = True
rcParams['font.family'] = 'serif'
rcParams['figure.figsize'] = (16.0, 9.0)
rcParams['figure.frameon'] = True
rcParams['figure.edgecolor'] = 'k'
rcParams['grid.color'] = 'k'
rcParams['grid.linestyle'] = ':'
rcParams['grid.linewidth'] = 0.5
rcParams['axes.linewidth'] = 1
rcParams['axes.edgecolor'] = 'k'
rcParams['axes.grid.which'] = 'both'
rcParams['legend.frameon'] = 'True'
rcParams['legend.framealpha'] = 1

rcParams['ytick.major.size'] = 12
rcParams['ytick.major.width'] = 1.5
rcParams['ytick.minor.size'] = 6
rcParams['ytick.minor.width'] = 1
rcParams['xtick.major.size'] = 12
rcParams['xtick.major.width'] = 1.5
rcParams['xtick.minor.size'] = 6
rcParams['xtick.minor.width'] = 1

marker=['x', '^', 'D', 'o', 's', 'h', '*', 'v', '<', ">"]
linestyles = ['-', '--', '-.', ':']



def fill_trajectory(performance_list, time_list, replace_nan=np.NaN):
    frame_dict = collections.OrderedDict()
    counter = np.arange(0, len(performance_list))
    for p, t, c in zip(performance_list, time_list, counter):
        if len(p) != len(t):
            raise ValueError("(%d) Array length mismatch: %d != %d" %
                             (c, len(p), len(t)))
        frame_dict[str(c)] = pd.Series(data=p, index=t)

    merged = pd.DataFrame(frame_dict)
    merged = merged.ffill()


    performance = merged.get_values()
    time_ = merged.index.values

    performance[np.isnan(performance)] = replace_nan

    if not np.isfinite(performance).all():
        raise ValueError("\nCould not merge lists, because \n"
                         "\t(a) one list is empty?\n"
                         "\t(b) the lists do not start with the same times and"
                         " replace_nan is not set?\n"
                         "\t(c) any other reason.")

    return performance, time_


parser = argparse.ArgumentParser()
parser.add_argument('--path', default='./', type=str, nargs='?', help='path to encodings or jsons for each algorithm')
parser.add_argument('--n_runs', default=10, type=int, nargs='?', help='number of runs to plot data for')
parser.add_argument('--output_path', default="./", type=str, nargs='?',
                    help='specifies the path where the plot will be saved')
parser.add_argument('--type', default="test", type=str, choices=["test", "validation"], help='to plot test/validation regret')
parser.add_argument('--name', default="comparison", type=str, help='file name for the PNG plot to be saved')

args = parser.parse_args()
print(args)

# path = "/home/kleinaa/experiments/nas_benchmark/nas_cifar10/comparison_nas_cifar10a/"
path = args.path # "/home/kleinaa/experiments/nas_benchmark/final_nasbench101/encoding_cs_a/"

n_runs = args.n_runs

# CIFAR A
# methods = ["bohb/final", "re/final", "hyperband", "random_search", "tpe", "de_gen1_cifarc", "dehb_evolve_0"]
# labels = ["BOHB", "RE", "HB", "RS", "TPE", "DE", "DEHB"]

# methods = ["bohb/final", "re/final", "de_gen1_cifarc", "dehb_final_old", "dehb_final_evolve", "dehb_final_dehbde_0.25", "dehb_final_dehbde_0.1"]
# labels = ["BOHB", "RE", "DE", "DEHB Old", "DEHB Evolve", "DEHB-DE 25\%", "DEHB-DE 10\%"]

# CIFAR C
# methods = ["bohb/hyperopt/min_bandwidth_0.6/bohb", "regularized_evolution", "hyperband",  "random_search", "tpe", "de_gen1_cifarc", "dehb_evolve_0"]
# labels = ["BOHB", "RE", "HB", "RS", "TPE", "DE", "DEHB"] #  "DE", "DEHB (dynamic)", "DEHB (fixed)"]

# methods = ["bohb/hyperopt/min_bandwidth_0.6/bohb", "regularized_evolution", "de_gen1_cifarc", "dehb_final_old",
#            "dehb_final_evolve", "dehb_final_dehbde_0.25", "dehb_final_dehbde_0.1"]
# labels = ["BOHB", "RE", "DE", "DEHB Old", "DEHB Evolve", "DEHB-DE 25\%", "DEHB-DE 10\%"]

# Protein
methods = ["bohb", "regularized_evolution", "hyperband", "random_search", "tpe", "de", "dehb_final_old",
           "dehb_final_evolve", "dehb_final_dehbde_0.25", "dehb_final_dehbde_0.1"]
labels = ["BOHB", "RE", "HB", "RS", "TPE", "DE", "DEHB Old", "DEHB Evolve", "DEHB-DE 25\%", "DEHB-DE 10\%"]

methods = ["de", "dehb_final_dehbde_0.25", "dehb_final_dehbde_0.1", "dehb_final_dehbde_0"]
labels = ["DE", "DEHB-DE 25\%", "DEHB-DE 10\%", "DEHB-DE 0\%"]

# Slice
# methods = ["bohb", "regularized_evolution", "hyperband", "random_search", "tpe", "de_200r_100i", "dehb_final_old",
#            "dehb_final_evolve", "dehb_final_dehbde_0.25", "dehb_final_dehbde_0.1"]
# labels = ["BOHB", "RE", "HB", "RS", "TPE", "DE", "DEHB Old", "DEHB Evolve", "DEHB-DE 25\%", "DEHB-DE 10\%"]
#
# # Naval
# methods = ["bohb", "regularized_evolution", "hyperband", "random_search", "tpe", "de", "dehb_final_old",
#            "dehb_final_evolve", "dehb_final_dehbde_0.25", "dehb_final_dehbde_0.1"]
# labels = ["BOHB", "RE", "HB", "RS", "TPE", "DE", "DEHB Old", "DEHB Evolve", "DEHB-DE 25\%", "DEHB-DE 10\%"]
#
# # Parkinsons
# methods = ["bohb", "regularized_evolution", "hyperband", "random_search", "tpe", "de", "dehb_final_old",
#            "dehb_final_dehbde_0.25", "dehb_final_dehbde_0.1"]
# labels = ["BOHB", "RE", "HB", "RS", "TPE", "DE", "DEHB Old", "DEHB-DE 25\%", "DEHB-DE 10\%"]

colors = ["C%d" % i for i in range(len(methods))]
plot_type = "regret_test" if args.type == 'test' else "regret_validation"
plot_y_axis = "test regret" if args.type == 'test' else "validation regret"
plot_name = args.name

plt.clf()
for index, m in enumerate(methods):

    regret = []
    runtimes = []
    for k, i in enumerate(np.arange(n_runs)):
        try:
            res = json.load(open(os.path.join(path, m, "run_%d.json" % i)))
        except Exception as e:
            print(m, i, e)
            continue
        _, idx = np.unique(res[plot_type], return_index=True)
        idx.sort()

        regret.append(np.array(res[plot_type])[idx])
        runtimes.append(np.array(res["runtime"])[idx])

    t = np.max([runtimes[i][0] for i in range(len(runtimes))])
    te, time = fill_trajectory(regret, runtimes, replace_nan=1)

    idx = time.tolist().index(t)
    te = te[idx:, :]
    time = time[idx:]

    idx = np.where(time < 1e7)[0]
    print("{}. Plotting for {}".format(index, m))
    plt.plot(time[idx], np.mean(te.T, axis=0)[idx], color=colors[index],
             linewidth=4, label=labels[index], linestyle=linestyles[index % len(linestyles)],
             marker=marker[index % len(marker)], markevery=(0.1,0.1), markersize=15)

plt.xscale("log")
plt.yscale("log")
# plt.tick_params(axis='x', which='minor')
plt.tick_params(which='both', direction="in")
# plt.legend(loc=0, fontsize=35, ncol=1)
plt.legend(loc='upper right', framealpha=1, prop={'size': 15})
# plt.title("NAS-CIFAR10", fontsize=50)
# plt.xlabel("estimated wall-clock time (seconds)", fontsize=50)
# plt.ylabel(plot_y_axis, fontsize=50)
plt.xlabel("wall clock time $[s]$", fontsize=50)
plt.ylabel("regret", fontsize=50)
# plt.xlim(1e1, 2e7)
plt.xlim(1e0, 2e7)  # naval, parkinsons
# plt.ylim(1e-5, 1-1)  # slice
plt.ylim(1e-6, 1-1)  # naval, parkinsons
# plt.ylim(1e-3, 1-1)
plt.grid(which='both', alpha=0.5, linewidth=0.5)
print(os.path.join(args.output_path, '{}.png'.format(plot_name)))
plt.savefig(os.path.join(args.output_path, '{}.png'.format(plot_name)), bbox_inches='tight', dpi=300)
