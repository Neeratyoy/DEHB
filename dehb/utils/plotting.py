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

    # creates a dataframe where the rows are indexed based on time
    # fills with NA for missing values for the respective timesteps
    merged = pd.DataFrame(frame_dict)
    # ffill() acts like a fillna() wherein a forward fill happens
    # only remaining NAs for in the beginning until a value is recorded
    merged = merged.ffill()

    performance = merged.get_values()  # converts to a 2D numpy array
    time_ = merged.index.values        # retrieves the timestamps

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
parser.add_argument('--type', default="wallclock", type=str, choices=["wallclock", "fevals"], help='to plot for wallclock times or # function evaluations')
parser.add_argument('--name', default="comparison", type=str, help='file name for the PNG plot to be saved')
parser.add_argument('--title', default="benchmark", type=str, help='title name for the plot')

args = parser.parse_args()
path = args.path
n_runs = args.n_runs
plot_type = args.type
plot_name = args.name

# methods = [("bohb", "BOHB"),
#         ("re", "RE"),
#         ("hyperband", "HB"),
#         ("random_search", "RS"),
#         ("tpe", "TPE"),
methods = [("de", "DE"),
           ("dehb_v1", "DEHB V1; $gen=1$;"),
           ("dehb_v2", "DEHB V2; $gen=1$; $rand=0.3$;"),
           ("dehb_v3", "DEHB V3; $gen=1$; $rand=0$;"),
           ("dehb_v3_pop10", "DEHB V3 $gen=1$; $rand=0$; $pop=10$"),
           ("dehb_v3_rand0.3", "DEHB V3; $gen=1$; $rand=0.3$;"),
           ("dehb_v3_pop10_rand0.3", "DEHB V3 $gen=1$; $rand=0.3$; $pop=10$")]
        # ("dehb_v2_rand", "DEHB V2 $rand=0.2$"),
        # ("dehb_v2_rand_0.9", "DEHB V2 $rand=0.9$")]

# plot limits
min_time = np.inf
max_time = 0
min_regret = 1
max_regret = 0

# plot setup
colors = ["C%d" % i for i in range(len(methods))]
plt.clf()

# looping and plotting for all methods
for index, (m, label) in enumerate(methods):
    regret = []
    runtimes = []
    for k, i in enumerate(np.arange(n_runs)):
        try:
            res = json.load(open(os.path.join(path, m, "run_%d.json" % i)))
            no_runs_found = False
        except Exception as e:
            print(m, i, e)
            no_runs_found = True
            continue
        _, idx = np.unique(res['regret_validation'], return_index=True)
        idx.sort()
        regret.append(np.array(res['regret_validation'])[idx])
        runtimes.append(np.array(res["runtime"])[idx])

    if not no_runs_found:
        # finds the latest time where the first measurement was made across runs
        t = np.max([runtimes[i][0] for i in range(len(runtimes))])
        min_time = min(min_time, t)
        te, time = fill_trajectory(regret, runtimes, replace_nan=1)

        idx = time.tolist().index(t)
        te = te[idx:, :]
        time = time[idx:]

        # Clips off all measurements after 10^7s
        idx = np.where(time < 1e7)[0]

        print("{}. Plotting for {}".format(index, m))
        # The mean plot
        plt.plot(time[idx], np.mean(te, axis=1)[idx], color=colors[index],
                 linewidth=4, label=label, linestyle=linestyles[index % len(linestyles)],
                 marker=marker[index % len(marker)], markevery=(0.1,0.1), markersize=15)
        # The error band
        plt.fill_between(time[idx],
                         np.mean(te, axis=1)[idx] + 2 * stats.sem(te[idx], axis=1),
                         np.mean(te[idx], axis=1)[idx] - 2 * stats.sem(te[idx], axis=1),
                         color="C%d" % index, alpha=0.2)

        # Stats to dynamically impose limits on the axes of the plots
        max_time = max(max_time, time[idx][-1])
        min_regret = min(min_regret, np.mean(te, axis=1)[idx][-1])
        max_regret = max(max_regret, np.mean(te, axis=1)[idx][0])

plt.xscale("log")
plt.yscale("log")
plt.tick_params(which='both', direction="in")
plt.legend(loc='upper right', framealpha=1, prop={'size': 15})
plt.title(args.title)
if plot_type == "wallclock":
    plt.xlabel("wall clock time $[s]$", fontsize=50)
elif plot_type == "fevals":
    plt.xlabel("number of function evaluations", fontsize=50)
plt.ylabel("regret", fontsize=50)
plt.xlim(max(min_time/10, 1e0), min(max_time*10, 1e7))
plt.ylim(min_regret, max_regret)
plt.grid(which='both', alpha=0.5, linewidth=0.5)
print(os.path.join(args.output_path, '{}.png'.format(plot_name)))
plt.savefig(os.path.join(args.output_path, '{}.png'.format(plot_name)), bbox_inches='tight', dpi=300)
