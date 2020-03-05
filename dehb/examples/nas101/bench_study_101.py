import os
import sys
sys.path.append(os.path.join(os.getcwd(), '../nas_benchmarks/'))
sys.path.append(os.path.join(os.getcwd(), '../nas_benchmarks-development/'))

import argparse
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import spearmanr as corr

from tabular_benchmarks import FCNetProteinStructureBenchmark, FCNetSliceLocalizationBenchmark,\
    FCNetNavalPropulsionBenchmark, FCNetParkinsonsTelemonitoringBenchmark
from tabular_benchmarks import NASCifar10A, NASCifar10B, NASCifar10C


################
# Common Utils #
################

def final_score_relation(sample_size=1e6, output=None):
    global b, cs
    x = []
    y = []
    for i in range(int(sample_size)):
        print("{:<6}/{:<6}".format(i+1, sample_size), end='\r')
        config = cs.sample_configuration()
        valid_score, _ = b.objective_function(config)
        test_score, _ = b.objective_function_test(config)
        x.append(valid_score)
        y.append(test_score)
    xlim = (min(x), max(x))
    ylim = (min(y), max(y))
    plt.clf()
    plt.plot([xlim[0], xlim[1]], [ylim[0], ylim[1]], color='black',
             alpha=0.5, linestyle='--', linewidth=1)
    plt.scatter(x, y)
    plt.xlabel('Validation scores')
    plt.ylabel('Test scores')
    if output is None:
        plt.show()
    else:
        plt.savefig(output, dpi=300)


def budget_correlation(sample_size, budgets, compare=False, output=None):
    global b, cs
    df = pd.DataFrame(columns=budgets, index=np.arange(sample_size))
    for i in range(int(sample_size)):
        print("{:<6}/{:<6}".format(i+1, sample_size), end='\r')
        config = cs.sample_configuration()
        for j, budget in enumerate(budgets):
            score, _ = b.objective_function(config, budget=budget)
            df.iloc[i, j] = score
    res = corr(df)
    corr_val = res.correlation

    plt.clf()
    ax = plt.gca()
    mat = ax.matshow(corr_val)
    for i in range(len(corr_val)):
        for j in range(len(corr_val[0])):
            ax.text(j, i, "{:0.5f}".format(corr_val[i][j]), ha="center", va="center", rotation=45)
    # Major ticks
    ax.set_xticks(np.arange(0, len(budgets), 1))
    ax.set_yticks(np.arange(0, len(budgets), 1))
    # Labels for major ticks
    ax.set_xticklabels(np.arange(1, len(budgets)+1, 1))
    ax.set_yticklabels(np.arange(1, len(budgets)+1, 1))
    # Minor ticks
    ax.set_xticks(np.arange(-.5, len(budgets), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(budgets), 1), minor=True)
    # Gridlines based on minor ticks
    ax.grid(which='minor', color='w', linestyle='-', linewidth=2)
    if compare:
        mat.set_clim(0, 1)
    else:
        mat.set_clim(np.min(corr_val), np.max(corr_val))
    plt.colorbar(mat)
    if output is None:
        plt.show()
    else:
        plt.savefig(output, dpi=300)


parser = argparse.ArgumentParser()
parser.add_argument('--benchmark', default='nas_cifar10a', type=str, nargs='?',
                    help='Choice of benchmark')
parser.add_argument('--sample_size', default=1e4, type=float, nargs='?',
                    help='# samples')
parser.add_argument('--compare', default='False', type=str, nargs='?',
                    help='If True, color limits set to [-1, 1], else dynamic')
args = parser.parse_args()
benchmark = args.benchmark
sample_size = args.sample_size
compare = True if args.compare == 'True' else False

names = {
    'nas_cifar10a': 'cifara',
    'nas_cifar10b': 'cifarb',
    'nas_cifar10c': 'cifarc',
    'protein_structure': 'protein',
    'naval_propulsion': 'naval',
    'slice_localization': 'slice',
    'parkinsons_telemonitoring': 'parkinsons'
}

#################################
# NAS-Bench-101 + NAS-HPO-Bench #
#################################

data_dir = "../nas_benchmarks-development/tabular_benchmarks/fcnet_tabular_benchmarks/"

def get_ready_101(benchmark):
    global b, cs, dimensions, budgets

    if benchmark == "nas_cifar10a":  # NAS-Bench-101
        budgets = [4, 12, 36, 108]
        b = NASCifar10A(data_dir=data_dir, multi_fidelity=True)

    elif benchmark == "nas_cifar10b":  # NAS-Bench-101
        budgets = [4, 12, 36, 108]
        b = NASCifar10B(data_dir=data_dir, multi_fidelity=True)

    elif benchmark == "nas_cifar10c":  # NAS-Bench-101
        budgets = [4, 12, 36, 108]
        b = NASCifar10C(data_dir=data_dir, multi_fidelity=True)

    elif benchmark == "protein_structure":  # NAS-HPO-Bench
        budgets = [3, 11, 33, 100]
        b = FCNetProteinStructureBenchmark(data_dir=data_dir)

    elif benchmark == "slice_localization":  # NAS-HPO-Bench
        budgets = [3, 11, 33, 100]
        b = FCNetSliceLocalizationBenchmark(data_dir=data_dir)

    elif benchmark == "naval_propulsion":  # NAS-HPO-Bench
        budgets = [3, 11, 33, 100]
        b = FCNetNavalPropulsionBenchmark(data_dir=data_dir)

    elif benchmark == "parkinsons_telemonitoring":  # NAS-HPO-Bench
        budgets = [3, 11, 33, 100]
        b = FCNetParkinsonsTelemonitoringBenchmark(data_dir=data_dir)

    cs = b.get_configuration_space()
    dimensions = len(cs.get_hyperparameters())


name = names[benchmark]
get_ready_101(benchmark)

if 'nas' not in benchmark:
    final_score_relation(sample_size,
                         output='dehb/examples/plots/correlation/{}_test_val.png'.format(name))
budget_correlation(sample_size, budgets=budgets, compare=compare,
                   output='dehb/examples/plots/correlation/{}_{}.png'.format(name, compare))
