import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import spearmanr as corr

from hpolib.benchmarks.synthetic_functions.counting_ones import CountingOnes


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
        valid_score = b.objective_function(config)['function_value']
        test_score = b.objective_function_test(config)['function_value']
        x.append(valid_score)
        y.append(test_score)
    xlim = (min(x), max(x))
    ylim = (min(y), max(y))
    plt.clf()
    plt.plot([xlim[0], xlim[1]], [ylim[0], ylim[1]], color='black',
             alpha=0.5, linestyle='--', linewidth=1)
    plt.scatter(x, y, s=5)
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
            score = b.objective_function(config, budget=budget)['function_value']
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


#################
# Counting Ones #
#################

def get_ready_countingones(n):
    global b, cs, dimensions, min_budget, max_budget
    b = CountingOnes()
    cs = b.get_configuration_space(n_continuous=n, n_categorical=n)
    dimensions = len(cs.get_hyperparameters())
    min_budget = 576 / dimensions
    max_budget = 93312 / dimensions

# 4+4
get_ready_countingones(4)
budgets = [144.,   432.,  1296.,  3888., 11664.]
final_score_relation(sample_size=10000, output='dehb/examples/plots/correlation/4+4_test_val.png')
budget_correlation(sample_size=10000, budgets=budgets, compare=True,
                   output='dehb/examples/plots/correlation/4+4_true.png')
budget_correlation(sample_size=10000, budgets=budgets, compare=False,
                   output='dehb/examples/plots/correlation/4+4_false.png')

# 8+8
get_ready_countingones(8)
budgets = [72.,  216.,  648., 1944., 5832.]
final_score_relation(sample_size=10000, output='dehb/examples/plots/correlation/8+8_test_val.png')
budget_correlation(sample_size=10000, budgets=budgets, compare=True,
                   output='dehb/examples/plots/correlation/8+8_true.png')
budget_correlation(sample_size=10000, budgets=budgets, compare=False,
                   output='dehb/examples/plots/correlation/8+8_false.png')

# 16+16
get_ready_countingones(16)
budgets = [36.,  108.,  324.,  972., 2916.]
final_score_relation(sample_size=10000, output='dehb/examples/plots/correlation/16+16_test_val.png')
budget_correlation(sample_size=10000, budgets=budgets, compare=True,
                   output='dehb/examples/plots/correlation/16+16_true.png')
budget_correlation(sample_size=10000, budgets=budgets, compare=False,
                   output='dehb/examples/plots/correlation/16+16_false.png')

# 32+32
get_ready_countingones(32)
budgets = [18.,   54.,  162.,  486., 1458.]
final_score_relation(sample_size=10000, output='dehb/examples/plots/correlation/32+32_test_val.png')
budget_correlation(sample_size=10000, budgets=budgets, compare=True,
                   output='dehb/examples/plots/correlation/32+32_true.png')
budget_correlation(sample_size=10000, budgets=budgets, compare=False,
                   output='dehb/examples/plots/correlation/32+32_false.png')
