import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import spearmanr as corr

from hpolib.benchmarks.surrogates.svm import SurrogateSVM as surrogate


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


#######
# SVM #
#######

b = surrogate()
cs = b.get_configuration_space()
dimensions = len(cs.get_hyperparameters())
budgets = [0.00411523, 0.01234568, 0.03703704, 0.11111111, 0.33333333, 1.]
final_score_relation(sample_size=10000, output='dehb/examples/plots/correlation/svm_test_val.png')
budget_correlation(sample_size=10000, budgets=budgets, compare=True,
                   output='dehb/examples/plots/correlation/svm_true.png')
budget_correlation(sample_size=10000, budgets=budgets, compare=False,
                   output='dehb/examples/plots/correlation/svm_false.png')
