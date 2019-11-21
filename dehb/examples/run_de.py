import sys
sys.path.append('../')

import os
import json
import pickle
import argparse
import numpy as np

from tabular_benchmarks import FCNetProteinStructureBenchmark, FCNetSliceLocalizationBenchmark,\
    FCNetNavalPropulsionBenchmark, FCNetParkinsonsTelemonitoringBenchmark
from tabular_benchmarks import NASCifar10A, NASCifar10B, NASCifar10C

from hpolib.benchmarks.surrogates.svm import SurrogateSVM as surrogate
from hpolib.benchmarks.synthetic_functions.counting_ones import CountingOnes

from optimizers import DE


def remove_invalid_configs(traj, runtime, history):
    idx = np.where(np.array(runtime)==0)
    runtime = np.delete(runtime, idx)
    traj = np.delete(np.array(traj), idx)
    history = np.delete(history, idx)
    return traj, runtime, history

def save(trajectory, runtime, history, path, run_id, filename="run"):
    global y_star_valid, y_star_test, inc_config
    res = {}
    res["runtime"] = np.cumsum(runtime).tolist()
    if np.max(traj) < 0:
        a_min = -np.inf
        a_max = 0
    else:
        a_min = 0
        a_max = np.inf
    res["regret_validation"] = np.array(np.clip(traj - y_star_valid,
                                                a_min=a_min, a_max=a_max)).tolist()
    res["history"] = history.tolist()
    res['y_star_valid'] = float(y_star_valid)
    res['y_star_test'] = float(y_star_test)
    # if inc_config is not None:
    #     inc_config = inc_config.tolist()
    res['inc_config'] = inc_config
    fh = open(os.path.join(output_path, '{}_{}.json'.format(filename, run_id)), 'w')
    json.dump(res, fh)
    fh.close()

def save_configspace(cs, path, filename='configspace'):
    fh = open(os.path.join(output_path, '{}.pkl'.format(filename)), 'wb')
    pickle.dump(cs, fh)
    fh.close()

def f(config, budget=None):
    if budget is not None:
        fitness, cost = b.objective_function(config, budget=int(budget))
    else:
        fitness, cost = b.objective_function(config)
    return fitness, cost

parser = argparse.ArgumentParser()
parser.add_argument('--run_id', default=0, type=int, nargs='?', help='unique number to identify this run')
parser.add_argument('--runs', default=None, type=int, nargs='?', help='number of runs to perform')
parser.add_argument('--run_start', default=0, type=int, nargs='?', help='run index to start with for multiple runs')
choices = ["protein_structure", "slice_localization", "naval_propulsion",
           "parkinsons_telemonitoring", "nas_cifar10a", "nas_cifar10b",
           "nas_cifar10c", "counting_*_*", "svm"]
parser.add_argument('--benchmark', default="protein_structure", help="specify the benchmark to run on from among {}".format(choices), type=str)
parser.add_argument('--gens', default=100, type=int, nargs='?', help='number of generations for DE to evolve')
parser.add_argument('--output_path', default="./", type=str, nargs='?',
                    help='specifies the path where the results will be saved')
parser.add_argument('--data_dir', default="../tabular_benchmarks/fcnet_tabular_benchmarks/", type=str, nargs='?',
                    help='specifies the path to the tabular data')
parser.add_argument('--pop_size', default=10, type=int, nargs='?', help='population size')
parser.add_argument('--strategy', default="rand1_bin", type=str, nargs='?', help='type of mutation & crossover scheme')
parser.add_argument('--mutation_factor', default=0.5, type=float, nargs='?', help='mutation factor value')
parser.add_argument('--crossover_prob', default=0.5, type=float, nargs='?', help='probability of crossover')
parser.add_argument('--max_budget', default=None, type=str, nargs='?', help='maximum wallclock time to run DE for')
parser.add_argument('--verbose', default='False', choices=['True', 'False'], nargs='?', help='to print progress or not')
parser.add_argument('--folder', default='de', type=str, nargs='?', help='name of folder where files will be dumped')

args = parser.parse_args()
args.verbose = True if args.verbose == 'True' else False

benchmark_type = "nasbench"

if args.benchmark == "nas_cifar10a":
    min_budget = 4
    max_budget = 108
    b = NASCifar10A(data_dir=args.data_dir, multi_fidelity=True)
    y_star_valid = b.y_star_valid
    y_star_test = b.y_star_test
    inc_config = None

elif args.benchmark == "nas_cifar10b":
    min_budget = 4
    max_budget = 108
    b = NASCifar10B(data_dir=args.data_dir, multi_fidelity=True)
    y_star_valid = b.y_star_valid
    y_star_test = b.y_star_test
    inc_config = None

elif args.benchmark == "nas_cifar10c":
    min_budget = 4
    max_budget = 108
    b = NASCifar10C(data_dir=args.data_dir, multi_fidelity=True)
    y_star_valid = b.y_star_valid
    y_star_test = b.y_star_test
    inc_config = None

elif args.benchmark == "protein_structure":
    min_budget = 4
    max_budget = 100
    b = FCNetProteinStructureBenchmark(data_dir=args.data_dir)
    inc_config, y_star_valid, y_star_test = b.get_best_configuration()

elif args.benchmark == "slice_localization":
    min_budget = 4
    max_budget = 100
    b = FCNetSliceLocalizationBenchmark(data_dir=args.data_dir)
    inc_config, y_star_valid, y_star_test = b.get_best_configuration()

elif args.benchmark == "naval_propulsion":
    min_budget = 4
    max_budget = 100
    b = FCNetNavalPropulsionBenchmark(data_dir=args.data_dir)
    inc_config, y_star_valid, y_star_test = b.get_best_configuration()

elif args.benchmark == "parkinsons_telemonitoring":
    min_budget = 4
    max_budget = 100
    b = FCNetParkinsonsTelemonitoringBenchmark(data_dir=args.data_dir)
    inc_config, y_star_valid, y_star_test = b.get_best_configuration()

elif "counting" in args.benchmark:
    assert len(args.benchmark.split('_')) == 3
    benchmark_type = "countingones"
    n_categorical = int(args.benchmark.split('_')[-2])
    n_continuous = int(args.benchmark.split('_')[-1])
    b = CountingOnes()
    min_budget = 9
    max_budget = 729
    inc_config, y_star_valid, y_star_test = (None, 0, 0)
    def f(config, budget=None):
        if budget is not None:
            res = b.objective_function(config, budget=budget)
            fitness = res["function_value"]
            cost = 1
        else:
            res = b.objective_function(config)
            fitness = res["function_value"]
            cost = 1
        return fitness, cost

elif "svm" in args.benchmark:
    benchmark_type = "svm"
    min_budget = 1 / 512
    max_budget = 1
    b = surrogate(path=None)
    inc_config, y_star_valid, y_star_test = (None, 0, 0)
    def f(config, budget=None):
        if budget is not None:
            res = b.objective_function(config, dataset_fraction=budget)
            fitness = res["function_value"]
            cost = res["cost"]
        else:
            res = b.objective_function(config)
            fitness = res["function_value"]
            cost = res["cost"]
        return fitness, cost

if "counting" in args.benchmark:
    cs = CountingOnes.get_configuration_space(n_categorical=n_categorical,
                                              n_continuous=n_continuous)
    args.output_path = os.path.join(args.output_path, "{}_{}".format(n_categorical, n_continuous))
else:
    cs = b.get_configuration_space()
dimensions = len(cs.get_hyperparameters())

output_path = os.path.join(args.output_path, args.folder)
os.makedirs(output_path, exist_ok=True)

de = DE(cs=cs, dimensions=dimensions, f=f, pop_size=args.pop_size,
        mutation_factor=args.mutation_factor, crossover_prob=args.crossover_prob,
        strategy=args.strategy, max_budget=args.max_budget)

if args.runs is None:
    traj, runtime, history = de.run(generations=args.gens, verbose=args.verbose)
    if 'cifar' in args.benchmark:
        save(traj, runtime, history, output_path, args.run_id, filename="raw_run")
        traj, runtime, history = remove_invalid_configs(traj, runtime, history)
    save(traj, runtime, history, output_path, args.run_id)
else:
    for run_id, _ in enumerate(range(args.runs), start=args.run_start):
        if args.verbose:
            print("\nRun #{:<3}\n{}".format(run_id + 1, '-' * 8))
        traj, runtime, history = de.run(generations=args.gens, verbose=args.verbose)
        if 'cifar' in args.benchmark:
            save(traj, runtime, history, output_path, run_id, filename="raw_run")
            traj, runtime, history = remove_invalid_configs(traj, runtime, history)
        save(traj, runtime, history, output_path, run_id)
        print("Run saved. Resetting...")
        de.reset()
        if benchmark_type == "nasbench":
            b.reset_tracker()

save_configspace(cs, output_path)
