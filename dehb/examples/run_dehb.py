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
from hpolib.benchmarks.surrogates.paramnet import SurrogateReducedParamNetTime


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
parser.add_argument('--fix_seed', default='False', type=str, choices=['True', 'False'], nargs='?', help='seed')
parser.add_argument('--run_id', default=0, type=int, nargs='?', help='unique number to identify this run')
parser.add_argument('--runs', default=None, type=int, nargs='?', help='number of runs to perform')
parser.add_argument('--run_start', default=0, type=int, nargs='?', help='run index to start with for multiple runs')
choices = ["protein_structure", "slice_localization", "naval_propulsion",
           "parkinsons_telemonitoring", "nas_cifar10a", "nas_cifar10b",
           "nas_cifar10c", "counting_*_*", "svm", "paramnet_*"]
parser.add_argument('--benchmark', default="protein_structure", help="specify the benchmark to run on from among {}".format(choices), type=str)
parser.add_argument('--n_iters', default=100, type=int, nargs='?', help='number of iterations for optimization method')
parser.add_argument('--output_path', default="./", type=str, nargs='?',
                    help='specifies the path where the results will be saved')
parser.add_argument('--data_dir', default="../tabular_benchmarks/fcnet_tabular_benchmarks/", type=str, nargs='?',
                    help='specifies the path to the tabular data')
parser.add_argument('--strategy', default="rand1_bin", type=str, nargs='?', help='type of mutation & crossover scheme')
parser.add_argument('--eta', default=3, type=int, nargs='?', help='eta for Successive Halving')
parser.add_argument('--min_clip', default=3, type=int, nargs='?', help='minimum number of configurations')
parser.add_argument('--max_clip', default=None, type=int, nargs='?', help='maximum number of configurations')
parser.add_argument('--randomize', default=None, type=float, help='fraction of population to randomize in v2')
parser.add_argument('--max_age', default=np.inf, type=float, help='maximum age an individual can survive')
parser.add_argument('--mutation_factor', default=0.5, type=float, nargs='?', help='mutation factor value')
parser.add_argument('--gens', default=1, type=int, nargs='?', help='number of generations')
parser.add_argument('--crossover_prob', default=0.5, type=float, nargs='?', help='probability of crossover')
parser.add_argument('--verbose', default='False', choices=['True', 'False'], nargs='?', help='to print progress or not')
parser.add_argument('--version', default='1', choices=['1', '2', '3'], nargs='?', help='the version of DEHB to run')
parser.add_argument('--debug', default='False', choices=['True', 'False'], nargs='?', help='for additional logs')
parser.add_argument('--folder', default='dehb', type=str, nargs='?', help='name of folder where files will be dumped')

args = parser.parse_args()
args.verbose = True if args.verbose == 'True' else False
args.debug = True if args.debug == 'True' else False
args.fix_seed = True if args.fix_seed == 'True' else False

if args.version == '1':
    from optimizers import DEHBV1 as DEHB
elif args.version == '2':
    from optimizers import DEHBV2 as DEHB
else:
    from optimizers import DEHBV3 as DEHB

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
    def f(config, budget=max_budget):
        res = b.objective_function(config, budget=budget)
        fitness = res["function_value"]
        cost = budget
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

elif "paramnet" in args.benchmark:
    assert len(args.benchmark.split('_')) == 2
    dataset = args.benchmark.split('_')[1]
    benchmark_type = "paramnet"
    budget_dict = {
        'adult': (9, 243),
        'higgs': (9, 243),
        'letter': (3, 81),
        'mnist': (9, 243),
        'optdigits': (1, 27) ,
        'poker': (81, 2187)
    }
    min_budget, max_budget = budget_dict[dataset]
    b = SurrogateReducedParamNetTime(dataset=dataset)
    inc_config, y_star_valid, y_star_test = (None, 0, 0)
    def f(config, budget=max_budget):
        res = b.objective_function(config, budget=budget)
        fitness = res["function_value"]
        cost = res["cost"]
        return fitness, cost

if "counting" in args.benchmark:
    cs = CountingOnes.get_configuration_space(n_categorical=n_categorical,
                                              n_continuous=n_continuous)
    args.output_path = os.path.join(args.output_path, "{}_{}".format(n_categorical, n_continuous))
elif "paramnet" in args.benchmark:
    cs = b.get_configuration_space()
    args.output_path = os.path.join(args.output_path, dataset)
else:
    cs = b.get_configuration_space()
dimensions = len(cs.get_hyperparameters())

output_path = os.path.join(args.output_path, args.folder)
os.makedirs(output_path, exist_ok=True)

dehb = DEHB(cs=cs, f=f, dimensions=dimensions, mutation_factor=args.mutation_factor,
            crossover_prob=args.crossover_prob, strategy=args.strategy, min_budget=min_budget,
            max_budget=max_budget, min_clip=args.min_clip, max_clip=args.max_clip,
            generations=args.gens, eta=args.eta, randomize=args.randomize, max_age=args.max_age)

if args.runs is None:
    traj, runtime, history = dehb.run(iterations=args.n_iters, verbose=args.verbose,
                                      debug=args.debug)
    if 'cifar' in args.benchmark:
        save(traj, runtime, history, output_path, args.run_id, filename="raw_run")
        traj, runtime, history = remove_invalid_configs(traj, runtime, history)
    save(traj, runtime, history, output_path, args.run_id)
else:
    for run_id, _ in enumerate(range(args.runs), start=args.run_start):
        if not args.fix_seed:
            np.random.seed(run_id)
        if args.verbose:
            print("\nRun #{:<3}\n{}".format(run_id + 1, '-' * 8))
        traj, runtime, history = dehb.run(iterations=args.n_iters, verbose=args.verbose,
                                          debug=args.debug)
        if 'cifar' in args.benchmark:
            save(traj, runtime, history, output_path, run_id, filename="raw_run")
            traj, runtime, history = remove_invalid_configs(traj, runtime, history)
        save(traj, runtime, history, output_path, run_id)
        print("Run saved. Resetting...")
        dehb.reset()
        if benchmark_type == "nasbench":
            b.reset_tracker()

save_configspace(cs, output_path)
