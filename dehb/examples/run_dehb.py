import sys
sys.path.append('../')

import os
import json
import argparse
import numpy as np

from tabular_benchmarks import FCNetProteinStructureBenchmark, FCNetSliceLocalizationBenchmark,\
    FCNetNavalPropulsionBenchmark, FCNetParkinsonsTelemonitoringBenchmark
from tabular_benchmarks import NASCifar10A, NASCifar10B, NASCifar10C


def remove_invalid_configs(traj, runtime):
    idx = np.where(np.array(runtime)==0)
    runtime = np.delete(runtime, idx)
    traj = np.delete(np.array(traj), idx)
    return traj, runtime

def save(trajectory, runtime, path, run_id, filename="run", logger=None):
    global y_star_valid
    res = {}
    res['runtime'] = np.cumsum(runtime).tolist()
    res['regret_validation'] = np.array(traj - y_star_valid).tolist()
    if logger is not None:
        res['logger'] = logger
    fh = open(os.path.join(output_path, '{}_{}.json'.format(filename, run_id)), 'w')
    json.dump(res, fh)
    fh.close()

parser = argparse.ArgumentParser()
parser.add_argument('--run_id', default=0, type=int, nargs='?', help='unique number to identify this run')
parser.add_argument('--runs', default=None, type=int, nargs='?', help='number of runs to perform')
parser.add_argument('--run_start', default=0, type=int, nargs='?', help='run index to start with for multiple runs')
parser.add_argument('--benchmark', default="protein_structure",
                    choices=["protein_structure", "slice_localization", "naval_propulsion",
                             "parkinsons_telemonitoring", "nas_cifar10a", "nas_cifar10b", "nas_cifar10c"],
                    type=str, nargs='?', help='specifies the benchmark')
parser.add_argument('--n_iters', default=100, type=int, nargs='?', help='number of iterations for optimization method')
parser.add_argument('--output_path', default="./", type=str, nargs='?',
                    help='specifies the path where the results will be saved')
parser.add_argument('--data_dir', default="../tabular_benchmarks", type=str, nargs='?',
                    help='specifies the path to the tabular data')
parser.add_argument('--strategy', default="rand1_bin", type=str, nargs='?', help='type of mutation & crossover scheme')
parser.add_argument('--eta', default=3, type=int, nargs='?', help='eta for Successive Halving')
parser.add_argument('--clip', default=3, type=int, nargs='?', help='minimum number of configurations')
parser.add_argument('--randomize', default=None, type=float, help='fraction of population to randomize in v2')
parser.add_argument('--mutation_factor', default=0.5, type=float, nargs='?', help='mutation factor value')
parser.add_argument('--gens', default=1, type=int, nargs='?', help='number of generations')
parser.add_argument('--crossover_prob', default=0.5, type=float, nargs='?', help='probability of crossover')
parser.add_argument('--verbose', default='False', choices=['True', 'False'], nargs='?', help='to print progress or not')
parser.add_argument('--version', default='1', choices=['1', '2', '3'], nargs='?', help='the version of DEHB to run')
parser.add_argument('--folder', default='dehb', type=str, nargs='?', help='name of folder where files will be dumped')

args = parser.parse_args()
args.verbose = True if args.verbose == 'True' else False

if args.version == '1':
    from optimizers import DEHBV1 as DEHB
elif args.version == '2':
    from optimizers import DEHBV2 as DEHB
else:
    from optimizers import DEHBV3 as DEHB

if args.benchmark == "nas_cifar10a":
    min_budget = 4
    max_budget = 108
    b = NASCifar10A(data_dir=args.data_dir, multi_fidelity=True)
    y_star_valid = b.y_star_valid

elif args.benchmark == "nas_cifar10b":
    min_budget = 4
    max_budget = 108
    b = NASCifar10B(data_dir=args.data_dir, multi_fidelity=True)
    y_star_valid = b.y_star_valid

elif args.benchmark == "nas_cifar10c":
    min_budget = 4
    max_budget = 108
    b = NASCifar10C(data_dir=args.data_dir, multi_fidelity=True)
    y_star_valid = b.y_star_valid

elif args.benchmark == "protein_structure":
    min_budget = 4
    max_budget = 100
    b = FCNetProteinStructureBenchmark(data_dir=args.data_dir)
    _, y_star_valid, _ = b.get_best_configuration()

elif args.benchmark == "slice_localization":
    min_budget = 4
    max_budget = 100
    b = FCNetSliceLocalizationBenchmark(data_dir=args.data_dir)
    _, y_star_valid, _ = b.get_best_configuration()

elif args.benchmark == "naval_propulsion":
    min_budget = 4
    max_budget = 100
    b = FCNetNavalPropulsionBenchmark(data_dir=args.data_dir)
    _, y_star_valid, _ = b.get_best_configuration()

elif args.benchmark == "parkinsons_telemonitoring":
    min_budget = 4
    max_budget = 100
    b = FCNetParkinsonsTelemonitoringBenchmark(data_dir=args.data_dir)
    _, y_star_valid, _ = b.get_best_configuration()

cs = b.get_configuration_space()
dimensions = len(cs.get_hyperparameters())

output_path = os.path.join(args.output_path, args.folder)
os.makedirs(output_path, exist_ok=True)

dehb = DEHB(b=b, cs=cs, dimensions=dimensions, mutation_factor=args.mutation_factor,
            crossover_prob=args.crossover_prob, strategy=args.strategy, min_budget=min_budget,
            max_budget=max_budget, generations=args.gens, eta=args.eta, randomize=args.randomize)

if args.runs is None:
    traj, runtime = dehb.run(iterations=args.n_iters, verbose=args.verbose)
    save(traj, runtime, output_path, args.run_id, filename="raw_run", logger=dehb.logger)
    if 'cifar' in args.benchmark:
        traj, runtime = remove_invalid_configs(traj, runtime)
    save(traj, runtime, output_path, args.run_id)
else:
    for run_id, _ in enumerate(range(args.runs), start=args.run_start):
        if args.verbose:
            print("\nRun #{:<3}\n{}".format(run_id + 1, '-' * 8))
        traj, runtime = dehb.run(iterations=args.n_iters, verbose=args.verbose)
        save(traj, runtime, output_path, run_id, filename="raw_run", logger=dehb.logger)
        if 'cifar' in args.benchmark:
            traj, runtime = remove_invalid_configs(traj, runtime)
        save(traj, runtime, output_path, run_id)
        print("Run saved. Resetting...")
        dehb.reset()
        b.reset_tracker()
