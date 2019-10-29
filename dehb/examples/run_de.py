import sys
sys.path.append('../')

import os
import json
import argparse
import numpy as np

from tabular_benchmarks import FCNetProteinStructureBenchmark, FCNetSliceLocalizationBenchmark,\
    FCNetNavalPropulsionBenchmark, FCNetParkinsonsTelemonitoringBenchmark
from tabular_benchmarks import NASCifar10A, NASCifar10B, NASCifar10C

from optimizers import DE


def remove_invalid_configs(traj, runtime):
    idx = np.where(np.array(runtime)==0)
    runtime = np.delete(runtime, idx)
    traj = np.delete(np.array(traj), idx)
    return traj, runtime

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
parser.add_argument('--pop_size', default=100, type=int, nargs='?', help='population size')
parser.add_argument('--strategy', default="rand1_bin", type=str, nargs='?', help='type of mutation & crossover scheme')
parser.add_argument('--mutation_factor', default=0.5, type=float, nargs='?', help='mutation factor value')
parser.add_argument('--crossover_prob', default=0.5, type=float, nargs='?', help='probability of crossover')
parser.add_argument('--max_budget', default=None, type=str, nargs='?', help='maximum wallclock time to run DE for')
parser.add_argument('--verbose', default='False', choices=['True', 'False'], nargs='?', help='to print progress or not')

args = parser.parse_args()
args.verbose = True if args.verbose == 'True' else False

if args.benchmark == "nas_cifar10a":
    b = NASCifar10A(data_dir=args.data_dir, multi_fidelity=False)
    y_star_valid = b.y_star_valid

elif args.benchmark == "nas_cifar10b":
    b = NASCifar10B(data_dir=args.data_dir, multi_fidelity=False)
    y_star_valid = b.y_star_valid

elif args.benchmark == "nas_cifar10c":
    b = NASCifar10C(data_dir=args.data_dir, multi_fidelity=False)
    y_star_valid = b.y_star_valid

elif args.benchmark == "protein_structure":
    b = FCNetProteinStructureBenchmark(data_dir=args.data_dir)
    _, y_star_valid, _ = b.get_best_configuration()

elif args.benchmark == "slice_localization":
    b = FCNetSliceLocalizationBenchmark(data_dir=args.data_dir)
    _, y_star_valid, _ = b.get_best_configuration()

elif args.benchmark == "naval_propulsion":
    b = FCNetNavalPropulsionBenchmark(data_dir=args.data_dir)
    _, y_star_valid, _ = b.get_best_configuration()

elif args.benchmark == "parkinsons_telemonitoring":
    b = FCNetParkinsonsTelemonitoringBenchmark(data_dir=args.data_dir)
    _, y_star_valid, _ = b.get_best_configuration()

cs = b.get_configuration_space()
dimensions = len(cs.get_hyperparameters())

output_path = os.path.join(args.output_path, "de")
os.makedirs(output_path, exist_ok=True)

de = DE(b=b, cs=cs, dimensions=dimensions, pop_size=args.pop_size,
        mutation_factor=args.mutation_factor, crossover_prob=args.crossover_prob,
        strategy=args.strategy, max_budget=args.max_budget)

if args.runs is None:
    traj, runtime = de.run(iterations=args.n_iters, verbose=args.verbose)
    if 'cifar' in args.benchmark:
        traj, runtime = remove_invalid_configs(traj, runtime)
    res = {}
    res['runtime'] = np.cumsum(runtime).tolist()
    res['regret_validation'] = np.array(traj - y_star_valid).tolist()
    fh = open(os.path.join(output_path, 'run_{}.json'.format(args.run_id)), 'w')
    json.dump(res, fh)
    fh.close()
else:
    for run_id, _ in enumerate(range(args.runs), start=args.run_start):
        if args.verbose:
            print("\nRun #{:<3}\n{}".format(run_id + 1, '-' * 8))
        traj, runtime = de.run(iterations=args.n_iters, verbose=args.verbose)
        if 'cifar' in args.benchmark:
            traj, runtime = remove_invalid_configs(traj, runtime)
        res = {}
        res['runtime'] = np.cumsum(runtime).tolist()
        res['regret_validation'] = np.array(traj - y_star_valid).tolist()
        fh = open(os.path.join(output_path, 'run_{}.json'.format(run_id)), 'w')
        json.dump(res, fh)
        fh.close()
        print("Run saved. Resetting...")
        de.reset()
        b.reset_tracker()
