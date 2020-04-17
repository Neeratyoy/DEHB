'''Runs BOHB on XGBoostBenchmark
'''

import os
import sys

import json
import pickle
import logging
import argparse
import numpy as np
import ConfigSpace

logging.basicConfig(level=logging.ERROR)

from hpbandster.optimizers.bohb import BOHB
import hpbandster.core.nameserver as hpns
from hpbandster.core.worker import Worker

sys.path.append('dehb/examples/')
from utils import util

sys.path.append(os.path.join(os.getcwd(), '../HPOlib3/'))
from hpolib.benchmarks.ml.xgboost_benchmark import XGBoostBenchmark as Benchmark
from hpolib.util.openml_data_manager import get_openmlcc18_taskids

from dehb import DE, DEHB_0, DEHB_1


# task_ids = get_openmlcc18_taskids()
task_ids = [126031, 189906, 167155]  # as suggested by Philip


def save_configspace(cs, path, filename='configspace'):
    fh = open(os.path.join(path, '{}.pkl'.format(filename)), 'wb')
    pickle.dump(cs, fh)
    fh.close()


class MyWorker(Worker):
    def compute(self, config, budget, **kwargs):
        global n_estimators, max_budget
        if budget is None:
            budget = max_budget
        res = b.objective_function(config, n_estimators=n_estimators, subsample=budget)
        fitness = res['function_value']
        cost = res['cost']
        res = b.objective_function_test(config, n_estimators=n_estimators)
        return ({
            'loss': float(fitness),
            'info': {'cost': float(cost), 'test_loss': float(res['function_value'])}
        })


parser = argparse.ArgumentParser()
parser.add_argument('--fix_seed', default='False', type=str, choices=['True', 'False'],
                    nargs='?', help='seed')
parser.add_argument('--run_id', default=0, type=int, nargs='?',
                    help='unique number to identify this run')
parser.add_argument('--runs', default=None, type=int, nargs='?', help='number of runs to perform')
parser.add_argument('--run_start', default=0, type=int, nargs='?',
                    help='run index to start with for multiple runs')
parser.add_argument('--task_id', default=task_ids[0], type=int,
                    help="specify the OpenML task id to run on from among {}".format(task_ids))
parser.add_argument('--n_estimators', default=64, type=int,
                    help="specify the number of estimators XGBoost will be trained with")
parser.add_argument('--n_iters', default=5, type=int, nargs='?',
                    help='number of iterations for optimization method')
parser.add_argument('--output_path', default="./results", type=str, nargs='?',
                    help='specifies the path where the results will be saved')
parser.add_argument('--strategy', default="sampling", type=str, nargs='?',
                    help='optimization strategy for the acquisition function')
parser.add_argument('--min_bandwidth', default=.3, type=float, nargs='?',
                    help='minimum bandwidth for KDE')
parser.add_argument('--bandwidth_factor', default=3, type=int, nargs='?',
                    help='factor multiplied to the bandwidth')
parser.add_argument('--num_samples', default=64, type=int, nargs='?',
                    help='number of samples for the acquisition function')
parser.add_argument('--random_fraction', default=.33, type=float, nargs='?',
                    help='fraction of random configurations')
parser.add_argument('--eta', default=3, type=int,
                    help='aggressive stopping rate (eta) for Hyperband')
parser.add_argument('--min_budget', default=1, type=float,
                    help='the minimum budget for the benchmark')
parser.add_argument('--max_budget', default=1, type=float,
                    help='the maximum budget for the benchmark')
parser.add_argument('--verbose', default='False', choices=['True', 'False'], nargs='?', type=str,
                    help='to print progress or not')
parser.add_argument('--folder', default='bohb', type=str, nargs='?',
                    help='name of folder where files will be dumped')

args = parser.parse_args()
args.verbose = True if args.verbose == 'True' else False
args.fix_seed = True if args.fix_seed == 'True' else False
n_estimators = args.n_estimators
min_budget = args.min_budget
max_budget = args.max_budget

task_ids = get_openmlcc18_taskids()
if args.task_id not in task_ids:
    raise "Incorrect task ID. Choose from: {}".format(task_ids)

b = Benchmark(task_id=args.task_id)
# Parameter space to be used by DE
cs = b.get_configuration_space()
dimensions = len(cs.get_hyperparameters())

output_path = os.path.join(args.output_path, str(args.task_id), args.folder)
os.makedirs(output_path, exist_ok=True)

runs = args.runs
for run_id in range(runs):
    print("Run {:>3}/{:>3}".format(run_id+1, runs))

    # hb_run_id = '0'
    hb_run_id = run_id

    NS = hpns.NameServer(run_id=hb_run_id, host='localhost', port=0)
    ns_host, ns_port = NS.start()

    num_workers = 1

    workers = []
    for i in range(num_workers):
        w = MyWorker(nameserver=ns_host, nameserver_port=ns_port,
                     run_id=hb_run_id,
                     id=i)
        w.run(background=True)
        workers.append(w)

    bohb = BOHB(configspace=cs,
                run_id=hb_run_id,
                eta=3, min_budget=min_budget, max_budget=max_budget,
                nameserver=ns_host,
                nameserver_port=ns_port,
                # optimization_strategy=args.strategy,
                num_samples=args.num_samples,
                random_fraction=args.random_fraction, bandwidth_factor=args.bandwidth_factor,
                ping_interval=10, min_bandwidth=args.min_bandwidth)

    results = bohb.run(args.n_iters, min_n_workers=num_workers)

    bohb.shutdown(shutdown_workers=True)
    NS.shutdown()

    fh = open(os.path.join(output_path, 'bohb_run_%d.pkl' % run_id), 'wb')
    pickle.dump(util.extract_results_to_pickle(results), fh)
    fh.close()
    print("Run saved. Resetting...")
