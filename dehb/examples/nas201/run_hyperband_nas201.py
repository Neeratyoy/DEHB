# Slightly modified version of:
# https://github.com/automl/nas_benchmarks/blob/development/experiment_scripts/run_bohb.py


import os
import sys
sys.path.append(os.path.join(os.getcwd(), '../nas201/'))
sys.path.append(os.path.join(os.getcwd(), '../AutoDL-Projects/lib/'))

import json
import pickle
import argparse
import numpy as np
import ConfigSpace

logging.basicConfig(level=logging.ERROR)

from hpbandster.optimizers.hyperband import HyperBand
import hpbandster.core.nameserver as hpns
from hpbandster.core.worker import Worker

from nas_201_api import NASBench201API as API
from models import CellStructure, get_search_spaces

from dehb import DE


# From https://github.com/D-X-Y/AutoDL-Projects/blob/master/exps/algos/BOHB.py
## Author: https://github.com/D-X-Y [Xuanyi.Dong@student.uts.edu.au]
def get_configuration_space(max_nodes, search_space):
  cs = ConfigSpace.ConfigurationSpace()
  #edge2index   = {}
  for i in range(1, max_nodes):
    for j in range(i):
      node_str = '{:}<-{:}'.format(i, j)
      cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter(node_str, search_space))
  return cs


# From https://github.com/D-X-Y/AutoDL-Projects/blob/master/exps/algos/BOHB.py
## Author: https://github.com/D-X-Y [Xuanyi.Dong@student.uts.edu.au]
def config2structure_func(max_nodes):
  def config2structure(config):
    genotypes = []
    for i in range(1, max_nodes):
      xlist = []
      for j in range(i):
        node_str = '{:}<-{:}'.format(i, j)
        op_name = config[node_str]
        xlist.append((op_name, j))
      genotypes.append( tuple(xlist) )
    return CellStructure( genotypes )
  return config2structure


def calculate_regrets(history, runtime):
    assert len(runtime) == len(history)
    global dataset, api, de, max_budget

    regret_test = []
    regret_validation = []
    inc = np.inf
    test_regret = 1
    validation_regret = 1
    for i in range(len(history)):
        config, valid_regret, budget = history[i]
        valid_regret = valid_regret - y_star_valid
        if valid_regret <= inc:
            inc = valid_regret
            config = de.vector_to_configspace(config)
            structure = config2structure(config)
            arch_index = api.query_index_by_arch(structure)
            info = api.get_more_info(arch_index, dataset, max_budget, False, False)
            test_regret = (1 - (info['test-accuracy'] / 100)) - y_star_test
        regret_validation.append(inc)
        regret_test.append(test_regret)
    res = {}
    res['regret_test'] = regret_test
    res['regret_validation'] = regret_validation
    res['runtime'] = np.cumsum(runtime).tolist()
    return res


def save_configspace(cs, path, filename='configspace'):
    fh = open(os.path.join(output_path, '{}.pkl'.format(filename)), 'wb')
    pickle.dump(cs, fh)
    fh.close()


def find_nas201_best(api, dataset):
    arch, y_star_test = api.find_best(dataset=dataset, metric_on_set='ori-test')
    _, y_star_valid = api.find_best(dataset=dataset, metric_on_set='x-valid')
    return 1 - (y_star_valid / 100), 1 - (y_star_test / 100)


parser = argparse.ArgumentParser()
parser.add_argument('--runs', default=1, type=int, nargs='?',
                    help='unique number to identify this run')
parser.add_argument('--dataset', default='cifar10-valid', type=str, nargs='?',
                    choices=['cifar10-valid', 'cifar100', 'ImageNet16-120'],
                    help='choose the dataset')
parser.add_argument('--n_iters', default=5, type=int, nargs='?',
                    help='number of iterations for optimization method')
parser.add_argument('--output_path', default="./results", type=str, nargs='?',
                    help='specifies the path where the results will be saved')
parser.add_argument('--data_dir', default="../tabular_benchmarks/fcnet_tabular_benchmarks/",
                    type=str, nargs='?', help='specifies the path to the tabular data')
parser.add_argument('--strategy', default="sampling", type=str, nargs='?',
                    help='optimization strategy for the acquisition function')
parser.add_argument('--min_bandwidth', default=.3, type=float, nargs='?',
                    help='minimum bandwidth for KDE')
parser.add_argument('--num_samples', default=64, type=int, nargs='?',
                    help='number of samples for the acquisition function')
parser.add_argument('--random_fraction', default=.33, type=float, nargs='?',
                    help='fraction of random configurations')
parser.add_argument('--bandwidth_factor', default=3, type=int, nargs='?',
                    help='factor multiplied to the bandwidth')

args = parser.parse_args()

output_path = os.path.join(args.output_path, "hyperband")
os.makedirs(os.path.join(output_path), exist_ok=True)

class MyWorker(Worker):
    def compute(self, config, budget, **kwargs):
        global dataset, api
        structure = config2structure(config)
        arch_index = api.query_index_by_arch(structure)
        if budget is not None:
            budget = int(budget)
        # From https://github.com/D-X-Y/AutoDL-Projects/blob/master/exps/algos/R_EA.py
        ## Author: https://github.com/D-X-Y [Xuanyi.Dong@student.uts.edu.au]
        info = api.get_more_info(arch_index, dataset, iepoch=budget,
                                 use_12epochs_result=False, is_random=True)
        try:
            fitness = info['valid-accuracy']
        except:
            fitness = info['valtest-accuracy']

        cost = info['train-all-time']
        try:
            cost += info['valid-all-time']
        except:
            cost += info['valtest-all-time']

        fitness = 1 - fitness / 100
        return ({
            'loss': float(fitness),
            'info': float(cost)})


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

    HB = HyperBand(configspace=cs,
                run_id=hb_run_id,
                eta=3, min_budget=min_budget, max_budget=max_budget,
                nameserver=ns_host,
                nameserver_port=ns_port,
                # optimization_strategy=args.strategy,
                num_samples=args.num_samples,
                random_fraction=args.random_fraction, bandwidth_factor=args.bandwidth_factor,
                ping_interval=10, min_bandwidth=args.min_bandwidth)

    results = HB.run(args.n_iters, min_n_workers=num_workers)

    HB.shutdown(shutdown_workers=True)
    NS.shutdown()

    fh = open(os.path.join(output_path, 'hyperband_run_%d.pkl' % run_id), 'w')
    pickle.dump(res, fh)
    fh.close()
    print("Run saved. Resetting...")
    b.reset_tracker()
