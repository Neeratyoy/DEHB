import sys
sys.path.append('../')

import numpy as np

from optimizers.de import DE


class DEHBBase():
    def __init__(self, b=None, cs=None, dimensions=None, mutation_factor=None,
                 crossover_prob=None, strategy='rand1_bin', generations=None, min_budget=None,
                 max_budget=None, eta=None, **kwargs):
        # Benchmark related variables
        self.b = b
        self.cs = self.b.get_configuration_space() if cs is None and b is not None else cs
        if dimensions is None and self.cs is not None:
            self.dimensions = len(self.cs.get_hyperparameters())
        else:
            self.dimensions = dimensions

        # DE related variables
        self.mutation_factor = mutation_factor
        self.crossover_prob = crossover_prob
        self.strategy = strategy
        self.generations = generations
        print("Gen: ", self.generations)

        # Hyperband related variables
        self.min_budget = min_budget
        self.max_budget = max_budget
        self.eta = eta

        # Precomputing budget spacing and number of configurations for HB iterations
        self.max_SH_iter = None
        self.budgets = None
        if self.min_budget is not None and \
           self.max_budget is not None and \
           self.eta is not None:
            self.max_SH_iter = -int(np.log(self.min_budget / self.max_budget) / np.log(self.eta)) + 1
            self.budgets = self.max_budget * np.power(self.eta,
                                                     -np.linspace(start=self.max_SH_iter - 1,
                                                                  stop=0, num=self.max_SH_iter))
            self.budgets = self.budgets.astype(int)

        # Miscellaneous
        self.output_path = kwargs['output_path'] if 'output_path' in kwargs else ''

        # Incumbent trackers
        self.inc_score = np.inf
        self.inc_config = None

    def init_population(self, pop_size=10):
        population = np.random.uniform(low=0.0, high=1.0, size=(pop_size, self.dimensions))
        return population

    def get_next_iteration(self, iteration, clip=None):
        '''Computes the Successive Halving spacing

        Given the iteration index, computes the budget spacing to be used and
        the number of configurations to be used for the SH iterations.

        Parameters
        ----------
        iteration : int
            Iteration index
        clip : int, {1, 2, 3, ..., None}
            If not None, clips the minimum number of configurations to 'clip'

        Returns
        -------
        ns : array
        budgets : array
        '''
        # number of 'SH runs'
        s = self.max_SH_iter - 1 - (iteration % self.max_SH_iter)
        # budget spacing for this iteration
        budgets = self.budgets[(-s-1):]
        # number of configurations in that bracket
        n0 = int(np.floor((self.max_SH_iter)/(s+1)) * self.eta**s)
        ns = [max(int(n0*(self.eta**(-i))), 1) for i in range(s+1)]
        if clip is not None:
            ns = np.clip(ns, a_min=clip, a_max=np.max(ns))

        return ns, budgets

    def f_objective(self):
        raise NotImplementedError("The function needs to be defined in the sub class.")

    def run(self):
        raise NotImplementedError("The function needs to be defined in the sub class.")


class DEHB(DEHBBase):
    def __init__(self, b=None, cs=None, dimensions=None, mutation_factor=None,
                 crossover_prob=None, strategy=None, min_budget=None, max_budget=None,
                 eta=None, generations=None, **kwargs):
        super().__init__(b=b, cs=cs, dimensions=dimensions, mutation_factor=mutation_factor,
                         crossover_prob=crossover_prob, strategy=strategy, min_budget=min_budget,
                         max_budget=max_budget, eta=eta, generations=generations)

    def run(self, iterations=100, verbose=True):
        # Book-keeping variables
        traj = []
        runtime = []
        # Performs DEHB iterations
        for iteration in range(iterations):
            # Retrieves SH budgets and number of configurations
            num_configs, budgets = self.get_next_iteration(iteration=iteration, clip=3)
            if verbose:
                print('Iteration #{:>3}\n{}'.format(iteration, '-' * 15))
                print(num_configs, budgets, self.inc_score)

            # Sets budget and population size for first SH iteration
            pop_size = num_configs[0]
            budget = budgets[0]

            num_SH_iters = len(budgets)  # number of SH iterations in this DEHB iteration

            de = DE(b=self.b, cs=self.cs, dimensions=self.dimensions, pop_size=pop_size,
                    mutation_factor=self.mutation_factor, crossover_prob=self.crossover_prob,
                    strategy=self.strategy, budget=budget)
            # warmstarting DE incumbent to be the global incumbent
            de.inc_score = self.inc_score
            de.inc_config = self.inc_config
            # creating new population for DEHB iteration to be used for the next SH steps
            de_traj, de_runtime = de.init_eval_pop()
            # update global incumbent with new population scores
            self.inc_score = de.inc_score
            self.inc_config = de.inc_config
            traj.extend(de_traj)
            runtime.extend(de_runtime)

            # Successive Halving iterations
            for i_sh in range(num_SH_iters):
                # Repeating DE over entire population to create generations
                for gen in range(self.generations):
                    # DE sweep : Evolving the population for a single generation
                    for j in range(pop_size):
                        fitness, cost = de.step(j, budget=budget)
                        if fitness < self.inc_score:
                            self.inc_score = fitness
                            self.inc_config = de.inc_config
                        traj.append(self.inc_score)
                        runtime.append(cost)

                # Ranking evolved population
                de.population = de.population[np.argsort(de.fitness)]
                de.fitness = np.sort(de.fitness)

                if i_sh < num_SH_iters-1:  # when not final SH iteration
                    pop_size = num_configs[i_sh+1]
                    budget = budgets[i_sh+1]
                    # Selecting top individuals to fit pop_size of next SH iteration
                    de.population = de.population[:pop_size]
                    de.fitness = de.fitness[:pop_size]
                    de.pop_size = pop_size

        if verbose:
            print("\nRun complete!")

        return np.array(traj), np.array(runtime)
