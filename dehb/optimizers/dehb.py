import numpy as np

from .de import DE, AsyncDE


class DEHBBase():
    def __init__(self, cs=None, f=None, dimensions=None, mutation_factor=None,
                 crossover_prob=None, strategy=None, generations=None, min_budget=None,
                 max_budget=None, eta=None, min_clip=3, max_clip=None, **kwargs):
        # Benchmark related variables
        self.cs = cs
        if dimensions is None and self.cs is not None:
            self.dimensions = len(self.cs.get_hyperparameters())
        else:
            self.dimensions = dimensions
        self.f = f

        # DE related variables
        self.mutation_factor = mutation_factor
        self.crossover_prob = crossover_prob
        self.strategy = strategy
        self.generations = generations

        # Hyperband related variables
        self.min_budget = min_budget
        self.max_budget = max_budget
        self.eta = eta
        self.min_clip = min_clip
        self.max_clip = max_clip

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
            # self.budgets = self.budgets.astype(int)

        # Miscellaneous
        self.output_path = kwargs['output_path'] if 'output_path' in kwargs else ''

        # Global trackers
        self.population = None
        self.fitness = None
        self.inc_score = np.inf
        self.inc_config = None
        self.history = []

    def reset(self):
        self.inc_score = np.inf
        self.inc_config = None
        self.population = None
        self.fitness = None
        self.history = []

    def init_population(self, pop_size=10):
        population = np.random.uniform(low=0.0, high=1.0, size=(pop_size, self.dimensions))
        return population

    def get_next_iteration(self, iteration):
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
        if self.min_clip is not None and self.max_clip is not None:
            ns = np.clip(ns, a_min=self.min_clip, a_max=self.max_clip)
        elif self.min_clip is not None:
            ns = np.clip(ns, a_min=self.min_clip, a_max=np.max(ns))

        return ns, budgets

    def f_objective(self):
        raise NotImplementedError("The function needs to be defined in the sub class.")

    def run(self):
        raise NotImplementedError("The function needs to be defined in the sub class.")


class DEHB_0(DEHBBase):
    '''DEHB with Async. DE
    '''
    def __init__(self, async_strategy='orig', **kwargs):
        super().__init__(**kwargs)
        self.max_age = np.inf
        self.min_clip = 0
        self.async_strategy = async_strategy
        self.async_strategy = 'orig'  # {'random', 'basic'}

    def concat_pops(self, exclude_budget=None):
        budgets = list(self.budgets)
        if exclude_budget is not None:
            budgets.remove(exclude_budget)
        pop = []
        for budget in budgets:
            pop.extend(self.de[budget].population.tolist())
        return np.array(pop)

    def run(self, iterations=1, verbose=False, debug=False):
        # Book-keeping variables
        traj = []
        runtime = []
        history = []

        # Determining the maximum pop size for a budget subpopulation as per SH
        self._max_pop_size = {}
        for i in range(self.max_SH_iter):
            n, r = self.get_next_iteration(i)
            for j, r_j in enumerate(r):
                self._max_pop_size[r_j] = \
                    max(n[j], self._max_pop_size[r_j]) if r_j in self._max_pop_size.keys() else n[j]

        # List of DE objects corresponding to the budgets (fidelities)
        self.de = {}
        for i, b in enumerate(self._max_pop_size.keys()):
            self.de[b] = AsyncDE(cs=self.cs, f=self.f, dimensions=self.dimensions,
                            pop_size=self._max_pop_size[b], mutation_factor=self.mutation_factor,
                            crossover_prob=self.crossover_prob, strategy=self.strategy,
                            budget=b, max_age=self.max_age)

        # Performs DEHB iterations
        for iteration in range(iterations):

            # Retrieves SH budgets and number of configurations
            num_configs, budgets = self.get_next_iteration(iteration=iteration)
            if verbose:
                print('Iteration #{:>3}\n{}'.format(iteration, '-' * 15))
                print(num_configs, budgets, self.inc_score)

            # Sets budget and population size for first SH iteration
            pop_size = num_configs[0]
            budget = budgets[0]
            self.de[budget].pop_size = pop_size

            # Number of SH iterations in this DEHB iteration
            num_SH_iters = len(budgets)

            if iteration > 0 and len(self.de[budget].population) < self._max_pop_size[budget]:
                # the previous iteration should have filled up the population slots
                # for certain budget spacings, this slot may be empty by one or two slots
                filler = self._max_pop_size[budget] - len(self.de[budget].population)
                if debug:
                    print("Adding {} individual(s) for the budget {}".format(filler, budget))
                self.de[budget].population, self.de[budget].fitness, self.de[budget].age = \
                    self.de[budget]._add_random_population(pop_size=filler)

            if iteration == 0:  # first HB bracket, first iteration
                for i_sh in range(num_SH_iters):
                    # warmstart DE with global incumbents
                    self.de[budget].inc_score = self.inc_score
                    self.de[budget].inc_config = self.inc_config
                    if i_sh == 0:
                        de_traj, de_runtime, de_history = self.de[budget].init_eval_pop(budget)
                    else:
                        # population is evaluated for pop_size, pop_size can be < len(population)
                        de_traj, de_runtime, de_history, _, _ = \
                                self.de[budget].eval_pop(budget=budget)
                    traj.extend(de_traj)
                    runtime.extend(de_runtime)
                    history.extend(de_history)
                    # update global incumbent with new population scores
                    self.inc_score = self.de[budget].inc_score
                    self.inc_config = self.de[budget].inc_config

                    if i_sh < (num_SH_iters - 1):  # when not final SH iteration
                        pop_size = num_configs[i_sh + 1]
                        next_budget = budgets[i_sh + 1]
                        rank = np.sort(np.argsort(self.de[budget].fitness)[:pop_size])

                        # initializing only the required pop size for the current SH iteration
                        ## remaining population slots to be filled in subsequent iterations
                        self.de[next_budget].population = self.de[budget].population[rank]
                        self.de[next_budget].fitness = self.de[budget].fitness[rank]
                        self.de[next_budget].age = self.de[budget].age[rank]
                        # print("Budget: ", next_budget, "Age: ", self.de[next_budget].age)
                        self.de[next_budget].pop_size = pop_size

                        # updating budget for next SH step
                        budget = next_budget


            elif iteration < self.max_SH_iter:  # first HB bracket, second iteration onwards
                for i_sh in range(num_SH_iters):
                    # warmstart DE with global incumbents
                    self.de[budget].inc_score = self.inc_score
                    self.de[budget].inc_config = self.inc_config

                    if i_sh == 0:
                        if debug:
                            print("Evolving {} on {}".format(self.de[budget].pop_size, budget))

                        de_traj, de_runtime, de_history = \
                                self.de[budget].evolve_generation(budget=budget,
                                                                  best=self.inc_config,
                                                                  async_strategy=self.async_strategy)
                    else:
                        if alt_population is None:
                            if debug:
                                print("Evaluating {} for {}".format(self.de[budget].pop_size,
                                                                    budget))
                            de_traj, de_runtime, de_history, _, _ = \
                                self.de[budget].eval_pop(budget=budget)
                        else:
                            if debug:
                                print("Evolving {} on {}".format(self.de[budget].pop_size, budget))

                            de_traj, de_runtime, de_history = \
                                self.de[budget].evolve_generation(budget=budget,
                                                                  best=self.inc_config,
                                                                  alt_pop=alt_population,
                                                                  async_strategy=self.async_strategy)
                    traj.extend(de_traj)
                    runtime.extend(de_runtime)
                    history.extend(de_history)

                    # update global incumbent with new population scores
                    self.inc_score = self.de[budget].inc_score
                    self.inc_config = self.de[budget].inc_config

                    if i_sh < (num_SH_iters - 1):  # when not final SH iteration
                        pop_size = num_configs[i_sh + 1]
                        next_budget = budgets[i_sh + 1]
                        rank = np.sort(np.argsort(self.de[budget].fitness)[:pop_size])
                        if len(self.de[next_budget].population) < self._max_pop_size[next_budget]:
                            self.de[next_budget].population = \
                                np.concatenate((self.de[budget].population[rank],
                                                self.de[next_budget].population))
                            self.de[next_budget].fitness = \
                                np.concatenate((np.array([np.inf] * pop_size),
                                                self.de[next_budget].fitness))
                            self.de[next_budget].age = \
                                np.concatenate((np.array([self.max_age] * pop_size),
                                                self.de[next_budget].age))
                            alt_population = None
                        else:
                            alt_population = self.de[budget].population[rank]
                        self.de[next_budget].pop_size = pop_size
                        budget = next_budget


            else:  # second HB bracket onwards
                alt_population = None
                for i_sh in range(num_SH_iters):
                    # warmstart DE with global incumbents
                    self.de[budget].inc_score = self.inc_score
                    self.de[budget].inc_config = self.inc_config
                    if debug:
                        print("Evolving {} for {}".format(self.de[budget].pop_size, budget))

                    if alt_population is not None and \
                            len(alt_population) < self.de[budget]._min_pop_size:
                        filler = self.de[budget]._min_pop_size - len(alt_population) + 1
                        if debug:
                            print("Adding {} individuals for mutation on "
                                  "budget {}".format(filler, budget))
                        new_pop = \
                            self.de[budget]._init_mutant_population(filler, self.concat_pops(),
                                                                    target=None,
                                                                    best=self.inc_config)
                        alt_population = np.concatenate((alt_population, new_pop))
                        if debug:
                            print("Mutation population size: {}".format(filler, budget))

                    de_traj, de_runtime, de_history = \
                            self.de[budget].evolve_generation(budget=budget,
                                                              best=self.inc_config,
                                                              alt_pop=alt_population,
                                                              async_strategy=self.async_strategy)
                    traj.extend(de_traj)
                    runtime.extend(de_runtime)
                    history.extend(de_history)

                    # update global incumbent with new population scores
                    self.inc_score = self.de[budget].inc_score
                    self.inc_config = self.de[budget].inc_config

                    if i_sh < (num_SH_iters - 1):  # when not final SH iteration
                        pop_size = num_configs[i_sh + 1]
                        next_budget = budgets[i_sh + 1]
                        rank = np.sort(np.argsort(self.de[budget].fitness)[:pop_size])

                        alt_population = self.de[budget].population[rank]
                        self.de[next_budget].pop_size = pop_size

                        budget = next_budget

                        if self.async_strategy in ['orig', 'basic'] and \
                                pop_size < len(self.de[budget].population):
                            # reordering to have the top individuals in front
                            rank_include = np.sort(np.argsort(self.de[budget].fitness)[:pop_size])
                            rank_exclude = list(set(np.arange(len(self.de[budget].population))) - \
                                                set(rank_include))
                            self.de[budget].population = \
                                np.concatenate((self.de[budget].population[rank_include],
                                                self.de[budget].population[rank_exclude]))
                            self.de[budget].fitness = \
                                np.concatenate((self.de[budget].fitness[rank_include],
                                                self.de[budget].fitness[rank_exclude]))
                            self.de[budget].age = \
                                np.concatenate((self.de[budget].age[rank_include],
                                                self.de[budget].age[rank_exclude]))

        return np.array(traj), np.array(runtime), np.array(history)
