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
        self.traj = []
        self.runtime = []
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
    def __init__(self, async_strategy='deferred', **kwargs):
        super().__init__(**kwargs)
        self.max_age = np.inf
        self.min_clip = 0
        self.async_strategy = async_strategy

        self.reset()
        self._get_pop_sizes()
        self._init_subpop()

    def reset(self):
        super().reset()
        self.de = {}

    def _get_pop_sizes(self):
        '''Determines maximum pop size for each budget
        '''
        self._max_pop_size = {}
        for i in range(self.max_SH_iter):
            n, r = self.get_next_iteration(i)
            for j, r_j in enumerate(r):
                self._max_pop_size[r_j] = \
                    max(n[j], self._max_pop_size[r_j]) if r_j in self._max_pop_size.keys() else n[j]

    def _init_subpop(self):
        # List of DE objects corresponding to the budgets (fidelities)
        self.de = {}
        for i, b in enumerate(self._max_pop_size.keys()):
            self.de[b] = AsyncDE(cs=self.cs, f=self.f, dimensions=self.dimensions,
                            pop_size=self._max_pop_size[b], mutation_factor=self.mutation_factor,
                            crossover_prob=self.crossover_prob, strategy=self.strategy,
                            budget=b, max_age=self.max_age)

    def _concat_pops(self, exclude_budget=None):
        '''Concatenates all subpopulations
        '''
        budgets = list(self.budgets)
        if exclude_budget is not None:
            budgets.remove(exclude_budget)
        pop = []
        for budget in budgets:
            pop.extend(self.de[budget].population.tolist())
        return np.array(pop)

    def _update_trackers(self, inc_score, inc_config, traj, runtime, history, budget):
        self.inc_score = self.de[budget].inc_score
        self.inc_config = self.de[budget].inc_config
        self.traj.extend(traj)
        self.runtime.extend(runtime)
        self.history.extend(history)

    def run(self, iterations=1, verbose=False, debug=False, reset=True):
        # Book-keeping variables
        if reset:
            self.reset()
            self._init_subpop()

        if len(self.traj) > 0 and len(self.runtime) > 0 and len(self.history) > 0:
            start = self.max_SH_iter
        else:
            start = 0

        # Performs DEHB iterations
        for iteration in range(start, iterations + start):
            # Retrieves budgets and number of configurations for the SH bracket
            num_configs, budgets = self.get_next_iteration(iteration=iteration)
            if verbose:
                print('Iteration #{:>3}\n{}'.format(iteration - start, '-' * 15))
                print(num_configs, budgets, self.inc_score)

            # Sets budget and population size for first iteration in the SH bracket
            pop_size = num_configs[0]
            budget = budgets[0]
            self.de[budget].pop_size = pop_size

            # Number of SH iterations in this DEHB iteration
            num_SH_iters = len(budgets)

            # if iteration > 0 and len(self.de[budget].population) < self._max_pop_size[budget]:
            #     # the previous iteration should have filled up the population slots
            #     # for certain budget spacings, this slot may be empty by one or two slots
            #     filler = self._max_pop_size[budget] - len(self.de[budget].population)
            #     if debug:
            #         print("Adding {} individual(s) for the budget {}".format(filler, budget))
            #     self.de[budget].population, self.de[budget].fitness, self.de[budget].age = \
            #         self.de[budget]._add_random_population(pop_size=filler)

            if iteration == 0:  # first HB bracket's first iteration (first SH bracket)
                for i_sh in range(num_SH_iters):
                    # warmstart DE incumbents with global incumbents
                    ## significant for iteration==0 when DEHB optimisation is continued
                    self.de[budget].inc_score = self.inc_score
                    self.de[budget].inc_config = self.inc_config
                    if i_sh == 0:
                        # initializes population and evaluates them on the 'budget'
                        # evaluations are counted as function evaluations for this iteration
                        de_traj, de_runtime, de_history = self.de[budget].init_eval_pop(budget)
                    else:
                        # population is evaluated for pop_size on the 'budget'
                        # pop_size can be < len(population)
                        de_traj, de_runtime, de_history, _, _ = \
                                self.de[budget].eval_pop(budget=budget)

                    self._update_trackers(self.de[budget].inc_score, self.de[budget].inc_config,
                                          de_traj, de_runtime, de_history, budget)

                    if i_sh < (num_SH_iters - 1):  # when not final SH iteration
                        pop_size = num_configs[i_sh + 1]
                        next_budget = budgets[i_sh + 1]
                        # finding the top individuals for the pop_size required
                        rank = np.sort(np.argsort(self.de[budget].fitness)[:pop_size])

                        # initializing the required pop_size for the higher budget populations
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
                    # warmstart DE incumbents with global incumbents
                    self.de[budget].inc_score = self.inc_score
                    self.de[budget].inc_config = self.inc_config

                    if i_sh == 0:  # first iteration in the SH bracket
                        if debug:
                            print("Evolving {} on {}".format(self.de[budget].pop_size, budget))
                        # evolves the subpopulation for 'budget' for one generation
                        de_traj, de_runtime, de_history = \
                                self.de[budget].evolve_generation(budget=budget,
                                                                  best=self.inc_config)
                    else:
                        if alt_population is None:
                            if debug:
                                print("Evaluating {} on {}".format(self.de[budget].pop_size,
                                                                    budget))
                            de_traj, de_runtime, de_history, _, _ = \
                                self.de[budget].eval_pop(budget=budget)
                        else:
                            if debug:
                                print("Evolving {} on {}".format(self.de[budget].pop_size, budget))

                            de_traj, de_runtime, de_history = \
                                self.de[budget].evolve_generation(budget=budget,
                                                                  best=self.inc_config,
                                                                  alt_pop=alt_population)

                    self._update_trackers(self.de[budget].inc_score, self.de[budget].inc_config,
                                          de_traj, de_runtime, de_history, budget)

                    if i_sh < (num_SH_iters - 1):  # when not final SH iteration
                        pop_size = num_configs[i_sh + 1]
                        next_budget = budgets[i_sh + 1]
                        # finding the top individuals for the pop_size required
                        rank = np.sort(np.argsort(self.de[budget].fitness)[:pop_size])
                        # checking if slots available
                        if len(self.de[next_budget].population) < self._max_pop_size[next_budget]:
                            # appending top individuals from the lower budget as part of next_budget
                            ## population of pop_size is appended to the front so they are evaluated
                            ## in the next iteration -- if size exceeds, weakest individuals
                            ## are dropped from the current population
                            required = self._max_pop_size[next_budget] - \
                                len(self.de[next_budget].population)
                            extra = required - pop_size
                            if extra < 0:
                                # removing weakest individuals from current population
                                extra = np.abs(extra)
                                top_rank = \
                                    np.sort(np.argsort(self.de[next_budget].fitness)[:-extra])
                                self.de[next_budget].population = \
                                    self.de[next_budget].population[top_rank]
                                self.de[next_budget].fitness = \
                                    self.de[next_budget].fitness[top_rank]
                                self.de[next_budget].age = \
                                    self.de[next_budget].age[top_rank]
                            self.de[next_budget].population = \
                                np.concatenate((self.de[budget].population[rank],
                                                self.de[next_budget].population))
                            # the individuals are evaluated on a lower budget
                            ## for a fair comparison during selection, all individuals should be
                            ## evaluated on the same budget level
                            ## the fitness values are set as infinity to not waste function
                            ## evaluations -- this is not a problem since the individuals will
                            ## participate in mutation and the new trial will replace it
                            self.de[next_budget].fitness = \
                                np.concatenate((np.array([np.inf] * pop_size),
                                                self.de[next_budget].fitness))
                            self.de[next_budget].age = \
                                np.concatenate((np.array([self.max_age] * pop_size),
                                                self.de[next_budget].age))
                            alt_population = None
                        else:
                            # the top individuals from the current budget are the candidates for
                            ## mutation in the next higher budget
                            alt_population = self.de[budget].population[rank]
                        postlen = len(self.de[next_budget].population)
                        self.de[next_budget].pop_size = pop_size
                        budget = next_budget

            else:  # second HB bracket onwards (DEHB brackets)
                alt_population = None
                for i_sh in range(num_SH_iters):
                    # warmstart DE with global incumbents
                    self.de[budget].inc_score = self.inc_score
                    self.de[budget].inc_config = self.inc_config
                    if debug:
                        print("Evolving {} for {}".format(self.de[budget].pop_size, budget))

                    # when the size of mutation candidate population is lesser than that required
                    ## for the chosen mutation strategy, new individuals of infinite fitness are
                    ## introduced by creating mutants from the total global population formed
                    ## by concatenating all the subpopulations associated with all the budgets
                    if alt_population is not None and \
                            len(alt_population) < self.de[budget]._min_pop_size:
                        filler = self.de[budget]._min_pop_size - len(alt_population) + 1
                        if debug:
                            print("Adding {} individuals for mutation on "
                                  "budget {}".format(filler, budget))
                        new_pop = \
                            self.de[budget]._init_mutant_population(filler, self._concat_pops(),
                                                                    target=None,
                                                                    best=self.inc_config)
                        alt_population = np.concatenate((alt_population, new_pop))
                        if debug:
                            print("Mutation population size: {}".format(filler, budget))

                    # evolving subpopulation on 'budget' for one generation
                    ## the targets in te evolution process are the individuals themselves
                    ## the mutants are created from the alt_population that is passed
                    de_traj, de_runtime, de_history = \
                            self.de[budget].evolve_generation(budget=budget,
                                                              best=self.inc_config,
                                                              alt_pop=alt_population)

                    self._update_trackers(self.de[budget].inc_score, self.de[budget].inc_config,
                                          de_traj, de_runtime, de_history, budget)

                    if i_sh < (num_SH_iters - 1):  # when not final SH iteration
                        pop_size = num_configs[i_sh + 1]
                        next_budget = budgets[i_sh + 1]
                        rank = np.sort(np.argsort(self.de[budget].fitness)[:pop_size])
                        # the top individuals from the current 'budget' serve as mutation
                        ## candidates for the DE evolution in the higher next_budget
                        alt_population = self.de[budget].population[rank]
                        self.de[next_budget].pop_size = pop_size

                        budget = next_budget

                        if self.async_strategy in ['deferred', 'immediate'] and \
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

        return np.array(self.traj), np.array(self.runtime), np.array(self.history)


class DEHB_1(DEHB_0):
    def __init__(self, async_strategy='immediate', **kwargs):
        super().__init__(**kwargs)
        self.async_strategy = async_strategy


class DEHB_2(DEHB_0):
    def __init__(self, async_strategy='random', **kwargs):
        super().__init__(**kwargs)
        self.async_strategy = async_strategy


class DEHB_3(DEHB_0):
    def __init__(self, async_strategy='worst', **kwargs):
        super().__init__(**kwargs)
        self.async_strategy = async_strategy


# class DEHB_1(DEHBBase):
#     '''DEHB with Async. DE with changes to max_pop_size, etc. -- sort
#     '''
#     def __init__(self, async_strategy='orig', **kwargs):
#         super().__init__(**kwargs)
#         self.max_age = np.inf
#         self.min_clip = 0
#         self.async_strategy = async_strategy
#         self.async_strategy = 'orig'  # {'random', 'basic'}
#
#         self.reset()
#         self._get_pop_sizes()
#         self._init_subpop()
#
#     def reset(self):
#         super().reset()
#         self.de = {}
#
#     def _get_pop_sizes(self):
#         '''Determines maximum pop size for each budget
#         '''
#         self._max_pop_size = {}
#         for i in range(self.max_SH_iter):
#             n, r = self.get_next_iteration(i)
#             for j, r_j in enumerate(r):
#                 if i == 0:
#                     self._max_pop_size[r_j] = 0
#                 self._max_pop_size[r_j] += n[j]
#
#     def _init_subpop(self):
#         # List of DE objects corresponding to the budgets (fidelities)
#         self.de = {}
#         for i, b in enumerate(self._max_pop_size.keys()):
#             self.de[b] = AsyncDE(cs=self.cs, f=self.f, dimensions=self.dimensions,
#                             pop_size=self._max_pop_size[b], mutation_factor=self.mutation_factor,
#                             crossover_prob=self.crossover_prob, strategy=self.strategy,
#                             budget=b, max_age=self.max_age)
#
#     def _concat_pops(self, exclude_budget=None):
#         '''Concatenates all subpopulations
#         '''
#         budgets = list(self.budgets)
#         if exclude_budget is not None:
#             budgets.remove(exclude_budget)
#         pop = []
#         for budget in budgets:
#             pop.extend(self.de[budget].population.tolist())
#         return np.array(pop)
#
#     def _update_trackers(self, inc_score, inc_config, traj, runtime, history, budget):
#         self.inc_score = self.de[budget].inc_score
#         self.inc_config = self.de[budget].inc_config
#         self.traj.extend(traj)
#         self.runtime.extend(runtime)
#         self.history.extend(history)
#
#     def run(self, iterations=1, verbose=False, debug=False, reset=True):
#         # Book-keeping variables
#         if reset:
#             self.reset()
#             self._init_subpop()
#
#         # Performs DEHB iterations
#         for iteration in range(iterations):
#             # Retrieves budgets and number of configurations for the SH bracket
#             num_configs, budgets = self.get_next_iteration(iteration=iteration)
#             if verbose:
#                 print('Iteration #{:>3}\n{}'.format(iteration, '-' * 15))
#                 print(num_configs, budgets, self.inc_score)
#
#             # Sets budget and population size for first iteration in the SH bracket
#             pop_size = num_configs[0]
#             budget = budgets[0]
#             self.de[budget].pop_size = pop_size
#
#             # Number of SH iterations in this DEHB iteration
#             num_SH_iters = len(budgets)
#
#             if iteration == 0:  # first HB bracket's first iteration (first SH bracket)
#                 for i_sh in range(num_SH_iters):
#                     # warmstart DE incumbents with global incumbents
#                     ## significant for iteration==0 when DEHB optimisation is continued
#                     self.de[budget].inc_score = self.inc_score
#                     self.de[budget].inc_config = self.inc_config
#                     if i_sh == 0:
#                         # initializes population and evaluates them on the 'budget'
#                         # evaluations are counted as function evaluations for this iteration
#                         de_traj, de_runtime, de_history = self.de[budget].init_eval_pop(budget)
#                     else:
#                         # population is evaluated for pop_size on the 'budget'
#                         # pop_size can be < len(population)
#                         de_traj, de_runtime, de_history, _, _ = \
#                                 self.de[budget].eval_pop(budget=budget)
#
#                     self._update_trackers(self.de[budget].inc_score, self.de[budget].inc_config,
#                                           de_traj, de_runtime, de_history, budget)
#
#                     if i_sh < (num_SH_iters - 1):  # when not final SH iteration
#                         pop_size = num_configs[i_sh + 1]
#                         next_budget = budgets[i_sh + 1]
#                         # finding the top individuals for the pop_size required
#                         rank = np.sort(np.argsort(self.de[budget].fitness)[:pop_size])
#
#                         # initializing the required pop_size for the higher budget populations
#                         ## remaining population slots to be filled in subsequent iterations
#                         self.de[next_budget].population = self.de[budget].population[rank]
#                         self.de[next_budget].fitness = self.de[budget].fitness[rank]
#                         self.de[next_budget].age = self.de[budget].age[rank]
#                         # print("Budget: ", next_budget, "Age: ", self.de[next_budget].age)
#                         self.de[next_budget].pop_size = pop_size
#
#                         # updating budget for next SH step
#                         budget = next_budget
#
#             elif iteration < self.max_SH_iter:  # first HB bracket, second iteration onwards
#                 for i_sh in range(num_SH_iters):
#                     # warmstart DE incumbents with global incumbents
#                     self.de[budget].inc_score = self.inc_score
#                     self.de[budget].inc_config = self.inc_config
#
#                     if i_sh == 0:  # first iteration in the SH bracket
#                         if debug:
#                             print("Adding {} to {}".format(self.de[budget].pop_size, budget))
#                         # Adds to the subpopulation
#                         population = self.de[budget].population
#                         fitness = self.de[budget].fitness
#                         age = self.de[budget].age
#                         de_traj, de_runtime, de_history = self.de[budget].init_eval_pop(budget)
#                         self.de[budget].population = np.concatenate((self.de[budget].population,
#                                                                      population))
#                         self.de[budget].fitness = np.concatenate((self.de[budget].fitness,
#                                                                      fitness))
#                         self.de[budget].age = np.concatenate((self.de[budget].age,
#                                                                      age))
#                     else:
#                         if alt_population is None:
#                             if debug:
#                                 print("Evaluating {} on {}".format(self.de[budget].pop_size,
#                                                                     budget))
#                             de_traj, de_runtime, de_history, _, _ = \
#                                 self.de[budget].eval_pop(budget=budget)
#                         else:
#                             if debug:
#                                 print("Evolving {} on {}".format(self.de[budget].pop_size, budget))
#
#                             de_traj, de_runtime, de_history = \
#                                 self.de[budget].evolve_generation(budget=budget,
#                                                                   best=self.inc_config,
#                                                                   alt_pop=alt_population,
#                                                                   async_strategy=self.async_strategy)
#
#                     self._update_trackers(self.de[budget].inc_score, self.de[budget].inc_config,
#                                           de_traj, de_runtime, de_history, budget)
#
#                     if i_sh < (num_SH_iters - 1):  # when not final SH iteration
#                         pop_size = num_configs[i_sh + 1]
#                         next_budget = budgets[i_sh + 1]
#                         # finding the top individuals for the pop_size required
#                         rank = np.sort(np.argsort(self.de[budget].fitness)[:pop_size])
#                         if len(self.de[next_budget].population) < self._max_pop_size[next_budget]:
#                             # appending top individuals from the lower budget as part of next_budget
#                             ## population of pop_size is appended to the front so they are evaluated
#                             ## in the next iteration
#                             self.de[next_budget].population = \
#                                 np.concatenate((self.de[budget].population[rank],
#                                                 self.de[next_budget].population))
#                             self.de[next_budget].fitness = \
#                                 np.concatenate((self.de[budget].fitness[rank],
#                                                 self.de[next_budget].fitness))
#                             self.de[next_budget].age = \
#                                 np.concatenate((self.de[budget].age[rank],
#                                                 self.de[next_budget].age))
#                             alt_population = None
#                         else:
#                             # the top individuals from the current budget are the candidates for
#                             ## mutation in the next higher budget
#                             alt_population = self.de[budget].population[rank]
#                             self.de[next_budget]._sort_pop()
#
#                         self.de[next_budget].pop_size = pop_size
#                         budget = next_budget
#
#             else:  # second HB bracket onwards (DEHB brackets)
#                 alt_population = None
#                 for i_sh in range(num_SH_iters):
#                     # warmstart DE with global incumbents
#                     self.de[budget].inc_score = self.inc_score
#                     self.de[budget].inc_config = self.inc_config
#                     if debug:
#                         print("Evolving {} for {}".format(self.de[budget].pop_size, budget))
#
#                     # when the size of mutation candidate population is lesser than that required
#                     ## for the chosen mutation strategy, new individuals of infinite fitness are
#                     ## introduced by creating mutants from the total global population formed
#                     ## by concatenating all the subpopulations associated with all the budgets
#                     if alt_population is not None and \
#                             len(alt_population) < self.de[budget]._min_pop_size:
#                         filler = self.de[budget]._min_pop_size - len(alt_population) + 1
#                         if debug:
#                             print("Adding {} individuals for mutation on "
#                                   "budget {}".format(filler, budget))
#                         new_pop = \
#                             self.de[budget]._init_mutant_population(filler, self._concat_pops(),
#                                                                     target=None,
#                                                                     best=self.inc_config)
#                         alt_population = np.concatenate((alt_population, new_pop))
#                         if debug:
#                             print("Mutation population size: {}".format(filler, budget))
#
#                     # evolving subpopulation on 'budget' for one generation
#                     ## the targets in te evolution process are the individuals themselves
#                     ## the mutants are created from the alt_population that is passed
#                     de_traj, de_runtime, de_history = \
#                             self.de[budget].evolve_generation(budget=budget,
#                                                               best=self.inc_config,
#                                                               alt_pop=alt_population,
#                                                               async_strategy=self.async_strategy)
#
#                     self._update_trackers(self.de[budget].inc_score, self.de[budget].inc_config,
#                                           de_traj, de_runtime, de_history, budget)
#
#                     if i_sh < (num_SH_iters - 1):  # when not final SH iteration
#                         pop_size = num_configs[i_sh + 1]
#                         next_budget = budgets[i_sh + 1]
#                         rank = np.sort(np.argsort(self.de[budget].fitness)[:pop_size])
#                         # the top individuals from the current 'budget' serve as mutation
#                         ## candidates for the DE evolution in the higher next_budget
#                         alt_population = self.de[budget].population[rank]
#                         self.de[next_budget].pop_size = pop_size
#
#                         budget = next_budget
#
#                         self.de[next_budget]._sort_pop()
#
#         return np.array(self.traj), np.array(self.runtime), np.array(self.history)
#
#
# class DEHB_2(DEHBBase):
#     '''DEHB with Async. DE with changes to max_pop_size, etc. -- shuffle
#     '''
#     def __init__(self, async_strategy='orig', **kwargs):
#         super().__init__(**kwargs)
#         self.max_age = np.inf
#         self.min_clip = 0
#         self.async_strategy = async_strategy
#         self.async_strategy = 'orig'  # {'random', 'basic'}
#
#         self.reset()
#         self._get_pop_sizes()
#         self._init_subpop()
#
#     def reset(self):
#         super().reset()
#         self.de = {}
#
#     def _get_pop_sizes(self):
#         '''Determines maximum pop size for each budget
#         '''
#         self._max_pop_size = {}
#         for i in range(self.max_SH_iter):
#             n, r = self.get_next_iteration(i)
#             for j, r_j in enumerate(r):
#                 if i == 0:
#                     self._max_pop_size[r_j] = 0
#                 self._max_pop_size[r_j] += n[j]
#
#     def _init_subpop(self):
#         # List of DE objects corresponding to the budgets (fidelities)
#         self.de = {}
#         for i, b in enumerate(self._max_pop_size.keys()):
#             self.de[b] = AsyncDE(cs=self.cs, f=self.f, dimensions=self.dimensions,
#                             pop_size=self._max_pop_size[b], mutation_factor=self.mutation_factor,
#                             crossover_prob=self.crossover_prob, strategy=self.strategy,
#                             budget=b, max_age=self.max_age)
#
#     def _concat_pops(self, exclude_budget=None):
#         '''Concatenates all subpopulations
#         '''
#         budgets = list(self.budgets)
#         if exclude_budget is not None:
#             budgets.remove(exclude_budget)
#         pop = []
#         for budget in budgets:
#             pop.extend(self.de[budget].population.tolist())
#         return np.array(pop)
#
#     def _update_trackers(self, inc_score, inc_config, traj, runtime, history, budget):
#         self.inc_score = self.de[budget].inc_score
#         self.inc_config = self.de[budget].inc_config
#         self.traj.extend(traj)
#         self.runtime.extend(runtime)
#         self.history.extend(history)
#
#     def run(self, iterations=1, verbose=False, debug=False, reset=True):
#         # Book-keeping variables
#         if reset:
#             self.reset()
#             self._init_subpop()
#
#         # Performs DEHB iterations
#         for iteration in range(iterations):
#             # Retrieves budgets and number of configurations for the SH bracket
#             num_configs, budgets = self.get_next_iteration(iteration=iteration)
#             if verbose:
#                 print('Iteration #{:>3}\n{}'.format(iteration, '-' * 15))
#                 print(num_configs, budgets, self.inc_score)
#
#             # Sets budget and population size for first iteration in the SH bracket
#             pop_size = num_configs[0]
#             budget = budgets[0]
#             self.de[budget].pop_size = pop_size
#
#             # Number of SH iterations in this DEHB iteration
#             num_SH_iters = len(budgets)
#
#             if iteration == 0:  # first HB bracket's first iteration (first SH bracket)
#                 for i_sh in range(num_SH_iters):
#                     # warmstart DE incumbents with global incumbents
#                     ## significant for iteration==0 when DEHB optimisation is continued
#                     self.de[budget].inc_score = self.inc_score
#                     self.de[budget].inc_config = self.inc_config
#                     if i_sh == 0:
#                         # initializes population and evaluates them on the 'budget'
#                         # evaluations are counted as function evaluations for this iteration
#                         de_traj, de_runtime, de_history = self.de[budget].init_eval_pop(budget)
#                     else:
#                         # population is evaluated for pop_size on the 'budget'
#                         # pop_size can be < len(population)
#                         de_traj, de_runtime, de_history, _, _ = \
#                                 self.de[budget].eval_pop(budget=budget)
#
#                     self._update_trackers(self.de[budget].inc_score, self.de[budget].inc_config,
#                                           de_traj, de_runtime, de_history, budget)
#
#                     if i_sh < (num_SH_iters - 1):  # when not final SH iteration
#                         pop_size = num_configs[i_sh + 1]
#                         next_budget = budgets[i_sh + 1]
#                         # finding the top individuals for the pop_size required
#                         rank = np.sort(np.argsort(self.de[budget].fitness)[:pop_size])
#
#                         # initializing the required pop_size for the higher budget populations
#                         ## remaining population slots to be filled in subsequent iterations
#                         self.de[next_budget].population = self.de[budget].population[rank]
#                         self.de[next_budget].fitness = self.de[budget].fitness[rank]
#                         self.de[next_budget].age = self.de[budget].age[rank]
#                         # print("Budget: ", next_budget, "Age: ", self.de[next_budget].age)
#                         self.de[next_budget].pop_size = pop_size
#
#                         # updating budget for next SH step
#                         budget = next_budget
#
#             elif iteration < self.max_SH_iter:  # first HB bracket, second iteration onwards
#                 for i_sh in range(num_SH_iters):
#                     # warmstart DE incumbents with global incumbents
#                     self.de[budget].inc_score = self.inc_score
#                     self.de[budget].inc_config = self.inc_config
#
#                     if i_sh == 0:  # first iteration in the SH bracket
#                         if debug:
#                             print("Adding {} to {}".format(self.de[budget].pop_size, budget))
#                         # Adds to the subpopulation
#                         population = self.de[budget].population
#                         fitness = self.de[budget].fitness
#                         age = self.de[budget].age
#                         de_traj, de_runtime, de_history = self.de[budget].init_eval_pop(budget)
#                         self.de[budget].population = np.concatenate((self.de[budget].population,
#                                                                      population))
#                         self.de[budget].fitness = np.concatenate((self.de[budget].fitness,
#                                                                      fitness))
#                         self.de[budget].age = np.concatenate((self.de[budget].age,
#                                                                      age))
#                     else:
#                         if alt_population is None:
#                             if debug:
#                                 print("Evaluating {} on {}".format(self.de[budget].pop_size,
#                                                                     budget))
#                             de_traj, de_runtime, de_history, _, _ = \
#                                 self.de[budget].eval_pop(budget=budget)
#                         else:
#                             if debug:
#                                 print("Evolving {} on {}".format(self.de[budget].pop_size, budget))
#
#                             de_traj, de_runtime, de_history = \
#                                 self.de[budget].evolve_generation(budget=budget,
#                                                                   best=self.inc_config,
#                                                                   alt_pop=alt_population,
#                                                                   async_strategy=self.async_strategy)
#
#                     self._update_trackers(self.de[budget].inc_score, self.de[budget].inc_config,
#                                           de_traj, de_runtime, de_history, budget)
#
#                     if i_sh < (num_SH_iters - 1):  # when not final SH iteration
#                         pop_size = num_configs[i_sh + 1]
#                         next_budget = budgets[i_sh + 1]
#                         # finding the top individuals for the pop_size required
#                         rank = np.sort(np.argsort(self.de[budget].fitness)[:pop_size])
#                         if len(self.de[next_budget].population) < self._max_pop_size[next_budget]:
#                             # appending top individuals from the lower budget as part of next_budget
#                             ## population of pop_size is appended to the front so they are evaluated
#                             ## in the next iteration
#                             self.de[next_budget].population = \
#                                 np.concatenate((self.de[budget].population[rank],
#                                                 self.de[next_budget].population))
#                             self.de[next_budget].fitness = \
#                                 np.concatenate((self.de[budget].fitness[rank],
#                                                 self.de[next_budget].fitness))
#                             self.de[next_budget].age = \
#                                 np.concatenate((self.de[budget].age[rank],
#                                                 self.de[next_budget].age))
#                             alt_population = None
#                         else:
#                             # the top individuals from the current budget are the candidates for
#                             ## mutation in the next higher budget
#                             alt_population = self.de[budget].population[rank]
#                             self.de[next_budget]._shuffle_pop()
#
#                         self.de[next_budget].pop_size = pop_size
#                         budget = next_budget
#
#             else:  # second HB bracket onwards (DEHB brackets)
#                 alt_population = None
#                 for i_sh in range(num_SH_iters):
#                     # warmstart DE with global incumbents
#                     self.de[budget].inc_score = self.inc_score
#                     self.de[budget].inc_config = self.inc_config
#                     if debug:
#                         print("Evolving {} for {}".format(self.de[budget].pop_size, budget))
#
#                     # when the size of mutation candidate population is lesser than that required
#                     ## for the chosen mutation strategy, new individuals of infinite fitness are
#                     ## introduced by creating mutants from the total global population formed
#                     ## by concatenating all the subpopulations associated with all the budgets
#                     if alt_population is not None and \
#                             len(alt_population) < self.de[budget]._min_pop_size:
#                         filler = self.de[budget]._min_pop_size - len(alt_population) + 1
#                         if debug:
#                             print("Adding {} individuals for mutation on "
#                                   "budget {}".format(filler, budget))
#                         new_pop = \
#                             self.de[budget]._init_mutant_population(filler, self._concat_pops(),
#                                                                     target=None,
#                                                                     best=self.inc_config)
#                         alt_population = np.concatenate((alt_population, new_pop))
#                         if debug:
#                             print("Mutation population size: {}".format(filler, budget))
#
#                     # evolving subpopulation on 'budget' for one generation
#                     ## the targets in te evolution process are the individuals themselves
#                     ## the mutants are created from the alt_population that is passed
#                     de_traj, de_runtime, de_history = \
#                             self.de[budget].evolve_generation(budget=budget,
#                                                               best=self.inc_config,
#                                                               alt_pop=alt_population,
#                                                               async_strategy=self.async_strategy)
#
#                     self._update_trackers(self.de[budget].inc_score, self.de[budget].inc_config,
#                                           de_traj, de_runtime, de_history, budget)
#
#                     if i_sh < (num_SH_iters - 1):  # when not final SH iteration
#                         pop_size = num_configs[i_sh + 1]
#                         next_budget = budgets[i_sh + 1]
#                         rank = np.sort(np.argsort(self.de[budget].fitness)[:pop_size])
#                         # the top individuals from the current 'budget' serve as mutation
#                         ## candidates for the DE evolution in the higher next_budget
#                         alt_population = self.de[budget].population[rank]
#                         self.de[next_budget].pop_size = pop_size
#
#                         budget = next_budget
#
#                         self.de[budget]._shuffle_pop()
#
#         return np.array(self.traj), np.array(self.runtime), np.array(self.history)
