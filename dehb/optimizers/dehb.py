import numpy as np

from .de import DE


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


class DEHBV1(DEHBBase):
    '''Version 1.0 of DEHB

    Each DEHB iteration is initialized with a new random population.
    In each of the DEHB iteration, Successive Halving (SH) takes place where
        the number of SH iterations, budget spacing, number of configurations,
        are determined dynamically based on the iteration number.
        The top performing individuals are carried forward to the next higher budget.
    Each SH iteration in each DEHB iteration is evolved for a certain number of generations.
    '''
    def __init__(self, max_age=np.inf, **kwargs):
        super().__init__(**kwargs)
        self.max_age = max_age

    def run(self, iterations=1, verbose=False, debug=False):
        # Book-keeping variables
        traj = []
        runtime = []
        history = []

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

            # Number of SH iterations in this DEHB iteration
            num_SH_iters = len(budgets)

            # Initializing DE object that will be used across this DEHB iteration
            # The DE object is initialized with the current pop_size and budget
            de = DE(cs=self.cs, f=self.f, dimensions=self.dimensions, pop_size=pop_size,
                    mutation_factor=self.mutation_factor, crossover_prob=self.crossover_prob,
                    strategy=self.strategy, budget=budget, max_age=self.max_age)

            # Warmstarting DE incumbent to be the global incumbent
            de.inc_score = self.inc_score
            de.inc_config = self.inc_config

            # Creating new population for current DEHB iteration
            de_traj, de_runtime, de_history = de.init_eval_pop(budget)
            traj.extend(de_traj)
            runtime.extend(de_runtime)
            history.extend(de_history)

            # Incorporating global incumbent into the new DE population
            if np.array_equal(de.inc_config, self.inc_config):
                # if new population has no better individual, randomly
                # replace an individual with the incumbent so far
                idx = np.random.choice(np.arange(len(de.population)))
                de.population[idx] = self.inc_config
                de.fitness[idx] = self.inc_score
                de.age[idx] = de.max_age
            else:
                # if new population has a better individual, update
                # the global incumbent and fitness
                self.inc_score = de.inc_score
                self.inc_config = de.inc_config

            # Successive Halving iterations
            for i_sh in range(num_SH_iters):
                # Repeating DE over entire population 'generations' times
                for gen in range(self.generations):
                    de_traj, de_runtime, de_history = de.evolve_generation(budget=budget,
                                                                           best=de.inc_config)
                    traj.extend(de_traj)
                    runtime.extend(de_runtime)
                    history.extend(de_history)

                    # killing/replacing parents that have not changed/has aged
                    # conditional on max_age not being set to inf
                    if self.max_age < np.inf:
                        if debug:
                            print("  Generation #{}: Ages -- {}".format(gen + 1, de.age))
                        de_traj, de_runtime, de_history = de.kill_aged_pop(budget, debug)
                        traj.extend(de_traj)
                        runtime.extend(de_runtime)
                        history.extend(de_history)

                # Updating global incumbent after each DE step
                self.inc_score = de.inc_score
                self.inc_config = de.inc_config

                # Retrieving budget, pop_size, population for the next SH iteration
                if i_sh < num_SH_iters - 1:  # when not final SH iteration
                    pop_size = num_configs[i_sh + 1]
                    budget = budgets[i_sh + 1]
                    # Selecting top individuals to fit pop_size of next SH iteration
                    self.rank = np.sort(np.argsort(de.fitness)[:pop_size])
                    de.population = de.population[self.rank]
                    de.fitness = de.fitness[self.rank]
                    de.age = de.age[self.rank]
                    de.pop_size = pop_size

        if verbose:
            print("\nRun complete!")

        return (np.array(traj), np.array(runtime), np.array(history))


class DEHBV1_1(DEHBBase):
    '''Version 1.1 of DEHB

    Each DEHB iteration is initialized with a new random population.
    In each of the DEHB iteration, Successive Halving (SH) takes place where
        the number of SH iterations, budget spacing, number of configurations,
        are determined dynamically based on the iteration number.
        The top performing individuals are carried forward to the next higher budget.
    Each SH iteration in each DEHB iteration is evolved for a certain number of generations.
    The population from the highest budget is collected in a global population.
    After the first such DEHB iteration, individuals for mutation are sampled from
        the global population.
    '''
    def __init__(self, max_age=np.inf, **kwargs):
        super().__init__(**kwargs)
        self.max_age = max_age

    def run(self, iterations=1, verbose=False, debug=False):
        # Book-keeping variables
        traj = []
        runtime = []
        history = []

        # Specifying details for the global full budget population
        global_pop_size = 0
        for i in range(self.max_SH_iter):
            num_configs, _ = self.get_next_iteration(iteration=i)
            global_pop_size += num_configs[-1]
        self.global_pop = np.array([None] * (global_pop_size))
        self.global_fitness = np.array([None] * (global_pop_size))

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

            # Number of SH iterations in this DEHB iteration
            num_SH_iters = len(budgets)

            # Initializing DE object that will be used across this DEHB iteration
            # The DE object is initialized with the current pop_size and budget
            de = DE(cs=self.cs, f=self.f, dimensions=self.dimensions, pop_size=pop_size,
                    mutation_factor=self.mutation_factor, crossover_prob=self.crossover_prob,
                    strategy=self.strategy, budget=budget, max_age=self.max_age)

            # Warmstarting DE incumbent to be the global incumbent
            de.inc_score = self.inc_score
            de.inc_config = self.inc_config

            # Creating new population for current DEHB iteration
            de_traj, de_runtime, de_history = de.init_eval_pop(budget=budget)
            traj.extend(de_traj)
            runtime.extend(de_runtime)
            history.extend(de_history)

            # Incorporating global incumbent into the new DE population
            if np.array_equal(de.inc_config, self.inc_config):
                # if new population has no better individual, randomly
                # replace an individual with the incumbent so far
                idx = np.random.choice(np.arange(len(de.population)))
                de.population[idx] = self.inc_config
                de.fitness[idx] = self.inc_score
                de.age[idx] = de.max_age
            else:
                # if new population has a better individual, update
                # the global incumbent and fitness
                self.inc_score = de.inc_score
                self.inc_config = de.inc_config

            # Successive Halving iterations
            for i_sh in range(num_SH_iters):
                # Repeating DE over entire population 'generations' times
                for gen in range(self.generations):
                    de_traj, de_runtime, de_history = de.evolve_generation(budget=budget,
                                                                           best=de.inc_config,
                                                                           alt_pop=self.global_pop)
                    traj.extend(de_traj)
                    runtime.extend(de_runtime)
                    history.extend(de_history)

                    # killing/replacing parents that have not changed/has aged
                    # conditional on max_age not being set to inf
                    if self.max_age < np.inf:
                        if debug:
                            print("  Generation #{}: Ages -- {}".format(gen + 1, de.age))
                        de_traj, de_runtime, de_history = de.kill_aged_pop(budget, debug)
                        traj.extend(de_traj)
                        runtime.extend(de_runtime)
                        history.extend(de_history)

                # Updating global incumbent after each DE evolution step
                self.inc_score = de.inc_score
                self.inc_config = de.inc_config

                # Retrieving budget, pop_size, population for the next SH iteration
                if i_sh < num_SH_iters - 1:  # when not final SH iteration
                    pop_size = num_configs[i_sh + 1]
                    budget = budgets[i_sh + 1]
                    # Selecting top individuals to fit pop_size of next SH iteration
                    self.rank = np.sort(np.argsort(de.fitness)[:pop_size])
                    de.population = de.population[self.rank]
                    de.fitness = de.fitness[self.rank]
                    de.age = de.age[self.rank]
                    de.pop_size = pop_size
                else:  # final SH iteration with max budget
                    pop_size = num_configs[-1]
                    idx = [indv is None for indv in self.global_pop]
                    if any(idx):  # None exists in global population
                        # Initialize global population with positions assigned randomly
                        idx = np.where(np.array(idx) == True)[0]
                        idx = np.random.choice(idx, pop_size, replace=False)
                    else:
                        self.global_pop = np.stack(self.global_pop)
                        self.global_fitness = np.stack(self.global_fitness)
                        # Find the weakest individuals in the global population to replace them
                        idx = np.sort(np.argsort(-self.global_fitness)[:pop_size])
                    for index, id in enumerate(idx):
                        # Updating global pop
                        self.global_pop[id] = de.population[index]
                        self.global_fitness[id] = de.fitness[index]

        if verbose:
            print("\nRun complete!")

        return (np.array(traj), np.array(runtime), np.array(history))


class DEHBV2(DEHBBase):
    '''Version 2 of DEHB

    Only the first DEHB iteration is initialized with a new random population.
    In each of the DEHB iteration, Successive Halving (SH) takes place where
        the number of SH iterations, budget spacing, number of configurations,
        are determined dynamically based on the iteration number.
        The top performing individuals are carried forward to the next higher budget.
    Each SH iteration in each DEHB iteration is evolved for only one generation,
        using the best individuals from the evolved population from previous iteration.
    '''
    def __init__(self, max_age=np.inf, randomize=None, **kwargs):
        super().__init__(**kwargs)
        self.randomize = randomize
        self.max_age = max_age

    def run(self, iterations=1, verbose=False, debug=False):
        # Book-keeping variables
        traj = []
        runtime = []
        history = []

        # To retrieve the maximal pop_size and initialize a single DE object for all DEHB runs
        num_configs, budgets = self.get_next_iteration(iteration=0)
        de = DE(cs=self.cs, f=self.f, dimensions=self.dimensions, pop_size=num_configs[0],
                mutation_factor=self.mutation_factor, crossover_prob=self.crossover_prob,
                strategy=self.strategy, budget=budgets[0], max_age=self.max_age)

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
            # Number of SH iterations in this DEHB iteration
            num_SH_iters = len(budgets)

            # The first DEHB iteration - only time when a random population is initialized
            if iteration == 0:
                # creating new population for DEHB iteration to be used for the next SH steps
                de_traj, de_runtime, de_history = de.init_eval_pop(budget)
                # maintaining global copy of random population created
                self.population = de.population
                self.fitness = de.fitness
                self.age = de.age
                # update global incumbent with new population scores
                self.inc_score = de.inc_score
                self.inc_config = de.inc_config
                traj.extend(de_traj)
                runtime.extend(de_runtime)
                history.extend(de_history)
            elif pop_size == len(self.population) and \
                 self.randomize is not None and self.randomize > 0:
                # executes in the first step of every SH iteration other than first DEHB iteration
                # also conditional on whether a randomization fraction has been specified
                num_replace = np.ceil(self.randomize * pop_size).astype(int)
                # fetching the worst performing individuals
                idxs = np.sort(np.argsort(-self.fitness)[:num_replace])
                if debug:
                    print("Replacing {}/{} -- {}".format(num_replace, pop_size, idxs))
                new_pop = self.init_population(pop_size=num_replace)
                self.population[idxs] = new_pop
                self.age[idxs] = de.max_age
                de.inc_score = self.inc_score
                de.inc_config = self.inc_config
                # evaluating new individuals
                for i in idxs:
                    fitness, cost = de.f_objective(self.population[i], budget)
                    self.fitness[i] = fitness
                    if self.fitness[i] < self.inc_score:
                        self.inc_score = self.fitness[i]
                        self.inc_config = self.population[i]
                    traj.append(self.inc_score)
                    runtime.append(cost)
                    history.append((self.population[i].tolist(),
                                    float(self.fitness[i]), float(budget or 0)))

            # Ranking current population
            self.rank = np.sort(np.argsort(self.fitness)[:pop_size])
            # Passing onto DE-SH steps a subset of top individuals from global population
            de.population = self.population[self.rank]
            de.fitness = np.array(self.fitness)[self.rank]
            de.age = np.array(self.age)[self.rank]
            de.pop_size = pop_size

            # Successive Halving iterations carrying out DE
            for i_sh in range(num_SH_iters):
                if debug:
                    print(i_sh, self.rank)
                # Repeating DE over entire population 'generations' times
                for gen in range(self.generations):
                    de_traj, de_runtime, de_history = de.evolve_generation(budget)
                    traj.extend(de_traj)
                    runtime.extend(de_runtime)
                    history.extend(de_history)
                    if debug:
                        print("  Generation #{}: Ages -- {}".format(gen+1, de.age))
                    # killing/replacing parents that have not changed/has aged
                    de_traj, de_runtime, de_history = de.kill_aged_pop(budget, debug)
                    traj.extend(de_traj)
                    runtime.extend(de_runtime)
                    history.extend(de_history)

                # Updating global incumbent after each DE step
                self.inc_score = de.inc_score
                self.inc_config = de.inc_config
                # Updating global population with evolved individuals
                self.population[self.rank] = de.population
                self.fitness[self.rank] = de.fitness
                self.age[self.rank] = de.age

                # Retrieving budget, pop_size, population for the next SH iteration
                if i_sh < num_SH_iters-1:  # when not final SH iteration
                    pop_size = num_configs[i_sh+1]
                    budget = budgets[i_sh+1]
                    # Selecting top individuals to fit pop_size of next SH iteration
                    self.de_rank = np.sort(np.argsort(de.fitness)[:pop_size])
                    # Saving index of new DE population from the global population in 'self.rank'
                    self.rank = self.rank[self.de_rank]
                    de.population = de.population[self.de_rank]
                    de.fitness = np.array(de.fitness)[self.de_rank]
                    de.age = np.array(de.age)[self.de_rank]
                    de.pop_size = pop_size

        if verbose:
            print("\nRun complete!")

        return (np.array(traj), np.array(runtime), np.array(history))


class DEHBV3(DEHBBase):
    '''Version 3.0 of DEHB

    At anytime, each set of population contains the best individuals from that budget
    '''
    def __init__(self, randomize=None, max_age=np.inf, **kwargs):
        super().__init__(**kwargs)
        self.randomize = randomize
        self.max_age = max_age

    def run(self, iterations=1, verbose=False, debug=False):
        # Book-keeping variables
        traj = []
        runtime = []
        history = []

        # To retrieve the population and budget ranges
        num_configs, budgets = self.get_next_iteration(iteration=0)
        full_budget = budgets[-1]
        small_budget = budgets[0]

        # List of DE objects corresponding to the populations
        de = {}
        for i, b in enumerate(budgets):
            de[b] = DE(cs=self.cs, f=self.f, dimensions=self.dimensions, pop_size=num_configs[i],
                       mutation_factor=self.mutation_factor, crossover_prob=self.crossover_prob,
                       strategy=self.strategy, budget=b, max_age=self.max_age)

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

            # Number of SH iterations in this DEHB iteration
            num_SH_iters = len(budgets)

            # Index determining the population to begin DE with for current iteration number
            de_idx = iteration % len(de)

            # The first DEHB iteration - only time when a random population is initialized
            if iteration == 0:
                # creating new population for DEHB iteration to be used for the next SH steps
                de_traj, de_runtime, de_history = de[budget].init_eval_pop(budget)
                # maintaining global copy of random population created
                self.population = de[budget].population
                self.fitness = de[budget].fitness
                # update global incumbent with new population scores
                self.inc_score = de[budget].inc_score
                self.inc_config = de[budget].inc_config
                traj.extend(de_traj)
                runtime.extend(de_runtime)
                history.extend(de_history)

            elif budget == small_budget and self.randomize is not None and self.randomize > 0:
                # executes in the first step of every SH iteration other than first DEHB iteration
                # also conditional on whether a randomization fraction has been specified
                num_replace = np.ceil(self.randomize * pop_size).astype(int)

                # fetching the worst performing individuals
                idxs = np.sort(np.argsort(-de[budget].fitness)[:num_replace])
                if debug:
                    print("Replacing {}/{} -- {}".format(num_replace, pop_size, idxs))
                new_pop = self.init_population(pop_size=num_replace)
                de[budget].population[idxs] = new_pop
                de[budget].age[idxs] = de[budget].max_age
                de[budget].inc_score = self.inc_score
                de[budget].inc_config = self.inc_config

                # evaluating new individuals
                for i in idxs:
                    de[budget].fitness[i], cost = \
                        de[budget].f_objective(de[budget].population[i], budget)
                    if self.fitness[i] < self.inc_score:
                        self.inc_score = de[budget].fitness[i]
                        self.inc_config = de[budget].population[i]
                    traj.append(self.inc_score)
                    runtime.append(cost)
                    history.append((de[budget].population[i].tolist(),
                                    float(de[budget].fitness[i]), float(budget or 0)))

            elif pop_size > de[budget].pop_size:
                # compensating for extra individuals if SH pop size is larger
                filler = pop_size - de[budget].pop_size
                if debug:
                    print("Adding {} random individuals for budget {}".format(filler, budget))
                new_pop = self.init_population(pop_size=filler)
                de[budget].inc_score = self.inc_score
                de[budget].inc_config = self.inc_config
                de[budget].age = np.hstack((de[budget].age, [de[budget].max_age] * filler))
                for i in range(filler):
                    fitness, cost = de[budget].f_objective(new_pop[i], budget)
                    de[budget].population = np.vstack((de[budget].population, new_pop[i]))
                    de[budget].fitness = np.append(de[budget].fitness, fitness)
                    if fitness < self.inc_score:
                        self.inc_score = fitness
                        self.inc_config = new_pop[i]
                    traj.append(self.inc_score)
                    runtime.append(cost)
                    history.append((new_pop[i].tolist(), float(fitness), float(budget or 0)))

                de[budget].inc_score = self.inc_score
                de[budget].inc_config = self.inc_config
                de[budget].pop_size = pop_size
                if debug:
                    print("Pop size: {}; Len pop: {}".format(de[budget].pop_size,
                                                             len(de[budget].population)))

            elif pop_size < de[budget].pop_size:
                # compensating for extra individuals if SH pop size is smaller
                if debug:
                    print("Reducing population from {} to {} "
                          "for budget {}".format(de[budget].pop_size, pop_size, budget))
                # keeping only the top performing individuals/discarding the weak ones
                rank = np.sort(np.argsort(de[budget].fitness)[:pop_size])
                de[budget].population = de[budget].population[rank]
                de[budget].fitness = de[budget].fitness[rank]
                de[budget].age = de[budget].age[rank]
                de[budget].pop_size = pop_size

            # Successive Halving iterations carrying out DE
            for i_sh in range(num_SH_iters):

                # Warmstarting DE incumbent
                de[budget].inc_score = self.inc_score
                de[budget].inc_config = self.inc_config
                if debug:
                    print("Pop size: {}; DE budget: {}".format(de[budget].pop_size, budget))
                best = self.inc_config

                # Repeating DE over entire population 'generations' times
                for gen in range(self.generations):
                    de_traj, de_runtime, de_history = \
                        de[budget].evolve_generation(budget=budget, best=best)
                    traj.extend(de_traj)
                    runtime.extend(de_runtime)
                    history.extend(de_history)

                    # killing/replacing parents that have not changed/has aged
                    # conditional on max_age not being set to inf
                    if de[budget].max_age < np.inf:
                        if debug:
                            print("  Generation #{}: Ages -- {}".format(gen + 1, de[budget].age))
                        de_traj, de_runtime, de_history = de[budget].kill_aged_pop(budget, debug)
                        traj.extend(de_traj)
                        runtime.extend(de_runtime)
                        history.extend(de_history)

                # Updating global incumbent after each DE step
                self.inc_score = de[budget].inc_score
                self.inc_config = de[budget].inc_config

                # Retrieving budget, pop_size, population for the next SH iteration
                if i_sh < num_SH_iters-1:  # when not final SH iteration
                    pop_size = num_configs[i_sh + 1]
                    next_budget = budgets[i_sh + 1]
                    # selecting top ranking individuals from lower budget
                    ## to be evaluated on higher budget and be eligible for competition
                    rank = np.sort(np.argsort(de[budget].fitness)[:pop_size])
                    rival_population = de[budget].population[rank]

                    if de[next_budget].population is not None:
                        # warmstarting DE incumbents to maintain global trajectory
                        de[next_budget].inc_score = self.inc_score
                        de[next_budget].inc_config = self.inc_config

                        # ranking individuals to determine population for next SH step
                        de_traj, de_runtime, de_history = \
                            de[next_budget].ranked_selection(rival_population, pop_size,
                                                             budget, debug)
                        self.inc_score = de[next_budget].inc_score
                        self.inc_config = de[next_budget].inc_config
                        traj.extend(de_traj)
                        runtime.extend(de_runtime)
                        history.extend(de_history)
                    else:
                        # equivalent to iteration == 0
                        # no ranked selection happens, rather top ranked individuals are selected
                        if debug:
                            print("Iteration: ", iteration)
                        de[next_budget].population = rival_population
                        de[next_budget].fitness = de[budget].fitness[rank]
                        de[next_budget].age = np.array([de[next_budget].max_age] * \
                                                        de[next_budget].pop_size)
                    budget = next_budget
        if verbose:
            print("\nRun complete!")

        return (np.array(traj), np.array(runtime), np.array(history))


class DEHBV3_1(DEHBBase):
    '''Version 3.1 of DEHB

    At anytime, each set of population contains the best individuals from that budget
    Maintains a global population of the full budget individuals to sample for mutations
    '''
    def __init__(self, randomize=None, max_age=np.inf, **kwargs):
        super().__init__(**kwargs)
        self.randomize = randomize
        self.max_age = max_age

    def run(self, iterations=1, verbose=False, debug=False):
        # Book-keeping variables
        traj = []
        runtime = []
        history = []

        # Specifying details for the global full budget population
        global_pop_size = 0
        for i in range(self.max_SH_iter):
            num_configs, _ = self.get_next_iteration(iteration=i)
            global_pop_size += num_configs[-1]
        self.global_pop = np.array([None] * (global_pop_size))
        self.global_fitness = np.array([None] * (global_pop_size))

        # To retrieve the population and budget ranges
        num_configs, budgets = self.get_next_iteration(iteration=0)
        full_budget = budgets[-1]
        small_budget = budgets[0]

        # List of DE objects corresponding to the populations
        de = {}
        for i, b in enumerate(budgets):
            de[b] = DE(cs=self.cs, f=self.f, dimensions=self.dimensions, pop_size=num_configs[i],
                       mutation_factor=self.mutation_factor, crossover_prob=self.crossover_prob,
                       strategy=self.strategy, budget=b, max_age=self.max_age)

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

            # Number of SH iterations in this DEHB iteration
            num_SH_iters = len(budgets)

            # Index determining the population to begin DE with for current iteration number
            de_idx = iteration % len(de)

            # The first DEHB iteration - only time when a random population is initialized
            if iteration == 0:
                # creating new population for DEHB iteration to be used for the next SH steps
                de_traj, de_runtime, de_history = de[budget].init_eval_pop(budget)

                # maintaining global copy of random population created
                self.population = de[budget].population
                self.fitness = de[budget].fitness

                # update global incumbent with new population scores
                self.inc_score = de[budget].inc_score
                self.inc_config = de[budget].inc_config
                traj.extend(de_traj)
                runtime.extend(de_runtime)
                history.extend(de_history)

            elif budget == small_budget and self.randomize is not None and self.randomize > 0:
                # executes in the first step of every SH iteration other than first DEHB iteration
                # also conditional on whether a randomization fraction has been specified
                num_replace = np.ceil(self.randomize * pop_size).astype(int)

                # fetching the worst performing individuals
                idxs = np.sort(np.argsort(-de[budget].fitness[:num_replace]))
                if debug:
                    print("Replacing {}/{} -- {}".format(num_replace, pop_size, idxs))
                new_pop = self.init_population(pop_size=num_replace)
                de[budget].population[idxs] = new_pop
                de[budget].age[idxs] = de[budget].max_age
                de[budget].inc_score = self.inc_score
                de[budget].inc_config = self.inc_config

                # evaluating new individuals
                for i in idxs:
                    de[budget].fitness[i], cost = \
                        de[budget].f_objective(de[budget].population[i], budget)
                    if self.fitness[i] < self.inc_score:
                        self.inc_score = de[budget].fitness[i]
                        self.inc_config = de[budget].population[i]
                    traj.append(self.inc_score)
                    runtime.append(cost)
                    history.append((de[budget].population[i].tolist(),
                                    float(de[budget].fitness[i]), float(budget or 0)))

            elif pop_size > de[budget].pop_size:
                # compensating for extra individuals if SH pop size is larger
                filler = pop_size - de[budget].pop_size
                if debug:
                    print("Adding {} random individuals for budget {}".format(filler, budget))
                new_pop = self.init_population(pop_size=filler)
                de[budget].inc_score = self.inc_score
                de[budget].inc_config = self.inc_config
                de[budget].age = np.hstack((de[budget].age, [de[budget].max_age] * filler))
                for i in range(filler):
                    fitness, cost = de[budget].f_objective(new_pop[i], budget)
                    de[budget].population = np.vstack((de[budget].population, new_pop[i]))
                    de[budget].fitness = np.append(de[budget].fitness, fitness)
                    if fitness < self.inc_score:
                        self.inc_score = fitness
                        self.inc_config = new_pop[i]
                    traj.append(self.inc_score)
                    runtime.append(cost)
                    history.append((new_pop[i].tolist(), float(fitness), float(budget or 0)))

                de[budget].inc_score = self.inc_score
                de[budget].inc_config = self.inc_config
                de[budget].pop_size = pop_size
                if debug:
                    print("Pop size: {}; Len pop: {}".format(de[budget].pop_size,
                                                             len(de[budget].population)))

            elif pop_size < de[budget].pop_size:
                # compensating for extra individuals if SH pop size is smaller
                if debug:
                    print("Reducing population from {} to {} "
                          "for budget {}".format(de[budget].pop_size, pop_size, budget))
                # keeping only the top performing individuals/discarding the weak ones
                rank = np.sort(np.argsort(de[budget].fitness)[:pop_size])
                de[budget].population = de[budget].population[rank]
                de[budget].fitness = de[budget].fitness[rank]
                de[budget].age = de[budget].age[rank]
                de[budget].pop_size = pop_size

            # Successive Halving iterations carrying out DE
            for i_sh in range(num_SH_iters):

                # Warmstarting DE incumbent
                de[budget].inc_score = self.inc_score
                de[budget].inc_config = self.inc_config
                if debug:
                    print("Pop size: {}; DE budget: {}".format(de[budget].pop_size, budget))
                best = self.inc_config

                # Repeating DE over entire population 'generations' times
                for gen in range(self.generations):
                    de_traj, de_runtime, de_history = \
                        de[budget].evolve_generation(budget=budget, best=best,
                                                     alt_pop=self.global_pop)
                    traj.extend(de_traj)
                    runtime.extend(de_runtime)
                    history.extend(de_history)

                    # killing/replacing parents that have not changed/has aged
                    # conditional on max_age not being set to inf
                    if de[budget].max_age < np.inf:
                        if debug:
                            print("  Generation #{}: Ages -- {}".format(gen + 1, de[budget].age))
                        de_traj, de_runtime, de_history = de[budget].kill_aged_pop(budget, debug)
                        traj.extend(de_traj)
                        runtime.extend(de_runtime)
                        history.extend(de_history)

                # Updating global incumbent after each DE step
                self.inc_score = de[budget].inc_score
                self.inc_config = de[budget].inc_config

                # Retrieving budget, pop_size, population for the next SH iteration
                if i_sh < num_SH_iters-1:  # when not final SH iteration
                    pop_size = num_configs[i_sh + 1]
                    next_budget = budgets[i_sh + 1]

                    # selecting top ranking individuals from lower budget
                    ## to be evaluated on higher budget and be eligible for competition
                    rank = np.sort(np.argsort(de[budget].fitness)[:pop_size])
                    rival_population = de[budget].population[rank]

                    if de[next_budget].population is not None:
                        # warmstarting DE incumbents to maintain global trajectory
                        de[next_budget].inc_score = self.inc_score
                        de[next_budget].inc_config = self.inc_config

                        # ranking individuals to determine population for next SH step
                        de_traj, de_runtime, de_history = \
                            de[next_budget].ranked_selection(rival_population, pop_size,
                                                             budget, debug)
                        self.inc_score = de[next_budget].inc_score
                        self.inc_config = de[next_budget].inc_config
                        traj.extend(de_traj)
                        runtime.extend(de_runtime)
                        history.extend(de_history)
                    else:
                        # equivalent to iteration == 0
                        # no ranked selection happens, rather top ranked individuals are selected
                        if debug:
                            print("Iteration: ", iteration)
                        de[next_budget].population = rival_population
                        de[next_budget].fitness = de[budget].fitness[rank]
                        de[next_budget].age = np.array([de[next_budget].max_age] * \
                                                        de[next_budget].pop_size)
                    budget = next_budget

                else:  # final SH iteration with max budget
                    pop_size = num_configs[-1]
                    idx = [indv is None for indv in self.global_pop]
                    if any(idx):  # None exists in global population
                        # Initialize global population with positions assigned randomly
                        idx = np.where(np.array(idx) == True)[0]
                        idx = np.random.choice(idx, pop_size, replace=False)
                    else:
                        self.global_pop = np.stack(self.global_pop)
                        self.global_fitness = np.stack(self.global_fitness)
                        # Find the weakest individuals in the global population to replace them
                        idx = np.sort(np.argsort(-self.global_fitness)[:pop_size])
                    for index, id in enumerate(idx):
                        # Updating global pop
                        self.global_pop[id] = de[full_budget].population[index]
                        self.global_fitness[id] = de[full_budget].fitness[index]
        if verbose:
            print("\nRun complete!")

        return (np.array(traj), np.array(runtime), np.array(history))


class DEHBV3_2(DEHBBase):
    '''Version 3.2 of DEHB

    At anytime, each set of population contains the best individuals from that budget
    The top individuals from the population evaluated on the previous budget serves as the
        parents for mutation in the next higher budget
    '''

    def __init__(self, randomize=None, max_age=np.inf, **kwargs):
        super().__init__(**kwargs)
        self.randomize = randomize
        self.max_age = max_age

    def run(self, iterations=1, verbose=False, debug=False):
        # Book-keeping variables
        traj = []
        runtime = []
        history = []

        # To retrieve the population and budget ranges
        num_configs, budgets = self.get_next_iteration(iteration=0)
        full_budget = budgets[-1]
        small_budget = budgets[0]

        # List of DE objects corresponding to the populations
        de = {}
        for i, b in enumerate(budgets):
            de[b] = DE(cs=self.cs, f=self.f, dimensions=self.dimensions, pop_size=num_configs[i],
                       mutation_factor=self.mutation_factor, crossover_prob=self.crossover_prob,
                       strategy=self.strategy, budget=b, max_age=self.max_age)

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

            # Number of SH iterations in this DEHB iteration
            num_SH_iters = len(budgets)

            # Index determining the population to begin DE with for current iteration number
            de_idx = iteration % len(de)

            # The first DEHB iteration - only time when a random population is initialized
            if iteration == 0:
                # creating new population for DEHB iteration to be used for the next SH steps
                de_traj, de_runtime, de_history = de[budget].init_eval_pop(budget)
                # maintaining global copy of random population created
                self.population = de[budget].population
                self.fitness = de[budget].fitness
                # update global incumbent with new population scores
                self.inc_score = de[budget].inc_score
                self.inc_config = de[budget].inc_config
                traj.extend(de_traj)
                runtime.extend(de_runtime)
                history.extend(de_history)

            elif budget == small_budget and self.randomize is not None and self.randomize > 0:
                # executes in the first step of every SH iteration other than first DEHB iteration
                # also conditional on whether a randomization fraction has been specified
                num_replace = np.ceil(self.randomize * pop_size).astype(int)

                # fetching the worst performing individuals
                idxs = np.sort(np.argsort(-de[budget].fitness)[:num_replace])
                if debug:
                    print("Replacing {}/{} -- {}".format(num_replace, pop_size, idxs))
                new_pop = self.init_population(pop_size=num_replace)
                de[budget].population[idxs] = new_pop
                de[budget].age[idxs] = de[budget].max_age
                de[budget].inc_score = self.inc_score
                de[budget].inc_config = self.inc_config

                # evaluating new individuals
                for i in idxs:
                    de[budget].fitness[i], cost = \
                        de[budget].f_objective(de[budget].population[i], budget)
                    if self.fitness[i] < self.inc_score:
                        self.inc_score = de[budget].fitness[i]
                        self.inc_config = de[budget].population[i]
                    traj.append(self.inc_score)
                    runtime.append(cost)
                    history.append((de[budget].population[i].tolist(),
                                    float(de[budget].fitness[i]), float(budget or 0)))

            elif pop_size > de[budget].pop_size:
                # compensating for extra individuals if SH pop size is larger
                filler = pop_size - de[budget].pop_size
                if debug:
                    print("Adding {} random individuals for budget {}".format(filler, budget))
                new_pop = self.init_population(pop_size=filler)
                de[budget].inc_score = self.inc_score
                de[budget].inc_config = self.inc_config
                de[budget].age = np.hstack((de[budget].age, [de[budget].max_age] * filler))
                for i in range(filler):
                    fitness, cost = de[budget].f_objective(new_pop[i], budget)
                    de[budget].population = np.vstack((de[budget].population, new_pop[i]))
                    de[budget].fitness = np.append(de[budget].fitness, fitness)
                    if fitness < self.inc_score:
                        self.inc_score = fitness
                        self.inc_config = new_pop[i]
                    traj.append(self.inc_score)
                    runtime.append(cost)
                    history.append((new_pop[i].tolist(), float(fitness), float(budget or 0)))

                de[budget].inc_score = self.inc_score
                de[budget].inc_config = self.inc_config
                de[budget].pop_size = pop_size
                if debug:
                    print("Pop size: {}; Len pop: {}".format(de[budget].pop_size,
                                                             len(de[budget].population)))

            elif pop_size < de[budget].pop_size:
                # compensating for extra individuals if SH pop size is smaller
                if debug:
                    print("Reducing population from {} to {} "
                          "for budget {}".format(de[budget].pop_size, pop_size, budget))
                # keeping only the top performing individuals/discarding the weak ones
                rank = np.sort(np.argsort(de[budget].fitness)[:pop_size])
                de[budget].population = de[budget].population[rank]
                de[budget].fitness = de[budget].fitness[rank]
                de[budget].age = de[budget].age[rank]
                de[budget].pop_size = pop_size

            # Represents the best individuals from the previous lower SH budget
            # which will serve as the parents for mutation
            alt_population = None

            # Successive Halving iterations carrying out DE
            for i_sh in range(num_SH_iters):

                # Warmstarting DE incumbent
                de[budget].inc_score = self.inc_score
                de[budget].inc_config = self.inc_config
                if debug:
                    print("Pop size: {}; DE budget: {}".format(de[budget].pop_size, budget))
                best = self.inc_config

                # Repeating DE over entire population 'generations' times
                for gen in range(self.generations):
                    de_traj, de_runtime, de_history = \
                        de[budget].evolve_generation(budget=budget, best=best,
                                                     alt_pop=alt_population)
                    traj.extend(de_traj)
                    runtime.extend(de_runtime)
                    history.extend(de_history)

                    # killing/replacing parents that have not changed/has aged
                    # conditional on max_age not being set to inf
                    if de[budget].max_age < np.inf:
                        if debug:
                            print("  Generation #{}: Ages -- {}".format(gen + 1, de[budget].age))
                        de_traj, de_runtime, de_history = de[budget].kill_aged_pop(budget, debug)
                        traj.extend(de_traj)
                        runtime.extend(de_runtime)
                        history.extend(de_history)

                # Updating global incumbent after each DE step
                self.inc_score = de[budget].inc_score
                self.inc_config = de[budget].inc_config

                # Retrieving budget, pop_size, population for the next SH iteration
                if i_sh < num_SH_iters - 1:  # when not final SH iteration
                    pop_size = num_configs[i_sh + 1]
                    next_budget = budgets[i_sh + 1]
                    # selecting top ranking individuals from lower budget
                    # that will be the mutation parents for the next higher budget
                    rank = np.sort(np.argsort(de[budget].fitness)[:pop_size])
                    alt_population = de[budget].population[rank]

                    if de[next_budget].population is not None:
                        # updating DE incumbents to maintain global trajectory
                        de[next_budget].inc_score = self.inc_score
                        de[next_budget].inc_config = self.inc_config
                    else:
                        # equivalent to iteration == 0
                        # top ranked individuals are selected from the lower budget, assigned as
                        # the population for the next higher budget, and evaluated on it
                        if debug:
                            print("Iteration: ", iteration)
                        de[next_budget].population = alt_population
                        de[next_budget].fitness = [None] * pop_size
                        for index, indv in enumerate(de[next_budget].population):
                            fitness, cost = de[next_budget].f_objective(indv, next_budget)
                            de[next_budget].fitness[index] = fitness
                            if fitness < self.inc_score:
                                self.inc_score = fitness
                                self.inc_config = indv
                            traj.append(self.inc_score)
                            runtime.append(cost)
                            history.append((indv.tolist(), float(fitness), float(budget or 0)))
                        de[next_budget].inc_config = self.inc_config
                        de[next_budget].inc_score = self.inc_score
                        de[next_budget].age = np.array([de[next_budget].max_age] * \
                                                       de[next_budget].pop_size)
                    budget = next_budget
        if verbose:
            print("\nRun complete!")

        return (np.array(traj), np.array(runtime), np.array(history))


class DEHBV4(DEHBBase):
    '''Version 4.0 of DEHB

    At anytime, each set of population contains the best individuals from that budget
    '''
    def __init__(self, randomize=None, max_age=np.inf, **kwargs):
        super().__init__(**kwargs)
        self.randomize = randomize
        self.max_age = max_age
        self.generations = None

    def get_next_iteration(self, iteration, type='original'):
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
        if type == 'original':
            n0 = int(np.floor((self.max_SH_iter)/(s+1)) * self.eta**s)
        else:  # type = 'custom'
            n0 = int(np.floor((self.max_SH_iter)/(s+1)) * self.eta**(s-1))
        ns = [max(int(n0*(self.eta**(-i))), 1) for i in range(s+1)]
        if type == 'original':
            if self.min_clip is not None and self.max_clip is not None:
                ns = np.clip(ns, a_min=self.min_clip, a_max=self.max_clip)
            elif self.min_clip is not None:
                ns = np.clip(ns, a_min=self.min_clip, a_max=np.max(ns))

        return ns, budgets

    def run(self, iterations=1, verbose=False, debug=False):
        # Book-keeping variables
        traj = []
        runtime = []
        history = []

        # To retrieve the population and budget ranges
        num_configs, budgets = self.get_next_iteration(iteration=0)
        full_budget = budgets[-1]
        small_budget = budgets[0]

        # List of DE objects corresponding to the populations
        de = {}
        for i, b in enumerate(budgets):
            de[b] = DE(cs=self.cs, f=self.f, dimensions=self.dimensions, pop_size=num_configs[i],
                       mutation_factor=self.mutation_factor, crossover_prob=self.crossover_prob,
                       strategy=self.strategy, budget=b, max_age=self.max_age)

        # Performs DEHB iterations
        for iteration in range(iterations):

            # Retrieves SH budgets and number of configurations
            num_configs, budgets = self.get_next_iteration(iteration=iteration)
            num_gens, _ = self.get_next_iteration(iteration=iteration, type='custom')
            if verbose:
                print('Iteration #{:>3}\n{}'.format(iteration, '-' * 15))
                print(num_configs, budgets, num_gens, self.inc_score)

            # Sets budget and population size for first SH iteration
            pop_size = num_configs[0]
            budget = budgets[0]
            gens = num_gens[0]

            # Number of SH iterations in this DEHB iteration
            num_SH_iters = len(budgets)

            # Index determining the population to begin DE with for current iteration number
            de_idx = iteration % len(de)

            # The first DEHB iteration - only time when a random population is initialized
            if iteration == 0:
                # creating new population for DEHB iteration to be used for the next SH steps
                de_traj, de_runtime, de_history = de[budget].init_eval_pop(budget)
                # maintaining global copy of random population created
                self.population = de[budget].population
                self.fitness = de[budget].fitness
                # update global incumbent with new population scores
                self.inc_score = de[budget].inc_score
                self.inc_config = de[budget].inc_config
                traj.extend(de_traj)
                runtime.extend(de_runtime)
                history.extend(de_history)

            elif budget == small_budget and self.randomize is not None and self.randomize > 0:
                # executes in the first step of every SH iteration other than first DEHB iteration
                # also conditional on whether a randomization fraction has been specified
                num_replace = np.ceil(self.randomize * pop_size).astype(int)

                # fetching the worst performing individuals
                idxs = np.sort(np.argsort(-de[budget].fitness)[:num_replace])
                if debug:
                    print("Replacing {}/{} -- {}".format(num_replace, pop_size, idxs))
                new_pop = self.init_population(pop_size=num_replace)
                de[budget].population[idxs] = new_pop
                de[budget].age[idxs] = de[budget].max_age
                de[budget].inc_score = self.inc_score
                de[budget].inc_config = self.inc_config

                # evaluating new individuals
                for i in idxs:
                    de[budget].fitness[i], cost = \
                        de[budget].f_objective(de[budget].population[i], budget)
                    if self.fitness[i] < self.inc_score:
                        self.inc_score = de[budget].fitness[i]
                        self.inc_config = de[budget].population[i]
                    traj.append(self.inc_score)
                    runtime.append(cost)
                    history.append((de[budget].population[i].tolist(),
                                    float(de[budget].fitness[i]), float(budget or 0)))

            elif pop_size > de[budget].pop_size:
                # compensating for extra individuals if SH pop size is larger
                filler = pop_size - de[budget].pop_size
                if debug:
                    print("Adding {} random individuals for budget {}".format(filler, budget))
                new_pop = self.init_population(pop_size=filler)
                de[budget].inc_score = self.inc_score
                de[budget].inc_config = self.inc_config
                de[budget].age = np.hstack((de[budget].age, [de[budget].max_age] * filler))
                for i in range(filler):
                    fitness, cost = de[budget].f_objective(new_pop[i], budget)
                    de[budget].population = np.vstack((de[budget].population, new_pop[i]))
                    de[budget].fitness = np.append(de[budget].fitness, fitness)
                    if fitness < self.inc_score:
                        self.inc_score = fitness
                        self.inc_config = new_pop[i]
                    traj.append(self.inc_score)
                    runtime.append(cost)
                    history.append((new_pop[i].tolist(), float(fitness), float(budget or 0)))

                de[budget].inc_score = self.inc_score
                de[budget].inc_config = self.inc_config
                de[budget].pop_size = pop_size
                if debug:
                    print("Pop size: {}; Len pop: {}".format(de[budget].pop_size,
                                                             len(de[budget].population)))

            elif pop_size < de[budget].pop_size:
                # compensating for extra individuals if SH pop size is smaller
                if debug:
                    print("Reducing population from {} to {} "
                          "for budget {}".format(de[budget].pop_size, pop_size, budget))
                # keeping only the top performing individuals/discarding the weak ones
                rank = np.sort(np.argsort(de[budget].fitness)[:pop_size])
                de[budget].population = de[budget].population[rank]
                de[budget].fitness = de[budget].fitness[rank]
                de[budget].age = de[budget].age[rank]
                de[budget].pop_size = pop_size

            # Successive Halving iterations carrying out DE
            for i_sh in range(num_SH_iters):

                # Warmstarting DE incumbent
                de[budget].inc_score = self.inc_score
                de[budget].inc_config = self.inc_config
                if debug:
                    print("Pop size: {}; DE budget: {}".format(de[budget].pop_size, budget))
                best = self.inc_config

                # Repeating DE over entire population 'generations' times
                for gen in range(gens):
                    de_traj, de_runtime, de_history = \
                        de[budget].evolve_generation(budget=budget, best=best)
                    traj.extend(de_traj)
                    runtime.extend(de_runtime)
                    history.extend(de_history)

                    # killing/replacing parents that have not changed/has aged
                    # conditional on max_age not being set to inf
                    if de[budget].max_age < np.inf:
                        if debug:
                            print("  Generation #{}: Ages -- {}".format(gen + 1, de[budget].age))
                        de_traj, de_runtime, de_history = de[budget].kill_aged_pop(budget, debug)
                        traj.extend(de_traj)
                        runtime.extend(de_runtime)
                        history.extend(de_history)

                # Updating global incumbent after each DE step
                self.inc_score = de[budget].inc_score
                self.inc_config = de[budget].inc_config

                # Retrieving budget, pop_size, population for the next SH iteration
                if i_sh < num_SH_iters-1:  # when not final SH iteration
                    pop_size = num_configs[i_sh + 1]
                    next_budget = budgets[i_sh + 1]
                    gens = num_gens[i_sh + 1]

                    # selecting top ranking individuals from lower budget
                    ## to be evaluated on higher budget and be eligible for competition
                    rank = np.sort(np.argsort(de[budget].fitness)[:pop_size])
                    rival_population = de[budget].population[rank]

                    if de[next_budget].population is not None:
                        # warmstarting DE incumbents to maintain global trajectory
                        de[next_budget].inc_score = self.inc_score
                        de[next_budget].inc_config = self.inc_config

                        # ranking individuals to determine population for next SH step
                        de_traj, de_runtime, de_history = \
                            de[next_budget].ranked_selection(rival_population, pop_size,
                                                             budget, debug)
                        self.inc_score = de[next_budget].inc_score
                        self.inc_config = de[next_budget].inc_config
                        traj.extend(de_traj)
                        runtime.extend(de_runtime)
                        history.extend(de_history)
                    else:
                        # equivalent to iteration == 0
                        # no ranked selection happens, rather top ranked individuals are selected
                        if debug:
                            print("Iteration: ", iteration)
                        de[next_budget].population = rival_population
                        de[next_budget].fitness = de[budget].fitness[rank]
                        de[next_budget].age = np.array([de[next_budget].max_age] * \
                                                        de[next_budget].pop_size)
                    budget = next_budget
        if verbose:
            print("\nRun complete!")

        return (np.array(traj), np.array(runtime), np.array(history))


class DEHBV4_1(DEHBBase):
    '''Version 4.1 of DEHB

    At anytime, each set of population contains the best individuals from that budget
    Maintains a global population of the full budget individuals to sample for mutations
    '''
    def __init__(self, randomize=None, max_age=np.inf, **kwargs):
        super().__init__(**kwargs)
        self.randomize = randomize
        self.max_age = max_age
        self.generations = None

    def get_next_iteration(self, iteration, type='original'):
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
        if type == 'original':
            n0 = int(np.floor((self.max_SH_iter)/(s+1)) * self.eta**s)
        else:  # type = 'custom'
            n0 = int(np.floor((self.max_SH_iter)/(s+1)) * self.eta**(s-1))
        ns = [max(int(n0*(self.eta**(-i))), 1) for i in range(s+1)]
        if type == 'original':
            if self.min_clip is not None and self.max_clip is not None:
                ns = np.clip(ns, a_min=self.min_clip, a_max=self.max_clip)
            elif self.min_clip is not None:
                ns = np.clip(ns, a_min=self.min_clip, a_max=np.max(ns))

        return ns, budgets

    def run(self, iterations=1, verbose=False, debug=False):
        # Book-keeping variables
        traj = []
        runtime = []
        history = []

        # Specifying details for the global full budget population
        global_pop_size = 0
        for i in range(self.max_SH_iter):
            num_configs, _ = self.get_next_iteration(iteration=i)
            global_pop_size += num_configs[-1]
        self.global_pop = np.array([None] * (global_pop_size))
        self.global_fitness = np.array([None] * (global_pop_size))

        # To retrieve the population and budget ranges
        num_configs, budgets = self.get_next_iteration(iteration=0)
        full_budget = budgets[-1]
        small_budget = budgets[0]

        # List of DE objects corresponding to the populations
        de = {}
        for i, b in enumerate(budgets):
            de[b] = DE(cs=self.cs, f=self.f, dimensions=self.dimensions, pop_size=num_configs[i],
                       mutation_factor=self.mutation_factor, crossover_prob=self.crossover_prob,
                       strategy=self.strategy, budget=b, max_age=self.max_age)

        # Performs DEHB iterations
        for iteration in range(iterations):

            # Retrieves SH budgets and number of configurations
            num_configs, budgets = self.get_next_iteration(iteration=iteration)
            num_gens, _ = self.get_next_iteration(iteration=iteration, type='custom')
            if verbose:
                print('Iteration #{:>3}\n{}'.format(iteration, '-' * 15))
                print(num_configs, budgets, self.inc_score)

            # Sets budget and population size for first SH iteration
            pop_size = num_configs[0]
            budget = budgets[0]
            gens = num_gens[0]

            # Number of SH iterations in this DEHB iteration
            num_SH_iters = len(budgets)

            # Index determining the population to begin DE with for current iteration number
            de_idx = iteration % len(de)

            # The first DEHB iteration - only time when a random population is initialized
            if iteration == 0:
                # creating new population for DEHB iteration to be used for the next SH steps
                de_traj, de_runtime, de_history = de[budget].init_eval_pop(budget)

                # maintaining global copy of random population created
                self.population = de[budget].population
                self.fitness = de[budget].fitness

                # update global incumbent with new population scores
                self.inc_score = de[budget].inc_score
                self.inc_config = de[budget].inc_config
                traj.extend(de_traj)
                runtime.extend(de_runtime)
                history.extend(de_history)

            elif budget == small_budget and self.randomize is not None and self.randomize > 0:
                # executes in the first step of every SH iteration other than first DEHB iteration
                # also conditional on whether a randomization fraction has been specified
                num_replace = np.ceil(self.randomize * pop_size).astype(int)

                # fetching the worst performing individuals
                idxs = np.sort(np.argsort(-de[budget].fitness[:num_replace]))
                if debug:
                    print("Replacing {}/{} -- {}".format(num_replace, pop_size, idxs))
                new_pop = self.init_population(pop_size=num_replace)
                de[budget].population[idxs] = new_pop
                de[budget].age[idxs] = de[budget].max_age
                de[budget].inc_score = self.inc_score
                de[budget].inc_config = self.inc_config

                # evaluating new individuals
                for i in idxs:
                    de[budget].fitness[i], cost = \
                        de[budget].f_objective(de[budget].population[i], budget)
                    if self.fitness[i] < self.inc_score:
                        self.inc_score = de[budget].fitness[i]
                        self.inc_config = de[budget].population[i]
                    traj.append(self.inc_score)
                    runtime.append(cost)
                    history.append((de[budget].population[i].tolist(),
                                    float(de[budget].fitness[i]), float(budget or 0)))

            elif pop_size > de[budget].pop_size:
                # compensating for extra individuals if SH pop size is larger
                filler = pop_size - de[budget].pop_size
                if debug:
                    print("Adding {} random individuals for budget {}".format(filler, budget))
                new_pop = self.init_population(pop_size=filler)
                de[budget].inc_score = self.inc_score
                de[budget].inc_config = self.inc_config
                de[budget].age = np.hstack((de[budget].age, [de[budget].max_age] * filler))
                for i in range(filler):
                    fitness, cost = de[budget].f_objective(new_pop[i], budget)
                    de[budget].population = np.vstack((de[budget].population, new_pop[i]))
                    de[budget].fitness = np.append(de[budget].fitness, fitness)
                    if fitness < self.inc_score:
                        self.inc_score = fitness
                        self.inc_config = new_pop[i]
                    traj.append(self.inc_score)
                    runtime.append(cost)
                    history.append((new_pop[i].tolist(), float(fitness), float(budget or 0)))

                de[budget].inc_score = self.inc_score
                de[budget].inc_config = self.inc_config
                de[budget].pop_size = pop_size
                if debug:
                    print("Pop size: {}; Len pop: {}".format(de[budget].pop_size,
                                                             len(de[budget].population)))

            elif pop_size < de[budget].pop_size:
                # compensating for extra individuals if SH pop size is smaller
                if debug:
                    print("Reducing population from {} to {} "
                          "for budget {}".format(de[budget].pop_size, pop_size, budget))
                # keeping only the top performing individuals/discarding the weak ones
                rank = np.sort(np.argsort(de[budget].fitness)[:pop_size])
                de[budget].population = de[budget].population[rank]
                de[budget].fitness = de[budget].fitness[rank]
                de[budget].age = de[budget].age[rank]
                de[budget].pop_size = pop_size

            # Successive Halving iterations carrying out DE
            for i_sh in range(num_SH_iters):

                # Warmstarting DE incumbent
                de[budget].inc_score = self.inc_score
                de[budget].inc_config = self.inc_config
                if debug:
                    print("Pop size: {}; DE budget: {}".format(de[budget].pop_size, budget))
                best = self.inc_config

                # Repeating DE over entire population 'generations' times
                for gen in range(gens):
                    de_traj, de_runtime, de_history = \
                        de[budget].evolve_generation(budget=budget, best=best,
                                                     alt_pop=self.global_pop)
                    traj.extend(de_traj)
                    runtime.extend(de_runtime)
                    history.extend(de_history)

                    # killing/replacing parents that have not changed/has aged
                    # conditional on max_age not being set to inf
                    if de[budget].max_age < np.inf:
                        if debug:
                            print("  Generation #{}: Ages -- {}".format(gen + 1, de[budget].age))
                        de_traj, de_runtime, de_history = de[budget].kill_aged_pop(budget, debug)
                        traj.extend(de_traj)
                        runtime.extend(de_runtime)
                        history.extend(de_history)

                # Updating global incumbent after each DE step
                self.inc_score = de[budget].inc_score
                self.inc_config = de[budget].inc_config

                # Retrieving budget, pop_size, population for the next SH iteration
                if i_sh < num_SH_iters-1:  # when not final SH iteration
                    pop_size = num_configs[i_sh + 1]
                    next_budget = budgets[i_sh + 1]
                    gens = num_gens[i_sh + 1]

                    # selecting top ranking individuals from lower budget
                    ## to be evaluated on higher budget and be eligible for competition
                    rank = np.sort(np.argsort(de[budget].fitness)[:pop_size])
                    rival_population = de[budget].population[rank]

                    if de[next_budget].population is not None:
                        # warmstarting DE incumbents to maintain global trajectory
                        de[next_budget].inc_score = self.inc_score
                        de[next_budget].inc_config = self.inc_config

                        # ranking individuals to determine population for next SH step
                        de_traj, de_runtime, de_history = \
                            de[next_budget].ranked_selection(rival_population, pop_size,
                                                             budget, debug)
                        self.inc_score = de[next_budget].inc_score
                        self.inc_config = de[next_budget].inc_config
                        traj.extend(de_traj)
                        runtime.extend(de_runtime)
                        history.extend(de_history)
                    else:
                        # equivalent to iteration == 0
                        # no ranked selection happens, rather top ranked individuals are selected
                        if debug:
                            print("Iteration: ", iteration)
                        de[next_budget].population = rival_population
                        de[next_budget].fitness = de[budget].fitness[rank]
                        de[next_budget].age = np.array([de[next_budget].max_age] * \
                                                        de[next_budget].pop_size)
                    budget = next_budget

                else:  # final SH iteration with max budget
                    pop_size = num_configs[-1]
                    idx = [indv is None for indv in self.global_pop]
                    if any(idx):  # None exists in global population
                        # Initialize global population with positions assigned randomly
                        idx = np.where(np.array(idx) == True)[0]
                        idx = np.random.choice(idx, pop_size, replace=False)
                    else:
                        self.global_pop = np.stack(self.global_pop)
                        self.global_fitness = np.stack(self.global_fitness)
                        # Find the weakest individuals in the global population to replace them
                        idx = np.sort(np.argsort(-self.global_fitness)[:pop_size])
                    for index, id in enumerate(idx):
                        # Updating global pop
                        self.global_pop[id] = de[full_budget].population[index]
                        self.global_fitness[id] = de[full_budget].fitness[index]
        if verbose:
            print("\nRun complete!")

        return (np.array(traj), np.array(runtime), np.array(history))


class DEHBV4_2(DEHBBase):
    '''Version 4.2 of DEHB

    At anytime, each set of population contains the best individuals from that budget
    The top individuals from the population evaluated on the previous budget serves as the
        parents for mutation in the next higher budget
    '''

    def __init__(self, randomize=None, max_age=np.inf, **kwargs):
        super().__init__(**kwargs)
        self.randomize = randomize
        self.max_age = max_age
        self.generations = None

    def get_next_iteration(self, iteration, type='original'):
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
        if type == 'original':
            n0 = int(np.floor((self.max_SH_iter)/(s+1)) * self.eta**s)
        else:  # type = 'custom'
            n0 = int(np.floor((self.max_SH_iter)/(s+1)) * self.eta**(s-1))
        ns = [max(int(n0*(self.eta**(-i))), 1) for i in range(s+1)]
        if type == 'original':
            if self.min_clip is not None and self.max_clip is not None:
                ns = np.clip(ns, a_min=self.min_clip, a_max=self.max_clip)
            elif self.min_clip is not None:
                ns = np.clip(ns, a_min=self.min_clip, a_max=np.max(ns))

        return ns, budgets


    def run(self, iterations=1, verbose=False, debug=False):
        # Book-keeping variables
        traj = []
        runtime = []
        history = []

        # To retrieve the population and budget ranges
        num_configs, budgets = self.get_next_iteration(iteration=0)
        full_budget = budgets[-1]
        small_budget = budgets[0]

        # List of DE objects corresponding to the populations
        de = {}
        for i, b in enumerate(budgets):
            de[b] = DE(cs=self.cs, f=self.f, dimensions=self.dimensions, pop_size=num_configs[i],
                       mutation_factor=self.mutation_factor, crossover_prob=self.crossover_prob,
                       strategy=self.strategy, budget=b, max_age=self.max_age)

        # Performs DEHB iterations
        for iteration in range(iterations):

            # Retrieves SH budgets and number of configurations
            num_configs, budgets = self.get_next_iteration(iteration=iteration)
            num_gens, _ = self.get_next_iteration(iteration=iteration, type='custom')
            if verbose:
                print('Iteration #{:>3}\n{}'.format(iteration, '-' * 15))
                print(num_configs, budgets, self.inc_score)

            # Sets budget and population size for first SH iteration
            pop_size = num_configs[0]
            budget = budgets[0]
            gens = num_gens[0]

            # Number of SH iterations in this DEHB iteration
            num_SH_iters = len(budgets)

            # Index determining the population to begin DE with for current iteration number
            de_idx = iteration % len(de)

            # The first DEHB iteration - only time when a random population is initialized
            if iteration == 0:
                # creating new population for DEHB iteration to be used for the next SH steps
                de_traj, de_runtime, de_history = de[budget].init_eval_pop(budget)
                # maintaining global copy of random population created
                self.population = de[budget].population
                self.fitness = de[budget].fitness
                # update global incumbent with new population scores
                self.inc_score = de[budget].inc_score
                self.inc_config = de[budget].inc_config
                traj.extend(de_traj)
                runtime.extend(de_runtime)
                history.extend(de_history)

            elif budget == small_budget and self.randomize is not None and self.randomize > 0:
                # executes in the first step of every SH iteration other than first DEHB iteration
                # also conditional on whether a randomization fraction has been specified
                num_replace = np.ceil(self.randomize * pop_size).astype(int)

                # fetching the worst performing individuals
                idxs = np.sort(np.argsort(-de[budget].fitness)[:num_replace])
                if debug:
                    print("Replacing {}/{} -- {}".format(num_replace, pop_size, idxs))
                new_pop = self.init_population(pop_size=num_replace)
                de[budget].population[idxs] = new_pop
                de[budget].age[idxs] = de[budget].max_age
                de[budget].inc_score = self.inc_score
                de[budget].inc_config = self.inc_config

                # evaluating new individuals
                for i in idxs:
                    de[budget].fitness[i], cost = \
                        de[budget].f_objective(de[budget].population[i], budget)
                    if self.fitness[i] < self.inc_score:
                        self.inc_score = de[budget].fitness[i]
                        self.inc_config = de[budget].population[i]
                    traj.append(self.inc_score)
                    runtime.append(cost)
                    history.append((de[budget].population[i].tolist(),
                                    float(de[budget].fitness[i]), float(budget or 0)))

            elif pop_size > de[budget].pop_size:
                # compensating for extra individuals if SH pop size is larger
                filler = pop_size - de[budget].pop_size
                if debug:
                    print("Adding {} random individuals for budget {}".format(filler, budget))
                new_pop = self.init_population(pop_size=filler)
                de[budget].inc_score = self.inc_score
                de[budget].inc_config = self.inc_config
                de[budget].age = np.hstack((de[budget].age, [de[budget].max_age] * filler))
                for i in range(filler):
                    fitness, cost = de[budget].f_objective(new_pop[i], budget)
                    de[budget].population = np.vstack((de[budget].population, new_pop[i]))
                    de[budget].fitness = np.append(de[budget].fitness, fitness)
                    if fitness < self.inc_score:
                        self.inc_score = fitness
                        self.inc_config = new_pop[i]
                    traj.append(self.inc_score)
                    runtime.append(cost)
                    history.append((new_pop[i].tolist(), float(fitness), float(budget or 0)))

                de[budget].inc_score = self.inc_score
                de[budget].inc_config = self.inc_config
                de[budget].pop_size = pop_size
                if debug:
                    print("Pop size: {}; Len pop: {}".format(de[budget].pop_size,
                                                             len(de[budget].population)))

            elif pop_size < de[budget].pop_size:
                # compensating for extra individuals if SH pop size is smaller
                if debug:
                    print("Reducing population from {} to {} "
                          "for budget {}".format(de[budget].pop_size, pop_size, budget))
                # keeping only the top performing individuals/discarding the weak ones
                rank = np.sort(np.argsort(de[budget].fitness)[:pop_size])
                de[budget].population = de[budget].population[rank]
                de[budget].fitness = de[budget].fitness[rank]
                de[budget].age = de[budget].age[rank]
                de[budget].pop_size = pop_size

            # Represents the best individuals from the previous lower SH budget
            # which will serve as the parents for mutation
            alt_population = None

            # Successive Halving iterations carrying out DE
            for i_sh in range(num_SH_iters):

                # Warmstarting DE incumbent
                de[budget].inc_score = self.inc_score
                de[budget].inc_config = self.inc_config
                if debug:
                    print("Pop size: {}; DE budget: {}".format(de[budget].pop_size, budget))
                best = self.inc_config

                # Repeating DE over entire population 'generations' times
                for gen in range(gens):
                    de_traj, de_runtime, de_history = \
                        de[budget].evolve_generation(budget=budget, best=best,
                                                     alt_pop=alt_population)
                    traj.extend(de_traj)
                    runtime.extend(de_runtime)
                    history.extend(de_history)

                    # killing/replacing parents that have not changed/has aged
                    # conditional on max_age not being set to inf
                    if de[budget].max_age < np.inf:
                        if debug:
                            print("  Generation #{}: Ages -- {}".format(gen + 1, de[budget].age))
                        de_traj, de_runtime, de_history = de[budget].kill_aged_pop(budget, debug)
                        traj.extend(de_traj)
                        runtime.extend(de_runtime)
                        history.extend(de_history)

                # Updating global incumbent after each DE step
                self.inc_score = de[budget].inc_score
                self.inc_config = de[budget].inc_config

                # Retrieving budget, pop_size, population for the next SH iteration
                if i_sh < num_SH_iters - 1:  # when not final SH iteration
                    pop_size = num_configs[i_sh + 1]
                    next_budget = budgets[i_sh + 1]
                    gens = num_gens[i_sh + 1]

                    # selecting top ranking individuals from lower budget
                    # that will be the mutation parents for the next higher budget
                    rank = np.sort(np.argsort(de[budget].fitness)[:pop_size])
                    alt_population = de[budget].population[rank]

                    if de[next_budget].population is not None:
                        # updating DE incumbents to maintain global trajectory
                        de[next_budget].inc_score = self.inc_score
                        de[next_budget].inc_config = self.inc_config
                    else:
                        # equivalent to iteration == 0
                        # top ranked individuals are selected from the lower budget, assigned as
                        # the population for the next higher budget, and evaluated on it
                        if debug:
                            print("Iteration: ", iteration)
                        de[next_budget].population = alt_population
                        de[next_budget].fitness = [None] * pop_size
                        for index, indv in enumerate(de[next_budget].population):
                            fitness, cost = de[next_budget].f_objective(indv, next_budget)
                            de[next_budget].fitness[index] = fitness
                            if fitness < self.inc_score:
                                self.inc_score = fitness
                                self.inc_config = indv
                            traj.append(self.inc_score)
                            runtime.append(cost)
                            history.append((indv.tolist(), float(fitness), float(budget or 0)))
                        de[next_budget].inc_config = self.inc_config
                        de[next_budget].inc_score = self.inc_score
                        de[next_budget].age = np.array([de[next_budget].max_age] * \
                                                       de[next_budget].pop_size)
                    budget = next_budget
        if verbose:
            print("\nRun complete!")

        return (np.array(traj), np.array(runtime), np.array(history))


class DEHBV4_2_2(DEHBBase):
    '''Version 4.2 of DEHB

    At anytime, each set of population contains the best individuals from that budget
    The top individuals from the population evaluated on the previous budget serves as the
        parents for mutation in the next higher budget
    '''

    def __init__(self, randomize=None, max_age=np.inf, **kwargs):
        super().__init__(**kwargs)
        self.randomize = randomize
        self.max_age = max_age
        self.generations = 1

    def get_next_iteration(self, iteration, type='original'):
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
        if type == 'original':
            n0 = int(np.floor((self.max_SH_iter)/(s+1)) * self.eta**s)
        else:  # type = 'custom'
            n0 = int(np.floor((self.max_SH_iter)/(s+1)) * self.eta**(s-1))
        ns = [max(int(n0*(self.eta**(-i))), 1) for i in range(s+1)]
        if type == 'original':
            if self.min_clip is not None and self.max_clip is not None:
                ns = np.clip(ns, a_min=self.min_clip, a_max=self.max_clip)
            elif self.min_clip is not None:
                ns = np.clip(ns, a_min=self.min_clip, a_max=np.max(ns))

        return ns, budgets

    def run(self, iterations=1, verbose=False, debug=False):
        # Book-keeping variables
        traj = []
        runtime = []
        history = []

        # To retrieve the population and budget ranges
        num_configs, budgets = self.get_next_iteration(iteration=0)
        full_budget = budgets[-1]
        small_budget = budgets[0]

        # List of DE objects corresponding to the populations
        de = {}
        for i, b in enumerate(budgets):
            de[b] = DE(cs=self.cs, f=self.f, dimensions=self.dimensions, pop_size=num_configs[i],
                       mutation_factor=self.mutation_factor, crossover_prob=self.crossover_prob,
                       strategy=self.strategy, budget=b, max_age=self.max_age)

        # Performs DEHB iterations
        for iteration in range(iterations):

            # Retrieves SH budgets and number of configurations
            num_configs, budgets = self.get_next_iteration(iteration=iteration)
            num_gens, _ = self.get_next_iteration(iteration=iteration, type='custom')
            if verbose:
                print('Iteration #{:>3}\n{}'.format(iteration, '-' * 15))
                print(num_configs, budgets, self.inc_score)

            # Sets budget and population size for first SH iteration
            pop_size = num_configs[0]
            budget = budgets[0]
            gens = num_gens[0]

            # Number of SH iterations in this DEHB iteration
            num_SH_iters = len(budgets)

            # Index determining the population to begin DE with for current iteration number
            de_idx = iteration % len(de)

            # The first DEHB iteration - only time when a random population is initialized
            if iteration == 0:
                # creating new population for DEHB iteration to be used for the next SH steps
                de_traj, de_runtime, de_history = de[budget].init_eval_pop(budget)
                # maintaining global copy of random population created
                self.population = de[budget].population
                self.fitness = de[budget].fitness
                # update global incumbent with new population scores
                self.inc_score = de[budget].inc_score
                self.inc_config = de[budget].inc_config
                traj.extend(de_traj)
                runtime.extend(de_runtime)
                history.extend(de_history)

            elif budget == small_budget and self.randomize is not None and self.randomize > 0:
                # executes in the first step of every SH iteration other than first DEHB iteration
                # also conditional on whether a randomization fraction has been specified
                num_replace = np.ceil(self.randomize * pop_size).astype(int)

                # fetching the worst performing individuals
                idxs = np.sort(np.argsort(-de[budget].fitness)[:num_replace])
                if debug:
                    print("Replacing {}/{} -- {}".format(num_replace, pop_size, idxs))
                new_pop = self.init_population(pop_size=num_replace)
                de[budget].population[idxs] = new_pop
                de[budget].age[idxs] = de[budget].max_age
                de[budget].inc_score = self.inc_score
                de[budget].inc_config = self.inc_config

                # evaluating new individuals
                for i in idxs:
                    de[budget].fitness[i], cost = \
                        de[budget].f_objective(de[budget].population[i], budget)
                    if self.fitness[i] < self.inc_score:
                        self.inc_score = de[budget].fitness[i]
                        self.inc_config = de[budget].population[i]
                    traj.append(self.inc_score)
                    runtime.append(cost)
                    history.append((de[budget].population[i].tolist(),
                                    float(de[budget].fitness[i]), float(budget or 0)))

            elif pop_size > de[budget].pop_size:
                # compensating for extra individuals if SH pop size is larger
                filler = pop_size - de[budget].pop_size
                if debug:
                    print("Adding {} random individuals for budget {}".format(filler, budget))
                new_pop = self.init_population(pop_size=filler)
                de[budget].inc_score = self.inc_score
                de[budget].inc_config = self.inc_config
                de[budget].age = np.hstack((de[budget].age, [de[budget].max_age] * filler))
                for i in range(filler):
                    fitness, cost = de[budget].f_objective(new_pop[i], budget)
                    de[budget].population = np.vstack((de[budget].population, new_pop[i]))
                    de[budget].fitness = np.append(de[budget].fitness, fitness)
                    if fitness < self.inc_score:
                        self.inc_score = fitness
                        self.inc_config = new_pop[i]
                    traj.append(self.inc_score)
                    runtime.append(cost)
                    history.append((new_pop[i].tolist(), float(fitness), float(budget or 0)))

                de[budget].inc_score = self.inc_score
                de[budget].inc_config = self.inc_config
                de[budget].pop_size = pop_size
                if debug:
                    print("Pop size: {}; Len pop: {}".format(de[budget].pop_size,
                                                             len(de[budget].population)))

            elif pop_size < de[budget].pop_size:
                # compensating for extra individuals if SH pop size is smaller
                if debug:
                    print("Reducing population from {} to {} "
                          "for budget {}".format(de[budget].pop_size, pop_size, budget))
                # keeping only the top performing individuals/discarding the weak ones
                rank = np.sort(np.argsort(de[budget].fitness)[:pop_size])
                de[budget].population = de[budget].population[rank]
                de[budget].fitness = de[budget].fitness[rank]
                de[budget].age = de[budget].age[rank]
                de[budget].pop_size = pop_size

            # Represents the best individuals from the previous lower SH budget
            # which will serve as the parents for mutation
            alt_population = None

            # Successive Halving iterations carrying out DE
            for i_sh in range(num_SH_iters):

                gens = self.generations if iteration < 1 else gens

                # Warmstarting DE incumbent
                de[budget].inc_score = self.inc_score
                de[budget].inc_config = self.inc_config
                if debug:
                    print("Pop size: {}; DE budget: {}; "
                          "Gens: {}".format(de[budget].pop_size, budget, gens))
                best = self.inc_config

                # Repeating DE over entire population 'generations' times
                for gen in range(gens):
                    de_traj, de_runtime, de_history = \
                        de[budget].evolve_generation(budget=budget, best=best,
                                                     alt_pop=alt_population)
                    traj.extend(de_traj)
                    runtime.extend(de_runtime)
                    history.extend(de_history)

                    # killing/replacing parents that have not changed/has aged
                    # conditional on max_age not being set to inf
                    if de[budget].max_age < np.inf:
                        if debug:
                            print("  Generation #{}: Ages -- {}".format(gen + 1, de[budget].age))
                        de_traj, de_runtime, de_history = de[budget].kill_aged_pop(budget, debug)
                        traj.extend(de_traj)
                        runtime.extend(de_runtime)
                        history.extend(de_history)

                # Updating global incumbent after each DE step
                self.inc_score = de[budget].inc_score
                self.inc_config = de[budget].inc_config

                # Retrieving budget, pop_size, population for the next SH iteration
                if i_sh < num_SH_iters - 1:  # when not final SH iteration
                    pop_size = num_configs[i_sh + 1]
                    next_budget = budgets[i_sh + 1]
                    gens = num_gens[i_sh + 1]

                    # selecting top ranking individuals from lower budget
                    # that will be the mutation parents for the next higher budget
                    rank = np.sort(np.argsort(de[budget].fitness)[:pop_size])
                    alt_population = de[budget].population[rank]

                    if de[next_budget].population is not None:
                        # updating DE incumbents to maintain global trajectory
                        de[next_budget].inc_score = self.inc_score
                        de[next_budget].inc_config = self.inc_config
                    else:
                        # equivalent to iteration == 0
                        # top ranked individuals are selected from the lower budget, assigned as
                        # the population for the next higher budget, and evaluated on it
                        if debug:
                            print("Iteration: ", iteration)
                        de[next_budget].population = alt_population
                        de[next_budget].fitness = [None] * pop_size
                        for index, indv in enumerate(de[next_budget].population):
                            fitness, cost = de[next_budget].f_objective(indv, next_budget)
                            de[next_budget].fitness[index] = fitness
                            if fitness < self.inc_score:
                                self.inc_score = fitness
                                self.inc_config = indv
                            traj.append(self.inc_score)
                            runtime.append(cost)
                            history.append((indv.tolist(), float(fitness), float(budget or 0)))
                        de[next_budget].inc_config = self.inc_config
                        de[next_budget].inc_score = self.inc_score
                        de[next_budget].age = np.array([de[next_budget].max_age] * \
                                                       de[next_budget].pop_size)
                    budget = next_budget
        if verbose:
            print("\nRun complete!")

        return (np.array(traj), np.array(runtime), np.array(history))


class DEHBV4_2_3(DEHBBase):
    '''Version 4.2 of DEHB

    At anytime, each set of population contains the best individuals from that budget
    The top individuals from the population evaluated on the previous budget serves as the
        parents for mutation in the next higher budget
    '''

    def __init__(self, randomize=None, max_age=np.inf, **kwargs):
        super().__init__(**kwargs)
        self.randomize = randomize
        self.max_age = max_age
        self.generations = 1

    def get_next_iteration(self, iteration, type='original'):
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
        if type == 'original':
            n0 = int(np.floor((self.max_SH_iter)/(s+1)) * self.eta**s)
        else:  # type = 'custom'
            n0 = int(np.floor((self.max_SH_iter)/(s+1)) * self.eta**(s-1))
        ns = [max(int(n0*(self.eta**(-i))), 1) for i in range(s+1)]
        if type == 'original':
            if self.min_clip is not None and self.max_clip is not None:
                ns = np.clip(ns, a_min=self.min_clip, a_max=self.max_clip)
            elif self.min_clip is not None:
                ns = np.clip(ns, a_min=self.min_clip, a_max=np.max(ns))

        return ns, budgets

    def run(self, iterations=1, verbose=False, debug=False):
        # Book-keeping variables
        traj = []
        runtime = []
        history = []

        # To retrieve the population and budget ranges
        num_configs, budgets = self.get_next_iteration(iteration=0)
        full_budget = budgets[-1]
        small_budget = budgets[0]

        # List of DE objects corresponding to the populations
        de = {}
        for i, b in enumerate(budgets):
            de[b] = DE(cs=self.cs, f=self.f, dimensions=self.dimensions, pop_size=num_configs[i],
                       mutation_factor=self.mutation_factor, crossover_prob=self.crossover_prob,
                       strategy=self.strategy, budget=b, max_age=self.max_age)

        # Performs DEHB iterations
        for iteration in range(iterations):

            # Retrieves SH budgets and number of configurations
            num_configs, budgets = self.get_next_iteration(iteration=iteration)
            num_gens, _ = self.get_next_iteration(iteration=iteration, type='custom')
            if verbose:
                print('Iteration #{:>3}\n{}'.format(iteration, '-' * 15))
                print(num_configs, budgets, self.inc_score)

            # Sets budget and population size for first SH iteration
            pop_size = num_configs[0]
            budget = budgets[0]
            gens = num_gens[0]

            # Number of SH iterations in this DEHB iteration
            num_SH_iters = len(budgets)

            # Index determining the population to begin DE with for current iteration number
            de_idx = iteration % len(de)

            # The first DEHB iteration - only time when a random population is initialized
            if iteration == 0:
                # creating new population for DEHB iteration to be used for the next SH steps
                de_traj, de_runtime, de_history = de[budget].init_eval_pop(budget)
                # maintaining global copy of random population created
                self.population = de[budget].population
                self.fitness = de[budget].fitness
                # update global incumbent with new population scores
                self.inc_score = de[budget].inc_score
                self.inc_config = de[budget].inc_config
                traj.extend(de_traj)
                runtime.extend(de_runtime)
                history.extend(de_history)

            elif budget == small_budget and self.randomize is not None and self.randomize > 0:
                # executes in the first step of every SH iteration other than first DEHB iteration
                # also conditional on whether a randomization fraction has been specified
                num_replace = np.ceil(self.randomize * pop_size).astype(int)

                # fetching the worst performing individuals
                idxs = np.sort(np.argsort(-de[budget].fitness)[:num_replace])
                if debug:
                    print("Replacing {}/{} -- {}".format(num_replace, pop_size, idxs))
                new_pop = self.init_population(pop_size=num_replace)
                de[budget].population[idxs] = new_pop
                de[budget].age[idxs] = de[budget].max_age
                de[budget].inc_score = self.inc_score
                de[budget].inc_config = self.inc_config

                # evaluating new individuals
                for i in idxs:
                    de[budget].fitness[i], cost = \
                        de[budget].f_objective(de[budget].population[i], budget)
                    if self.fitness[i] < self.inc_score:
                        self.inc_score = de[budget].fitness[i]
                        self.inc_config = de[budget].population[i]
                    traj.append(self.inc_score)
                    runtime.append(cost)
                    history.append((de[budget].population[i].tolist(),
                                    float(de[budget].fitness[i]), float(budget or 0)))

            elif pop_size > de[budget].pop_size:
                # compensating for extra individuals if SH pop size is larger
                filler = pop_size - de[budget].pop_size
                if debug:
                    print("Adding {} random individuals for budget {}".format(filler, budget))
                new_pop = self.init_population(pop_size=filler)
                de[budget].inc_score = self.inc_score
                de[budget].inc_config = self.inc_config
                de[budget].age = np.hstack((de[budget].age, [de[budget].max_age] * filler))
                for i in range(filler):
                    fitness, cost = de[budget].f_objective(new_pop[i], budget)
                    de[budget].population = np.vstack((de[budget].population, new_pop[i]))
                    de[budget].fitness = np.append(de[budget].fitness, fitness)
                    if fitness < self.inc_score:
                        self.inc_score = fitness
                        self.inc_config = new_pop[i]
                    traj.append(self.inc_score)
                    runtime.append(cost)
                    history.append((new_pop[i].tolist(), float(fitness), float(budget or 0)))

                de[budget].inc_score = self.inc_score
                de[budget].inc_config = self.inc_config
                de[budget].pop_size = pop_size
                if debug:
                    print("Pop size: {}; Len pop: {}".format(de[budget].pop_size,
                                                             len(de[budget].population)))

            elif pop_size < de[budget].pop_size:
                # compensating for extra individuals if SH pop size is smaller
                if debug:
                    print("Reducing population from {} to {} "
                          "for budget {}".format(de[budget].pop_size, pop_size, budget))
                # keeping only the top performing individuals/discarding the weak ones
                rank = np.sort(np.argsort(de[budget].fitness)[:pop_size])
                de[budget].population = de[budget].population[rank]
                de[budget].fitness = de[budget].fitness[rank]
                de[budget].age = de[budget].age[rank]
                de[budget].pop_size = pop_size

            # Represents the best individuals from the previous lower SH budget
            # which will serve as the parents for mutation
            alt_population = None

            # Successive Halving iterations carrying out DE
            for i_sh in range(num_SH_iters):

                gens = self.generations if iteration < self.max_SH_iter else gens

                # Warmstarting DE incumbent
                de[budget].inc_score = self.inc_score
                de[budget].inc_config = self.inc_config
                if debug:
                    print("Pop size: {}; DE budget: {}; "
                          "Gens: {}".format(de[budget].pop_size, budget, gens))
                best = self.inc_config

                # Repeating DE over entire population 'generations' times
                for gen in range(gens):
                    de_traj, de_runtime, de_history = \
                        de[budget].evolve_generation(budget=budget, best=best,
                                                     alt_pop=alt_population)
                    traj.extend(de_traj)
                    runtime.extend(de_runtime)
                    history.extend(de_history)

                    # killing/replacing parents that have not changed/has aged
                    # conditional on max_age not being set to inf
                    if de[budget].max_age < np.inf:
                        if debug:
                            print("  Generation #{}: Ages -- {}".format(gen + 1, de[budget].age))
                        de_traj, de_runtime, de_history = de[budget].kill_aged_pop(budget, debug)
                        traj.extend(de_traj)
                        runtime.extend(de_runtime)
                        history.extend(de_history)

                # Updating global incumbent after each DE step
                self.inc_score = de[budget].inc_score
                self.inc_config = de[budget].inc_config

                # Retrieving budget, pop_size, population for the next SH iteration
                if i_sh < num_SH_iters - 1:  # when not final SH iteration
                    pop_size = num_configs[i_sh + 1]
                    next_budget = budgets[i_sh + 1]
                    gens = num_gens[i_sh + 1]

                    # selecting top ranking individuals from lower budget
                    # that will be the mutation parents for the next higher budget
                    rank = np.sort(np.argsort(de[budget].fitness)[:pop_size])
                    alt_population = de[budget].population[rank]

                    if de[next_budget].population is not None:
                        # updating DE incumbents to maintain global trajectory
                        de[next_budget].inc_score = self.inc_score
                        de[next_budget].inc_config = self.inc_config
                    else:
                        # equivalent to iteration == 0
                        # top ranked individuals are selected from the lower budget, assigned as
                        # the population for the next higher budget, and evaluated on it
                        if debug:
                            print("Iteration: ", iteration)
                        de[next_budget].population = alt_population
                        de[next_budget].fitness = [None] * pop_size
                        for index, indv in enumerate(de[next_budget].population):
                            fitness, cost = de[next_budget].f_objective(indv, next_budget)
                            de[next_budget].fitness[index] = fitness
                            if fitness < self.inc_score:
                                self.inc_score = fitness
                                self.inc_config = indv
                            traj.append(self.inc_score)
                            runtime.append(cost)
                            history.append((indv.tolist(), float(fitness), float(budget or 0)))
                        de[next_budget].inc_config = self.inc_config
                        de[next_budget].inc_score = self.inc_score
                        de[next_budget].age = np.array([de[next_budget].max_age] * \
                                                       de[next_budget].pop_size)
                    budget = next_budget
        if verbose:
            print("\nRun complete!")

        return (np.array(traj), np.array(runtime), np.array(history))


class DEHBV5(DEHBBase):
    '''Version 5.0 of DEHB
    Similar to Version 4.0
    At anytime, each set of population contains the best individuals from that budget
    At each SH iteration, the score to measure quality of a DE population is computed
    Based on which, the number of generations passed is allocated.
    '''
    def __init__(self, randomize=None, max_age=np.inf, **kwargs):
        super().__init__(**kwargs)
        self.randomize = randomize
        self.max_age = max_age
        self.generations = None

    def get_next_iteration(self, iteration, type='original'):
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
        if type == 'original':
            n0 = int(np.floor((self.max_SH_iter)/(s+1)) * self.eta**s)
        else:  # type = 'custom'
            n0 = int(np.floor((self.max_SH_iter)/(s+1)) * self.eta**(s-1))
        ns = [max(int(n0*(self.eta**(-i))), 1) for i in range(s+1)]
        if type == 'original':
            if self.min_clip is not None and self.max_clip is not None:
                ns = np.clip(ns, a_min=self.min_clip, a_max=self.max_clip)
            elif self.min_clip is not None:
                ns = np.clip(ns, a_min=self.min_clip, a_max=np.max(ns))

        return ns, budgets

    def convergence_score(self, budget):
        std_range = np.max(self.de[budget].fitness) - np.min(self.de[budget].fitness)
        if std_range == 0:
            return 0
        median_fitness = np.median(self.de[budget].fitness) / std_range
        min_fitness = np.min(self.de[budget].fitness) / std_range
        var = median_fitness - min_fitness
        return var

    def run(self, iterations=1, verbose=False, debug=False, debug2=False):
        # Book-keeping variables
        traj = []
        runtime = []
        history = []

        # To retrieve the population and budget ranges
        num_configs, budgets = self.get_next_iteration(iteration=0)
        full_budget = budgets[-1]
        small_budget = budgets[0]

        # Tracks the variance found in fitness on b in previous iteration for that b
        self.tracker = {}
        self.penalty = {}
        self.b_eta = {}
        for b in budgets:
            self.tracker[b] = 0
            self.penalty[b] = False
            self.b_eta[b] = self.eta

        # List of DE objects corresponding to the populations
        self.de = {}
        for i, b in enumerate(budgets):
            self.de[b] = DE(cs=self.cs, f=self.f, dimensions=self.dimensions,
                            pop_size=num_configs[i], mutation_factor=self.mutation_factor,
                            crossover_prob=self.crossover_prob, strategy=self.strategy,
                            budget=b, max_age=self.max_age)

        # Performs DEHB iterations
        for iteration in range(iterations):

            # Retrieves SH budgets and number of configurations
            num_configs, budgets = self.get_next_iteration(iteration=iteration)
            num_gens, _ = self.get_next_iteration(iteration=iteration, type='custom')
            if verbose:
                print('Iteration #{:>3}\n{}'.format(iteration, '-' * 15))
                print(num_configs, budgets, num_gens, self.inc_score)

            # Sets budget and population size for first SH iteration
            pop_size = num_configs[0]
            budget = budgets[0]
            gens = num_gens[0]

            # Number of SH iterations in this DEHB iteration
            num_SH_iters = len(budgets)

            # Index determining the population to begin DE with for current iteration number
            de_idx = iteration % len(self.de)

            # The first DEHB iteration - only time when a random population is initialized
            if iteration == 0:
                # creating new population for DEHB iteration to be used for the next SH steps
                de_traj, de_runtime, de_history = self.de[budget].init_eval_pop(budget)
                # maintaining global copy of random population created
                self.population = self.de[budget].population
                self.fitness = self.de[budget].fitness
                # update global incumbent with new population scores
                self.inc_score = self.de[budget].inc_score
                self.inc_config = self.de[budget].inc_config
                traj.extend(de_traj)
                runtime.extend(de_runtime)
                history.extend(de_history)

            elif budget == small_budget and self.randomize is not None and self.randomize > 0:
                # executes in the first step of every SH iteration other than first DEHB iteration
                # also conditional on whether a randomization fraction has been specified
                num_replace = np.ceil(self.randomize * pop_size).astype(int)

                # fetching the worst performing individuals
                idxs = np.sort(np.argsort(-self.de[budget].fitness)[:num_replace])
                if debug:
                    print("Replacing {}/{} -- {}".format(num_replace, pop_size, idxs))
                new_pop = self.init_population(pop_size=num_replace)
                self.de[budget].population[idxs] = new_pop
                self.de[budget].age[idxs] = self.de[budget].max_age
                self.de[budget].inc_score = self.inc_score
                self.de[budget].inc_config = self.inc_config

                # evaluating new individuals
                for i in idxs:
                    self.de[budget].fitness[i], cost = \
                        self.de[budget].f_objective(self.de[budget].population[i], budget)
                    if self.fitness[i] < self.inc_score:
                        self.inc_score = self.de[budget].fitness[i]
                        self.inc_config = self.de[budget].population[i]
                    traj.append(self.inc_score)
                    runtime.append(cost)
                    history.append((self.de[budget].population[i].tolist(),
                                    float(self.de[budget].fitness[i]), float(budget or 0)))

            elif pop_size > self.de[budget].pop_size:
                # compensating for extra individuals if SH pop size is larger
                filler = pop_size - self.de[budget].pop_size
                if debug:
                    print("Adding {} random individuals for budget {}".format(filler, budget))
                new_pop = self.init_population(pop_size=filler)
                self.de[budget].inc_score = self.inc_score
                self.de[budget].inc_config = self.inc_config
                self.de[budget].age = np.hstack((self.de[budget].age,
                                                 [self.de[budget].max_age] * filler))
                for i in range(filler):
                    fitness, cost = self.de[budget].f_objective(new_pop[i], budget)
                    self.de[budget].population = np.vstack((self.de[budget].population,
                                                            new_pop[i]))
                    self.de[budget].fitness = np.append(self.de[budget].fitness, fitness)
                    if fitness < self.inc_score:
                        self.inc_score = fitness
                        self.inc_config = new_pop[i]
                    traj.append(self.inc_score)
                    runtime.append(cost)
                    history.append((new_pop[i].tolist(), float(fitness), float(budget or 0)))

                self.de[budget].inc_score = self.inc_score
                self.de[budget].inc_config = self.inc_config
                self.de[budget].pop_size = pop_size
                if debug:
                    print("Pop size: {}; Len pop: {}".format(self.de[budget].pop_size,
                                                             len(self.de[budget].population)))

            elif pop_size < self.de[budget].pop_size:
                # compensating for extra individuals if SH pop size is smaller
                if debug:
                    print("Reducing population from {} to {} "
                          "for budget {}".format(self.de[budget].pop_size, pop_size, budget))
                # keeping only the top performing individuals/discarding the weak ones
                rank = np.sort(np.argsort(self.de[budget].fitness)[:pop_size])
                self.de[budget].population = self.de[budget].population[rank]
                self.de[budget].fitness = self.de[budget].fitness[rank]
                self.de[budget].age = self.de[budget].age[rank]
                self.de[budget].pop_size = pop_size

            # Successive Halving iterations carrying out DE
            for i_sh in range(num_SH_iters):

                # Warmstarting DE incumbent
                self.de[budget].inc_score = self.inc_score
                self.de[budget].inc_config = self.inc_config
                if debug:
                    print("Pop size: {}; DE budget: {}".format(self.de[budget].pop_size, budget))
                best = self.inc_config

                # Checking population quality to allocate DE budget evolutions
                if debug2:
                    print("Budget: ", budget," --- Pre-gen: ", gens, "; Post-gen: ", end='')
                if iteration > 0:
                    if self.penalty[budget]:
                        gens = int(max(1, np.floor(gens / self.b_eta[budget])))
                        # Consecutive penalty increases the factor by which gens is reduced
                        self.b_eta[budget] = self.b_eta[budget] * self.eta
                        self.penalty[budget] = False
                    else:
                        # If no penalty, then the next penalty, gens will be reduced eta times
                        self.b_eta[budget] = self.eta
                if debug2:
                    print(gens)

                # Repeating DE over entire population 'generations' times
                for gen in range(gens):
                    de_traj, de_runtime, de_history = \
                        self.de[budget].evolve_generation(budget=budget, best=best)
                    traj.extend(de_traj)
                    runtime.extend(de_runtime)
                    history.extend(de_history)

                    # killing/replacing parents that have not changed/has aged
                    # conditional on max_age not being set to inf
                    if self.de[budget].max_age < np.inf:
                        if debug:
                            print("  Generation #{}: Ages -- {}".format(gen + 1,
                                                                        self.de[budget].age))
                        de_traj, de_runtime, de_history = self.de[budget].kill_aged_pop(budget,
                                                                                        debug)
                        traj.extend(de_traj)
                        runtime.extend(de_runtime)
                        history.extend(de_history)

                # Updating global incumbent after each DE step
                self.inc_score = self.de[budget].inc_score
                self.inc_config = self.de[budget].inc_config

                # Score calculation only when not full budget
                if budget < self.max_budget:
                    var = self.convergence_score(budget)
                    # if score has improved, it hints at partial convergence for that DE[budget]
                    # incur penalty to restrict generations evolved
                    if var <= self.tracker[budget]:
                        self.penalty[budget] = True
                    self.tracker[budget] = var

                # Retrieving budget, pop_size, population for the next SH iteration
                if i_sh < num_SH_iters-1:  # when not final SH iteration
                    pop_size = num_configs[i_sh + 1]
                    next_budget = budgets[i_sh + 1]
                    gens = num_gens[i_sh + 1]
                    # selecting top ranking individuals from lower budget
                    ## to be evaluated on higher budget and be eligible for competition
                    rank = np.sort(np.argsort(self.de[budget].fitness)[:pop_size])
                    rival_population = self.de[budget].population[rank]

                    if self.de[next_budget].population is not None:
                        # warmstarting DE incumbents to maintain global trajectory
                        self.de[next_budget].inc_score = self.inc_score
                        self.de[next_budget].inc_config = self.inc_config

                        # ranking individuals to determine population for next SH step
                        de_traj, de_runtime, de_history = \
                            self.de[next_budget].ranked_selection(rival_population, pop_size,
                                                             budget, debug)
                        self.inc_score = self.de[next_budget].inc_score
                        self.inc_config = self.de[next_budget].inc_config
                        traj.extend(de_traj)
                        runtime.extend(de_runtime)
                        history.extend(de_history)
                    else:
                        # equivalent to iteration == 0
                        # no ranked selection happens, rather top ranked individuals are selected
                        if debug:
                            print("Iteration: ", iteration)
                        self.de[next_budget].population = rival_population
                        self.de[next_budget].fitness = self.de[budget].fitness[rank]
                        self.de[next_budget].age = np.array([self.de[next_budget].max_age] * \
                                                        self.de[next_budget].pop_size)
                    budget = next_budget

        if verbose:
            print("\nRun complete!")

        return (np.array(traj), np.array(runtime), np.array(history))


class DEHBV5_2(DEHBBase):
    '''Version 5.2 of DEHB
    At anytime, each set of population contains the best individuals from that budget
    The top individuals from the population evaluated on the previous budget serves as the
        parents for mutation in the next higher budget
    At each SH iteration, the score to measure quality of a DE population is computed
    Based on which, the number of generations passed is allocated.
    '''

    def __init__(self, randomize=None, max_age=np.inf, **kwargs):
        super().__init__(**kwargs)
        self.randomize = randomize
        self.max_age = max_age
        self.generations = None

    def get_next_iteration(self, iteration, type='original'):
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
        if type == 'original':
            n0 = int(np.floor((self.max_SH_iter)/(s+1)) * self.eta**s)
        else:  # type = 'custom'
            n0 = int(np.floor((self.max_SH_iter)/(s+1)) * self.eta**(s-1))
        ns = [max(int(n0*(self.eta**(-i))), 1) for i in range(s+1)]
        if type == 'original':
            if self.min_clip is not None and self.max_clip is not None:
                ns = np.clip(ns, a_min=self.min_clip, a_max=self.max_clip)
            elif self.min_clip is not None:
                ns = np.clip(ns, a_min=self.min_clip, a_max=np.max(ns))

        return ns, budgets

    def convergence_score(self, budget, de):
        std_range = np.max(de[budget].fitness) - np.min(de[budget].fitness)
        if std_range == 0:
            return 0
        median_fitness = np.median(de[budget].fitness) / std_range
        min_fitness = np.min(de[budget].fitness) / std_range
        var = median_fitness - min_fitness
        return var

    def run(self, iterations=1, verbose=False, debug=False, debug2=False):
        # Book-keeping variables
        traj = []
        runtime = []
        history = []

        # To retrieve the population and budget ranges
        num_configs, budgets = self.get_next_iteration(iteration=0)
        full_budget = budgets[-1]
        small_budget = budgets[0]

        # Tracks the variance found in fitness on b in previous iteration for that b
        self.tracker = {}
        self.penalty = {}
        self.b_eta = {}
        for b in budgets:
            self.tracker[b] = 0
            self.penalty[b] = False
            self.b_eta[b] = self.eta

        # List of DE objects corresponding to the populations
        de = {}
        for i, b in enumerate(budgets):
            de[b] = DE(cs=self.cs, f=self.f, dimensions=self.dimensions, pop_size=num_configs[i],
                       mutation_factor=self.mutation_factor, crossover_prob=self.crossover_prob,
                       strategy=self.strategy, budget=b, max_age=self.max_age)

        # Performs DEHB iterations
        for iteration in range(iterations):

            # Retrieves SH budgets and number of configurations
            num_configs, budgets = self.get_next_iteration(iteration=iteration)
            num_gens, _ = self.get_next_iteration(iteration=iteration, type='custom')
            if verbose:
                print('Iteration #{:>3}\n{}'.format(iteration, '-' * 15))
                print(num_configs, budgets, self.inc_score)

            # Sets budget and population size for first SH iteration
            pop_size = num_configs[0]
            budget = budgets[0]
            gens = num_gens[0]

            # Number of SH iterations in this DEHB iteration
            num_SH_iters = len(budgets)

            # Index determining the population to begin DE with for current iteration number
            de_idx = iteration % len(de)

            # The first DEHB iteration - only time when a random population is initialized
            if iteration == 0:
                # creating new population for DEHB iteration to be used for the next SH steps
                de_traj, de_runtime, de_history = de[budget].init_eval_pop(budget)
                # maintaining global copy of random population created
                self.population = de[budget].population
                self.fitness = de[budget].fitness
                # update global incumbent with new population scores
                self.inc_score = de[budget].inc_score
                self.inc_config = de[budget].inc_config
                traj.extend(de_traj)
                runtime.extend(de_runtime)
                history.extend(de_history)

            elif budget == small_budget and self.randomize is not None and self.randomize > 0:
                # executes in the first step of every SH iteration other than first DEHB iteration
                # also conditional on whether a randomization fraction has been specified
                num_replace = np.ceil(self.randomize * pop_size).astype(int)

                # fetching the worst performing individuals
                idxs = np.sort(np.argsort(-de[budget].fitness)[:num_replace])
                if debug:
                    print("Replacing {}/{} -- {}".format(num_replace, pop_size, idxs))
                new_pop = self.init_population(pop_size=num_replace)
                de[budget].population[idxs] = new_pop
                de[budget].age[idxs] = de[budget].max_age
                de[budget].inc_score = self.inc_score
                de[budget].inc_config = self.inc_config

                # evaluating new individuals
                for i in idxs:
                    de[budget].fitness[i], cost = \
                        de[budget].f_objective(de[budget].population[i], budget)
                    if self.fitness[i] < self.inc_score:
                        self.inc_score = de[budget].fitness[i]
                        self.inc_config = de[budget].population[i]
                    traj.append(self.inc_score)
                    runtime.append(cost)
                    history.append((de[budget].population[i].tolist(),
                                    float(de[budget].fitness[i]), float(budget or 0)))

            elif pop_size > de[budget].pop_size:
                # compensating for extra individuals if SH pop size is larger
                filler = pop_size - de[budget].pop_size
                if debug:
                    print("Adding {} random individuals for budget {}".format(filler, budget))
                new_pop = self.init_population(pop_size=filler)
                de[budget].inc_score = self.inc_score
                de[budget].inc_config = self.inc_config
                de[budget].age = np.hstack((de[budget].age, [de[budget].max_age] * filler))
                for i in range(filler):
                    fitness, cost = de[budget].f_objective(new_pop[i], budget)
                    de[budget].population = np.vstack((de[budget].population, new_pop[i]))
                    de[budget].fitness = np.append(de[budget].fitness, fitness)
                    if fitness < self.inc_score:
                        self.inc_score = fitness
                        self.inc_config = new_pop[i]
                    traj.append(self.inc_score)
                    runtime.append(cost)
                    history.append((new_pop[i].tolist(), float(fitness), float(budget or 0)))

                de[budget].inc_score = self.inc_score
                de[budget].inc_config = self.inc_config
                de[budget].pop_size = pop_size
                if debug:
                    print("Pop size: {}; Len pop: {}".format(de[budget].pop_size,
                                                             len(de[budget].population)))

            elif pop_size < de[budget].pop_size:
                # compensating for extra individuals if SH pop size is smaller
                if debug:
                    print("Reducing population from {} to {} "
                          "for budget {}".format(de[budget].pop_size, pop_size, budget))
                # keeping only the top performing individuals/discarding the weak ones
                rank = np.sort(np.argsort(de[budget].fitness)[:pop_size])
                de[budget].population = de[budget].population[rank]
                de[budget].fitness = de[budget].fitness[rank]
                de[budget].age = de[budget].age[rank]
                de[budget].pop_size = pop_size

            # Represents the best individuals from the previous lower SH budget
            # which will serve as the parents for mutation
            alt_population = None

            # Successive Halving iterations carrying out DE
            for i_sh in range(num_SH_iters):

                # Warmstarting DE incumbent
                de[budget].inc_score = self.inc_score
                de[budget].inc_config = self.inc_config
                if debug:
                    print("Pop size: {}; DE budget: {}".format(de[budget].pop_size, budget))
                best = self.inc_config

                # Checking population quality to allocate DE budget evolutions
                if debug2:
                    print("Budget: ", budget," --- Pre-gen: ", gens, "; Post-gen: ", end='')
                if iteration > 0:
                    if self.penalty[budget]:
                        gens = int(max(1, np.floor(gens / self.b_eta[budget])))
                        # Consecutive penalty increases the factor by which gens is reduced
                        self.b_eta[budget] = self.b_eta[budget] * self.eta
                        self.penalty[budget] = False
                    else:
                        # If no penalty, then the next penalty, gens will be reduced eta times
                        self.b_eta[budget] = self.eta
                if debug2:
                    print(gens)

                # Repeating DE over entire population 'generations' times
                for gen in range(gens):
                    de_traj, de_runtime, de_history = \
                        de[budget].evolve_generation(budget=budget, best=best,
                                                     alt_pop=alt_population)
                    traj.extend(de_traj)
                    runtime.extend(de_runtime)
                    history.extend(de_history)

                    # killing/replacing parents that have not changed/has aged
                    # conditional on max_age not being set to inf
                    if de[budget].max_age < np.inf:
                        if debug:
                            print("  Generation #{}: Ages -- {}".format(gen + 1, de[budget].age))
                        de_traj, de_runtime, de_history = de[budget].kill_aged_pop(budget, debug)
                        traj.extend(de_traj)
                        runtime.extend(de_runtime)
                        history.extend(de_history)

                # Updating global incumbent after each DE step
                self.inc_score = de[budget].inc_score
                self.inc_config = de[budget].inc_config

                # Score calculation only when not full budget
                if budget < self.max_budget:
                    var = self.convergence_score(budget, de)
                    # if score has improved, it hints at partial convergence for that DE[budget]
                    # incur penalty to restrict generations evolved
                    if var <= self.tracker[budget]:
                        self.penalty[budget] = True
                    self.tracker[budget] = var

                # Retrieving budget, pop_size, population for the next SH iteration
                if i_sh < num_SH_iters - 1:  # when not final SH iteration
                    pop_size = num_configs[i_sh + 1]
                    next_budget = budgets[i_sh + 1]
                    gens = num_gens[i_sh + 1]

                    # selecting top ranking individuals from lower budget
                    # that will be the mutation parents for the next higher budget
                    rank = np.sort(np.argsort(de[budget].fitness)[:pop_size])
                    alt_population = de[budget].population[rank]

                    if de[next_budget].population is not None:
                        # updating DE incumbents to maintain global trajectory
                        de[next_budget].inc_score = self.inc_score
                        de[next_budget].inc_config = self.inc_config
                    else:
                        # equivalent to iteration == 0
                        # top ranked individuals are selected from the lower budget, assigned as
                        # the population for the next higher budget, and evaluated on it
                        if debug:
                            print("Iteration: ", iteration)
                        de[next_budget].population = alt_population
                        de[next_budget].fitness = [None] * pop_size
                        for index, indv in enumerate(de[next_budget].population):
                            fitness, cost = de[next_budget].f_objective(indv, next_budget)
                            de[next_budget].fitness[index] = fitness
                            if fitness < self.inc_score:
                                self.inc_score = fitness
                                self.inc_config = indv
                            traj.append(self.inc_score)
                            runtime.append(cost)
                            history.append((indv.tolist(), float(fitness), float(budget or 0)))
                        de[next_budget].inc_config = self.inc_config
                        de[next_budget].inc_score = self.inc_score
                        de[next_budget].age = np.array([de[next_budget].max_age] * \
                                                       de[next_budget].pop_size)
                    budget = next_budget
        if verbose:
            print("\nRun complete!")

        return (np.array(traj), np.array(runtime), np.array(history))


class DEHBV6_0(DEHBBase):
    '''DEHB-BOHB-Random
    DEHB version like BOHB with random sampling and no evolution but evaluation on higher budgets
        Only the first iteration of a SH bracket is evolved 'generations' times
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.max_age = np.inf
        self.min_clip = 0
        self.randomize = None

    def run(self, iterations=1, verbose=False, debug=False):
        # Book-keeping variables
        traj = []
        runtime = []
        history = []

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

            # Number of SH iterations in this DEHB iteration
            num_SH_iters = len(budgets)

            # Initializing DE object that will be used across this DEHB iteration
            # The DE object is initialized with the current pop_size and budget
            de = DE(cs=self.cs, f=self.f, dimensions=self.dimensions, pop_size=pop_size,
                    mutation_factor=self.mutation_factor, crossover_prob=self.crossover_prob,
                    strategy=self.strategy, budget=budget, max_age=self.max_age)

            # Warmstarting DE incumbent to be the global incumbent
            de.inc_score = self.inc_score
            de.inc_config = self.inc_config

            # Creating new population for current DEHB iteration
            de.init_eval_pop(budget, eval=False)

            # Successive Halving iterations
            for i_sh in range(num_SH_iters):

                if i_sh == 0:  # first iteration in the SH bracket
                    # Repeating DE over entire population 'generations' times
                    for gen in range(self.generations):
                        de_traj, de_runtime, de_history = de.evolve_generation(budget=budget,
                                                                               best=de.inc_config)
                        traj.extend(de_traj)
                        runtime.extend(de_runtime)
                        history.extend(de_history)
                else:
                    de_traj, de_runtime, de_history = de.eval_pop(budget=budget)
                    traj.extend(de_traj)
                    runtime.extend(de_runtime)
                    history.extend(de_history)

                # Retrieving budget, pop_size, population for the next SH iteration
                if i_sh < num_SH_iters - 1:  # when not final SH iteration
                    pop_size = num_configs[i_sh + 1]
                    budget = budgets[i_sh + 1]
                    # Selecting top individuals to fit pop_size of next SH iteration
                    self.rank = np.sort(np.argsort(de.fitness)[:pop_size])
                    de.population = de.population[self.rank]
                    de.fitness = de.fitness[self.rank]
                    de.age = de.age[self.rank]
                    de.pop_size = pop_size
                    de.budget = budget

            # Updating global incumbent after each DEHB iteration
            self.inc_score = de.inc_score
            self.inc_config = de.inc_config

        if verbose:
            print("\nRun complete!")

        return (np.array(traj), np.array(runtime), np.array(history))


class DEHBV6_1(DEHBBase):
    '''DEHB-BOHB-Previous

    Maintains a separate population for each budget. Only the very first SH iteration of the very
        first DEHB iteration is randomly sampled. Top individuals are forwarded to the next higher
        budget and so on.
    For an iteration i for a budget b1, it chooses the population for b2 by performing a
        ranked selection for the population for b2 from iteration i-1 and the population forwarded
        from b1 in iteration i.
    For the case when the population size suggested by SH differs from the current population size
        contained by DE populations for each budget, some adjustments are made.
        If suggested pop size > current pop size
            Sample (suggested pop size - current pop size) individuals randomly and set their
            fitness to infinity, and continue.
        else if suggested pop size < current pop size
            Retain only the top 'suggested pop size' individuals based on their fitness values
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.max_age = np.inf
        self.min_clip = 0
        self.randomize = None

    def run(self, iterations=1, verbose=False, debug=False):
        # Book-keeping variables
        traj = []
        runtime = []
        history = []

        # To retrieve the population and budget ranges
        num_configs, budgets = self.get_next_iteration(iteration=0)
        full_budget = budgets[-1]
        small_budget = budgets[0]

        # List of DE objects corresponding to the budgets (fidelities)
        de = {}
        for i, b in enumerate(budgets):
            de[b] = DE(cs=self.cs, f=self.f, dimensions=self.dimensions, pop_size=num_configs[i],
                       mutation_factor=self.mutation_factor, crossover_prob=self.crossover_prob,
                       strategy=self.strategy, budget=b, max_age=self.max_age)

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

            # Number of SH iterations in this DEHB iteration
            num_SH_iters = len(budgets)

            # The first DEHB iteration - only time when a random population is initialized
            if iteration == 0:
                # creating new population for DEHB iteration to be used for the next SH steps
                de[budget].init_eval_pop(budget, eval=False)
                # maintaining global copy of random population created
                self.population = de[budget].population
                self.fitness = de[budget].fitness
                # update global incumbent with new population scores
                self.inc_score = de[budget].inc_score
                self.inc_config = de[budget].inc_config

            elif pop_size > de[budget].pop_size:
                # compensating for extra individuals if SH pop size is larger
                filler = pop_size - de[budget].pop_size
                if debug:
                    print("Adding {} random individuals for budget {}".format(filler, budget))
                # randomly sample to fill the size and set fitness to infinity (no evaluation)
                new_pop = self.init_population(pop_size=filler)
                de[budget].inc_score = self.inc_score
                de[budget].inc_config = self.inc_config
                de[budget].population = np.vstack((de[budget].population, new_pop))
                de[budget].age = np.hstack((de[budget].age, [de[budget].max_age] * filler))
                de[budget].fitness = np.hstack((de[budget].fitness, [np.inf] * filler))

            elif pop_size < de[budget].pop_size:
                # compensating for extra individuals if SH pop size is smaller
                if debug:
                    print("Reducing population from {} to {} "
                          "for budget {}".format(de[budget].pop_size, pop_size, budget))
                # keeping only the top performing individuals/discarding the weak ones
                rank = np.sort(np.argsort(de[budget].fitness)[:pop_size])
                de[budget].population = de[budget].population[rank]
                de[budget].fitness = de[budget].fitness[rank]
                de[budget].age = de[budget].age[rank]

            # Adjusting pop size parameter
            de[budget].pop_size = pop_size

            # Successive Halving iterations carrying out DE
            for i_sh in range(num_SH_iters):

                # Warmstarting DE incumbent
                de[budget].inc_score = self.inc_score
                de[budget].inc_config = self.inc_config
                if debug:
                    print("Pop size: {}; DE budget: {}".format(de[budget].pop_size, budget))
                best = self.inc_config

                if i_sh == 0:  # first iteration in the SH bracket
                    # Repeating DE over entire population 'generations' times
                    for gen in range(self.generations):
                        de_traj, de_runtime, de_history = \
                            de[budget].evolve_generation(budget=budget, best=best)
                        traj.extend(de_traj)
                        runtime.extend(de_runtime)
                        history.extend(de_history)

                elif i_sh > 0 and iteration == 0:
                    de_traj, de_runtime, de_history = de[budget].eval_pop(budget=budget)
                    traj.extend(de_traj)
                    runtime.extend(de_runtime)
                    history.extend(de_history)

                # Updating global incumbent after each DE step for a budget
                self.inc_score = de[budget].inc_score
                self.inc_config = de[budget].inc_config

                # Retrieving budget, pop_size, population for the next SH iteration
                if i_sh < num_SH_iters-1:  # when not final SH iteration
                    pop_size = num_configs[i_sh + 1]
                    next_budget = budgets[i_sh + 1]
                    # selecting top ranking individuals from lower budget
                    ## to be evaluated on higher budget and be eligible for competition
                    rank = np.sort(np.argsort(de[budget].fitness)[:pop_size])
                    rival_population = de[budget].population[rank]

                    if iteration > 0:
                        # warmstarting DE incumbents to maintain global trajectory
                        de[next_budget].inc_score = self.inc_score
                        de[next_budget].inc_config = self.inc_config
                        de[next_budget].pop_size = pop_size

                        # ranking individuals to determine population for next SH step
                        de_traj, de_runtime, de_history = \
                            de[next_budget].ranked_selection(rival_population, pop_size,
                                                             next_budget, debug)
                        self.inc_score = de[next_budget].inc_score
                        self.inc_config = de[next_budget].inc_config
                        traj.extend(de_traj)
                        runtime.extend(de_runtime)
                        history.extend(de_history)
                    else:
                        # equivalent to iteration == 0
                        # no ranked selection happens, rather top ranked individuals are selected
                        if debug:
                            print("Iteration: ", iteration)
                        de[next_budget].population = rival_population
                        de[next_budget].fitness = de[budget].fitness[rank]
                        de[next_budget].age = np.array([de[next_budget].max_age] * \
                                                        de[next_budget].pop_size)
                        de[next_budget].pop_size = pop_size
                    budget = next_budget

        if verbose:
            print("\nRun complete!")

        return (np.array(traj), np.array(runtime), np.array(history))


class DEHBV6_1_2(DEHBBase):
    '''DEHB-BOHB-Previous

    Maintains a separate population for each budget. Only the very first SH iteration of the very
        first DEHB iteration is randomly sampled. Top individuals are forwarded to the next higher
        budget and so on.
    For an iteration i for a budget b1, it chooses the population for b2 by performing a
        ranked selection for the population for b2 from iteration i-1 and the population forwarded
        from b1 in iteration i.
    For the case when the population size suggested by SH differs from the current population size
        contained by DE populations for each budget, some adjustments are made.
        If suggested pop size > current pop size
            Sample (suggested pop size - current pop size) individuals randomly and set their
            fitness to infinity, and continue.
        else if suggested pop size < current pop size
            Retain only the top 'suggested pop size' individuals based on their fitness values
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.max_age = np.inf
        self.min_clip = 0
        self.randomize = None

    def run(self, iterations=1, verbose=False, debug=False):
        # Book-keeping variables
        traj = []
        runtime = []
        history = []

        # To retrieve the population and budget ranges
        num_configs, budgets = self.get_next_iteration(iteration=0)
        full_budget = budgets[-1]
        small_budget = budgets[0]

        # List of DE objects corresponding to the budgets (fidelities)
        de = {}
        for i, b in enumerate(budgets):
            de[b] = DE(cs=self.cs, f=self.f, dimensions=self.dimensions, pop_size=num_configs[i],
                       mutation_factor=self.mutation_factor, crossover_prob=self.crossover_prob,
                       strategy=self.strategy, budget=b, max_age=self.max_age)

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

            # Number of SH iterations in this DEHB iteration
            num_SH_iters = len(budgets)

            # The first DEHB iteration - only time when a random population is initialized
            if iteration == 0:
                # creating new population for DEHB iteration to be used for the next SH steps
                de[budget].init_eval_pop(budget, eval=False)
                # maintaining global copy of random population created
                self.population = de[budget].population
                self.fitness = de[budget].fitness
                # update global incumbent with new population scores
                self.inc_score = de[budget].inc_score
                self.inc_config = de[budget].inc_config

            elif pop_size > de[budget].pop_size:
                # compensating for extra individuals if SH pop size is larger
                filler = pop_size - de[budget].pop_size
                if debug:
                    print("Adding {} random individuals for budget {}".format(filler, budget))
                # randomly sample to fill the size and set fitness to infinity (no evaluation)
                new_pop = self.init_population(pop_size=filler)
                de[budget].inc_score = self.inc_score
                de[budget].inc_config = self.inc_config
                de[budget].population = np.vstack((de[budget].population, new_pop))
                de[budget].age = np.hstack((de[budget].age, [de[budget].max_age] * filler))
                de[budget].fitness = np.hstack((de[budget].fitness, [np.inf] * filler))

            elif pop_size < de[budget].pop_size:
                # compensating for extra individuals if SH pop size is smaller
                if debug:
                    print("Reducing population from {} to {} "
                          "for budget {}".format(de[budget].pop_size, pop_size, budget))
                # shuffling to have the top individuals at the beginning of the population
                rank_include = np.sort(np.argsort(de[budget].fitness)[:pop_size])
                rank_ignore = np.sort(list(set(np.arange(len(de[budget].fitness))) -
                                           set(rank_include)))
                de[budget].population = np.vstack((de[budget].population[rank_include],
                                                   de[budget].population[rank_ignore]))
                de[budget].fitness = np.vstack((de[budget].fitness[rank_include],
                                                de[budget].fitness[rank_ignore]))
                de[budget].age = np.vstack((de[budget].age[rank_include],
                                                de[budget].age[rank_ignore]))

            # Adjusting pop size parameter
            de[budget].pop_size = pop_size

            # Successive Halving iterations carrying out DE
            for i_sh in range(num_SH_iters):

                # Warmstarting DE incumbent
                de[budget].inc_score = self.inc_score
                de[budget].inc_config = self.inc_config
                if debug:
                    print("Pop size: {}; DE budget: {}".format(de[budget].pop_size, budget))
                best = self.inc_config

                if i_sh == 0:  # first iteration in the SH bracket
                    # Repeating DE over entire population 'generations' times
                    for gen in range(self.generations):
                        de_traj, de_runtime, de_history = \
                            de[budget].evolve_generation(budget=budget, best=best)
                        traj.extend(de_traj)
                        runtime.extend(de_runtime)
                        history.extend(de_history)

                elif i_sh > 0 and iteration == 0:
                    de_traj, de_runtime, de_history = de[budget].eval_pop(budget=budget)
                    traj.extend(de_traj)
                    runtime.extend(de_runtime)
                    history.extend(de_history)

                # Updating global incumbent after each DE step for a budget
                self.inc_score = de[budget].inc_score
                self.inc_config = de[budget].inc_config

                # Retrieving budget, pop_size, population for the next SH iteration
                if i_sh < num_SH_iters-1:  # when not final SH iteration
                    pop_size = num_configs[i_sh + 1]
                    next_budget = budgets[i_sh + 1]
                    # selecting top ranking individuals from lower budget
                    ## to be evaluated on higher budget and be eligible for competition
                    rank = np.sort(np.argsort(de[budget].fitness)[:pop_size])
                    rival_population = de[budget].population[rank]

                    if iteration > 0:
                        # warmstarting DE incumbents to maintain global trajectory
                        de[next_budget].inc_score = self.inc_score
                        de[next_budget].inc_config = self.inc_config
                        de[next_budget].pop_size = pop_size

                        # ranking individuals to determine population for next SH step
                        de_traj, de_runtime, de_history = \
                            de[next_budget].ranked_selection(rival_population, pop_size,
                                                             next_budget, debug)
                        self.inc_score = de[next_budget].inc_score
                        self.inc_config = de[next_budget].inc_config
                        traj.extend(de_traj)
                        runtime.extend(de_runtime)
                        history.extend(de_history)
                    else:
                        # equivalent to iteration == 0
                        # no ranked selection happens, rather top ranked individuals are selected
                        if debug:
                            print("Iteration: ", iteration)
                        de[next_budget].population = rival_population
                        de[next_budget].fitness = de[budget].fitness[rank]
                        de[next_budget].age = np.array([de[next_budget].max_age] * \
                                                        de[next_budget].pop_size)
                        de[next_budget].pop_size = pop_size
                    budget = next_budget

        if verbose:
            print("\nRun complete!")

        return (np.array(traj), np.array(runtime), np.array(history))


class DEHBV6_2(DEHBBase):
    '''DEHB-BOHB-Previous

    Maintains a separate population for each budget. Only the very first SH iteration of the very
        first DEHB iteration is randomly sampled. Top individuals are forwarded to the next higher
        budget and so on.
    Uses the next budget population for mutation sampling.
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.max_age = np.inf
        self.min_clip = 0
        self.randomize = None

    def run(self, iterations=1, verbose=False, debug=False):
        # Book-keeping variables
        traj = []
        runtime = []
        history = []

        # To retrieve the population and budget ranges
        num_configs, budgets = self.get_next_iteration(iteration=0)
        full_budget = budgets[-1]
        small_budget = budgets[0]

        # List of DE objects corresponding to the budgets (fidelities)
        de = {}
        for i, b in enumerate(budgets):
            de[b] = DE(cs=self.cs, f=self.f, dimensions=self.dimensions, pop_size=num_configs[i],
                       mutation_factor=self.mutation_factor, crossover_prob=self.crossover_prob,
                       strategy=self.strategy, budget=b, max_age=self.max_age)

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

            # Number of SH iterations in this DEHB iteration
            num_SH_iters = len(budgets)

            # The first DEHB iteration - only time when a random population is initialized
            if iteration == 0:
                # creating new population for DEHB iteration to be used for the next SH steps
                de[budget].init_eval_pop(budget, eval=False)
                # maintaining global copy of random population created
                self.population = de[budget].population
                self.fitness = de[budget].fitness
                # update global incumbent with new population scores
                self.inc_score = de[budget].inc_score
                self.inc_config = de[budget].inc_config

            elif pop_size > de[budget].pop_size:
                # compensating for extra individuals if SH pop size is larger
                filler = pop_size - de[budget].pop_size
                if debug:
                    print("Adding {} random individuals for budget {}".format(filler, budget))
                # randomly sample to fill the size and set fitness to infinity (no evaluation)
                new_pop = self.init_population(pop_size=filler)
                de[budget].inc_score = self.inc_score
                de[budget].inc_config = self.inc_config
                de[budget].population = np.vstack((de[budget].population, new_pop))
                de[budget].age = np.hstack((de[budget].age, [de[budget].max_age] * filler))
                de[budget].fitness = np.hstack((de[budget].fitness, [np.inf] * filler))

            elif pop_size < de[budget].pop_size:
                # compensating for extra individuals if SH pop size is smaller
                if debug:
                    print("Reducing population from {} to {} "
                          "for budget {}".format(de[budget].pop_size, pop_size, budget))
                # keeping only the top performing individuals/discarding the weak ones
                rank = np.sort(np.argsort(de[budget].fitness)[:pop_size])
                de[budget].population = de[budget].population[rank]
                de[budget].fitness = de[budget].fitness[rank]
                de[budget].age = de[budget].age[rank]

            # Adjusting pop size parameter
            de[budget].pop_size = pop_size

            # Successive Halving iterations carrying out DE
            for i_sh in range(num_SH_iters):

                # Warmstarting DE incumbent
                de[budget].inc_score = self.inc_score
                de[budget].inc_config = self.inc_config
                if debug:
                    print("Pop size: {}; DE budget: {}".format(de[budget].pop_size, budget))
                best = self.inc_config

                # Population for mutation sampling
                ## For the first iteration, the current population itself
                ## For subsequent iterations, the next higher population
                ## For the full budget, the current population itself
                next_budget = budgets[min(i_sh + 1, num_SH_iters - 1)]
                alt_population = de[next_budget].population

                if i_sh == 0:  # first iteration in the SH bracket
                    # Repeating DE over entire population 'generations' times
                    for gen in range(self.generations):
                        de_traj, de_runtime, de_history = \
                            de[budget].evolve_generation(budget=budget, best=best,
                                                         alt_pop=alt_population)
                        traj.extend(de_traj)
                        runtime.extend(de_runtime)
                        history.extend(de_history)

                elif i_sh > 0 and iteration == 0:
                    de_traj, de_runtime, de_history = de[budget].eval_pop(budget=budget)
                    traj.extend(de_traj)
                    runtime.extend(de_runtime)
                    history.extend(de_history)

                # Updating global incumbent after each DE step for a budget
                self.inc_score = de[budget].inc_score
                self.inc_config = de[budget].inc_config

                # Retrieving budget, pop_size, population for the next SH iteration
                if i_sh < num_SH_iters-1:  # when not final SH iteration
                    pop_size = num_configs[i_sh + 1]
                    next_budget = budgets[i_sh + 1]
                    # selecting top ranking individuals from lower budget
                    # to be evaluated on higher budget and be eligible for competition
                    rank = np.sort(np.argsort(de[budget].fitness)[:pop_size])
                    rival_population = de[budget].population[rank]
                    # alt_population = de[next_budget].population
                    # if alt_population is not None and len(alt_population) < 3:
                    #     alt_population = np.vstack((alt_population, rival_population))

                    if iteration > 0:
                        # warmstarting DE incumbents to maintain global trajectory
                        de[next_budget].inc_score = self.inc_score
                        de[next_budget].inc_config = self.inc_config
                        de[next_budget].pop_size = pop_size

                        # ranking individuals to determine population for next SH step
                        ## top individuals from budget are evaluated on next_budget
                        ## inside ranked_selection
                        de_traj, de_runtime, de_history = \
                            de[next_budget].ranked_selection(rival_population, pop_size,
                                                             next_budget, debug)
                        self.inc_score = de[next_budget].inc_score
                        self.inc_config = de[next_budget].inc_config
                        traj.extend(de_traj)
                        runtime.extend(de_runtime)
                        history.extend(de_history)
                    else:
                        # equivalent to iteration == 0
                        # no ranked selection happens, top ranked individuals are selected
                        if debug:
                            print("Iteration: ", iteration)
                        de[next_budget].population = rival_population
                        de[next_budget].fitness = de[budget].fitness[rank]
                        de[next_budget].age = np.array([de[next_budget].max_age] * \
                                                        de[next_budget].pop_size)
                        de[next_budget].pop_size = pop_size
                    budget = next_budget

        if verbose:
            print("\nRun complete!")

        return (np.array(traj), np.array(runtime), np.array(history))


class DEHBV6_2_2(DEHBBase):
    '''DEHB-BOHB-Previous

    Maintains a separate population for each budget. Only the very first SH iteration of the very
        first DEHB iteration is randomly sampled. Top individuals are forwarded to the next higher
        budget and so on.
    Uses the next budget population for mutation sampling.
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.max_age = np.inf
        self.min_clip = 0
        self.randomize = None

    def run(self, iterations=1, verbose=False, debug=False):
        # Book-keeping variables
        traj = []
        runtime = []
        history = []

        # To retrieve the population and budget ranges
        num_configs, budgets = self.get_next_iteration(iteration=0)
        full_budget = budgets[-1]
        small_budget = budgets[0]

        # List of DE objects corresponding to the budgets (fidelities)
        de = {}
        for i, b in enumerate(budgets):
            de[b] = DE(cs=self.cs, f=self.f, dimensions=self.dimensions, pop_size=num_configs[i],
                       mutation_factor=self.mutation_factor, crossover_prob=self.crossover_prob,
                       strategy=self.strategy, budget=b, max_age=self.max_age)

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

            # Number of SH iterations in this DEHB iteration
            num_SH_iters = len(budgets)

            # The first DEHB iteration - only time when a random population is initialized
            if iteration == 0:
                # creating new population for DEHB iteration to be used for the next SH steps
                de[budget].init_eval_pop(budget, eval=False)
                # maintaining global copy of random population created
                self.population = de[budget].population
                self.fitness = de[budget].fitness
                # update global incumbent with new population scores
                self.inc_score = de[budget].inc_score
                self.inc_config = de[budget].inc_config

            elif pop_size > de[budget].pop_size:
                # compensating for extra individuals if SH pop size is larger
                filler = pop_size - de[budget].pop_size
                if debug:
                    print("Adding {} random individuals for budget {}".format(filler, budget))
                # randomly sample to fill the size and set fitness to infinity (no evaluation)
                new_pop = self.init_population(pop_size=filler)
                de[budget].inc_score = self.inc_score
                de[budget].inc_config = self.inc_config
                de[budget].population = np.vstack((de[budget].population, new_pop))
                de[budget].age = np.hstack((de[budget].age, [de[budget].max_age] * filler))
                de[budget].fitness = np.hstack((de[budget].fitness, [np.inf] * filler))

            elif pop_size < de[budget].pop_size:
                # compensating for extra individuals if SH pop size is smaller
                if debug:
                    print("Reducing population from {} to {} "
                          "for budget {}".format(de[budget].pop_size, pop_size, budget))
                # reordering to have the top individuals at the beginning of the population
                ## de[budget].pop_size controls the view over the entire list
                rank_include = np.sort(np.argsort(de[budget].fitness)[:pop_size])
                rank_ignore = np.sort(list(set(np.arange(len(de[budget].fitness))) -
                                           set(rank_include)))
                de[budget].population = np.vstack((de[budget].population[rank_include],
                                                   de[budget].population[rank_ignore]))
                de[budget].fitness = np.vstack((de[budget].fitness[rank_include],
                                                de[budget].fitness[rank_ignore]))
                de[budget].age = np.vstack((de[budget].age[rank_include],
                                                de[budget].age[rank_ignore]))

            # Adjusting pop size parameter
            de[budget].pop_size = pop_size

            # Successive Halving iterations carrying out DE
            for i_sh in range(num_SH_iters):

                # Warmstarting DE incumbent
                de[budget].inc_score = self.inc_score
                de[budget].inc_config = self.inc_config
                if debug:
                    print("Pop size: {}; DE budget: {}".format(de[budget].pop_size, budget))
                best = self.inc_config

                # Population for mutation sampling
                ## For the first iteration, the current population itself
                ## For subsequent iterations, the next higher population
                ## For the full budget, the current population itself
                next_budget = budgets[min(i_sh + 1, num_SH_iters - 1)]
                alt_population = de[next_budget].population

                if i_sh == 0:  # first iteration in the SH bracket
                    # Repeating DE over entire population 'generations' times
                    for gen in range(self.generations):
                        de_traj, de_runtime, de_history = \
                            de[budget].evolve_generation(budget=budget, best=best,
                                                         alt_pop=alt_population)
                        traj.extend(de_traj)
                        runtime.extend(de_runtime)
                        history.extend(de_history)

                elif i_sh > 0 and iteration == 0:
                    de_traj, de_runtime, de_history = de[budget].eval_pop(budget=budget)
                    traj.extend(de_traj)
                    runtime.extend(de_runtime)
                    history.extend(de_history)

                # Updating global incumbent after each DE step for a budget
                self.inc_score = de[budget].inc_score
                self.inc_config = de[budget].inc_config

                # Retrieving budget, pop_size, population for the next SH iteration
                if i_sh < num_SH_iters-1:  # when not final SH iteration
                    pop_size = num_configs[i_sh + 1]
                    next_budget = budgets[i_sh + 1]
                    # selecting top ranking individuals from lower budget
                    # to be evaluated on higher budget and be eligible for competition
                    rank = np.sort(np.argsort(de[budget].fitness)[:pop_size])
                    rival_population = de[budget].population[rank]

                    if iteration > 0:
                        # warmstarting DE incumbents to maintain global trajectory
                        de[next_budget].inc_score = self.inc_score
                        de[next_budget].inc_config = self.inc_config
                        de[next_budget].pop_size = pop_size

                        # ranking individuals to determine population for next SH step
                        de_traj, de_runtime, de_history = \
                            de[next_budget].ranked_selection(rival_population, pop_size,
                                                             next_budget, debug)
                        self.inc_score = de[next_budget].inc_score
                        self.inc_config = de[next_budget].inc_config
                        traj.extend(de_traj)
                        runtime.extend(de_runtime)
                        history.extend(de_history)
                    else:
                        # equivalent to iteration == 0
                        # no ranked selection happens, top ranked individuals are selected
                        if debug:
                            print("Iteration: ", iteration)
                        de[next_budget].population = rival_population
                        de[next_budget].fitness = de[budget].fitness[rank]
                        de[next_budget].age = np.array([de[next_budget].max_age] * \
                                                        de[next_budget].pop_size)
                        de[next_budget].pop_size = pop_size
                    budget = next_budget

        if verbose:
            print("\nRun complete!")

        return (np.array(traj), np.array(runtime), np.array(history))


class DEHBV6_3(DEHBBase):
    '''DEHB-BOHB-Previous

    Maintains a separate population for each budget. Only the very first SH iteration of the very
        first DEHB iteration is randomly sampled. Top individuals are forwarded to the next higher
        budget and so on.
    Uses the next budget population for mutation sampling.
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.max_age = np.inf
        self.min_clip = 0
        self.randomize = None
        self.generations = None

    def get_next_iteration(self, iteration, type='original'):
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
        if type == 'original':
            n0 = int(np.floor((self.max_SH_iter)/(s+1)) * self.eta**s)
        else:  # type = 'custom'
            n0 = int(np.floor((self.max_SH_iter)/(s+1)) * self.eta**(s-1))
        ns = [max(int(n0*(self.eta**(-i))), 1) for i in range(s+1)]
        if type == 'original':
            if self.min_clip is not None and self.max_clip is not None:
                ns = np.clip(ns, a_min=self.min_clip, a_max=self.max_clip)
            elif self.min_clip is not None:
                ns = np.clip(ns, a_min=self.min_clip, a_max=np.max(ns))

        return ns, budgets

    def run(self, iterations=1, verbose=False, debug=False):
        # Book-keeping variables
        traj = []
        runtime = []
        history = []

        # To retrieve the population and budget ranges
        num_configs, budgets = self.get_next_iteration(iteration=0)
        full_budget = budgets[-1]
        small_budget = budgets[0]

        # List of DE objects corresponding to the budgets (fidelities)
        de = {}
        for i, b in enumerate(budgets):
            de[b] = DE(cs=self.cs, f=self.f, dimensions=self.dimensions, pop_size=num_configs[i],
                       mutation_factor=self.mutation_factor, crossover_prob=self.crossover_prob,
                       strategy=self.strategy, budget=b, max_age=self.max_age)

        # Performs DEHB iterations
        for iteration in range(iterations):

            # Retrieves SH budgets and number of configurations
            num_configs, budgets = self.get_next_iteration(iteration=iteration)
            num_gens, _ = self.get_next_iteration(iteration=iteration, type='custom')
            if verbose:
                print('Iteration #{:>3}\n{}'.format(iteration, '-' * 15))
                print(num_configs, budgets, self.inc_score)

            # Sets budget, population size, generations for first SH iteration
            pop_size = num_configs[0]
            budget = budgets[0]
            gens = num_gens[0]

            # Number of SH iterations in this DEHB iteration
            num_SH_iters = len(budgets)

            # The first DEHB iteration - only time when a random population is initialized
            if iteration == 0:
                # creating new population for DEHB iteration to be used for the next SH steps
                de[budget].init_eval_pop(budget, eval=False)
                # maintaining global copy of random population created
                self.population = de[budget].population
                self.fitness = de[budget].fitness
                # update global incumbent with new population scores
                self.inc_score = de[budget].inc_score
                self.inc_config = de[budget].inc_config

            elif pop_size > de[budget].pop_size:
                # compensating for extra individuals if SH pop size is larger
                filler = pop_size - de[budget].pop_size
                if debug:
                    print("Adding {} random individuals for budget {}".format(filler, budget))
                # randomly sample to fill the size and set fitness to infinity (no evaluation)
                new_pop = self.init_population(pop_size=filler)
                de[budget].inc_score = self.inc_score
                de[budget].inc_config = self.inc_config
                de[budget].population = np.vstack((de[budget].population, new_pop))
                de[budget].age = np.hstack((de[budget].age, [de[budget].max_age] * filler))
                de[budget].fitness = np.hstack((de[budget].fitness, [np.inf] * filler))

            elif pop_size < de[budget].pop_size:
                # compensating for extra individuals if SH pop size is smaller
                if debug:
                    print("Reducing population from {} to {} "
                          "for budget {}".format(de[budget].pop_size, pop_size, budget))
                # reordering to have the top individuals at the beginning of the population
                ## de[budget].pop_size controls the view over the entire list
                rank_include = np.sort(np.argsort(de[budget].fitness)[:pop_size])
                rank_ignore = np.sort(list(set(np.arange(len(de[budget].fitness))) -
                                           set(rank_include)))
                de[budget].population = np.vstack((de[budget].population[rank_include],
                                                   de[budget].population[rank_ignore]))
                de[budget].fitness = np.vstack((de[budget].fitness[rank_include],
                                                de[budget].fitness[rank_ignore]))
                de[budget].age = np.vstack((de[budget].age[rank_include],
                                                de[budget].age[rank_ignore]))

            # Adjusting pop size parameter
            de[budget].pop_size = pop_size

            # Successive Halving iterations carrying out DE
            for i_sh in range(num_SH_iters):

                # Warmstarting DE incumbent
                de[budget].inc_score = self.inc_score
                de[budget].inc_config = self.inc_config
                if debug:
                    print("Pop size: {}; DE budget: {}".format(de[budget].pop_size, budget))
                best = self.inc_config

                # Population for mutation sampling
                ## For the first iteration, the current population itself
                ## For subsequent iterations, the next higher population
                ## For the full budget, the current population itself
                next_budget = budgets[min(i_sh + 1, num_SH_iters - 1)]
                alt_population = de[next_budget].population

                if i_sh == 0:  # first iteration in the SH bracket
                    # Repeating DE over entire population 'generations' times
                    for gen in range(gens):
                        de_traj, de_runtime, de_history = \
                            de[budget].evolve_generation(budget=budget, best=best,
                                                         alt_pop=alt_population)
                        traj.extend(de_traj)
                        runtime.extend(de_runtime)
                        history.extend(de_history)

                elif i_sh > 0 and iteration == 0:
                    de_traj, de_runtime, de_history = de[budget].eval_pop(budget=budget)
                    traj.extend(de_traj)
                    runtime.extend(de_runtime)
                    history.extend(de_history)

                # Updating global incumbent after each DE step for a budget
                self.inc_score = de[budget].inc_score
                self.inc_config = de[budget].inc_config

                # Retrieving budget, pop_size, population for the next SH iteration
                if i_sh < num_SH_iters-1:  # when not final SH iteration
                    pop_size = num_configs[i_sh + 1]
                    next_budget = budgets[i_sh + 1]
                    # selecting top ranking individuals from lower budget
                    # to be evaluated on higher budget and be eligible for competition
                    rank = np.sort(np.argsort(de[budget].fitness)[:pop_size])
                    rival_population = de[budget].population[rank]

                    if iteration > 0:
                        # warmstarting DE incumbents to maintain global trajectory
                        de[next_budget].inc_score = self.inc_score
                        de[next_budget].inc_config = self.inc_config
                        de[next_budget].pop_size = pop_size

                        # ranking individuals to determine population for next SH step
                        de_traj, de_runtime, de_history = \
                            de[next_budget].ranked_selection(rival_population, pop_size,
                                                             next_budget, debug)
                        self.inc_score = de[next_budget].inc_score
                        self.inc_config = de[next_budget].inc_config
                        traj.extend(de_traj)
                        runtime.extend(de_runtime)
                        history.extend(de_history)
                    else:
                        # equivalent to iteration == 0
                        # no ranked selection happens, top ranked individuals are selected
                        if debug:
                            print("Iteration: ", iteration)
                        de[next_budget].population = rival_population
                        de[next_budget].fitness = de[budget].fitness[rank]
                        de[next_budget].age = np.array([de[next_budget].max_age] * \
                                                        de[next_budget].pop_size)
                        de[next_budget].pop_size = pop_size
                    budget = next_budget

        if verbose:
            print("\nRun complete!")

        return (np.array(traj), np.array(runtime), np.array(history))


class DEHBV6_3_2(DEHBBase):
    '''DEHB-BOHB-Previous

    Maintains a separate population for each budget. Only the very first SH iteration of the very
        first DEHB iteration is randomly sampled. Top individuals are forwarded to the next higher
        budget and so on.
    Uses the next budget population for mutation sampling.
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.max_age = np.inf
        self.min_clip = 0
        self.randomize = None
        self.generations = 1 if self.generations is None else self.generations

    def get_next_iteration(self, iteration, type='original'):
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
        if type == 'original':
            n0 = int(np.floor((self.max_SH_iter)/(s+1)) * self.eta**s)
        else:  # type = 'custom'
            n0 = int(np.floor((self.max_SH_iter)/(s+1)) * self.eta**(s-1))
        ns = [max(int(n0*(self.eta**(-i))), 1) for i in range(s+1)]
        if type == 'original':
            if self.min_clip is not None and self.max_clip is not None:
                ns = np.clip(ns, a_min=self.min_clip, a_max=self.max_clip)
            elif self.min_clip is not None:
                ns = np.clip(ns, a_min=self.min_clip, a_max=np.max(ns))

        return ns, budgets

    def run(self, iterations=1, verbose=False, debug=False):
        # Book-keeping variables
        traj = []
        runtime = []
        history = []

        # To retrieve the population and budget ranges
        num_configs, budgets = self.get_next_iteration(iteration=0)
        full_budget = budgets[-1]
        small_budget = budgets[0]

        # List of DE objects corresponding to the budgets (fidelities)
        de = {}
        for i, b in enumerate(budgets):
            de[b] = DE(cs=self.cs, f=self.f, dimensions=self.dimensions, pop_size=num_configs[i],
                       mutation_factor=self.mutation_factor, crossover_prob=self.crossover_prob,
                       strategy=self.strategy, budget=b, max_age=self.max_age)

        # Performs DEHB iterations
        for iteration in range(iterations):

            # Retrieves SH budgets and number of configurations
            num_configs, budgets = self.get_next_iteration(iteration=iteration)
            num_gens, _ = self.get_next_iteration(iteration=iteration, type='custom')
            if verbose:
                print('Iteration #{:>3}\n{}'.format(iteration, '-' * 15))
                print(num_configs, budgets, self.inc_score)

            # Sets budget, population size, generations for first SH iteration
            pop_size = num_configs[0]
            budget = budgets[0]
            gens = num_gens[0]

            # Number of SH iterations in this DEHB iteration
            num_SH_iters = len(budgets)

            # The first DEHB iteration - only time when a random population is initialized
            if iteration == 0:
                # creating new population for DEHB iteration to be used for the next SH steps
                de[budget].init_eval_pop(budget, eval=False)
                # maintaining global copy of random population created
                self.population = de[budget].population
                self.fitness = de[budget].fitness
                # update global incumbent with new population scores
                self.inc_score = de[budget].inc_score
                self.inc_config = de[budget].inc_config

            elif pop_size > de[budget].pop_size:
                # compensating for extra individuals if SH pop size is larger
                filler = pop_size - de[budget].pop_size
                if debug:
                    print("Adding {} random individuals for budget {}".format(filler, budget))
                # randomly sample to fill the size and set fitness to infinity (no evaluation)
                new_pop = self.init_population(pop_size=filler)
                de[budget].inc_score = self.inc_score
                de[budget].inc_config = self.inc_config
                de[budget].population = np.vstack((de[budget].population, new_pop))
                de[budget].age = np.hstack((de[budget].age, [de[budget].max_age] * filler))
                de[budget].fitness = np.hstack((de[budget].fitness, [np.inf] * filler))

            elif pop_size < de[budget].pop_size:
                # compensating for extra individuals if SH pop size is smaller
                if debug:
                    print("Reducing population from {} to {} "
                          "for budget {}".format(de[budget].pop_size, pop_size, budget))
                # reordering to have the top individuals at the beginning of the population
                ## de[budget].pop_size controls the view over the entire list
                rank_include = np.sort(np.argsort(de[budget].fitness)[:pop_size])
                rank_ignore = np.sort(list(set(np.arange(len(de[budget].fitness))) -
                                           set(rank_include)))
                de[budget].population = np.vstack((de[budget].population[rank_include],
                                                   de[budget].population[rank_ignore]))
                de[budget].fitness = np.vstack((de[budget].fitness[rank_include],
                                                de[budget].fitness[rank_ignore]))
                de[budget].age = np.vstack((de[budget].age[rank_include],
                                                de[budget].age[rank_ignore]))

            # Adjusting pop size parameter
            de[budget].pop_size = pop_size

            # Successive Halving iterations carrying out DE
            for i_sh in range(num_SH_iters):

                # Warmstarting DE incumbent
                de[budget].inc_score = self.inc_score
                de[budget].inc_config = self.inc_config
                if debug:
                    print("Pop size: {}; DE budget: {}".format(de[budget].pop_size, budget))
                best = self.inc_config

                # Population for mutation sampling
                ## For the first iteration, the current population itself
                ## For subsequent iterations, the next higher population
                ## For the full budget, the current population itself
                next_budget = budgets[min(i_sh + 1, num_SH_iters - 1)]
                alt_population = de[next_budget].population

                if i_sh == 0:  # first iteration in the SH bracket
                    # Repeating DE over entire population 'generations' times
                    gens = gens if iteration > 0 else self.generations
                    for gen in range(gens):
                        de_traj, de_runtime, de_history = \
                            de[budget].evolve_generation(budget=budget, best=best,
                                                         alt_pop=alt_population)
                        traj.extend(de_traj)
                        runtime.extend(de_runtime)
                        history.extend(de_history)

                elif i_sh > 0 and iteration == 0:
                    de_traj, de_runtime, de_history = de[budget].eval_pop(budget=budget)
                    traj.extend(de_traj)
                    runtime.extend(de_runtime)
                    history.extend(de_history)

                # Updating global incumbent after each DE step for a budget
                self.inc_score = de[budget].inc_score
                self.inc_config = de[budget].inc_config

                # Retrieving budget, pop_size, population for the next SH iteration
                if i_sh < num_SH_iters-1:  # when not final SH iteration
                    pop_size = num_configs[i_sh + 1]
                    next_budget = budgets[i_sh + 1]
                    # selecting top ranking individuals from lower budget
                    # to be evaluated on higher budget and be eligible for competition
                    rank = np.sort(np.argsort(de[budget].fitness)[:pop_size])
                    rival_population = de[budget].population[rank]

                    if iteration > 0:
                        # warmstarting DE incumbents to maintain global trajectory
                        de[next_budget].inc_score = self.inc_score
                        de[next_budget].inc_config = self.inc_config
                        de[next_budget].pop_size = pop_size

                        # ranking individuals to determine population for next SH step
                        de_traj, de_runtime, de_history = \
                            de[next_budget].ranked_selection(rival_population, pop_size,
                                                             next_budget, debug)
                        self.inc_score = de[next_budget].inc_score
                        self.inc_config = de[next_budget].inc_config
                        traj.extend(de_traj)
                        runtime.extend(de_runtime)
                        history.extend(de_history)
                    else:
                        # equivalent to iteration == 0
                        # no ranked selection happens, top ranked individuals are selected
                        if debug:
                            print("Iteration: ", iteration)
                        de[next_budget].population = rival_population
                        de[next_budget].fitness = de[budget].fitness[rank]
                        de[next_budget].age = np.array([de[next_budget].max_age] * \
                                                        de[next_budget].pop_size)
                        de[next_budget].pop_size = pop_size
                    budget = next_budget

        if verbose:
            print("\nRun complete!")

        return (np.array(traj), np.array(runtime), np.array(history))


class DEHBV7_0(DEHBBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.max_age = np.inf
        self.min_clip = 0
        self.randomize = None
        self.generations = None
        self.update_budgets0()

    def update_budgets0(self):
        '''Original Successive Halving and Hyperband'''
        self.max_SH_iter = -int(np.log(self.min_budget / self.max_budget) / np.log(self.eta)) + 1
        self.budgets = self.max_budget * np.power(self.eta,
                                                  -np.linspace(start=self.max_SH_iter - 1,
                                                               stop=0, num=self.max_SH_iter))

    def update_budgets1(self):
        '''Modified Successive Halving and Hyperband

        Plays a role after the first DEHB iteration/first SH bracket
        '''
        self.max_SH_iter = -int(np.log(self.min_budget / self.max_budget) / np.log(self.eta))
        self.budgets = self.max_budget * np.power(self.eta,
                                                  -np.linspace(start=self.max_SH_iter,
                                                               stop=1, num=self.max_SH_iter) + 1)

    def reset(self):
        super().reset()
        self.update_budgets0()

    def get_next_iteration(self, iteration):
        return self.get_next_iteration0(iteration)

    def get_next_iteration0(self, iteration):
        '''Computes the Modified Successive Halving spacing

        Sacrifices the smallest budget in SH bracket and allocates its resources to the next budget

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
        self.update_budgets0()
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

    def get_next_iteration1(self, iteration, HB_bracket_size=None):
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
        self.update_budgets1()
        if HB_bracket_size is not None:
            # important step to adjust the change in max_SH_iter
            iteration = iteration - HB_bracket_size
        # number of 'SH runs'
        s = self.max_SH_iter - (iteration % self.max_SH_iter)
        # budget spacing for this iteration
        budgets = self.budgets[(-s):]
        # number of configurations in that bracket
        n0 = int(np.ceil(self.max_SH_iter / s) * self.eta ** (s-1))
        ns = [int(np.floor(n0 * (self.eta ** (-j + 1))))  for j in range(1, s+1)]
        if self.min_clip is not None and self.max_clip is not None:
            ns = np.clip(ns, a_min=self.min_clip, a_max=self.max_clip)
        elif self.min_clip is not None:
            ns = np.clip(ns, a_min=self.min_clip, a_max=np.max(ns))

        return ns, budgets

    def run(self, iterations=1, verbose=False, debug=False):
        # Book-keeping variables
        traj = []
        runtime = []
        history = []

        # To retrieve the population and budget ranges
        num_configs, budgets = self.get_next_iteration0(iteration=0)
        HB_bracket_size = self.max_SH_iter

        # List of DE objects corresponding to the budgets (fidelities)
        de = {}
        for i, b in enumerate(budgets):
            de[b] = DE(cs=self.cs, f=self.f, dimensions=self.dimensions, pop_size=num_configs[i],
                       mutation_factor=self.mutation_factor, crossover_prob=self.crossover_prob,
                       strategy=self.strategy, budget=b, max_age=self.max_age)

        # Performs DEHB iterations
        for iteration in range(iterations):

            if iteration < HB_bracket_size:
                # Retrieves SH budgets and number of configurations
                num_configs, budgets = self.get_next_iteration0(iteration)
                generations = 1
            else:
                # Retrieves SH budgets and number of configurations
                num_configs, budgets = self.get_next_iteration1(iteration, HB_bracket_size)
                generations = 2

            if verbose:
                print('Iteration #{:>3}\n{}'.format(iteration, '-' * 15))
                print(num_configs, budgets, self.inc_score)

            # Sets budget, population size, generations for first SH iteration
            pop_size = num_configs[0]
            budget = budgets[0]

            # Number of SH iterations in this DEHB iteration
            num_SH_iters = len(budgets)

            # The first DEHB iteration - only time when a random population is initialized
            if iteration == 0:
                # creating new population for DEHB iteration to be used for the next SH steps
                de[budget].init_eval_pop(budget, eval=False)
                # maintaining global copy of random population created
                self.population = de[budget].population
                self.fitness = de[budget].fitness
                # update global incumbent with new population scores
                self.inc_score = de[budget].inc_score
                self.inc_config = de[budget].inc_config

            elif pop_size > de[budget].pop_size:
                # compensating for extra individuals if SH pop size is larger
                filler = pop_size - de[budget].pop_size
                if debug:
                    print("Adding {} random individuals for budget {}".format(filler, budget))
                # randomly sample to fill the size and set fitness to infinity (no evaluation)
                new_pop = self.init_population(pop_size=filler)
                de[budget].inc_score = self.inc_score
                de[budget].inc_config = self.inc_config
                de[budget].population = np.vstack((de[budget].population, new_pop))
                de[budget].age = np.hstack((de[budget].age, [de[budget].max_age] * filler))
                de[budget].fitness = np.hstack((de[budget].fitness, [np.inf] * filler))

            elif pop_size < de[budget].pop_size:
                # compensating for extra individuals if SH pop size is smaller
                if debug:
                    print("Reducing population from {} to {} "
                          "for budget {}".format(de[budget].pop_size, pop_size, budget))
                # reordering to have the top individuals at the beginning of the population
                ## de[budget].pop_size controls the view over the entire list
                rank_include = np.sort(np.argsort(de[budget].fitness)[:pop_size])
                rank_ignore = np.sort(list(set(np.arange(len(de[budget].fitness))) -
                                           set(rank_include)))
                de[budget].population = np.vstack((de[budget].population[rank_include],
                                                   de[budget].population[rank_ignore]))
                de[budget].fitness = np.vstack((de[budget].fitness[rank_include],
                                                de[budget].fitness[rank_ignore]))
                de[budget].age = np.vstack((de[budget].age[rank_include],
                                            de[budget].age[rank_ignore]))

            # Adjusting pop size parameter
            de[budget].pop_size = pop_size

            # Successive Halving iterations carrying out DE
            for i_sh in range(num_SH_iters):

                # Warmstarting DE incumbent
                de[budget].inc_score = self.inc_score
                de[budget].inc_config = self.inc_config
                if debug:
                    print("Pop size: {}; DE budget: {}".format(de[budget].pop_size, budget))
                best = self.inc_config

                # Population for mutation sampling
                ## For the first iteration, the current population itself
                ## For subsequent iterations, the next higher population
                ## For the full budget, the current population itself
                next_budget = budgets[min(i_sh + 1, num_SH_iters - 1)]
                alt_population = de[next_budget].population

                if i_sh == 0:  # first iteration in the SH bracket
                    # Repeating DE over entire population 'generations' times
                    if debug:
                        print("Evolving {} individuals for {} generations "
                              "on {} budget".format(pop_size, generations, budget))
                    for gen in range(generations):
                        de_traj, de_runtime, de_history = \
                            de[budget].evolve_generation(budget=budget, best=best,
                                                         alt_pop=alt_population)
                        traj.extend(de_traj)
                        runtime.extend(de_runtime)
                        history.extend(de_history)

                elif i_sh > 0 and iteration == 0:
                    de_traj, de_runtime, de_history = de[budget].eval_pop(budget=budget)
                    traj.extend(de_traj)
                    runtime.extend(de_runtime)
                    history.extend(de_history)

                # Updating global incumbent after each DE step for a budget
                self.inc_score = de[budget].inc_score
                self.inc_config = de[budget].inc_config

                # Retrieving budget, pop_size, population for the next SH iteration
                if i_sh < num_SH_iters - 1:  # when not final SH iteration
                    pop_size = num_configs[i_sh + 1]
                    next_budget = budgets[i_sh + 1]
                    # selecting top ranking individuals from lower budget
                    # to be evaluated on higher budget and be eligible for competition
                    rank = np.sort(np.argsort(de[budget].fitness)[:pop_size])
                    rival_population = de[budget].population[rank]

                    if iteration > 0:
                        # warmstarting DE incumbents to maintain global trajectory
                        de[next_budget].inc_score = self.inc_score
                        de[next_budget].inc_config = self.inc_config
                        de[next_budget].pop_size = pop_size

                        # ranking individuals to determine population for next SH step
                        de_traj, de_runtime, de_history = \
                            de[next_budget].ranked_selection(rival_population, pop_size,
                                                             next_budget, debug)
                        self.inc_score = de[next_budget].inc_score
                        self.inc_config = de[next_budget].inc_config
                        traj.extend(de_traj)
                        runtime.extend(de_runtime)
                        history.extend(de_history)
                    else:
                        # equivalent to iteration == 0
                        # no ranked selection happens, top ranked individuals are selected
                        de[next_budget].population = rival_population
                        de[next_budget].fitness = de[budget].fitness[rank]
                        de[next_budget].age = np.array([de[next_budget].max_age] * \
                                                       de[next_budget].pop_size)
                        de[next_budget].pop_size = pop_size
                    budget = next_budget

        if verbose:
            print("\nRun complete!")

        return (np.array(traj), np.array(runtime), np.array(history))


class DEHBV7_1(DEHBBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.max_age = np.inf
        self.min_clip = 0
        self.randomize = None
        self.generations = None
        self.update_budgets0()

    def update_budgets0(self):
        '''Original Successive Halving and Hyperband'''
        self.max_SH_iter = -int(np.log(self.min_budget / self.max_budget) / np.log(self.eta)) + 1
        self.budgets = self.max_budget * np.power(self.eta,
                                                  -np.linspace(start=self.max_SH_iter - 1,
                                                               stop=0, num=self.max_SH_iter))

    def update_budgets1(self):
        '''Modified Successive Halving and Hyperband

        Plays a role after the first DEHB iteration/first SH bracket
        '''
        self.max_SH_iter = -int(np.log(self.min_budget / self.max_budget) / np.log(self.eta))
        self.budgets = self.max_budget * np.power(self.eta,
                                                  -np.linspace(start=self.max_SH_iter,
                                                               stop=1, num=self.max_SH_iter) + 1)

    def reset(self):
        super().reset()
        self.update_budgets0()

    def get_next_iteration(self, iteration):
        return self.get_next_iteration0(iteration)

    def get_next_iteration0(self, iteration):
        '''Computes the Modified Successive Halving spacing

        Sacrifices the smallest budget in SH bracket and allocates its resources to the next budget

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
        # self.update_budgets0()
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

    def get_next_iteration1(self, iteration, HB_bracket_size=None):
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
        # self.update_budgets1()
        if HB_bracket_size is not None:
            # important step to adjust the change in max_SH_iter
            iteration = iteration - HB_bracket_size
        # number of 'SH runs'
        s = self.max_SH_iter - (iteration % self.max_SH_iter)
        # budget spacing for this iteration
        budgets = self.budgets[(-s):]
        # number of configurations in that bracket
        n0 = int(np.ceil(self.max_SH_iter / s) * self.eta ** (s-1))
        ns = [int(np.floor(n0 * (self.eta ** (-j + 1))))  for j in range(1, s+1)]
        if self.min_clip is not None and self.max_clip is not None:
            ns = np.clip(ns, a_min=self.min_clip, a_max=self.max_clip)
        elif self.min_clip is not None:
            ns = np.clip(ns, a_min=self.min_clip, a_max=np.max(ns))

        return ns, budgets

    def run(self, iterations=1, verbose=False, debug=False):
        # Book-keeping variables
        traj = []
        runtime = []
        history = []

        # To retrieve the population and budget ranges
        num_configs, budgets = self.get_next_iteration0(iteration=0)
        HB_bracket_size_0 = self.max_SH_iter
        HB_bracket_size_1 = 0
        HB_bracket_size = self.max_SH_iter

        # List of DE objects corresponding to the budgets (fidelities)
        de = {}
        for i, b in enumerate(budgets):
            de[b] = DE(cs=self.cs, f=self.f, dimensions=self.dimensions, pop_size=num_configs[i],
                       mutation_factor=self.mutation_factor, crossover_prob=self.crossover_prob,
                       strategy=self.strategy, budget=b, max_age=self.max_age)

        # Performs DEHB iterations
        for iteration in range(iterations):

            if HB_bracket_size_0 > 0:
                iter = self.max_SH_iter - HB_bracket_size_0
                num_configs, budgets = self.get_next_iteration0(iter)
                generations = 1
                HB_bracket_size_0 -= 1
                if HB_bracket_size_0 == 0:
                    self.update_budgets1()
                    HB_bracket_size_1 = self.max_SH_iter

            elif HB_bracket_size_1 > 0:
                iter = self.max_SH_iter - HB_bracket_size_1
                num_configs, budgets = self.get_next_iteration1(iter)
                generations = 2
                HB_bracket_size_1 -= 1
                if HB_bracket_size_1 == 0:
                    self.update_budgets0()
                    HB_bracket_size_0 = self.max_SH_iter

            if verbose:
                print('Iteration #{:>3}\n{}'.format(iteration, '-' * 15))
                print(num_configs, budgets, self.inc_score, iter)

            # Sets budget, population size, generations for first SH iteration
            pop_size = num_configs[0]
            budget = budgets[0]

            # Number of SH iterations in this DEHB iteration
            num_SH_iters = len(budgets)

            # The first DEHB iteration - only time when a random population is initialized
            if iteration == 0:
                # creating new population for DEHB iteration to be used for the next SH steps
                de[budget].init_eval_pop(budget, eval=False)
                # maintaining global copy of random population created
                self.population = de[budget].population
                self.fitness = de[budget].fitness
                # update global incumbent with new population scores
                self.inc_score = de[budget].inc_score
                self.inc_config = de[budget].inc_config

            elif pop_size > de[budget].pop_size:
                # compensating for extra individuals if SH pop size is larger
                filler = pop_size - de[budget].pop_size
                if debug:
                    print("Adding {} random individuals for budget {}".format(filler, budget))
                # randomly sample to fill the size and set fitness to infinity (no evaluation)
                new_pop = self.init_population(pop_size=filler)
                de[budget].inc_score = self.inc_score
                de[budget].inc_config = self.inc_config
                de[budget].population = np.vstack((de[budget].population, new_pop))
                de[budget].age = np.hstack((de[budget].age, [de[budget].max_age] * filler))
                de[budget].fitness = np.hstack((de[budget].fitness, [np.inf] * filler))

            elif pop_size < de[budget].pop_size:
                # compensating for extra individuals if SH pop size is smaller
                if debug:
                    print("Reducing population from {} to {} "
                          "for budget {}".format(de[budget].pop_size, pop_size, budget))
                # reordering to have the top individuals at the beginning of the population
                ## de[budget].pop_size controls the view over the entire list
                rank_include = np.sort(np.argsort(de[budget].fitness)[:pop_size])
                rank_ignore = np.sort(list(set(np.arange(len(de[budget].fitness))) -
                                           set(rank_include)))
                de[budget].population = np.vstack((de[budget].population[rank_include],
                                                   de[budget].population[rank_ignore]))
                de[budget].fitness = np.vstack((de[budget].fitness[rank_include],
                                                de[budget].fitness[rank_ignore]))
                de[budget].age = np.vstack((de[budget].age[rank_include],
                                            de[budget].age[rank_ignore]))

            # Adjusting pop size parameter
            de[budget].pop_size = pop_size

            # Successive Halving iterations carrying out DE
            for i_sh in range(num_SH_iters):

                # Warmstarting DE incumbent
                de[budget].inc_score = self.inc_score
                de[budget].inc_config = self.inc_config
                if debug:
                    print("Pop size: {}; DE budget: {}".format(de[budget].pop_size, budget))
                best = self.inc_config

                # Population for mutation sampling
                ## For the first iteration, the current population itself
                ## For subsequent iterations, the next higher population
                ## For the full budget, the current population itself
                next_budget = budgets[min(i_sh + 1, num_SH_iters - 1)]
                alt_population = de[next_budget].population

                if i_sh == 0:  # first iteration in the SH bracket
                    # Repeating DE over entire population 'generations' times
                    if debug:
                        print("Evolving {} individuals for {} generations "
                              "on {} budget".format(pop_size, generations, budget))
                    for gen in range(generations):
                        de_traj, de_runtime, de_history = \
                            de[budget].evolve_generation(budget=budget, best=best,
                                                         alt_pop=alt_population)
                        traj.extend(de_traj)
                        runtime.extend(de_runtime)
                        history.extend(de_history)

                elif i_sh > 0 and iteration == 0:
                    de_traj, de_runtime, de_history = de[budget].eval_pop(budget=budget)
                    traj.extend(de_traj)
                    runtime.extend(de_runtime)
                    history.extend(de_history)

                # Updating global incumbent after each DE step for a budget
                self.inc_score = de[budget].inc_score
                self.inc_config = de[budget].inc_config

                # Retrieving budget, pop_size, population for the next SH iteration
                if i_sh < num_SH_iters - 1:  # when not final SH iteration
                    pop_size = num_configs[i_sh + 1]
                    next_budget = budgets[i_sh + 1]
                    # selecting top ranking individuals from lower budget
                    # to be evaluated on higher budget and be eligible for competition
                    rank = np.sort(np.argsort(de[budget].fitness)[:pop_size])
                    rival_population = de[budget].population[rank]

                    if iteration > 0:
                        # warmstarting DE incumbents to maintain global trajectory
                        de[next_budget].inc_score = self.inc_score
                        de[next_budget].inc_config = self.inc_config
                        de[next_budget].pop_size = pop_size

                        # ranking individuals to determine population for next SH step
                        de_traj, de_runtime, de_history = \
                            de[next_budget].ranked_selection(rival_population, pop_size,
                                                             next_budget, debug)
                        self.inc_score = de[next_budget].inc_score
                        self.inc_config = de[next_budget].inc_config
                        traj.extend(de_traj)
                        runtime.extend(de_runtime)
                        history.extend(de_history)
                    else:
                        # equivalent to iteration == 0
                        # no ranked selection happens, top ranked individuals are selected
                        de[next_budget].population = rival_population
                        de[next_budget].fitness = de[budget].fitness[rank]
                        de[next_budget].age = np.array([de[next_budget].max_age] * \
                                                       de[next_budget].pop_size)
                        de[next_budget].pop_size = pop_size
                    budget = next_budget

        if verbose:
            print("\nRun complete!")

        return (np.array(traj), np.array(runtime), np.array(history))
