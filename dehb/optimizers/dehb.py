import sys
sys.path.append('../')

import numpy as np

from optimizers.de import DE


class DEHBBase():
    def __init__(self, b=None, cs=None, dimensions=None, mutation_factor=None,
                 crossover_prob=None, strategy='rand1_bin', generations=None, min_budget=None,
                 max_budget=None, eta=None, min_clip=3, max_clip=None, **kwargs):
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
            self.budgets = self.budgets.astype(int)

        # Miscellaneous
        self.output_path = kwargs['output_path'] if 'output_path' in kwargs else ''

        # Global trackers
        self.population = None
        self.fitness = None
        self.inc_score = np.inf
        self.inc_config = None

    def reset(self):
        self.inc_score = np.inf
        self.inc_config = None
        self.population = None
        self.fitness = None

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
    '''Version 1 of DEHB

    Each DEHB iteration is initialized with a new random population.
    In each of the DEHB iteration, Successive Halving (SH) takes place where
        the number of SH iterations, budget spacing, number of configurations,
        are determined dynamically based on the iteration number.
        The top performing individuals are carried forward to the next higher budget.
    Each SH iteration in each DEHB iteration is evolved for a certain number of generations.
    '''
    def __init__(self, b=None, cs=None, dimensions=None, mutation_factor=None,
                 crossover_prob=None, strategy=None, min_budget=None, max_budget=None,
                 eta=None, generations=None, min_clip=3, max_clip=None, **kwargs):
        super().__init__(b=b, cs=cs, dimensions=dimensions, mutation_factor=mutation_factor,
                         crossover_prob=crossover_prob, strategy=strategy, min_budget=min_budget,
                         max_budget=max_budget, eta=eta, generations=generations,
                         min_clip=min_clip, max_clip=max_clip)
        self.logger = []

    def reset(self):
        super().reset()
        self.logger = []

    def run(self, iterations=100, verbose=True):
        # Book-keeping variables
        traj = []
        runtime = []
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
            de = DE(b=self.b, cs=self.cs, dimensions=self.dimensions, pop_size=pop_size,
                    mutation_factor=self.mutation_factor, crossover_prob=self.crossover_prob,
                    strategy=self.strategy, budget=budget)
            # Warmstarting DE incumbent to be the global incumbent
            de.inc_score = self.inc_score
            de.inc_config = self.inc_config
            # Creating new population for current DEHB iteration
            de_traj, de_runtime = de.init_eval_pop(budget)
            traj.extend(de_traj)
            runtime.extend(de_runtime)

            # Incorporating global incumbent into the new DE population
            if all(de.inc_config == self.inc_config):
                # if new population has no better individual, randomly
                # replace an individual with the incumbent so far
                idx = np.random.choice(np.arange(len(de.population)))
                de.population[idx] = self.inc_config
                de.fitness[idx] = self.inc_score
            else:
                # if new population has a better individual, update
                # the global incumbent and fitness
                self.inc_score = de.inc_score
                self.inc_config = de.inc_config

            # Successive Halving iterations
            for i_sh in range(num_SH_iters):
                # Repeating DE over entire population 'generations' times
                for gen in range(self.generations):
                    de_traj, de_runtime = de.evolve_generation(budget)
                    traj.extend(de_traj)
                    runtime.extend(de_runtime)

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
                    de.pop_size = pop_size

        if verbose:
            print("\nRun complete!")

        return np.array(traj), np.array(runtime)


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
    def __init__(self, b=None, cs=None, dimensions=None, mutation_factor=None,
                 crossover_prob=None, strategy=None, min_budget=None, max_budget=None,
                 eta=None, generations=None,  min_clip=3, max_clip=None,
                 randomize=None, **kwargs):
        super().__init__(b=b, cs=cs, dimensions=dimensions, mutation_factor=mutation_factor,
                         crossover_prob=crossover_prob, strategy=strategy, min_budget=min_budget,
                         max_budget=max_budget, eta=eta, generations=generations,
                         min_clip=min_clip, max_clip=max_clip)
        self.randomize = randomize
        self.logger = []
        # Fixing to 1 -- specific attribute of version 2 of DEHB
        # self.generations = 1

    def reset(self):
        super().reset()
        self.logger = []

    def run(self, iterations=100, verbose=True):
        # Book-keeping variables
        traj = []
        runtime = []

        # To retrieve the maximal pop_size and initialize a single DE object for all DEHB runs
        num_configs, budgets = self.get_next_iteration(iteration=0)
        de = DE(b=self.b, cs=self.cs, dimensions=self.dimensions, pop_size=num_configs[0],
                mutation_factor=self.mutation_factor, crossover_prob=self.crossover_prob,
                strategy=self.strategy, budget=budgets[0])

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
                de_traj, de_runtime = de.init_eval_pop(budget)
                # maintaining global copy of random population created
                self.population = de.population
                self.fitness = de.fitness
                # update global incumbent with new population scores
                self.inc_score = de.inc_score
                self.inc_config = de.inc_config
                traj.extend(de_traj)
                runtime.extend(de_runtime)
            elif pop_size == len(self.population) and \
                 self.randomize is not None and self.randomize > 0:
                # executes in the first step of every SH iteration other than first DEHB iteration
                # also conditional on whether a randomization fraction has been specified
                num_replace = np.ceil(self.randomize * pop_size).astype(int)
                # fetching the worst performing individuals
                idxs = np.sort(np.argsort(-self.fitness)[:num_replace])
                print("Replacing {}/{} -- {}".format(num_replace, pop_size, idxs))
                new_pop = self.init_population(pop_size=num_replace)
                self.population[idxs] = new_pop
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

            # Ranking current population
            self.rank = np.sort(np.argsort(self.fitness)[:pop_size])
            # Passing onto DE-SH steps a subset of top individuals from global population
            de.population = self.population[self.rank]
            de.fitness = np.array(self.fitness)[self.rank]
            de.pop_size = pop_size

            # Successive Halving iterations carrying out DE
            for i_sh in range(num_SH_iters):
                print(i_sh, self.rank)
                # Repeating DE over entire population 'generations' times
                for gen in range(self.generations):
                    de_traj, de_runtime = de.evolve_generation(budget)
                    traj.extend(de_traj)
                    runtime.extend(de_runtime)

                # Updating global incumbent after each DE step
                self.inc_score = de.inc_score
                self.inc_config = de.inc_config
                # Updating global population with evolved individuals
                self.population[self.rank] = de.population
                self.fitness[self.rank] = de.fitness

                # Retrieving budget, pop_size, population for the next SH iteration
                if i_sh < num_SH_iters-1:  # when not final SH iteration
                    pop_size = num_configs[i_sh+1]
                    budget = budgets[i_sh+1]
                    # Selecting top individuals to fit pop_size of next SH iteration
                    self.de_rank = np.sort(np.argsort(de.fitness)[:pop_size])
                    # Saving index of new DE population from the global population
                    self.rank = self.rank[self.de_rank]
                    de.population = de.population[self.de_rank]
                    de.fitness = np.array(de.fitness)[self.de_rank]
                    de.pop_size = pop_size

        if verbose:
            print("\nRun complete!")

        return np.array(traj), np.array(runtime)


class DEHBV3(DEHBBase):
    '''Version 3 of DEHB

    At anytime, each set of population contains the best individuals from that budget
    '''
    def __init__(self, b=None, cs=None, dimensions=None, mutation_factor=None,
                 crossover_prob=None, strategy=None, min_budget=None, max_budget=None,
                 eta=None, generations=None,  min_clip=3, max_clip=None,
                 randomize=None, **kwargs):
        super().__init__(b=b, cs=cs, dimensions=dimensions, mutation_factor=mutation_factor,
                         crossover_prob=crossover_prob, strategy=strategy, min_budget=min_budget,
                         max_budget=max_budget, eta=eta, generations=generations,
                         min_clip=min_clip, max_clip=max_clip)
        self.randomize = randomize
        self.logger = []

    def reset(self):
        super().reset()
        self.logger = []

    def run(self, iterations=100, verbose=True):
        # Book-keeping variables
        traj = []
        runtime = []

        # To retrieve the population and budget ranges
        num_configs, budgets = self.get_next_iteration(iteration=0)
        # List of DE objects corresponding to the populations
        de = []
        for i in range(len(budgets)):
            de.append(DE(b=self.b, cs=self.cs, dimensions=self.dimensions, pop_size=num_configs[i],
                      mutation_factor=self.mutation_factor, crossover_prob=self.crossover_prob,
                      strategy=self.strategy, budget=budgets[i]))

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
                de_traj, de_runtime = de[0].init_eval_pop(budget)
                # maintaining global copy of random population created
                self.population = de[0].population
                self.fitness = de[0].fitness
                # update global incumbent with new population scores
                self.inc_score = de[0].inc_score
                self.inc_config = de[0].inc_config
                traj.extend(de_traj)
                runtime.extend(de_runtime)
            elif de_idx == 0 and self.randomize is not None and self.randomize > 0:
                # executes in the first step of every SH iteration other than first DEHB iteration
                # also conditional on whether a randomization fraction has been specified
                num_replace = np.ceil(self.randomize * pop_size).astype(int)
                # fetching the worst performing individuals
                idxs = np.sort(np.argsort(-de[0].fitness)[:num_replace])
                print("Replacing {}/{} -- {}".format(num_replace, pop_size, idxs))
                new_pop = self.init_population(pop_size=num_replace)
                de[0].population[idxs] = new_pop
                de[0].inc_score = self.inc_score
                de[0].inc_config = self.inc_config
                # evaluating new individuals
                for i in idxs:
                    de[0].fitness[i], cost = de[0].f_objective(de[0].population[i], budget)
                    if self.fitness[i] < self.inc_score:
                        self.inc_score = de[0].fitness[i]
                        self.inc_config = de[0].population[i]
                    traj.append(self.inc_score)
                    runtime.append(cost)

            # Successive Halving iterations carrying out DE
            for i_sh in range(num_SH_iters):
                de_curr = de_idx + i_sh
                # Warmstarting DE incumbent
                de[de_curr].inc_score = self.inc_score
                de[de_curr].inc_config = self.inc_config
                # Repeating DE over entire population 'generations' times
                print("DE index: {}; DE budget: {}".format(de_curr, budget))
                for gen in range(self.generations):
                    de_traj, de_runtime = de[de_curr].evolve_generation(budget)
                    traj.extend(de_traj)
                    runtime.extend(de_runtime)
                # Updating global incumbent after each DE step
                self.inc_score = de[de_curr].inc_score
                self.inc_config = de[de_curr].inc_config

                # Retrieving budget, pop_size, population for the next SH iteration
                if i_sh < num_SH_iters-1:  # when not final SH iteration
                    pop_size = num_configs[i_sh + 1]
                    budget = budgets[i_sh + 1]
                    # selecting top ranking individuals from lower budget
                    ## to be evaluated on higher budget and be eligible for competition
                    rank = np.sort(np.argsort(de[de_curr].fitness)[:pop_size])
                    rival_population = de[de_curr].population[rank]
                    # print(i_sh, rank)
                    if de[de_curr + 1].population is not None:
                        # warmstarting DE incumbents to maintain global trajectory
                        de[de_curr + 1].inc_score = self.inc_score
                        de[de_curr + 1].inc_config = self.inc_config
                        de_traj, de_runtime = de[de_curr + 1].ranked_selection(rival_population,
                                                                               budget)
                        self.inc_score = de[de_curr + 1].inc_score
                        self.inc_config = de[de_curr + 1].inc_config
                        traj.extend(de_traj)
                        runtime.extend(de_runtime)
                    else:  # equivalent to iteration == 0
                        print("Iteration: ", iteration)
                        de[de_curr + 1].population = rival_population
                        de[de_curr + 1].fitness = de[de_curr].fitness[rank]
        if verbose:
            print("\nRun complete!")

        return np.array(traj), np.array(runtime)
