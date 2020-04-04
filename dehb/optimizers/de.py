
import numpy as np
import ConfigSpace


class DEBase():
    def __init__(self, cs=None, f=None, dimensions=None, pop_size=None, max_age=None,
                 mutation_factor=None, crossover_prob=None, strategy=None, budget=None, **kwargs):
        # Benchmark related variables
        self.cs = cs
        self.f = f
        if dimensions is None and self.cs is not None:
            self.dimensions = len(self.cs.get_hyperparameters())
        else:
            self.dimensions = dimensions

        # DE related variables
        self.pop_size = pop_size
        self.max_age = max_age
        self.mutation_factor = mutation_factor
        self.crossover_prob = crossover_prob
        self.strategy = strategy
        self.budget = budget

        # Miscellaneous
        self.output_path = kwargs['output_path'] if 'output_path' in kwargs else ''

        # Global trackers
        self.inc_score = np.inf
        self.inc_config = None
        self.population = None
        self.fitness = None
        self.age = None
        self.history = []

    def reset(self):
        self.inc_score = np.inf
        self.inc_config = None
        self.population = None
        self.fitness = None
        self.age = None
        self.history = []

    def _shuffle_pop(self):
        pop_order = np.arange(len(self.population))
        np.random.shuffle(pop_order)
        self.population = self.population[pop_order]
        self.fitness = self.fitness[pop_order]
        self.age = self.age[pop_order]

    def _set_min_pop_size(self):
        if self.mutation_strategy in ['rand1', 'rand2dir', 'randtobest1']:
            self._min_pop_size = 3
        elif self.mutation_strategy in ['currenttobest1', 'best1']:
            self._min_pop_size = 2
        elif self.mutation_strategy in ['best2']:
            self._min_pop_size = 4
        elif self.mutation_strategy in ['rand2']:
            self._min_pop_size = 5
        else:
            self._min_pop_size = 1

        return self._min_pop_size

    def init_population(self, pop_size=10):
        population = np.random.uniform(low=0.0, high=1.0, size=(pop_size, self.dimensions))
        return population

    def sample_population(self, size=3, alt_pop=None):
        '''Samples 'size' individuals

        If alt_pop is None or a list/array of None, sample from own population
        Else sample from the specified alternate population
        '''
        if isinstance(alt_pop, list) or isinstance(alt_pop, np.ndarray):
            idx = [indv is None for indv in alt_pop]
            if any(idx):
                selection = np.random.choice(np.arange(len(self.population)), size, replace=False)
                return self.population[selection]
            else:
                if len(alt_pop) < 3:
                    alt_pop = np.vstack((alt_pop, self.population))
                # # Sample from the joint population
                # selection = np.random.choice(np.arange(len(alt_pop) + len(self.population)),
                #                              size, replace=False)
                # return np.concatenate((alt_pop, self.population))[selection]
                selection = np.random.choice(np.arange(len(alt_pop)), size, replace=False)
                alt_pop = np.stack(alt_pop)
                return alt_pop[selection]
        else:
            selection = np.random.choice(np.arange(len(self.population)), size, replace=False)
            return self.population[selection]

    def boundary_check(self, vector, fix_type='random'):
        '''
        Checks whether each of the dimensions of the input vector are within [0, 1].
        If not, values of those dimensions are replaced with the type of fix selected.

        Parameters
        ----------
        vector : array
            The vector describing the individual from the population
        fix_type : str, {'random', 'clip'}
            if 'random', the values are replaced with a random sampling from (0,1)
            if 'clip', the values are clipped to the closest limit from {0, 1}

        Returns
        -------
        array
        '''
        violations = np.where((vector > 1) | (vector < 0))[0]
        if len(violations) == 0:
            return vector
        if fix_type == 'random':
            vector[violations] = np.random.uniform(low=0.0, high=1.0, size=len(violations))
        else:
            vector[violations] = np.clip(vector[violations], a_min=0, a_max=1)
        return vector

    def vector_to_configspace(self, vector):
        '''Converts numpy array to ConfigSpace object

        Works when self.cs is a ConfigSpace object and the input vector is in the domain [0, 1].
        '''
        new_config = self.cs.sample_configuration()
        for i, hyper in enumerate(self.cs.get_hyperparameters()):
            if type(hyper) == ConfigSpace.OrdinalHyperparameter:
                ranges = np.arange(start=0, stop=1, step=1/len(hyper.sequence))
                param_value = hyper.sequence[np.where((vector[i] < ranges) == False)[0][-1]]
            elif type(hyper) == ConfigSpace.CategoricalHyperparameter:
                ranges = np.arange(start=0, stop=1, step=1/len(hyper.choices))
                param_value = hyper.choices[np.where((vector[i] < ranges) == False)[0][-1]]
            else:  # handles UniformFloatHyperparameter & UniformIntegerHyperparameter
                # rescaling continuous values
                if hyper.log:
                    log_range = np.log(hyper.upper) - np.log(hyper.lower)
                    param_value = np.exp(np.log(hyper.lower) + vector[i] * log_range)
                else:
                    param_value = hyper.lower + (hyper.upper - hyper.lower) * vector[i]
                if type(hyper) == ConfigSpace.UniformIntegerHyperparameter:
                    param_value = np.round(param_value).astype(int)   # converting to discrete (int)
            new_config[hyper.name] = param_value
        return new_config

    def f_objective(self):
        raise NotImplementedError("The function needs to be defined in the sub class.")

    def mutation(self):
        raise NotImplementedError("The function needs to be defined in the sub class.")

    def crossover(self):
        raise NotImplementedError("The function needs to be defined in the sub class.")

    def evolve(self):
        raise NotImplementedError("The function needs to be defined in the sub class.")

    def run(self):
        raise NotImplementedError("The function needs to be defined in the sub class.")


class DE(DEBase):
    def __init__(self, cs=None, f=None, dimensions=None, pop_size=None, max_age=np.inf,
                 mutation_factor=None, crossover_prob=None, strategy='rand1_bin',
                 budget=None, **kwargs):
        super().__init__(cs=cs, f=f, dimensions=dimensions, pop_size=pop_size, max_age=max_age,
                         mutation_factor=mutation_factor, crossover_prob=crossover_prob,
                         strategy=strategy, budget=budget, **kwargs)
        if self.strategy is not None:
            self.mutation_strategy = self.strategy.split('_')[0]
            self.crossover_strategy = self.strategy.split('_')[1]
        else:
            self.mutation_strategy = self.crossover_strategy = None

    def f_objective(self, x, budget=None):
        if self.f is None:
            raise NotImplementedError("An objective function needs to be passed.")
        config = self.vector_to_configspace(x)
        if budget is not None:  # to be used when called by multi-fidelity based optimizers
            fitness, cost = self.f(config, budget=budget)
        else:
            fitness, cost = self.f(config)
        return fitness, cost

    def init_eval_pop(self, budget=None, eval=True):
        '''Creates new population of 'pop_size' and evaluates individuals.
        '''
        self.population = self.init_population(self.pop_size)
        self.fitness = np.array([np.inf for i in range(self.pop_size)])
        self.age = np.array([self.max_age] * self.pop_size)

        traj = []
        runtime = []
        history = []

        if not eval:
            return traj, runtime, history

        for i in range(self.pop_size):
            config = self.population[i]
            self.fitness[i], cost = self.f_objective(config, budget)
            if self.fitness[i] < self.inc_score:
                self.inc_score = self.fitness[i]
                self.inc_config = config
            traj.append(self.inc_score)
            runtime.append(cost)
            history.append((config.tolist(), float(self.fitness[i]), float(budget or 0)))

        return traj, runtime, history

    def eval_pop(self, population=None, budget=None):
        pop = self.population if population is None else population
        pop_size = self.pop_size if population is None else len(pop)
        traj = []
        runtime = []
        history = []
        fitnesses = []
        costs = []
        ages = []
        for i in range(pop_size):
            fitness, cost = self.f_objective(pop[i], budget)
            if population is None:
                self.fitness[i] = fitness
            if fitness <= self.inc_score:
                self.inc_score = fitness
                self.inc_config = pop[i]
            traj.append(self.inc_score)
            runtime.append(cost)
            history.append((pop[i].tolist(), float(fitness), float(budget or 0)))
            fitnesses.append(fitness)
            costs.append(cost)
            ages.append(self.max_age)
        if population is None:
            self.fitness = np.array(fitnesses)
            return traj, runtime, history
        else:
            return traj, runtime, history, np.array(fitnesses), np.array(ages)

    def mutation_rand1(self, r1, r2, r3):
        '''Performs the 'rand1' type of DE mutation
        '''
        diff = r2 - r3
        mutant = r1 + self.mutation_factor * diff
        return mutant

    def mutation_rand2(self, r1, r2, r3, r4, r5):
        '''Performs the 'rand2' type of DE mutation
        '''
        diff1 = r2 - r3
        diff2 = r4 - r5
        mutant = r1 + self.mutation_factor * diff1 + self.mutation_factor * diff2
        return mutant

    def mutation_currenttobest1(self, current, best, r1, r2):
        diff1 = best - current
        diff2 = r1 - r2
        mutant = current + self.mutation_factor * diff1 + self.mutation_factor * diff2
        return mutant

    def mutation_rand2dir(self, r1, r2, r3):
        diff = r1 - r2 - r3
        mutant = r1 + self.mutation_factor * diff / 2
        return mutant

    def mutation(self, current=None, best=None, alt_pop=None):
        '''Performs DE mutation
        '''
        if self.mutation_strategy == 'rand1':
            r1, r2, r3 = self.sample_population(size=3, alt_pop=alt_pop)
            mutant = self.mutation_rand1(r1, r2, r3)

        elif self.mutation_strategy == 'rand2':
            r1, r2, r3, r4, r5 = self.sample_population(size=5, alt_pop=alt_pop)
            mutant = self.mutation_rand2(r1, r2, r3, r4, r5)

        elif self.mutation_strategy == 'rand2dir':
            r1, r2, r3 = self.sample_population(size=3, alt_pop=alt_pop)
            mutant = self.mutation_rand2dir(r1, r2, r3)

        elif self.mutation_strategy == 'best1':
            r1, r2 = self.sample_population(size=2, alt_pop=alt_pop)
            if best is None:
                best = self.population[np.argmin(self.fitness)]
            mutant = self.mutation_rand1(best, r1, r2)

        elif self.mutation_strategy == 'best2':
            r1, r2, r3, r4 = self.sample_population(size=4, alt_pop=alt_pop)
            if best is None:
                best = self.population[np.argmin(self.fitness)]
            mutant = self.mutation_rand2(best, r1, r2, r3, r4)

        elif self.mutation_strategy == 'currenttobest1':
            r1, r2 = self.sample_population(size=2, alt_pop=alt_pop)
            if best is None:
                best = self.population[np.argmin(self.fitness)]
            mutant = self.mutation_currenttobest1(current, best, r1, r2)

        elif self.mutation_strategy == 'randtobest1':
            r1, r2, r3 = self.sample_population(size=3, alt_pop=alt_pop)
            if best is None:
                best = self.population[np.argmin(self.fitness)]
            mutant = self.mutation_currenttobest1(r1, best, r2, r3)

        return mutant

    def crossover_bin(self, target, mutant):
        '''Performs the binomial crossover of DE
        '''
        cross_points = np.random.rand(self.dimensions) < self.crossover_prob
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dimensions)] = True
        offspring = np.where(cross_points, mutant, target)
        return offspring

    def crossover_exp(self, target, mutant):
        '''Performs the exponential crossover of DE
        '''
        n = np.random.randint(0, self.dimensions)
        L = 0
        while ((np.random.rand() < self.crossover_prob) and L < self.dimensions):
            idx = (n+L) % self.dimensions
            target[idx] = mutant[idx]
            L = L + 1
        return target

    def crossover(self, target, mutant):
        '''Performs DE crossover
        '''
        if self.crossover_strategy == 'bin':
            offspring = self.crossover_bin(target, mutant)
        elif self.crossover_strategy == 'exp':
            offspring = self.crossover_exp(target, mutant)
        return offspring

    def selection(self, trials, budget=None):
        '''Carries out a parent-offspring competition given a set of trial population
        '''
        # assert len(self.population) == len(trials), \
        #     "Pop len ({}) does not match trial population length ({})".format(len(self.population),
        #                                                                       len(trials))
        traj = []
        runtime = []
        history = []
        for i in range(len(trials)):
            # evaluation of the newly created individuals
            fitness, cost = self.f_objective(trials[i], budget)
            # selection -- competition between parent[i] -- child[i]
            ## equality is important for landscape exploration
            if fitness <= self.fitness[i]:
                self.population[i] = trials[i]
                self.fitness[i] = fitness
                # resetting age since new individual in the population
                self.age[i] = self.max_age
            else:
                # decreasing age by 1 of parent who is better than offspring/trial
                self.age[i] -= 1
            # updation of global incumbent for trajectory
            if self.fitness[i] < self.inc_score:
                self.inc_score = self.fitness[i]
                self.inc_config = self.population[i]
            traj.append(self.inc_score)
            runtime.append(cost)
            history.append((trials[i].tolist(), float(fitness), float(budget or 0)))
        return traj, runtime, history

    def ranked_selection(self, trials, final_pop_size=None, budget=None, debug=False):
        '''Returns the fittest individuals from two sets of population
        '''
        final_pop_size = self.pop_size if final_pop_size is None else final_pop_size

        traj, runtime, history, fitnesses, ages = self.eval_pop(population=trials, budget=budget)

        if debug:
            print("Ranking {} from {} originals + {} rivals".format(final_pop_size, self.pop_size,
                                                                    len(trials)))

        # Creating a total population of current individuals and rival individuals evaluated on
        # the same budget
        tot_pop = np.vstack((self.population, trials))
        tot_fitness = np.hstack((self.fitness, fitnesses))
        tot_age = np.hstack((self.age, ages))

        # Sorting the total population by fitness to keep only the top individuals from the pool
        rank = np.sort(np.argsort(tot_fitness)[:final_pop_size])
        self.population = tot_pop[rank]
        self.fitness = tot_fitness[rank]
        self.age = tot_age[rank]
        self.pop_size = final_pop_size
        return traj, runtime, history

    def kill_aged_pop(self, budget=None, debug=False):
        '''Replaces individuals with age older than max_age
        '''
        traj = []
        runtime = []
        history = []
        idxs = np.where(self.age <= 0)[0]
        if len(idxs) == 0:
            return traj, runtime, history
        if debug:
            print("Killing {} individual(s) for budget {}: {}".format(len(idxs), budget, self.age[idxs]))
        new_pop = self.init_population(pop_size=len(idxs))
        for i, index in enumerate(idxs):
            self.population[index] = new_pop[i]
            self.fitness[index], cost = self.f_objective(self.population[index], budget)
            self.age[index] = self.max_age
            if self.fitness[index] < self.inc_score:
                self.inc_score = self.fitness[index]
                self.inc_config = self.population[index]
            traj.append(self.inc_score)
            runtime.append(cost)
            history.append((self.population[index].tolist(), float(self.fitness[index]),
                            float(budget or 0)))
        return traj, runtime, history

    def evolve_generation(self, budget=None, best=None, alt_pop=None):
        '''Performs a complete DE evolution, mutation -> crossover -> selection
        '''
        trials = []
        for j in range(self.pop_size):
            target = self.population[j]
            donor = self.mutation(current=target, best=best, alt_pop=alt_pop)
            trial = self.crossover(target, donor)
            trial = self.boundary_check(trial)
            trials.append(trial)
        trials = np.array(trials)
        traj, runtime, history = self.selection(trials, budget)
        return traj, runtime, history

    def sample_mutants(self, size, population=None):
        if population is None:
            population = self.population
        elif len(population) < 3:
            population = np.vstack((self.population, population))

        old_strategy = self.mutation_strategy
        self.mutation_strategy = 'rand1'
        mutants = np.random.uniform(low=0.0, high=1.0, size=(size, self.dimensions))
        for i in range(size):
            mutant = self.mutation(current=None, best=None, alt_pop=population)
            mutants[i] = self.boundary_check(mutant, fix_type='clip')
        self.mutation_strategy = old_strategy

        return mutants

    def run(self, generations=1, verbose=False, budget=None):
        self.traj = []
        self.runtime = []
        self.history = []

        if verbose:
            print("Initializing and evaluating new population...")
        self.traj, self.runtime, self.history = self.init_eval_pop(budget=budget)

        if verbose:
            print("Running evolutionary search...")
        for i in range(generations):
            if verbose:
                print("Generation {:<2}/{:<2} -- {:<0.7}".format(i+1, generations, self.inc_score))
            traj, runtime, history = self.evolve_generation(budget=budget)
            self.traj.extend(traj)
            self.runtime.extend(runtime)
            self.history.extend(history)

        if verbose:
            print("\nRun complete!")

        return (np.array(self.traj), np.array(self.runtime), np.array(self.history))


class AsyncDE(DEBase):
    def __init__(self, cs=None, f=None, dimensions=None, pop_size=None, max_age=np.inf,
                 mutation_factor=None, crossover_prob=None, strategy='rand1_bin',
                 budget=None, async_strategy='random', **kwargs):
        super().__init__(cs=cs, f=f, dimensions=dimensions, pop_size=pop_size, max_age=max_age,
                         mutation_factor=mutation_factor, crossover_prob=crossover_prob,
                         strategy=strategy, budget=budget, **kwargs)
        if self.strategy is not None:
            self.mutation_strategy = self.strategy.split('_')[0]
            self.crossover_strategy = self.strategy.split('_')[1]
        else:
            self.mutation_strategy = self.crossover_strategy = None
        self._set_min_pop_size()
        self.async_strategy = async_strategy

    def _shuffle_pop(self):
        pop_order = np.arange(len(self.population))
        np.random.shuffle(pop_order)
        self.population = self.population[pop_order]
        self.fitness = self.fitness[pop_order]
        self.age = self.age[pop_order]

    def _set_min_pop_size(self):
        if self.mutation_strategy in ['rand1', 'rand2dir', 'randtobest1']:
            self._min_pop_size = 3
        elif self.mutation_strategy in ['currenttobest1', 'best1']:
            self._min_pop_size = 2
        elif self.mutation_strategy in ['best2']:
            self._min_pop_size = 4
        elif self.mutation_strategy in ['rand2']:
            self._min_pop_size = 5
        else:
            self._min_pop_size = 1

        return self._min_pop_size

    def _add_random_population(self, pop_size, population=None, fitness=[], age=[]):
        new_pop = self.init_population(pop_size=pop_size)
        new_fitness = np.array([np.inf] * pop_size)
        new_age = np.array([self.max_age] * pop_size)

        if population is None:
            population = self.population
            fitness = self.fitness
            age = self.age

        population = np.concatenate((population, new_pop))
        fitness = np.concatenate((fitness, new_fitness))
        age = np.concatenate((age, new_age))

        return population, fitness, age

    def _init_mutant_population(self, pop_size, population, target=None, best=None):
        mutants = np.random.uniform(low=0.0, high=1.0, size=(pop_size, self.dimensions))
        for i in range(pop_size):
            mutants[i] = self.mutation(current=target, best=best, alt_pop=population)
        return mutants

    def _new_sample_population(self, size=3, alt_pop=None, target=None):
        population = None
        if isinstance(alt_pop, list) or isinstance(alt_pop, np.ndarray):
            idx = [indv is None for indv in alt_pop]
            if any(idx):
                population = self.population
            else:
                population = alt_pop
        else:
            population = self.population
        if target is not None and len(population) > 1:
            # eliminating target from mutation sampling pool
            for i, pop in enumerate(population):
                if all(target == pop):
                    population = np.concatenate((population[:i], population[i + 1:]))
                    break
        if len(population) < self._min_pop_size:
            # compensate if target was part of the population and deleted earlier
            filler = self._min_pop_size - len(population)
            new_pop = self.init_population(pop_size=filler)
            population = np.concatenate((population, new_pop))

        selection = np.random.choice(np.arange(len(population)), size, replace=False)
        return population[selection]

    def sample_population(self, size=3, alt_pop=None, target=None):
        '''Samples 'size' individuals

        If alt_pop is None or a list/array of None, sample from own population
        Else sample from the specified alternate population
        '''
        return self._new_sample_population(size, alt_pop, target)

        population = None
        if isinstance(alt_pop, list) or isinstance(alt_pop, np.ndarray):
            idx = [indv is None for indv in alt_pop]
            if any(idx):
                population = self.population
            else:
                population = alt_pop
        else:
            population = self.population
        if target is not None and len(population) > 1:
            # eliminating target from mutation sampling pool
            for i, pop in enumerate(population):
                if all(target == pop):
                    population = np.concatenate((population[:i], population[i+1:]))
                    break
        if len(population) < self._min_pop_size:
            selection = np.random.choice(np.arange(len(population)), size, replace=True)
        else:
            selection = np.random.choice(np.arange(len(population)), size, replace=False)
        return population[selection]

    def f_objective(self, x, budget=None):
        if self.f is None:
            raise NotImplementedError("An objective function needs to be passed.")
        config = self.vector_to_configspace(x)
        if budget is not None:  # to be used when called by multi-fidelity based optimizers
            fitness, cost = self.f(config, budget=budget)
        else:
            fitness, cost = self.f(config)
        return fitness, cost

    def init_eval_pop(self, budget=None, eval=True):
        '''Creates new population of 'pop_size' and evaluates individuals.
        '''
        self.population = self.init_population(self.pop_size)
        self.fitness = np.array([np.inf for i in range(self.pop_size)])
        self.age = np.array([self.max_age] * self.pop_size)

        traj = []
        runtime = []
        history = []

        if not eval:
            return traj, runtime, history

        for i in range(self.pop_size):
            config = self.population[i]
            self.fitness[i], cost = self.f_objective(config, budget)
            if self.fitness[i] <= self.inc_score:
                self.inc_score = self.fitness[i]
                self.inc_config = config
            traj.append(self.inc_score)
            runtime.append(cost)
            history.append((config.tolist(), float(self.fitness[i]), float(budget or 0)))

        return traj, runtime, history

    def eval_pop(self, population=None, budget=None):
        pop = self.population if population is None else population
        pop_size = self.pop_size if population is None else len(pop)
        traj = []
        runtime = []
        history = []
        fitnesses = []
        costs = []
        ages = []
        for i in range(pop_size):
            fitness, cost = self.f_objective(pop[i], budget)
            if population is None:
                self.fitness[i] = fitness
            if fitness <= self.inc_score:
                self.inc_score = fitness
                self.inc_config = pop[i]
            traj.append(self.inc_score)
            runtime.append(cost)
            history.append((pop[i].tolist(), float(fitness), float(budget or 0)))
            fitnesses.append(fitness)
            costs.append(cost)
            ages.append(self.max_age)
        # if population is None:
        # #     self.fitness[:self.pop_size] = np.array(fitnesses)
        #     return traj, runtime, history
        # else:
        return traj, runtime, history, np.array(fitnesses), np.array(ages)

    def mutation_rand1(self, r1, r2, r3):
        '''Performs the 'rand1' type of DE mutation
        '''
        diff = r2 - r3
        mutant = r1 + self.mutation_factor * diff
        return mutant

    def mutation_rand2(self, r1, r2, r3, r4, r5):
        '''Performs the 'rand2' type of DE mutation
        '''
        diff1 = r2 - r3
        diff2 = r4 - r5
        mutant = r1 + self.mutation_factor * diff1 + self.mutation_factor * diff2
        return mutant

    def mutation_currenttobest1(self, current, best, r1, r2):
        diff1 = best - current
        diff2 = r1 - r2
        mutant = current + self.mutation_factor * diff1 + self.mutation_factor * diff2
        return mutant

    def mutation_rand2dir(self, r1, r2, r3):
        diff = r1 - r2 - r3
        mutant = r1 + self.mutation_factor * diff / 2
        return mutant

    def mutation(self, current=None, best=None, alt_pop=None):
        '''Performs DE mutation
        '''
        if self.mutation_strategy == 'rand1':
            r1, r2, r3 = self.sample_population(size=3, alt_pop=alt_pop, target=current)
            mutant = self.mutation_rand1(r1, r2, r3)

        elif self.mutation_strategy == 'rand2':
            r1, r2, r3, r4, r5 = self.sample_population(size=5, alt_pop=alt_pop, target=current)
            mutant = self.mutation_rand2(r1, r2, r3, r4, r5)

        elif self.mutation_strategy == 'rand2dir':
            r1, r2, r3 = self.sample_population(size=3, alt_pop=alt_pop, target=current)
            mutant = self.mutation_rand2dir(r1, r2, r3)

        elif self.mutation_strategy == 'best1':
            r1, r2 = self.sample_population(size=2, alt_pop=alt_pop, target=current)
            if best is None:
                best = self.population[np.argmin(self.fitness)]
            mutant = self.mutation_rand1(best, r1, r2)

        elif self.mutation_strategy == 'best2':
            r1, r2, r3, r4 = self.sample_population(size=4, alt_pop=alt_pop, target=current)
            if best is None:
                best = self.population[np.argmin(self.fitness)]
            mutant = self.mutation_rand2(best, r1, r2, r3, r4)

        elif self.mutation_strategy == 'currenttobest1':
            r1, r2 = self.sample_population(size=2, alt_pop=alt_pop, target=current)
            if best is None:
                best = self.population[np.argmin(self.fitness)]
            mutant = self.mutation_currenttobest1(current, best, r1, r2)

        elif self.mutation_strategy == 'randtobest1':
            r1, r2, r3 = self.sample_population(size=3, alt_pop=alt_pop, target=current)
            if best is None:
                best = self.population[np.argmin(self.fitness)]
            mutant = self.mutation_currenttobest1(r1, best, r2, r3)

        return mutant

    def crossover_bin(self, target, mutant):
        '''Performs the binomial crossover of DE
        '''
        cross_points = np.random.rand(self.dimensions) < self.crossover_prob
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dimensions)] = True
        offspring = np.where(cross_points, mutant, target)
        return offspring

    def crossover_exp(self, target, mutant):
        '''Performs the exponential crossover of DE
        '''
        n = np.random.randint(0, self.dimensions)
        L = 0
        while ((np.random.rand() < self.crossover_prob) and L < self.dimensions):
            idx = (n+L) % self.dimensions
            target[idx] = mutant[idx]
            L = L + 1
        return target

    def crossover(self, target, mutant):
        '''Performs DE crossover
        '''
        if self.crossover_strategy == 'bin':
            offspring = self.crossover_bin(target, mutant)
        elif self.crossover_strategy == 'exp':
            offspring = self.crossover_exp(target, mutant)
        return offspring

    def selection(self, trials, budget=None):
        '''Carries out a parent-offspring competition given a set of trial population
        '''
        assert(len(self.population) >= len(trials))
        traj = []
        runtime = []
        history = []
        for i in range(len(trials)):
            # evaluation of the newly created individuals
            fitness, cost = self.f_objective(trials[i], budget)
            # selection -- competition between parent[i] -- child[i]
            ## equality is important for landscape exploration
            if fitness <= self.fitness[i]:
                self.population[i] = trials[i]
                self.fitness[i] = fitness
                # resetting age since new individual in the population
                self.age[i] = self.max_age
            else:
                # decreasing age by 1 of parent who is better than offspring/trial
                self.age[i] -= 1
            # updation of global incumbent for trajectory
            if self.fitness[i] <= self.inc_score:
                self.inc_score = self.fitness[i]
                self.inc_config = self.population[i]
            traj.append(self.inc_score)
            runtime.append(cost)
            history.append((trials[i].tolist(), float(fitness), float(budget or 0)))
        return traj, runtime, history

    def ranked_selection(self, trials, final_pop_size=None, budget=None, debug=False):
        '''Returns the fittest individuals from two sets of population
        '''
        final_pop_size = self.pop_size if final_pop_size is None else final_pop_size

        traj, runtime, history, fitnesses, ages = self.eval_pop(population=trials, budget=budget)

        if debug:
            print("Ranking {} from {} originals + {} rivals".format(final_pop_size, self.pop_size,
                                                                    len(trials)))

        # Creating a total population of current individuals and rival individuals evaluated on
        # the same budget
        tot_pop = np.vstack((self.population, trials))
        tot_fitness = np.hstack((self.fitness, fitnesses))
        tot_age = np.hstack((self.age, ages))

        # Sorting the total population by fitness to keep only the top individuals from the pool
        rank = np.sort(np.argsort(tot_fitness)[:final_pop_size])
        self.population = tot_pop[rank]
        self.fitness = tot_fitness[rank]
        self.age = tot_age[rank]
        self.pop_size = final_pop_size
        return traj, runtime, history

    def kill_aged_pop(self, budget=None, debug=False):
        '''Replaces individuals with age older than max_age
        '''
        traj = []
        runtime = []
        history = []
        idxs = np.where(self.age <= 0)[0]
        if len(idxs) == 0:
            return traj, runtime, history
        if debug:
            print("Killing {} individual(s) for budget {}: {}".format(len(idxs), budget, self.age[idxs]))
        new_pop = self.init_population(pop_size=len(idxs))
        for i, index in enumerate(idxs):
            self.population[index] = new_pop[i]
            self.fitness[index], cost = self.f_objective(self.population[index], budget)
            self.age[index] = self.max_age
            if self.fitness[index] < self.inc_score:
                self.inc_score = self.fitness[index]
                self.inc_config = self.population[index]
            traj.append(self.inc_score)
            runtime.append(cost)
            history.append((self.population[index].tolist(), float(self.fitness[index]),
                            float(budget or 0)))
        return traj, runtime, history

    def evolve_generation(self, budget=None, best=None, alt_pop=None, async_strategy='orig'):
        '''Performs a complete DE evolution, mutation -> crossover -> selection
        '''
        traj = []
        runtime = []
        history = []

        if async_strategy == 'orig':
            trials = []
            for j in range(self.pop_size):
                target = self.population[j]
                donor = self.mutation(current=target, best=best, alt_pop=alt_pop)
                trial = self.crossover(target, donor)
                trial = self.boundary_check(trial)
                trials.append(trial)
            trials = np.array(trials)
            traj, runtime, history = self.selection(trials, budget)
            return traj, runtime, history

        elif async_strategy == 'basic':
            for i in range(self.pop_size):
                target = self.population[i]
                donor = self.mutation(current=target, best=best, alt_pop=alt_pop)
                trial = self.crossover(target, donor)
                trial = self.boundary_check(trial)
                de_traj, de_runtime, de_history, fitnesses, costs = \
                    self.eval_pop(trial.reshape(1, self.dimensions), budget=budget)
                # one-vs-one selection
                if fitnesses[0] <= self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = fitnesses[0]
                traj.extend(de_traj)
                runtime.extend(de_runtime)
                history.extend(de_history)
            return traj, runtime, history

        for count in range(self.pop_size):
            if async_strategy == 'random':
                i = np.random.choice(np.arange(self.pop_size))
            else:  # async_strategy == 'worst'
                i = np.argsort(-self.fitness)[0]
            target = self.population[i]
            donor = self.mutation(current=target, best=best, alt_pop=alt_pop)
            trial = self.crossover(target, donor)
            trial = self.boundary_check(trial)
            de_traj, de_runtime, de_history, fitnesses, costs = \
                self.eval_pop(trial.reshape(1, self.dimensions), budget=budget)
            # one-vs-one selection
            if fitnesses[0] <= self.fitness[i]:
                self.population[i] = trial
                self.fitness[i] = fitnesses[0]
            traj.extend(de_traj)
            runtime.extend(de_runtime)
            history.extend(de_history)

        return traj, runtime, history

    def sample_mutants(self, size, population=None, fix_type='random'):
        if population is None:
            population = self.population

        mutants = np.random.uniform(low=0.0, high=1.0, size=(size, self.dimensions))
        for i in range(size):
            j = np.random.choice(np.arange(len(population)))
            mutant = self.mutation(current=population[j], best=self.inc_config, alt_pop=population)
            mutants[i] = self.boundary_check(mutant, fix_type=fix_type)

        return mutants

    def run(self, generations=1, verbose=False, budget=None, async_strategy='basic'):
        self.traj = []
        self.runtime = []
        self.history = []

        if verbose:
            print("Initializing and evaluating new population...")
        self.traj, self.runtime, self.history = self.init_eval_pop(budget=budget)

        if verbose:
            print("Running evolutionary search...")
        for i in range(generations):
            if verbose:
                print("Generation {:<2}/{:<2} -- {:<0.7}".format(i+1, generations, self.inc_score))
            traj, runtime, history = self.evolve_generation(budget=budget, async_strategy=async_strategy)
            self.traj.extend(traj)
            self.runtime.extend(runtime)
            self.history.extend(history)

        if verbose:
            print("\nRun complete!")

        return (np.array(self.traj), np.array(self.runtime), np.array(self.history))
