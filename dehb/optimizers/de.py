import numpy as np
import ConfigSpace


class DEBase():
    def __init__(self, b=None, cs=None, dimensions=None, pop_size=None, mutation_factor=None,
                 crossover_prob=None, strategy=None, budget=None, **kwargs):
        # Benchmark related variables
        self.b = b
        self.cs = self.b.get_configuration_space() if cs is None and b is not None else cs
        if dimensions is None and self.cs is not None:
            self.dimensions = len(self.cs.get_hyperparameters())
        else:
            self.dimensions = dimensions

        # DE related variables
        self.pop_size = pop_size
        self.mutation_factor = mutation_factor
        self.crossover_prob = crossover_prob
        self.strategy = strategy
        self.budget = budget

        # Miscellaneous
        self.output_path = kwargs['output_path'] if 'output_path' in kwargs else ''

        # Incumbent trackers
        self.inc_score = np.inf
        self.inc_config = None

    def reset(self):
        self.inc_score = np.inf
        self.inc_config = None
        self.population = None

    def init_population(self, pop_size=10):
        self.population = np.random.uniform(low=0.0, high=1.0, size=(pop_size, self.dimensions))
        return self.population

    def sample_population(self, size=3):
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
    def __init__(self, b=None, cs=None, dimensions=None, pop_size=None, mutation_factor=None,
                 crossover_prob=None, strategy='rand1_bin', budget=None, **kwargs):
        super().__init__(b=b, cs=cs, dimensions=dimensions, pop_size=pop_size, mutation_factor=mutation_factor,
                         crossover_prob=crossover_prob, strategy=strategy, budget=budget, **kwargs)

        if self.strategy is not None:
            self.mutation_strategy = self.strategy.split('_')[0]
            self.crossover_strategy = self.strategy.split('_')[1]
        else:
            self.mutation_strategy = self.crossover_strategy = None

    def f_objective(self, x, budget=None):
        if self.b is None:
            raise NotImplementedError("The custom objective function needs to be defined here.")
        config = self.vector_to_configspace(x)
        if budget is not None:  # to be used when called by multi-fidelity based optimizers
            fitness, cost = self.b.objective_function(config, budget=budget)
        else:
            fitness, cost = self.b.objective_function(config)
        return fitness, cost

    def init_eval_pop(self, budget=None):
        '''Creates new population of 'pop_size' and evaluates individuals.
        '''
        self.population = self.init_population(self.pop_size)
        self.fitness = np.array([np.inf for i in range(self.pop_size)])

        traj = []
        runtime = []
        for i in range(self.pop_size):
            config = self.population[i]
            self.fitness[i], cost = self.f_objective(config, budget)
            if self.fitness[i] < self.inc_score:
                self.inc_score = self.fitness[i]
                self.inc_config = config
            traj.append(self.inc_score)
            runtime.append(cost)

        return traj, runtime

    def mutation_rand1(self, r1, r2, r3):
        '''Performs the 'rand1' type of DE mutation
        '''
        diff = r2 - r3
        mutant = r1 + self.mutation_factor * diff
        return mutant

    def mutation(self):
        '''Performs DE mutation
        '''
        if self.mutation_strategy == 'rand1':
            r1, r2, r3 = self.sample_population(size=3)
            mutant = self.mutation_rand1(r1, r2, r3)
        return mutant

    def crossover_bin(self, parent, mutant):
        '''Performs the binomial crossover of DE
        '''
        cross_points = np.random.rand(self.dimensions) < self.crossover_prob
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dimensions)] = True
        offspring = np.where(cross_points, mutant, parent)
        return offspring

    def crossover(self, parent, mutant):
        '''Performs DE crossover
        '''
        if self.crossover_strategy == 'bin':
            offspring = self.crossover_bin(parent, mutant)
        return offspring

    def evolve(self, current=None, best=None):
        '''Performs a DE mutation and crossover
        '''
        mutant = self.mutation()
        offspring = self.crossover(current, mutant)
        offspring = self.boundary_check(offspring)
        return offspring

    def step(self, j, budget=None):
        '''Performs a single DE evolution for an individual
        '''
        offspring = self.evolve(current=self.population[j])
        fitness, cost = self.f_objective(offspring, budget)
        if fitness < self.fitness[j]:
            self.fitness[j] = fitness
            self.population[j] = offspring
            if self.fitness[j] < self.inc_score:
                self.inc_score = self.fitness[j]
                self.inc_config = self.population[j]
        return self.inc_score, cost

    def run(self, iterations=100, verbose=False):
        self.traj = []
        self.runtime = []

        if verbose:
            print("Initializing and evaluating new population...")
        self.traj, self.runtime = self.init_eval_pop()

        if verbose:
            print("Running evolutionary search...")
        for i in range(iterations):
            for j in range(self.pop_size):
                if verbose:
                    print("Iteration {:<2}/{:<2} -- "
                          "Evaluating individual {:<2}/{:<2}".format(i+1, iterations, j+1,
                                                                     self.pop_size), end='\r')
                traj, runtime = self.step(j)
                self.traj.append(traj)
                self.runtime.append(runtime)

        if verbose:
            print("\nRun complete!")

        return np.array(self.traj), np.array(self.runtime)
