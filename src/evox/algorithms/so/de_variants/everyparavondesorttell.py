from evox import (
    Algorithm,
    State,
    jit_class,
)
from evox.operators.selection import select_rand_pbest
from evox.operators.crossover import (
    de_diff_sum,
    de_arith_recom,
    de_bin_cross,
    de_exp_cross,
)
from evox.utils import *


def adjacent_indices_array_wrapped_exclude_center(n, size=3):
    """
    Function to generate an array of indices for a 2D grid with wrapping at the edges.
    The center index is excluded from each sub-array.

    Parameters:
    n (int): Total number of elements.
    size (int): Size of the square sub-grid.

    Returns:
    np.array: Array of indices.
    """
    side_length = int(np.ceil(np.sqrt(n)))
    if side_length * (side_length - 1) >= n:
        rows, cols = side_length, side_length - 1
    else:
        rows, cols = side_length, side_length

    indices = np.arange(rows * cols) % n
    indices_padded = np.pad(indices.reshape((rows, cols)),
                            ((size // 2, size // 2), (size // 2, size // 2)),
                            'wrap')

    result_list = []

    for row in range(rows):
        for col in range(cols):
            window_indices = indices_padded[row:row + size, col:col + size].flatten()
            center_index = size * size // 2
            window_indices = np.concatenate([window_indices[:center_index], window_indices[center_index + 1:]])
            result_list.append(window_indices)

    result_array = np.array(result_list)[:n]

    for row in result_array:
        half_size = len(row) // 2
        for i in range(half_size):
            if row[i] > row[-1 - i]:
                row[i], row[-1 - i] = row[-1 - i], row[i]
    return result_array


@jit_class
class EveryParaVonDESortTell(Algorithm):
    """
    Class representing the EveryParaVonDESortTell algorithm, a variant of the Differential Evolution algorithm.
    This class is a child of the Algorithm class from the evox library.

    The class uses JAX for just-in-time compilation to speed up the computations.
    """
    def __init__(
            self,
            lb,
            ub,
            pop_size=10000,
            diff_padding_num=9,
            neighbor_index=None,
            convolution_size=3,
            replace=False,
            mean=None,
            stdvar=None,
    ):
        """
        Constructor for the EveryParaVonDESortTell class.

        Parameters:
        lb (np.array): Lower bounds for the population.
        ub (np.array): Upper bounds for the population.
        pop_size (int): Population size.
        diff_padding_num (int): Number of padding for the difference.
        neighbor_index (list): List of neighbor indices.
        convolution_size (int): Size of the convolution.
        replace (bool): If True, replacement is used in the selection.
        mean (float): Mean for the normal distribution used in the initialization.
        stdvar (float): Standard deviation for the normal distribution used in the initialization.
        """
        assert jnp.all(lb < ub)
        assert pop_size >= 4
        assert convolution_size % 2 == 1
        self.convolution_size = convolution_size
        self.dim = lb.shape[0]
        self.lb = lb
        self.ub = ub
        self.pop_size = pop_size
        self.neighbor_index = neighbor_index
        self.replace = replace
        self.mean = mean
        self.stdvar = stdvar
        self.diff_padding_num = diff_padding_num

    def setup(self, key):
        """
        Set up the initial state for the algorithm.

        Parameters:
        key (int): Random seed.

        Returns:
        State: Initial state.
        """
        state_key, init_key, param1_key, param2_key, param3_key, param4_key, param5_key = jax.random.split(key, 7)
        if self.neighbor_index is None:
            self.neighbor_index = list(range(self.convolution_size ** 2 // 2))

        if self.mean is not None and self.stdvar is not None:
            population = self.stdvar * jax.random.normal(
                init_key, shape=(self.pop_size, self.dim)
            )
            population = jnp.clip(population, self.lb, self.ub)
        else:
            population = jax.random.uniform(init_key, shape=(self.pop_size, self.dim))
            population = population * (self.ub - self.lb) + self.lb
        fitness = jnp.full((self.pop_size,), jnp.inf)
        best_index = 0
        start_index = 0
        adjlist = jnp.array(adjacent_indices_array_wrapped_exclude_center(self.pop_size,
                                                                          size=self.convolution_size))

        params_array = jax.random.uniform(param1_key, shape=(self.pop_size, 2))
        cross_strategy_array = jax.random.choice(param5_key, jnp.array([0, 1, 2]), shape=(self.pop_size,))
        params_list = jnp.column_stack((params_array, cross_strategy_array))

        return State(
            population=population,
            fitness=fitness,
            best_index=best_index,
            start_index=start_index,
            key=state_key,
            adjlist=adjlist,
            params=params_list,
            count=0,
        )

    def ask(self, state):
        """
        Generate trial vectors for the population.

        Parameters:
        state (State): Current state.

        Returns:
        np.array, State: Trial vectors and updated state.
        """
        key, ask_one_key = jax.random.split(state.key, 2)
        ask_one_keys = jax.random.split(ask_one_key, self.pop_size)
        indices = jnp.arange(self.pop_size)

        trial_vectors = vmap(
            partial(
                self._ask_one, state=state
            )
        )(index=indices, key=ask_one_keys, neighbor_choiced=state.adjlist)

        return trial_vectors, state.update(trial_vectors=trial_vectors, key=key)

    def _ask_one(self, state, index, key, neighbor_choiced):
        """
        Generate a trial vector for a single individual.

        Parameters:
        state (State): Current state.
        index (int): Index of the individual.
        key (int): Random seed.
        neighbor_choiced (np.array): Array of chosen neighbors.

        Returns:
        np.array: Trial vector.
        """
        select_key, pbest_key, crossover_key = jax.random.split(key, 3)
        population = state.population
        params = state.params
        param = params[index]

        base_vector = population[index]
        difference_sum = jnp.zeros_like(base_vector)
        index_mod = self.convolution_size ** 2 - 1
        for loop_index, i in enumerate(self.neighbor_index):
            first_index = (index_mod - i - 1)
            second_index = i
            difference_sum += population[neighbor_choiced[first_index], :] - population[neighbor_choiced[second_index],
                                                                             :]
        mutation_vector = (base_vector + difference_sum * param[0])
        """Crossover: 0 = bin, 1 = exp, 2 = arith"""
        cross_funcs = (
            de_bin_cross,
            de_exp_cross,
            lambda _key, x, y, z: de_arith_recom(x, y, z),
        )
        trial_vector = jax.lax.switch(
            param[2].astype(int),
            cross_funcs,
            crossover_key,
            mutation_vector,
            base_vector,
            param[1],
        )
        trial_vector = jnp.clip(trial_vector, self.lb, self.ub)
        return trial_vector

    def tell(self, state, trial_fitness):
        """
        Update the state based on the fitness of the trial vectors.

        Parameters:
        state (State): Current state.
        trial_fitness (np.array): Fitness values of the trial vectors.

        Returns:
        State: Updated state.
        """
        combined_population = jnp.concatenate([state.population, state.trial_vectors], axis=0)

        combined_fitness = jnp.concatenate([state.fitness, trial_fitness], axis=0)

        sorted_indices = jnp.argsort(combined_fitness)
        sorted_population = combined_population[sorted_indices]
        sorted_fitness = combined_fitness[sorted_indices]

        new_population = sorted_population[:self.pop_size]
        new_fitness = sorted_fitness[:self.pop_size]

        best_index = jnp.argmin(new_fitness)

        new_state = state.update(
            population=new_population,
            fitness=new_fitness,
            best_index=best_index,
        )
        return new_state

    def override(self, state, key, params):
        """
        Override the current state with a new one.

        Parameters:
        state (State): Current state.
        key (int): Random seed.
        params (np.array): Parameters for the new state.

        Returns:
        State: New state.
        """
        state = state | self.setup(key)
        return state.update(params=params)
# import random
#
# from evox import (
#     Algorithm,
#     State,
#     jit_class,
# )
# from evox.operators.selection import select_rand_pbest
# from evox.operators.crossover import (
#     de_diff_sum,
#     de_arith_recom,
#     de_bin_cross,
#     de_exp_cross,
# )
# from evox.utils import *
#
#
# def adjacent_indices_array_wrapped_exclude_center(n, size=3):
#     """
#     Function to generate an array of indices for a 2D grid with wrapping at the edges.
#     The center index is excluded from each sub-array.
#
#     Parameters:
#     n (int): Total number of elements.
#     size (int): Size of the square sub-grid.
#
#     Returns:
#     np.array: Array of indices.
#     """
#     side_length = int(np.ceil(np.sqrt(n)))
#     if side_length * (side_length - 1) >= n:
#         rows, cols = side_length, side_length - 1
#     else:
#         rows, cols = side_length, side_length
#
#     indices = np.arange(rows * cols) % n
#     indices_padded = np.pad(indices.reshape((rows, cols)),
#                             ((size // 2, size // 2), (size // 2, size // 2)),
#                             'wrap')
#
#     result_list = []
#
#     for row in range(rows):
#         for col in range(cols):
#             window_indices = indices_padded[row:row + size, col:col + size].flatten()
#             center_index = size * size // 2
#             window_indices = np.concatenate([window_indices[:center_index], window_indices[center_index + 1:]])
#             result_list.append(window_indices)
#
#     result_array = np.array(result_list)[:n]
#
#     for row in result_array:
#         half_size = len(row) // 2
#         for i in range(half_size):
#             if row[i] > row[-1 - i]:
#                 row[i], row[-1 - i] = row[-1 - i], row[i]
#     return result_array
#
#
# @jit_class
# class EveryParaVonDESortTell(Algorithm):
#     """
#     Class representing the EveryParaVonDESortTell algorithm, a variant of the Differential Evolution algorithm.
#     This class is a child of the Algorithm class from the evox library.
#
#     The class uses JAX for just-in-time compilation to speed up the computations.
#     """
#     def __init__(
#             self,
#             lb,
#             ub,
#             pop_size=10000,
#             diff_padding_num=9,
#             neighbor_index=None,
#             convolution_size=3,
#             replace=False,
#             mean=None,
#             stdvar=None,
#             # differential_neighbor_index=None,
#     ):
#         """
#         Constructor for the EveryParaVonDESortTell class.
#
#         Parameters:
#         lb (np.array): Lower bounds for the population.
#         ub (np.array): Upper bounds for the population.
#         pop_size (int): Population size.
#         diff_padding_num (int): Number of padding for the difference.
#         neighbor_index (list): List of neighbor indices.
#         convolution_size (int): Size of the convolution.
#         replace (bool): If True, replacement is used in the selection.
#         mean (float): Mean for the normal distribution used in the initialization.
#         stdvar (float): Standard deviation for the normal distribution used in the initialization.
#         """
#         assert jnp.all(lb < ub)
#         assert pop_size >= 4
#         assert convolution_size % 2 == 1
#         self.convolution_size = convolution_size
#         self.dim = lb.shape[0]
#         self.lb = lb
#         self.ub = ub
#         self.pop_size = pop_size
#         self.neighbor_index = neighbor_index
#         self.replace = replace
#         self.mean = mean
#         self.stdvar = stdvar
#         self.diff_padding_num = diff_padding_num
#         # self.differential_neighbor_index = differential_neighbor_index
#
#     def setup(self, key):
#         """
#         Set up the initial state for the algorithm.
#
#         Parameters:
#         key (int): Random seed.
#
#         Returns:
#         State: Initial state.
#         """
#         state_key, init_key, param1_key, param2_key, param3_key, param4_key, param5_key = jax.random.split(key, 7)
#         if self.neighbor_index is None:
#             self.neighbor_index = list(range(self.convolution_size ** 2 // 2))
#         result = []
#         # for _ in range(self.pop_size):
#         #     n = random.choice([2, 4, 6, 8])
#         #     nested_list = random.sample(range(3 ** 2 - 1), n)
#         #     nested_list.sort()
#         #     if n < 8:
#         #         fill_count = 8 - n
#         #         fill_index = n // 2
#         #         nested_list = nested_list[:fill_index] + [0] * fill_count + nested_list[fill_index:]
#         #     result.append(nested_list)
#         for _ in range(self.pop_size):
#             n = random.choice([2, 4, 6, 8])
#             if n == 2:
#                 nested_list = random.sample([3, 4], n)
#             elif n == 4:
#                 nested_list = random.sample([1, 3, 4, 6], n)
#             elif n == 6:
#                 nested_list = random.sample([1, 2, 3, 4, 5, 6], n)
#             else:  # n == 8
#                 nested_list = random.sample(range(8), n)
#
#             nested_list.sort()
#
#             if n < 8:
#                 fill_count = 8 - n
#                 fill_index = n // 2
#                 nested_list = nested_list[:fill_index] + [0] * fill_count + nested_list[fill_index:]
#
#             result.append(nested_list)
#         result = jnp.array(result)
#
#         if self.mean is not None and self.stdvar is not None:
#             population = self.stdvar * jax.random.normal(
#                 init_key, shape=(self.pop_size, self.dim)
#             )
#             population = jnp.clip(population, self.lb, self.ub)
#         else:
#             population = jax.random.uniform(init_key, shape=(self.pop_size, self.dim))
#             population = population * (self.ub - self.lb) + self.lb
#         fitness = jnp.full((self.pop_size,), jnp.inf)
#         best_index = 0
#         start_index = 0
#         adjlist = jnp.array(adjacent_indices_array_wrapped_exclude_center(self.pop_size,
#                                                                           size=self.convolution_size))
#
#         params_array = jax.random.uniform(param1_key, shape=(self.pop_size, 2))
#         cross_strategy_array = jax.random.choice(param5_key, jnp.array([0, 1, 2]), shape=(self.pop_size,))
#         params_list = jnp.column_stack((params_array, cross_strategy_array))
#
#         return State(
#             differential_neighbor_index=result,
#             population=population,
#             fitness=fitness,
#             best_index=best_index,
#             start_index=start_index,
#             key=state_key,
#             adjlist=adjlist,
#             params=params_list
#         )
#
#     def ask(self, state):
#         """
#         Generate trial vectors for the population.
#
#         Parameters:
#         state (State): Current state.
#
#         Returns:
#         np.array, State: Trial vectors and updated state.
#         """
#         key, ask_one_key = jax.random.split(state.key, 2)
#         ask_one_keys = jax.random.split(ask_one_key, self.pop_size)
#         indices = jnp.arange(self.pop_size)
#
#         trial_vectors = vmap(
#             partial(
#                 self._ask_one, state=state
#             )
#         )(index=indices, key=ask_one_keys, neighbor_choiced=state.adjlist, differential_neighbor_index=state.differential_neighbor_index)
#
#         return trial_vectors, state.update(trial_vectors=trial_vectors, key=key)
#
#     def _ask_one(self, state, index, key, neighbor_choiced, differential_neighbor_index):
#         """
#         Generate a trial vector for a single individual.
#
#         Parameters:
#         state (State): Current state.
#         index (int): Index of the individual.
#         key (int): Random seed.
#         neighbor_choiced (np.array): Array of chosen neighbors.
#
#         Returns:
#         np.array: Trial vector.
#         """
#         select_key, pbest_key, crossover_key = jax.random.split(key, 3)
#         population = state.population
#         params = state.params
#         param = params[index]
#
#         base_vector = population[index]
#         difference_sum = jnp.zeros_like(base_vector)
#         for i in range(self.convolution_size**2 // 2):
#             first_index = differential_neighbor_index[-i-1]
#             second_index = differential_neighbor_index[i]
#             difference_sum += population[neighbor_choiced[first_index], :] - population[neighbor_choiced[second_index], :]
#
#         mutation_vector = (base_vector + difference_sum * param[0])
#         """Crossover: 0 = bin, 1 = exp, 2 = arith"""
#         cross_funcs = (
#             de_bin_cross,
#             de_exp_cross,
#             lambda _key, x, y, z: de_arith_recom(x, y, z),
#         )
#         trial_vector = jax.lax.switch(
#             param[2].astype(int),
#             cross_funcs,
#             crossover_key,
#             mutation_vector,
#             base_vector,
#             param[1],
#         )
#         trial_vector = jnp.clip(trial_vector, self.lb, self.ub)
#         return trial_vector
#     def tell(self, state, trial_fitness):
#         """
#         Update the state based on the fitness of the trial vectors.
#
#         Parameters:
#         state (State): Current state.
#         trial_fitness (np.array): Fitness values of the trial vectors.
#
#         Returns:
#         State: Updated state.
#         """
#         start_index = state.start_index
#         batch_pop = jax.lax.dynamic_slice_in_dim(
#             state.population, start_index, self.pop_size, axis=0
#         )
#         batch_fitness = jax.lax.dynamic_slice_in_dim(
#             state.fitness, start_index, self.pop_size, axis=0
#         )
#
#         compare = trial_fitness <= batch_fitness
#
#         population_update = jnp.where(
#             compare[:, jnp.newaxis], state.trial_vectors, batch_pop
#         )
#         fitness_update = jnp.where(compare, trial_fitness, batch_fitness)
#
#         population = jax.lax.dynamic_update_slice_in_dim(
#             state.population, population_update, start_index, axis=0
#         )
#         fitness = jax.lax.dynamic_update_slice_in_dim(
#             state.fitness, fitness_update, start_index, axis=0
#         )
#         best_index = jnp.argmin(fitness)
#         start_index = (state.start_index + self.pop_size) % self.pop_size
#         return state.update(
#             population=population,
#             fitness=fitness,
#             best_index=best_index,
#             start_index=start_index,
#         )
#
#     # def tell(self, state, trial_fitness):
#     #     """
#     #     Update the state based on the fitness of the trial vectors.
#     #
#     #     Parameters:
#     #     state (State): Current state.
#     #     trial_fitness (np.array): Fitness values of the trial vectors.
#     #
#     #     Returns:
#     #     State: Updated state.
#     #     """
#     #     combined_population = jnp.concatenate([state.population, state.trial_vectors], axis=0)
#     #
#     #     combined_fitness = jnp.concatenate([state.fitness, trial_fitness], axis=0)
#     #
#     #     sorted_indices = jnp.argsort(combined_fitness)
#     #     sorted_population = combined_population[sorted_indices]
#     #     sorted_fitness = combined_fitness[sorted_indices]
#     #
#     #     new_population = sorted_population[:self.pop_size]
#     #     new_fitness = sorted_fitness[:self.pop_size]
#     #
#     #     best_index = jnp.argmin(new_fitness)
#     #
#     #     new_state = state.update(
#     #         population=new_population,
#     #         fitness=new_fitness,
#     #         best_index=best_index,
#     #     )
#     #     return new_state
#
#     def override(self, state, key, params):
#         """
#         Override the current state with a new one.
#
#         Parameters:
#         state (State): Current state.
#         key (int): Random seed.
#         params (np.array): Parameters for the new state.
#
#         Returns:
#         State: New state.
#         """
#         state = state | self.setup(key)
#         return state.update(params=params)
