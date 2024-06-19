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
class ParaVonDESortTell(Algorithm):
    def __init__(
            self,
            lb,
            ub,
            pop_size=10000,
            neighbor_index=None,
            base_vector="rand",
            convolution_size=3,
            differential_weight=0.2,
            differential_weight_lb=0.1,
            differential_weight_ub=0.3,
            is_diff_weight_random=False,
            cross_probability=0.9,
            replace=False,
            mean=None,
            stdvar=None,
            reverse_proportion=0.05,
    ):
        assert jnp.all(lb < ub)
        assert pop_size >= 4
        assert cross_probability > 0 and cross_probability <= 1
        assert base_vector in ["rand", "best"]
        assert convolution_size % 2 == 1
        self.convolution_size = convolution_size
        self.dim = lb.shape[0]
        self.lb = lb
        self.ub = ub
        self.pop_size = pop_size
        self.neighbor_index = neighbor_index
        self.is_diff_weight_random = is_diff_weight_random
        self.base_vector = base_vector
        self.replace = replace
        self.cross_probability = cross_probability
        self.reverse_proportion = reverse_proportion
        self.differential_weight = differential_weight
        self.differential_weight_lb = differential_weight_lb
        self.differential_weight_ub = differential_weight_ub
        self.mean = mean
        self.stdvar = stdvar

    def setup(self, key):
        state_key, init_key, param1_key, param2_key, param3_key = jax.random.split(key, 5)
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
        side_length = int(np.ceil(np.sqrt(self.pop_size)))
        if side_length * (side_length - 1) >= self.pop_size:
            rows, cols = side_length, side_length - 1
        else:
            rows, cols = side_length, side_length

        arr = np.arange(10000)
        matrix = arr.reshape(100, 100)
        splits = []
        belong_parts = jnp.zeros(10000, dtype=jnp.int32)
        part_number = 0
        for i in range(0, 100, 10):
            for j in range(0, 100, 10):
                small_matrix = matrix[i:i + 10, j:j + 10]
                splits.append(small_matrix.flatten())
                belong_parts = belong_parts.at[i * 100 + j: i * 100 + j + 10].set(part_number)
                belong_parts = belong_parts.at[i * 100 + j + 100: i * 100 + j + 110].set(part_number)
                belong_parts = belong_parts.at[i * 100 + j + 200: i * 100 + j + 210].set(part_number)
                belong_parts = belong_parts.at[i * 100 + j + 300: i * 100 + j + 310].set(part_number)
                belong_parts = belong_parts.at[i * 100 + j + 400: i * 100 + j + 410].set(part_number)
                belong_parts = belong_parts.at[i * 100 + j + 500: i * 100 + j + 510].set(part_number)
                belong_parts = belong_parts.at[i * 100 + j + 600: i * 100 + j + 610].set(part_number)
                belong_parts = belong_parts.at[i * 100 + j + 700: i * 100 + j + 710].set(part_number)
                belong_parts = belong_parts.at[i * 100 + j + 800: i * 100 + j + 810].set(part_number)
                belong_parts = belong_parts.at[i * 100 + j + 900: i * 100 + j + 910].set(part_number)
                part_number += 1

        num_diff_vects_choices = [2, 4]
        cross_strategy_choices = [0, 1, 2]

        params_array = jax.random.uniform(param1_key, shape=(100, 2))
        num_diff_vects_array = jax.random.choice(param2_key, jnp.array(num_diff_vects_choices), shape=(100,))
        cross_strategy_array = jax.random.choice(param3_key, jnp.array(cross_strategy_choices), shape=(100,))

        params_list = jnp.column_stack((params_array, num_diff_vects_array, cross_strategy_array))

        return State(
            belong_parts=belong_parts,
            population=population,
            fitness=fitness,
            best_index=best_index,
            start_index=start_index,
            key=state_key,
            adjlist=adjlist,
            splits=splits,
            params=params_list,
            count=0,
        )

    def ask(self, state):
        key, R_key = jax.random.split(state.key, 2)

        indices = jnp.arange(self.pop_size)

        neighbor_choices = state.adjlist

        total_elements = state.adjlist.size
        num_ones = int(self.reverse_proportion * total_elements)

        reversed_masks = jnp.zeros(total_elements, dtype=int)
        reversed_masks = reversed_masks.at[:num_ones].set(1)
        reversed_masks = jax.random.permutation(key, reversed_masks)[:self.pop_size]

        belong_parts = state.belong_parts

        trial_vectors = vmap(
            partial(
                self._ask_one, population=state.population, params=state.params, key=key
            )
        )(indices, neighbor_choice=neighbor_choices, reversed_mask=reversed_masks, belong_part=belong_parts)

        return trial_vectors, state.update(trial_vectors=trial_vectors, key=key)

    def _ask_one(self, index, population, neighbor_choice, reversed_mask, belong_part, params, key):
        select_key, pbest_key, crossover_key = jax.random.split(key, 3)
        base_vector = population[index, :]
        differential_vector = jnp.zeros_like(base_vector)
        param = params[belong_part]

        # 使用 jax.lax.switch 根据 param[3] 的值选择不同的 pairs
        def get_pairs_4():
            return [(0, 7), (1, 6), (2, 5), (3, 4)]

        def get_pairs_2():
            return [(1, 6), (3, 4), (0, 0), (0, 0)]  # 填充到相同长度

        # 使用 switch 选择 pairs
        pairs = jax.lax.switch(
            param[3].astype(int) // 2 - 1,
            [get_pairs_2, get_pairs_4]
        )

        for first_index, second_index in pairs:
            actual_first_index = jax.lax.select(reversed_mask, second_index, first_index)
            actual_second_index = jax.lax.select(reversed_mask, first_index, second_index)

            differential_vector += param[0] * (
                    population[neighbor_choice[actual_first_index], :] - population[
                                                                         neighbor_choice[actual_second_index], :]
            )

        mutation_vector = base_vector + differential_vector

        cross_funcs = (
            de_bin_cross,
            de_exp_cross,
            lambda _key, x, y, z: de_arith_recom(x, y, z),
        )
        trial_vector = jax.lax.switch(
            param[3].astype(int),
            cross_funcs,
            crossover_key,
            mutation_vector,
            base_vector,
            param[1],
        )

        trial_vector = jnp.clip(trial_vector, self.lb, self.ub)
        return trial_vector

    # def tell(self, state, trial_fitness):
    #     new_population = jnp.zeros_like(state.population)
    #     new_fitness = jnp.zeros_like(state.fitness)
    #
    #     for indices in state.splits:
    #         pop_part = state.population[indices]
    #         trial_part = state.trial_vectors[indices]
    #         fit_part = state.fitness[indices]
    #         trial_fit_part = trial_fitness[indices]
    #
    #         combined_population = jnp.concatenate([pop_part, trial_part], axis=0)
    #         combined_fitness = jnp.concatenate([fit_part, trial_fit_part], axis=0)
    #
    #         sorted_indices = jnp.argsort(combined_fitness)
    #         sorted_population = combined_population[sorted_indices]
    #         sorted_fitness = combined_fitness[sorted_indices]
    #
    #         selected_indices = sorted_indices[:100]
    #         new_population = new_population.at[indices].set(sorted_population[:100])
    #         new_fitness = new_fitness.at[indices].set(sorted_fitness[:100])
    #
    #     best_index = jnp.argmin(new_fitness)
    #     new_count = state.count + 1
    #
    #     '''每200代重新排序'''
    #     if new_count > 0 and new_count % 200 == 0:
    #         combined_population = jnp.concatenate([state.population, state.trial_vectors], axis=0)
    #
    #         combined_fitness = jnp.concatenate([state.fitness, trial_fitness], axis=0)
    #
    #         # 根据适应度值排序候选种群
    #         sorted_indices = jnp.argsort(combined_fitness)
    #         sorted_population = combined_population[sorted_indices]
    #         sorted_fitness = combined_fitness[sorted_indices]
    #
    #         # 选择适应度最小的pop_size个个体
    #         new_population = sorted_population[:self.pop_size]
    #         new_fitness = sorted_fitness[:self.pop_size]
    #
    #     new_state = state.update(
    #         population=new_population,
    #         fitness=new_fitness,
    #         best_index=best_index,
    #         count=new_count,
    #     )
    #
    #     return new_state
    def tell(self, state, trial_fitness):
        new_population = jnp.zeros_like(state.population)
        new_fitness = jnp.zeros_like(state.fitness)

        for indices in state.splits:
            pop_part = state.population[indices]
            trial_part = state.trial_vectors[indices]
            fit_part = state.fitness[indices]
            trial_fit_part = trial_fitness[indices]

            combined_population = jnp.concatenate([pop_part, trial_part], axis=0)
            combined_fitness = jnp.concatenate([fit_part, trial_fit_part], axis=0)

            sorted_indices = jnp.argsort(combined_fitness)
            sorted_population = combined_population[sorted_indices]
            sorted_fitness = combined_fitness[sorted_indices]

            selected_indices = sorted_indices[:100]
            new_population = new_population.at[indices].set(sorted_population[:100])
            new_fitness = new_fitness.at[indices].set(sorted_fitness[:100])

        best_index = jnp.argmin(new_fitness)
        new_count = state.count + 1

        def reorder_population():
            combined_population = jnp.concatenate([state.population, state.trial_vectors], axis=0)
            combined_fitness = jnp.concatenate([state.fitness, trial_fitness], axis=0)
            sorted_indices = jnp.argsort(combined_fitness)
            sorted_population = combined_population[sorted_indices]
            sorted_fitness = combined_fitness[sorted_indices]
            new_population = sorted_population[:self.pop_size]
            new_fitness = sorted_fitness[:self.pop_size]
            return new_population, new_fitness

        new_population, new_fitness = jax.lax.cond(
            (new_count > 0) & (new_count % 200 == 0),
            reorder_population,
            lambda: (new_population, new_fitness)
        )

        new_state = state.update(
            population=new_population,
            fitness=new_fitness,
            best_index=best_index,
            count=new_count,
        )

        return new_state

    def override(self, state, key, params):
        state = state | self.setup(key)
        return state.update(params=params)
