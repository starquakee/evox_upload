from functools import partial
import jax
import jax.numpy as jnp
from jax import vmap

from evox import Algorithm, State, jit_class

import numpy as np



def adjacent_indices_array_wrapped_exclude_center(n, size=3):
    # if size % 2 == 0:
    #     raise ValueError("Size must be an odd number.")
    #
    # # 计算可能的矩阵的行数和列数
    # side_length = int(np.ceil(np.sqrt(n)))
    # if side_length * (side_length - 1) >= n:
    #     rows, cols = side_length, side_length - 1
    # else:
    #     rows, cols = side_length, side_length
    #
    # # 创建一个填充有序号的矩阵，超出部分用wrap方式填充，扩大矩阵以处理边缘情况
    # indices = np.arange(rows * cols) % n
    # indices_padded = np.pad(indices.reshape((rows, cols)),
    #                         ((size // 2, size // 2), (size // 2, size // 2)),
    #                         'wrap')
    #
    # # 初始化结果列表，考虑到要排除中心元素，所以总数是size*size-1
    # result_list = []
    #
    # # 遍历每个元素，收集其周围size*size-1个元素的索引
    # for row in range(rows):
    #     for col in range(cols):
    #         window_indices = indices_padded[row:row + size, col:col + size].flatten()
    #         # 排除中心元素
    #         center_index = size * size // 2
    #         window_indices = np.concatenate([window_indices[:center_index], window_indices[center_index + 1:]])
    #         result_list.append(window_indices)
    #
    # # 将列表转换为数组，只保留前n个元素的索引（对于超出原始矩阵大小的部分不考虑）
    # result_array = np.array(result_list)[:n]

    # return result_array

    # 计算可能的矩阵的行数和列数
    side_length = int(np.ceil(np.sqrt(n)))
    if side_length * (side_length - 1) >= n:
        rows, cols = side_length, side_length - 1
    else:
        rows, cols = side_length, side_length

    # 创建一个填充有序号的矩阵，超出部分用wrap方式填充，扩大矩阵以处理边缘情况
    indices = np.arange(rows * cols) % n
    indices_padded = np.pad(indices.reshape((rows, cols)),
                            ((size // 2, size // 2), (size // 2, size // 2)),
                            'wrap')

    # 初始化结果列表，考虑到要排除中心元素，所以总数是size*size-1
    result_list = []

    # 遍历每个元素，收集其周围size*size-1个元素的索引
    for row in range(rows):
        for col in range(cols):
            window_indices = indices_padded[row:row + size, col:col + size].flatten()
            # 排除中心元素
            center_index = size * size // 2
            window_indices = np.concatenate([window_indices[:center_index], window_indices[center_index + 1:]])
            result_list.append(window_indices)

    # 将列表转换为数组，只保留前n个元素的索引（对于超出原始矩阵大小的部分不考虑）
    result_array = np.array(result_list)[:n]

    # 调整每行的元素顺序，如果前面的数大于后面的数，交换顺序
    for row in result_array:
        half_size = len(row) // 2
        for i in range(half_size):
            if row[i] > row[-1 - i]:
                row[i], row[-1 - i] = row[-1 - i], row[i]  # 差减好 88
    return result_array
    # reversed_result_array = result_array[:, ::-1]  # 好减差 14

    # for i in range(side_length):
    #     for j in range((size * size - 1) // 2):  # 只需要遍历到中间即可
    #         if np.random.rand() < reverse_probability:  # 使用随机数和给定的概率比较
    #             # 交换第j个和倒数第j个元素
    #             reversed_result_array[i][j], reversed_result_array[i][-j - 1] = reversed_result_array[i][-j - 1], \
    #             reversed_result_array[i][j]
    #
    # return reversed_result_array


@jit_class
class VonDESortTell(Algorithm):
    def __init__(
            self,
            lb,
            ub,
            pop_size,
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
        assert base_vector in [
            "rand",
            "best",
        ]
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
        self.adjlist = None
        self.rows = None
        self.cols = None
        self.reversed_masks = None

    def setup(self, key):
        state_key, init_key = jax.random.split(key)
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
        # self.adjlist = adjacent_indices_array_wrapped_exclude_center(self.pop_size, size=self.convolution_size)
        adjlist = jnp.array(adjacent_indices_array_wrapped_exclude_center(self.pop_size,
                                                                size=self.convolution_size))
        side_length = int(np.ceil(np.sqrt(self.pop_size)))
        if side_length * (side_length - 1) >= self.pop_size:
            self.rows, self.cols = side_length, side_length - 1
        else:
            self.rows, self.cols = side_length, side_length
        # print(self.adjlist)

        # total_elements = self.rows * self.cols
        # # 计算1的数量
        # num_ones = int(self.reverse_proportion * total_elements)
        #
        # # 创建一个全0矩阵
        # reversed_masks = jnp.zeros(total_elements, dtype=int)
        # # 将前num_ones个元素设置为1
        # reversed_masks = reversed_masks.at[:num_ones].set(1)
        # # 打乱矩阵元素顺序
        # self.reversed_masks = jax.random.permutation(key, reversed_masks)
        # matrix = np.zeros((100, 100))
        #
        # # 设置四个角的10x10区域的值为1
        # # 左上角
        # matrix[0:10, 0:10] = 1
        # # 右上角
        # matrix[0:10, 90:100] = 1
        # # 左下角
        # matrix[90:100, 0:10] = 1
        # # 右下角
        # matrix[90:100, 90:100] = 1
        # # 将100x100的二维数组转换为10000的一维数组
        # self.reversed_masks = matrix.flatten().astype(int)

        return State(
            population=population,
            fitness=fitness,
            best_index=best_index,
            start_index=start_index,
            key=state_key,
            adjlist=adjlist,
            population_key=init_key,
            trial_vectors=jnp.empty((self.pop_size, self.dim)),
        )

    def ask(self, state):
        key, R_key = jax.random.split(state.key, 2)

        indices = jnp.arange(self.pop_size)

        neighbor_choices = state.adjlist

        R = jax.random.choice(R_key, self.dim, shape=(self.pop_size,))
        masks_init = (
                jax.random.uniform(R_key, shape=(self.pop_size, self.dim))
                < self.cross_probability
        )
        tile_arange = jnp.tile(jnp.arange(self.dim), (self.pop_size, 1))
        tile_R = jnp.tile(R[:, jnp.newaxis], (1, self.dim))
        masks = jnp.where(tile_arange == tile_R, True, masks_init)

        total_elements = state.adjlist.size
        # 计算1的数量
        num_ones = int(self.reverse_proportion * total_elements)

        # 创建一个全0矩阵
        reversed_masks = jnp.zeros(total_elements, dtype=int)
        # 将前num_ones个元素设置为1
        reversed_masks = reversed_masks.at[:num_ones].set(1)
        # 打乱矩阵元素顺序
        self.reversed_masks = jax.random.permutation(key, reversed_masks)
        trial_vectors = vmap(
            partial(
                self._ask_one, population=state.population, best_index=state.best_index, key=key
            )
        )(indices, R, neighbor_choiced=neighbor_choices, mask=masks, reversed_mask=self.reversed_masks[:self.pop_size])

        return trial_vectors, state.update(trial_vectors=trial_vectors, key=key)

    def _ask_one(self, index, R, population, best_index, neighbor_choiced, mask, reversed_mask, key):
        base_vector = population[index, :]


        index_mod = self.convolution_size ** 2 - 1
        differential_vector = 0

        # Use random weights if specified, otherwise use a constant weight
        differential_weights = jax.random.uniform(key, (len(self.neighbor_index),),
                                                  minval=self.differential_weight_lb * self.is_diff_weight_random + self.differential_weight * (
                                                          1 - self.is_diff_weight_random),
                                                  maxval=self.differential_weight_ub * self.is_diff_weight_random + self.differential_weight * (
                                                          1 - self.is_diff_weight_random))

        # Compute differential vector using possibly random weights
        for loop_index, i in enumerate(self.neighbor_index):
            first_index = (1 - reversed_mask) * (index_mod - i - 1) + reversed_mask * i
            second_index = (1 - reversed_mask) * i + reversed_mask * (index_mod - i - 1)

            differential_vector += differential_weights[loop_index] * (
                    population[neighbor_choiced[first_index], :] - population[neighbor_choiced[second_index], :])

        mutation_vector = base_vector + differential_vector

        mutation_vector = jnp.clip(mutation_vector, self.lb, self.ub)

        # 创建试验向量
        trial_vector = jnp.where(mask, mutation_vector, population[index])

        return trial_vector

    def tell(self, state, trial_fitness):
        combined_population = jnp.concatenate([state.population, state.trial_vectors], axis=0)

        combined_fitness = jnp.concatenate([state.fitness, trial_fitness], axis=0)

        # 根据适应度值排序候选种群
        sorted_indices = jnp.argsort(combined_fitness)
        sorted_population = combined_population[sorted_indices]
        sorted_fitness = combined_fitness[sorted_indices]

        # 选择适应度最小的pop_size个个体
        new_population = sorted_population[:self.pop_size]
        new_fitness = sorted_fitness[:self.pop_size]

        # 找到新种群中适应度最好的个体索引
        best_index = jnp.argmin(new_fitness)

        # 更新状态
        new_state = state.update(
            population=new_population,
            fitness=new_fitness,
            best_index=best_index,
        )
        return new_state


    def override(self, state, key, params):
        # Override the algorithm parameters with new values.
        state = state | VonDESortTell.setup(self, key)
        return state.update(params=params)
