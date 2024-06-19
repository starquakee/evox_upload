# --------------------------------------------------------------------------------------
# 1. This code implements algorithms described in the following papers:
#
# Title: A Competitive Swarm Optimizer for Large Scale Optimization
# Link: https://ieeexplore.ieee.org/document/6819057
# --------------------------------------------------------------------------------------

import jax
import jax.numpy as jnp
from typing import Optional
from evox import Algorithm, State, Static, jit_class, dataclass
from dataclasses import field


@dataclass
class CSOState:
    population: jax.Array
    fitness: jax.Array
    velocity: jax.Array
    students: jax.Array
    key: jax.random.PRNGKey


@jit_class
@dataclass
class PCSO(Algorithm):
    lb: jax.Array
    ub: jax.Array
    pop_size: Static[int]
    phi: float = 0.0
    mean: Optional[jax.Array] = None
    stdev: Optional[jax.Array] = None
    dim: Static[int] = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "dim", self.lb.shape[0])

    def setup(self, key):
        state_key, init_key, param1_key, param2_key, param3_key = jax.random.split(key, 5)
        if self.mean is not None and self.stdev is not None:
            population = self.stdev * jax.random.normal(
                init_key, shape=(self.pop_size, self.dim)
            )
            population = jnp.clip(population, self.lb, self.ub)
        else:
            population = jax.random.uniform(init_key, shape=(self.pop_size, self.dim))
            population = population * (self.ub - self.lb) + self.lb
        velocity = jnp.zeros((self.pop_size, self.dim))
        fitness = jnp.full((self.pop_size,), jnp.inf)
        params1_array = jax.random.uniform(param1_key, shape=(self.pop_size, self.dim))
        params2_array = jax.random.uniform(param2_key, shape=(self.pop_size, self.dim))
        params3_array = jax.random.uniform(param3_key, shape=(self.pop_size, self.dim))
        params_list = jnp.column_stack((params1_array, params2_array, params3_array))
        return State(
            params=params_list,
            population=population,
            fitness=fitness,
            velocity=velocity,
            students=jnp.empty((self.pop_size // 2,), dtype=jnp.int32),
            key=state_key,
        )


    def init_ask(self, state):
        return state.population, state

    def init_tell(self, state, fitness):
        return state.update(fitness=fitness)

    def ask(self, state):
        key, pairing_key, lambda1_key, lambda2_key, lambda3_key = jax.random.split(
            state.key, num=5
        )
        randperm = jax.random.permutation(pairing_key, self.pop_size).reshape(2, -1)
        mask = state.fitness[randperm[0, :]] < state.fitness[randperm[1, :]]

        teachers = jnp.where(mask, randperm[0, :], randperm[1, :])
        students = jnp.where(mask, randperm[1, :], randperm[0, :])
        center = jnp.mean(state.population, axis=0)
        student_velocity = (
            state.params[teachers, :self.dim] * state.velocity[students]
            + state.params[teachers, self.dim:2*self.dim] * (state.population[teachers] - state.population[students])
            + self.phi * state.params[students, 2*self.dim:3*self.dim] * (center - state.population[students])
        )
        candidates = jnp.clip(
            state.population[students] + student_velocity, self.lb, self.ub
        )
        new_population = state.population.at[students].set(candidates)
        new_velocity = state.velocity.at[students].set(student_velocity)

        return (
            candidates,
            state.update(
                population=new_population,
                velocity=new_velocity,
                students=students,
                key=key,
            ),
        )

    def tell(self, state, fitness):
        fitness = state.fitness.at[state.students].set(fitness)
        return state.update(fitness=fitness)





# import jax
# import jax.numpy as jnp
# from jax import lax
# from typing import Optional
# from evox import Algorithm, State, Static, jit_class, dataclass
# from dataclasses import field
#
#
# @dataclass
# class PCSOState:
#     population: jax.Array
#     fitness: jax.Array
#     velocity: jax.Array
#     students: jax.Array
#     key: jax.random.PRNGKey
#
#
# @jit_class
# @dataclass
# class PCSO(Algorithm):
#     lb: jax.Array
#     ub: jax.Array
#     pop_size: Static[int]
#     phi: float = 0.0
#     mean: Optional[jax.Array] = None
#     stdev: Optional[jax.Array] = None
#     selection_strategy: jax.Array = None
#     competition_strategy: Static[int] = 0
#     velocity_update_strategy: Static[int] = 1
#     crossover_prob: float = 0.5
#     dim: Static[int] = field(init=False)
#
#     def __post_init__(self):
#         object.__setattr__(self, "dim", self.lb.shape[0])
#         selection_strategy = jnp.concatenate((jnp.zeros(self.pop_size // 2, dtype=jnp.int32),
#                                               0*jnp.ones(self.pop_size // 16, dtype=jnp.int32)))
#
#         # 打乱数组顺序
#         selection_strategy = jax.random.permutation(jax.random.PRNGKey(42), selection_strategy)
#         if self.selection_strategy is None:
#             # 默认策略是全部使用正向选择
#
#             object.__setattr__(self, "selection_strategy", selection_strategy)
#
#     def setup(self, key):
#         state_key, init_key, param1_key, param2_key, param3_key, selection_key = jax.random.split(key, 6)
#         if self.mean is not None and self.stdev is not None:
#             population = self.stdev * jax.random.normal(
#                 init_key, shape=(self.pop_size, self.dim)
#             )
#             population = jnp.clip(population, self.lb, self.ub)
#         else:
#             population = jax.random.uniform(init_key, shape=(self.pop_size, self.dim))
#             population = population * (self.ub - self.lb) + self.lb
#         velocity = jnp.zeros((self.pop_size, self.dim))
#         fitness = jnp.full((self.pop_size,), jnp.inf)
#         params1_array = jax.random.uniform(param1_key, shape=(self.pop_size, self.dim))
#         params2_array = jax.random.uniform(param2_key, shape=(self.pop_size, self.dim))
#         params3_array = jax.random.uniform(param3_key, shape=(self.pop_size, self.dim))
#         selection_strategy = jax.random.choice(selection_key, jnp.array([0, 1]), shape=(self.pop_size,))
#         params_list = jnp.column_stack((params1_array, params2_array, params3_array, selection_strategy))
#
#
#
#         return State(
#             selection_strategy=jnp.zeros(self.pop_size // 2, dtype=jnp.int32),
#             params=params_list,
#             population=population,
#             fitness=fitness,
#             velocity=velocity,
#             students=jnp.empty((self.pop_size // 2,), dtype=jnp.int32),
#             key=state_key,
#         )
#
#     def init_ask(self, state):
#         return state.population, state
#
#     def init_tell(self, state, fitness):
#         return state.update(fitness=fitness)
#
#     def ask(self, state):
#         key, pairing_key, update_key1, update_key2 = jax.random.split(state.key, num=4)
#         randperm = jax.random.permutation(pairing_key, self.pop_size).reshape(2, -1)
#
#         def select_pair(i):
#             return lax.cond(
#                 self.selection_strategy[i] == 0,
#                 lambda _: state.fitness[randperm[0, i]] < state.fitness[randperm[1, i]],
#                 lambda _: state.fitness[randperm[0, i]] > state.fitness[randperm[1, i]],
#                 operand=None
#             )
#
#         mask = jax.vmap(select_pair)(jnp.arange(self.pop_size // 2))
#
#         teachers = jnp.where(mask, randperm[0, :], randperm[1, :])
#         students = jnp.where(mask, randperm[1, :], randperm[0, :])
#
#         center = jnp.mean(state.population, axis=0)
#
#         if self.velocity_update_strategy == 0:
#             # 标准速度更新
#             student_velocity = (
#                     jax.random.uniform(update_key1, shape=(self.pop_size // 2, self.dim)) * state.velocity[students]
#                     + jax.random.uniform(update_key1, shape=(self.pop_size // 2, self.dim)) * (
#                                 state.population[teachers] - state.population[students])
#                     + self.phi * jax.random.uniform(update_key1, shape=(self.pop_size // 2, self.dim)) * (
#                                 center - state.population[students])
#             )
#         else:
#             # 其他速度更新策略，例如添加噪声
#             noise = jax.random.normal(update_key1, shape=(self.pop_size // 2, self.dim))
#             student_velocity = (
#                     jax.random.uniform(update_key1, shape=(self.pop_size // 2, self.dim)) * state.velocity[students]
#                     + jax.random.uniform(update_key1, shape=(self.pop_size // 2, self.dim)) * (
#                                 state.population[teachers] - state.population[students])
#                     + self.phi * jax.random.uniform(update_key1, shape=(self.pop_size // 2, self.dim)) * (
#                                 center - state.population[students])
#                     + noise
#             )
#
#         candidates = jnp.clip(
#             state.population[students] + student_velocity, self.lb, self.ub
#         )
#
#         # # 交叉操作
#         # crossover_mask = jax.random.uniform(key, shape=(self.pop_size // 2, self.dim)) < self.crossover_prob
#         # new_population = jnp.where(crossover_mask, candidates, state.population[students])
#         # new_velocity = jnp.where(crossover_mask, student_velocity, state.velocity[students])
#         #
#         # new_population = state.population.at[students].set(new_population)
#         # new_velocity = state.velocity.at[students].set(new_velocity)  不要cross
#         new_population = state.population.at[students].set(candidates)
#         new_velocity = state.velocity.at[students].set(student_velocity)
#
#         return (
#             candidates,
#             state.update(
#                 population=new_population,
#                 velocity=new_velocity,
#                 students=students,
#                 key=key,
#             ),
#         )
#
#     def tell(self, state, fitness):
#         fitness = state.fitness.at[state.students].set(fitness)
#         return state.update(fitness=fitness)


# 给定字符串列表 strs ，返回其中 最长的特殊序列 的长度。如果最长特殊序列不存在，返回 -1 。
#
# 特殊序列 定义如下：该序列为某字符串 独有的子序列（即不能是其他字符串的子序列）。
#
#  s 的 子序列可以通过删去字符串 s 中的某些字符实现。
#
# 例如，"abc" 是 "aebdc" 的子序列，因为您可以删除"aebdc"中的下划线字符来得到 "abc" 。"aebdc"的子序列还包括"aebdc"、 "aeb" 和 "" (空字符串)。