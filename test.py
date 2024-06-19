import jax.numpy as jnp
import numpy as np
import jax

from src.evox.algorithms.so.de_variants.everyparavondesorttell import adjacent_indices_array_wrapped_exclude_center

state_key, init_key = jax.random.split(jax.random.PRNGKey(42), 2)
population = jax.random.uniform(init_key, shape=(10, 4))
print(population)
print(population[0])
print(type(population[0]))
print()

differential_weight_range = (0, 1)
cross_probability_range = (0, 1)
num_diff_vects_choices = [2, 4]
cross_strategy_choices = [0, 1, 2]

# 使用 JAX 生成 params_list
params_array = jax.random.uniform(jax.random.PRNGKey(0), shape=(10, 2))
num_diff_vects_array = jax.random.choice(jax.random.PRNGKey(1), jnp.array(num_diff_vects_choices), shape=(10,))
cross_strategy_array = jax.random.choice(jax.random.PRNGKey(2), jnp.array(cross_strategy_choices), shape=(10,))

params_list = jnp.column_stack((params_array, num_diff_vects_array, cross_strategy_array))
print(type(params_list[0]))       # <class 'jax.numpy.DeviceArray'>
print(params_list)

# 验证 params_list 的内容和类型
print(type(params_list[0]))    # <class 'jax.numpy.DeviceArray'>
print(type(params_list[0, 0])) # <class 'jax.numpy.DeviceArray'>
print(type(params_list[0][2])) # <class 'jax.numpy.DeviceArray'>
print(params_list[params_list[0][2].astype(int)])

params1_array = jax.random.uniform(init_key, shape=(6, 2))
params2_array = jax.random.uniform(init_key, shape=(6, 2))
params3_array = jax.random.uniform(init_key, shape=(6, 2))
params_list = jnp.column_stack((params1_array, params2_array, params3_array))
print()
print(params_list[:, :2])
print()
lambda1 = jax.random.uniform(init_key, shape=(3, 2))
print(lambda1)

print()
import random
n = 4  # 例如 n 为 4

# 初始化差值和
difference_sum = 0

# 随机选择 n 个数
selected_indices = list.sort(random.sample(list(range(3 ** 2 - 1)), n))
# list.sort(selected_indices)
print(selected_indices)
print(list(range(3 ** 2-1)))

def generate_nested_list(size):
    result = []
    for _ in range(size):
        n = random.choice([2, 4, 6, 8])
        nested_list = random.sample(range(3 ** 2 - 1), n)
        nested_list.sort()
        result.append(nested_list)
    return result

# Generate the list of lists with 10,000 elements
nested_list = generate_nested_list(1000)
print(nested_list[:10])  # Display the first 10 elements for verification
for loop_index, i in enumerate(nested_list[0]):
    print(f"Index {loop_index}: {i}")
for loop_index, i in enumerate([0, 1, 2, 3, 4, 5, 6, 7]):
    first_index = (8 - i - 1)
    second_index = i
    print(first_index, second_index)

result = []
len = []
for _ in range(10):
    n = random.choice([2, 4, 6, 8])
    len.append(n)
    nested_list = random.sample(range(3 ** 2 - 1), n)
    nested_list.sort()
    result.append(nested_list)

# 将列表转换为NumPy数组

print(jnp.array(len))
print(jnp.array(adjacent_indices_array_wrapped_exclude_center(10, size=3)))
import random

# 生成列表
result = []
lengths = []

for _ in range(10):
    n = random.choice([2, 4, 6, 8])
    lengths.append(n)
    nested_list = random.sample(range(3 ** 2 - 1), n)
    nested_list.sort()
    if n < 8:
        fill_count = 8 - n
        fill_index = n // 2
        nested_list = nested_list[:fill_index] + [0] * fill_count + nested_list[fill_index:]
    result.append(nested_list)

# 输出结果
for lst in result:
    print(lst)


