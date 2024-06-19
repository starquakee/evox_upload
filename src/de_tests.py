import os
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.01"
import time
import matplotlib.pyplot as plt
import jax
from pyexpat import model
from tqdm import tqdm
from jax.example_libraries.optimizers import optimizer

from evox import algorithms, problems, workflows, monitors
import jax.numpy as jnp

from evox.algorithms import OpenES
from evox.monitors import StdSOMonitor
from evox.utils import TreeAndVector, rank_based_fitness

pop_size = 50000
dimension = 200
steps = 3_0000
ode_fitness_scores = []
de_fitness_scores = []
adjde_fitness_scores = []
openes_fitness_scores = []
# algorithm = algorithms.DE(
#     lb=jnp.full(shape=(dimension,), fill_value=-32),
#     ub=jnp.full(shape=(dimension,), fill_value=32),
#     pop_size=pop_size,
# )
algorithm = algorithms.EveryParaDESortTell(
    lb=jnp.full(shape=(dimension,), fill_value=-32),
    ub=jnp.full(shape=(dimension,), fill_value=32),
    pop_size=pop_size,
)
# algorithm = algorithms.VonDESortTell(
#     lb=jnp.full(shape=(dimension,), fill_value=-32),
#     ub=jnp.full(shape=(dimension,), fill_value=32),
#     pop_size=pop_size,
#     convolution_size=3,
#     is_diff_weight_random=False,
#     reverse_proportion=0.05,
# )
# algorithm = algorithms.VonDESortTellBatch(lb=jnp.full(shape=(dimension,), fill_value=-32),
#                                      ub=jnp.full(shape=(dimension,), fill_value=32),
#                                      pop_size=pop_size, )
# algorithm = algorithms.CSO(lb=jnp.full(shape=(dimension,), fill_value=-32),
#                            ub=jnp.full(shape=(dimension,), fill_value=32),
#                            pop_size=pop_size)
# algorithm = algorithms.VonDESortTellRand(
#     lb=jnp.full(shape=(dimension,), fill_value=-32),
#     ub=jnp.full(shape=(dimension,), fill_value=32),
#     pop_size=pop_size,
# )
monitor = monitors.EvalMonitor()
key = jax.random.PRNGKey(42)
problem = problems.numerical.Rosenbrock()
de_workflow = workflows.StdWorkflow(algorithm, problem, monitors=[monitor])

de_state = de_workflow.init(key)

for i in tqdm(range(steps), desc="Processing"):
    de_state = de_workflow.step(de_state)
    de_fitness_scores.append(monitor.get_best_fitness())
    if i % 200 == 0 and i > 0:
        # print(de_state.get_child_state("algorithm").fitness)  ### 打印种群当代的fitness
        print("de best fitness: ", monitor.get_best_fitness())

print("de best fitness: ", monitor.get_best_fitness())

# plt.plot(range(1, 1 + steps), de_fitness_scores, label="DE Fitness", marker="x")
#
# plt.title(
#     "DE Fitness Scores Over Steps with pop_size:"
#     + str(pop_size)
#     + " dim"
#     + str(dimension)
# )
# plt.xlabel("Step")
# plt.ylabel("Fitness Score")
# plt.legend()
# plt.grid(True)
# plt.show()
