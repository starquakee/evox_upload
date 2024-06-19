import jax.numpy as jnp
import pandas as pd
from jax import random
from tqdm import tqdm

from evox import workflows, algorithms, problems
from evox.monitors import StdMOMonitor
from src.evox.monitors import UsrMonitor2
from evox.metrics import IGD
import jax
import jax.numpy as jnp
from src.evox.problems.numerical import Optics

N = 32
lb = jnp.array([0.28182136, 0.06742916, 0.18156948, 0.25611739, 0.09106675, 0.02228855,
                -0.02817289, -0.03712761, - 0.07101222, -0.05984424, 0.05572402, 0.15531199,
                0.31151721, 0.05924405, -0.10207156, 0.21655084,
                1.74298208, 0.04520876, 0.359085, 0.21529327, 0.63155937, 0.68210329,
                0.6252486, 0.09740579, 0.7157511, 0.43154123, 0.51223919, 0.24646098,
                0.71836904, 1.21754272, 0.73992625, 0.27729453
                ]) - 0.1
lb = lb.at[16:].set(jnp.maximum(lb[16:], 0))
ub = jnp.array([0.28182136, 0.06742916, 0.18156948, 0.25611739, 0.09106675, 0.02228855,
                -0.02817289, -0.03712761, - 0.07101222, -0.05984424, 0.05572402, 0.15531199,
                0.31151721, 0.05924405, -0.10207156, 0.21655084,
                1.74298208, 0.04520876, 0.359085, 0.21529327, 0.63155937, 0.68210329,
                0.6252486, 0.09740579, 0.7157511, 0.43154123, 0.51223919, 0.24646098,
                0.71836904, 1.21754272, 0.73992625, 0.27729453
                ]) + 0.2
algorithm = algorithms.NSGA2(
    lb=jnp.full(shape=(N,), fill_value=lb),
    ub=jnp.full(shape=(N,), fill_value=ub),
    n_objs=2,
    pop_size=100,
)


def run_moea(algorithm, problem=Optics()):
    key = jax.random.PRNGKey(45)
    # monitor = StdMOMonitor(record_pf=False)
    monitor = UsrMonitor2()
    workflow = workflows.StdWorkflow(
        algorithm=algorithm,
        problem=problem,
        monitor=monitor,
    )
    state = workflow.init(key)
    # true_pf, state = problem.pf(state)

    for _ in tqdm(range(500)):
        state = workflow.step(state)



    fit = state.get_child_state("algorithm").fitness
    selected = (fit[:, 0] < 1e-2) * (fit[:, 1] < 1e-2)
    print(fit[selected])
    # print(monitor.get_last())
    pop = state.get_child_state("algorithm").population
    print(jnp.unique(pop[selected], axis=0).shape[0])
    unique_population = jnp.unique(pop[selected], axis=0)
    print(unique_population)
    print(unique_population[0])

    fig = monitor.plot(state, problem)
    fig.show()

    # df = pd.DataFrame(unique_population)
    #
    # # 将 DataFrame 保存为 CSV 文件
    # df.to_csv('matrix.csv', index=False, header=False)


run_moea(algorithm)


