import os
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.01"
from evox import algorithms, workflows
from evox.monitors import StdSOMonitor
from evox.problems.numerical.cec2022_so import CEC2022TestSuit
import jax
import jax.numpy as jnp
import time
import tqdm

from src.evox.algorithms import ParaVonDESortTell, EveryParaDESortTell, EveryParaVonDESortTell

func_list = jnp.arange(12) + 1
# func_list = [4, 8]
D = 10
steps = 9999999
pop_size = 10000
# key_start = 42
runs = 16  # number of independent runs. should be an even number
max_time = 60  ## 60
num_samples = 100  # history sample num
key = jax.random.PRNGKey(42)
optimizer_name = "EveryParaDESortTell"
# optimizer_name = "voncsosorttell10000"

# optimizer = algorithms.VonCSOSortTell(
#     lb=jnp.full(shape=(D,), fill_value=-100),
#     ub=jnp.full(shape=(D,), fill_value=100),
#     pop_size=pop_size,
# )
optimizer = EveryParaDESortTell(lb=jnp.full(shape=(D,), fill_value=-100),
                              ub=jnp.full(shape=(D,), fill_value=100), pop_size=pop_size,
                              is_diff_weight_random=False, reverse_proportion=0.05)


# optimizer = algorithms.PSO(lb=jnp.full(shape=(D,), fill_value=-100), ub=jnp.full(shape=(D,), fill_value=100), pop_size=4000)
# optimizer = algorithms.CSO(
#     lb=lower_bound,
#     ub=upper_bound,
#     pop_size=pop_size,
# )
# optimizer =algorithms.de_variants.DE(lb=jnp.full(shape=(D,), fill_value=-100), ub=jnp.full(shape=(D,), fill_value=100), pop_size=pop_size, base_vector="rand", num_difference_vectors=1, )
# optimizer =algorithms.de_variants.SaDE(lb=jnp.full(shape=(D,), fill_value=-100), ub=jnp.full(shape=(D,), fill_value=100), pop_size=pop_size, )
# optimizer =algorithms.de_variants.JaDE(lb=jnp.full(shape=(D,), fill_value=-100), ub=jnp.full(shape=(D,), fill_value=100), pop_size=pop_size, )
# optimizer =algorithms.de_variants.CoDE(lb=jnp.full(shape=(D,), fill_value=-100), ub=jnp.full(shape=(D,), fill_value=100), pop_size=pop_size, )
# optimizer =algorithms.de_variants.SHADE(lb=jnp.full(shape=(D,), fill_value=-100), ub=jnp.full(shape=(D,), fill_value=100), pop_size=pop_size, )

def sample_history(num_samples, fit_history):
    fit_history = jnp.array(fit_history)
    float_indices = jnp.linspace(0, len(fit_history) - 1, num_samples)
    indices = jnp.floor(float_indices).astype(int)
    sampled_arr = fit_history[indices]
    sampled_arr = sampled_arr.tolist()
    return sampled_arr


for txt_name in ['src/evox/result_' + optimizer_name + '.txt',
                 'src/evox/result_' + optimizer_name + '_history.txt']:
    with open(txt_name, 'w') as f:
        f.write(f'Problem_Dim: {D}, ')
        f.write(f'Time: {max_time}, ')
        f.write(f'Optimizer: {type(optimizer).__name__}, ')
        f.write(f'Popsize: {pop_size}, ')
        f.write(f'Iters: {steps}\n\n')

"""Run the algorithm"""
time_all = jnp.array([])
time_start = time.time()

for func_num in func_list:
    problem = CEC2022TestSuit.create(int(func_num))
    print(type(problem).__name__)

    with open('src/evox/result_' + optimizer_name + '.txt', 'a') as f:
        f.write(f'{type(problem).__name__}  ')
    with open('src/evox/result_' + optimizer_name + '_history.txt', 'a') as f:
        f.write(f'{type(problem).__name__}  ')

    best_fit_history_all = []  # history of all runs
    best_fit_all = []  # best fit of all runs

    for run_num in range(runs):
        start_time = time.time()
        monitor = StdSOMonitor(record_fit_history=False)

        # create a pipeline
        workflow = workflows.StdWorkflow(algorithm=optimizer, problem=problem, monitor=monitor, )

        # init the pipeline
        key, _ = jax.random.split(key)
        state = workflow.init(key)

        bestfit_history = []
        # run the pipeline for 100 steps
        for i in range(steps):
            state = workflow.step(state)
            # print(monitor.get_best_fitness())
            steps_iter = i + 1

            bestfit_step = monitor.get_best_fitness().item()  # record best fitness history
            bestfit_history.append(bestfit_step)

            end_time = time.time()
            elapsed_time = end_time - start_time
            if elapsed_time >= max_time:
                break

        """Record and print"""
        bestfit_history = sample_history(num_samples, bestfit_history)
        best_fit = monitor.get_best_fitness()
        print(f"min fitness: {best_fit}")
        print(f"Steps: {steps_iter} Runs: {run_num}")
        print(f"Time: {elapsed_time} s\n")

        if run_num >= 1:
            best_fit_all.append(best_fit)
            best_fit_history_all.append(bestfit_history)

            with open('src/evox/result_' + optimizer_name + '.txt', 'a') as f:
                f.write(f'{best_fit} ')

    FEs = steps_iter * pop_size
    with open('src/evox/result_' + optimizer_name + '.txt', 'a') as f:
        f.write(f'{FEs}\n')

    # find the median run for history
    sorted_best_fit_all = sorted(best_fit_all)
    median_value = sorted_best_fit_all[len(sorted_best_fit_all) // 2]
    median_index = best_fit_all.index(median_value)

    best_fit_history_median = best_fit_history_all[median_index]

    with open('src/evox/result_' + optimizer_name + '_history.txt', 'a') as f:
        f.write(f'{best_fit_history_median}\n')
print(f"Time: {time.time() - time_start} s\n")
