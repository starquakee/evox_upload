import warnings
from typing import Any

import jax
import jax.numpy as jnp

from evox import Monitor
import plotly
import plotly.express as px
import plotly.graph_objects as go
def plot_obj_space_2d(state, problem, fitness_history, sort_points=False, **kwargs):
    try:
        import plotly
        import plotly.express as px
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("The plot function requires plotly to be installed.")
    all_fitness = jnp.concatenate(fitness_history, axis=0)
    x_lb = jnp.min(all_fitness[:, 0])
    x_ub = jnp.max(all_fitness[:, 0])
    x_range = x_ub - x_lb
    x_lb = x_lb - 0.05 * x_range
    x_ub = x_ub + 0.05 * x_range
    y_lb = jnp.min(all_fitness[:, 1])
    y_ub = jnp.max(all_fitness[:, 1])
    y_range = y_ub - y_lb
    y_lb = y_lb - 0.05 * y_range
    y_ub = y_ub + 0.05 * y_range

    frames = []
    steps = []
    pf_fitness = None
    # pf_scatter = go.Scatter(
    #     x=problem_pf[:, 0],
    #     y=problem_pf[:, 1],
    #     mode="markers",
    #     marker={"color": "#FFA15A", "size": 2},
    #     name="Pareto Front",
    # )
    for i, fit in enumerate(fitness_history):
        # it will make the animation look nicer
        if sort_points:
            indices = jnp.lexsort(fit.T)
            fit = fit[indices]
        scatter = go.Scatter(
            x=fit[:, 0],
            y=fit[:, 1],
            mode="markers",
            marker={"color": "#636EFA"},
            name="Population",
        )
        frames.append(go.Frame(data=[scatter], name=str(i)))

        step = {
            "label": i,
            "method": "animate",
            "args": [
                [str(i)],
                {
                    "frame": {"duration": 200, "redraw": False},
                    "mode": "immediate",
                    "transition": {"duration": 200},
                },
            ],
        }
        steps.append(step)

    sliders = [
        {
            "currentvalue": {"prefix": "Generation: "},
            "pad": {"b": 1, "t": 10},
            "len": 0.8,
            "x": 0.2,
            "y": 0,
            "yanchor": "top",
            "xanchor": "left",
            "steps": steps,
        }
    ]
    fig = go.Figure(
        data=frames[0].data,
        layout=go.Layout(
            legend={
                "x": 1,
                "y": 1,
                "xanchor": "auto",
                "xanchor": "auto",
            },
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
            sliders=sliders,
            xaxis={"range": [x_lb, x_ub], "autorange": False},
            yaxis={"range": [y_lb, y_ub], "autorange": False},
            updatemenus=[
                {
                    "type": "buttons",
                    "buttons": [
                        {
                            "args": [
                                None,
                                {
                                    "frame": {"duration": 200, "redraw": False},
                                    "fromcurrent": True,
                                    "transition": {
                                        "duration": 200,
                                        "easing": "linear",
                                    },
                                },
                            ],
                            "label": "Play",
                            "method": "animate",
                        },
                        {
                            "args": [
                                [None],
                                {
                                    "frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate",
                                },
                            ],
                            "label": "Pause",
                            "method": "animate",
                        },
                    ],
                    "x": 0.2,
                    "xanchor": "right",
                    "y": 0,
                    "yanchor": "top",
                    "direction": "left",
                    "pad": {"r": 10, "t": 30},
                },
            ],
            **kwargs,
        ),
        frames=frames,
    )

    return fig



class UsrMonitor2(Monitor):
    """Population monitor,
    used to monitor the population inside the genetic algorithm.

    Parameters
    ----------
    population_name
        The name of the population in the state.
        Default to "population".
    fitness_name
        The name of the fitness in the state.
        Default to "fitness".
    to_host
        Whether to move the population and fitness to host memory (ram).
        Doing so can reduce memory usage on device (vram),
        but also introduces overhead of data transfer.
        Default to False.
    fitness_only
        Whether to only record the fitness.
        Setting it to True will disable the recording of population (decision space),
        only the fitness (objective space) will be recorded.
        This can reduce memory usage if you only care about the fitness.
        Default to False.
    """

    def __init__(
        self,
        population_name="population",
        fitness_name="fitness",
        to_host=False,
        fitness_only=False,
    ):
        super().__init__()
        self.population_name = population_name
        self.fitness_name = fitness_name
        self.to_host = to_host
        if to_host:
            self.host = jax.devices("cpu")[0]
        self.population_history = []
        self.fitness_history = []
        self.fitness_only = fitness_only

    def hooks(self):
        return ["post_step"]

    def post_step(self, state):
        if not self.fitness_only:
            population = getattr(
                state.get_child_state("algorithm"), self.population_name
            )
            if self.to_host:
                population = jax.device_put(population, self.host)
            self.population_history.append(population)

        fitness = getattr(state.get_child_state("algorithm"), self.fitness_name)
        if self.to_host:
            fitness = jax.device_put(fitness, self.host)
        self.fitness_history.append(fitness)

    def get_population_history(self):
        return self.population_history

    def get_fitness_history(self):
        return self.fitness_history

    def plot(self, state, problem, **kwargs):
        if not self.fitness_history:
            warnings.warn("No fitness history recorded, return None")
            return

        if self.fitness_history[0].ndim == 1:
            n_objs = 1
        else:
            n_objs = self.fitness_history[0].shape[1]

        if n_objs == 2:
            return plot_obj_space_2d(state, problem, self.fitness_history, **kwargs)
        else:
            warnings.warn("Not supported yet.")
