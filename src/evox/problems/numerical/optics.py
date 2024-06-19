import jax.numpy as jnp
import jax
from evox import State
from evox import Problem, jit_class
def _f1_func(x):
    n2 = 1.0
    N = 18
    l1 = 1e6
    efl = 1.0
    n_values = jnp.array([1.49821927, 1.0, 1.67773039, 1.0, 1.54560483, 1.0,
                          1.81361475, 1.0, 1.54560483, 1.0, 1.56999094, 1.0,
                          1.54560483, 1.0, 1.53656964, 1.0, 1.51787442, 1.0])
    c_values = jnp.concatenate((x[:16], jnp.array([0.0, 0.0])))
    t_values = jnp.concatenate((x[16:], jnp.array([0.21, 0.828])))
    for i in range(N):
        n1 = n2
        n2 = n_values[i]  # n_i
        l2 = n2 / ((n2 - n1) * c_values[i] - n1 / l1)
        l1 = t_values[i] - l2
        efl *= l2 / l1

    efl = -efl * l1
    return jnp.abs(efl - 8.46)
def f1_func(X):
    return jax.vmap(_f1_func)(X)

def _f2_func(x):
    t_values = jnp.concatenate((x[16:], jnp.array([0.21, 0.828])))
    return jnp.abs(jnp.sum(t_values) - 10.5)
def f2_func(X):
    return jax.vmap(_f2_func)(X)
@jit_class
class Optics(Problem):

    def __init__(self):
        self.n_values = jnp.array([1.49821927, 1.0, 1.67773039, 1.0, 1.54560483, 1.0,
                                   1.81361475, 1.0, 1.54560483, 1.0, 1.56999094, 1.0,
                                   1.54560483, 1.0, 1.53656964, 1.0, 1.51787442, 1.0])
        self.N = 18
        self.l1 = 1e6
        self.efl = 1.0
    def setup(self, key):
        return State(key=key)
    def evaluate(self, state, X):
        f1 = f1_func(X)
        f2 = f2_func(X)
        f = jnp.column_stack((f1, f2))
        return f, state
