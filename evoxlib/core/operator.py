from .module import *
from .state import State

class Operator(Module):
    def __call__(self, state: State, pop: chex.Array) -> chex.Array:
        pass
