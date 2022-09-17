import types
from typing import Tuple

import chex

from .module import *
from .state import State


class Algorithm(Module):
    """Base class for all algorithms

    """

    def ask(self, state: State) -> Tuple[State, chex.Array]:
        """Ask the algorithm

        Ask the algorithm for points to explore

        Parameters
        ----------
        state : dict
            The state of this algorithm.

        Returns
        -------
        dict : dict
            The new state of the algorithm.
        """
        pass

    def tell(self, state: State, fitness: chex.Array) -> State:
        """Tell the algorithm more information

        Tell the algorithm about the points it chose and their corresponding fitness

        Parameters
        ----------
        state : dict
            The state of this algorithm
        X : ndarray
            The points given by ``ask``
        F : ndarray
            The fitness

        Returns
        -------
        dict
            The new state of the algorithm
        """
        pass
