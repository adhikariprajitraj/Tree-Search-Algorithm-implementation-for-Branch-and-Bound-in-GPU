import random
import math
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, List
from .base import Problem

@dataclass
class KnapsackInstance:
    weights: np.ndarray  # shape (n,)
    values: np.ndarray   # shape (n,)
    capacity: int

class Knapsack(Problem):
    def generate_instance(self, n: int,
                          weight_range: Tuple[int, int] = (5, 50),
                          value_range: Tuple[int, int] = (10, 100),
                          capacity_factor: float = 0.5,
                          seed: Optional[int] = None) -> KnapsackInstance:
        """Generate a random 0-1 knapsack instance."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        weights = np.random.randint(weight_range[0], weight_range[1] + 1, size=n)
        values = np.random.randint(value_range[0], value_range[1] + 1, size=n)
        capacity = int(capacity_factor * weights.sum())
        return KnapsackInstance(weights=weights.astype(int),
                                values=values.astype(int),
                                capacity=capacity)

def fractional_knapsack_bound(weights: np.ndarray,
                              values: np.ndarray,
                              capacity: int,
                              taken_mask: int,
                              fixed_index: int) -> float:
    """
    Compute an upper bound by taking remaining items greedily (fractional allowed).
    Items are assumed to be pre-sorted by value/weight ratio.
    """
    n = len(weights)
    remaining_cap = capacity
    value = 0.0

    # Apply taken_mask to accumulate current weight/value
    for i in range(n):
        if taken_mask & (1 << i):
            remaining_cap -= weights[i]
            value += values[i]
    if remaining_cap < 0:
        return -math.inf

    # Fractional fill from fixed_index onward
    for i in range(fixed_index, n):
        if remaining_cap <= 0:
            break
        w = weights[i]
        v = values[i]
        if w <= remaining_cap:
            remaining_cap -= w
            value += v
        else:
            value += v * (remaining_cap / w)
            remaining_cap = 0
            break
    return value
