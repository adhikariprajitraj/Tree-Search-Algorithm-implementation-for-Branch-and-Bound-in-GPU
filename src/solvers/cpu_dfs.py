import time
import numpy as np
from dataclasses import dataclass
from typing import List, Any
from .base import Solver
from ..problems.knapsack import KnapsackInstance, fractional_knapsack_bound

@dataclass
class BnBResult:
    best_value: int
    best_items: List[int]
    nodes_explored: int
    time_sec: float
    optimal: bool

class CpuDfsSolver(Solver):
    def solve(self, instance: KnapsackInstance, **kwargs) -> BnBResult:
        """
        Classic depth-first Branch & Bound for 0-1 knapsack on CPU.
        """
        start = time.perf_counter()

        # Sort items by value/weight ratio
        ratio = instance.values / instance.weights
        order = np.argsort(-ratio)
        w = instance.weights[order]
        v = instance.values[order]
        cap = instance.capacity
        n = len(w)

        best_value = 0
        best_mask = 0
        nodes_explored = 0

        def dfs(idx: int, curr_weight: int, curr_value: int, mask: int):
            nonlocal best_value, best_mask, nodes_explored
            nodes_explored += 1

            # If overweight, prune
            if curr_weight > cap:
                return

            # Leaf: update incumbent
            if idx == n:
                if curr_value > best_value:
                    best_value = curr_value
                    best_mask = mask
                return

            # Upper bound check
            ub = fractional_knapsack_bound(w, v, cap, mask, idx)
            if ub <= best_value:
                return

            # Branch: try taking item idx
            dfs(idx + 1,
                curr_weight + int(w[idx]),
                curr_value + int(v[idx]),
                mask | (1 << idx))

            # Branch: skip item idx
            dfs(idx + 1, curr_weight, curr_value, mask)

        dfs(0, 0, 0, 0)

        # Map mask back to original indices
        taken_sorted = [i for i in range(n) if best_mask & (1 << i)]
        taken_orig = [int(order[i]) for i in taken_sorted]
        taken_orig.sort()

        elapsed = time.perf_counter() - start
        return BnBResult(best_value=int(best_value),
                         best_items=taken_orig,
                         nodes_explored=nodes_explored,
                         time_sec=elapsed,
                         optimal=True)
