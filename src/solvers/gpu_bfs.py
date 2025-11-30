import time
import math
import numpy as np
from typing import Optional, List
from dataclasses import dataclass
from .base import Solver
from ..problems.knapsack import KnapsackInstance

# Try importing torch and ipex
try:
    import torch
    HAS_TORCH = True
    try:
        import intel_extension_for_pytorch as ipex
        HAS_IPEX = True
    except ImportError:
        HAS_IPEX = False
except ImportError:
    HAS_TORCH = False
    HAS_IPEX = False

@dataclass
class BnBResult:
    best_value: int
    best_items: List[int]
    nodes_explored: int
    time_sec: float
    optimal: bool

class GpuBfsSolver(Solver):
    def solve(self, instance: KnapsackInstance,
              beam_width: Optional[int] = None,
              max_depth: Optional[int] = None,
              time_limit: Optional[float] = None,
              device: Optional[str] = None,
              use_torch: bool = True) -> BnBResult:
        
        if use_torch and HAS_TORCH:
            return self._solve_torch(instance, beam_width, max_depth, time_limit, device)
        else:
            return self._solve_numpy(instance, beam_width, max_depth, time_limit)

    def _solve_numpy(self, instance: KnapsackInstance,
                     beam_width: Optional[int],
                     max_depth: Optional[int],
                     time_limit: Optional[float]) -> BnBResult:
        """
        NumPy-based BFS solver with list-based solution tracking (no bitmasks).
        This avoids integer overflow for large N.
        """
        start = time.perf_counter()
        weights = instance.weights.copy()
        values = instance.values.copy()
        cap = instance.capacity
        n = len(weights)

        ratio = values / weights
        order = np.argsort(-ratio)
        w = weights[order]
        v = values[order]

        # Greedy initial solution
        remaining_cap = cap
        best_value = 0
        best_items_sorted = []
        for i in range(n):
            if w[i] <= remaining_cap:
                remaining_cap -= int(w[i])
                best_value += int(v[i])
                best_items_sorted.append(i)

        best_solution_value = best_value
        best_solution_items = best_items_sorted.copy()

        # Initialize BFS with empty solution
        curr_weights = np.array([0], dtype=np.int64)
        curr_values = np.array([0], dtype=np.int64)
        curr_solutions = [[]]  # List of lists tracking which items are selected
        curr_index = 0
        nodes_explored = 0

        prefix_w = np.cumsum(w)
        prefix_v = np.cumsum(v)

        def fractional_bound_batch(node_weights, node_values, next_index):
            if next_index >= n:
                return node_values.astype(float)
            rem_cap = cap - node_weights
            base_weight = prefix_w[next_index - 1] if next_index > 0 else 0.0
            base_value = prefix_v[next_index - 1] if next_index > 0 else 0.0
            target = base_weight + rem_cap
            idx = np.searchsorted(prefix_w, target, side="left")
            ub = node_values.astype(float).copy()
            for j in range(node_weights.size):
                if rem_cap[j] <= 0: continue
                if idx[j] >= n:
                    ub[j] += float(prefix_v[-1] - base_value)
                else:
                    prev_index = idx[j] - 1
                    if prev_index >= next_index:
                        ub[j] += float(prefix_v[prev_index] - base_value)
                    cap_used = 0.0
                    if prev_index >= next_index:
                        cap_used = float(prefix_w[prev_index] - base_weight)
                    remaining = float(rem_cap[j] - cap_used)
                    if remaining <= 0: continue
                    if math.isclose(remaining, float(w[idx[j]]), rel_tol=1e-9, abs_tol=1e-9):
                        ub[j] += float(v[idx[j]])
                    else:
                        item_w = float(w[idx[j]])
                        item_v = float(v[idx[j]])
                        frac = max(0.0, min(1.0, remaining / item_w))
                        ub[j] += item_v * frac
            return ub

        while curr_index < n and curr_weights.size > 0:
            if max_depth is not None and curr_index >= max_depth: break
            if time_limit is not None and (time.perf_counter() - start) > time_limit: break

            count = curr_weights.size
            # Branch: create children (skip and take)
            new_weights = np.concatenate([curr_weights, curr_weights])
            new_values = np.concatenate([curr_values, curr_values])
            new_solutions = curr_solutions + [sol + [curr_index] for sol in curr_solutions]

            # Add current item to second half
            new_weights[count:] += w[curr_index]
            new_values[count:] += v[curr_index]

            # Feasibility check
            feasible = new_weights <= cap
            new_weights = new_weights[feasible]
            new_values = new_values[feasible]
            new_solutions = [new_solutions[i] for i in range(len(new_solutions)) if feasible[i]]

            if new_weights.size == 0: break

            next_index = curr_index + 1
            if next_index < n:
                # Bounding and beam search
                ub = fractional_bound_batch(new_weights, new_values, next_index)
                keep = ub > best_solution_value
                new_weights = new_weights[keep]
                new_values = new_values[keep]
                new_solutions = [new_solutions[i] for i in range(len(new_solutions)) if keep[i]]
                ub = ub[keep]
                if new_weights.size == 0: break
                if beam_width is not None and new_weights.size > beam_width:
                    idx_sorted = np.argsort(-ub)
                    top_idx = idx_sorted[:beam_width]
                    new_weights = new_weights[top_idx]
                    new_values = new_values[top_idx]
                    new_solutions = [new_solutions[i] for i in top_idx]
            else:
                # Leaf nodes - update best solution
                for j in range(new_values.size):
                    if new_values[j] > best_solution_value:
                        best_solution_value = int(new_values[j])
                        best_solution_items = new_solutions[j].copy()
                nodes_explored += new_weights.size
                break

            curr_weights = new_weights
            curr_values = new_values
            curr_solutions = new_solutions
            curr_index = next_index
            nodes_explored += curr_weights.size

        # Map back to original indices
        taken_orig = [int(order[i]) for i in best_solution_items]
        taken_orig.sort()
        elapsed = time.perf_counter() - start
        optimal = (beam_width is None and max_depth is None and time_limit is None)
        return BnBResult(best_value=int(best_solution_value), best_items=taken_orig, nodes_explored=nodes_explored, time_sec=elapsed, optimal=optimal)

    def _solve_torch(self, instance: KnapsackInstance,
                     beam_width: Optional[int],
                     max_depth: Optional[int],
                     time_limit: Optional[float],
                     device: Optional[str]) -> BnBResult:
        """
        PyTorch-based BFS solver with list-based solution tracking (no bitmasks).
        This avoids integer overflow for large N.
        """
        if device is None:
            if HAS_IPEX and hasattr(torch, 'xpu') and torch.xpu.is_available():
                device = "xpu"
            elif torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        print(f"DEBUG: Using device '{device}' for PyTorch B&B")
        start = time.perf_counter()

        weights = torch.tensor(instance.weights, dtype=torch.float32, device=device)
        values = torch.tensor(instance.values, dtype=torch.float32, device=device)
        cap = float(instance.capacity)
        n = weights.numel()

        ratio = values / weights
        order = torch.argsort(-ratio)
        w = weights[order]
        v = values[order]

        # Greedy initial solution
        remaining_cap = cap
        best_value = 0.0
        best_items_sorted = []
        for i in range(n):
            wi = float(w[i])
            if wi <= remaining_cap:
                remaining_cap -= wi
                best_value += float(v[i])
                best_items_sorted.append(i)

        best_solution_value = best_value
        best_solution_items = best_items_sorted.copy()

        # Initialize BFS
        curr_weights = torch.zeros(1, dtype=torch.float32, device=device)
        curr_values = torch.zeros(1, dtype=torch.float32, device=device)
        curr_solutions = [[]]  # List of lists tracking which items are selected
        curr_index = 0
        nodes_explored = 0

        prefix_w = torch.cumsum(w, dim=0)
        prefix_v = torch.cumsum(v, dim=0)

        def fractional_bound_batch_torch(node_weights: torch.Tensor,
                                         node_values: torch.Tensor,
                                         next_index: int) -> torch.Tensor:
            if next_index >= n:
                return node_values
            
            rem_cap = cap - node_weights
            base_weight = float(prefix_w[next_index - 1]) if next_index > 0 else 0.0
            base_value = float(prefix_v[next_index - 1]) if next_index > 0 else 0.0

            target = base_weight + rem_cap
            idx = torch.searchsorted(prefix_w, target)
            ub = node_values.clone()
            
            mask_cap = rem_cap > 0
            mask_all = (idx >= n) & mask_cap
            if mask_all.any():
                ub[mask_all] += (prefix_v[-1] - base_value)

            mask_frac = (idx < n) & mask_cap
            if mask_frac.any():
                idx_frac = idx[mask_frac]
                rem_cap_frac = rem_cap[mask_frac]
                prev_index = idx_frac - 1
                
                val_add = torch.zeros_like(rem_cap_frac)
                cap_used = torch.zeros_like(rem_cap_frac)
                
                mask_c1 = prev_index >= next_index
                if mask_c1.any():
                    prev_idx_c1 = prev_index[mask_c1]
                    term_v = prefix_v[prev_idx_c1] - base_value
                    term_w = prefix_w[prev_idx_c1] - base_weight
                    val_add[mask_c1] = term_v
                    cap_used[mask_c1] = term_w

                remaining = rem_cap_frac - cap_used
                mask_rem = remaining > 0
                if mask_rem.any():
                    item_w = w[idx_frac[mask_rem]]
                    item_v = v[idx_frac[mask_rem]]
                    frac = torch.clamp(remaining[mask_rem] / item_w, 0.0, 1.0)
                    val_add[mask_rem] += item_v * frac
                
                ub[mask_frac] += val_add
            return ub

        while curr_index < n and curr_weights.numel() > 0:
            if max_depth is not None and curr_index >= max_depth: break
            if time_limit is not None and (time.perf_counter() - start) > time_limit: break

            count = curr_weights.numel()
            # Branch: create children (skip and take)
            new_weights = torch.cat([curr_weights, curr_weights])
            new_values = torch.cat([curr_values, curr_values])
            new_solutions = curr_solutions + [sol + [curr_index] for sol in curr_solutions]

            # Add current item to second half
            new_weights[count:] = new_weights[count:] + w[curr_index]
            new_values[count:] = new_values[count:] + v[curr_index]

            # Feasibility check
            feasible = new_weights <= cap
            new_weights = new_weights[feasible]
            new_values = new_values[feasible]
            feasible_cpu = feasible.cpu().numpy()
            new_solutions = [new_solutions[i] for i in range(len(new_solutions)) if feasible_cpu[i]]

            if new_weights.numel() == 0: break

            next_index = curr_index + 1
            if next_index < n:
                # Bounding and beam search
                ub = fractional_bound_batch_torch(new_weights, new_values, next_index)
                keep = ub > best_solution_value
                new_weights = new_weights[keep]
                new_values = new_values[keep]
                keep_cpu = keep.cpu().numpy()
                new_solutions = [new_solutions[i] for i in range(len(new_solutions)) if keep_cpu[i]]
                ub = ub[keep]
                if new_weights.numel() == 0: break
                if beam_width is not None and new_weights.numel() > beam_width:
                    ub_sorted, idx_sorted = torch.sort(ub, descending=True)
                    top_idx = idx_sorted[:beam_width]
                    new_weights = new_weights[top_idx]
                    new_values = new_values[top_idx]
                    top_idx_cpu = top_idx.cpu().numpy()
                    new_solutions = [new_solutions[i] for i in top_idx_cpu]
            else:
                # Leaf nodes - update best solution
                for j in range(new_values.numel()):
                    if float(new_values[j]) > best_solution_value:
                        best_solution_value = float(new_values[j])
                        best_solution_items = new_solutions[j].copy()
                nodes_explored += int(new_weights.numel())
                break

            curr_weights = new_weights
            curr_values = new_values
            curr_solutions = new_solutions
            curr_index = next_index
            nodes_explored += int(curr_weights.numel())

        # Map back to original indices
        order_np = order.cpu().numpy()
        taken_orig = [int(order_np[i]) for i in best_solution_items]
        taken_orig.sort()
        elapsed = time.perf_counter() - start
        optimal = (beam_width is None and max_depth is None and time_limit is None)
        return BnBResult(best_value=int(best_solution_value), best_items=taken_orig, nodes_explored=nodes_explored, time_sec=elapsed, optimal=optimal)
