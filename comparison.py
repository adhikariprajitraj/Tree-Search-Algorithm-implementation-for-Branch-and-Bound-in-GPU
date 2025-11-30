import time
import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np

try:
    import torch
    HAS_TORCH = True
    HAS_MPS = torch.backends.mps.is_available()
except Exception:
    HAS_TORCH = False
    HAS_MPS = False


@dataclass
class KnapsackInstance:
    weights: np.ndarray  # shape (n,)
    values: np.ndarray   # shape (n,)
    capacity: int


@dataclass
class BnBResult:
    best_value: int
    best_items: List[int]
    nodes_explored: int
    time_sec: float
    optimal: bool


def generate_knapsack_instance(n: int,
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


# ---------- Classic CPU depth-first Branch & Bound ----------

def fractional_knapsack_bound(weights: np.ndarray,
                              values: np.ndarray,
                              capacity: int,
                              taken_mask: int,
                              fixed_index: int) -> float:
    """
    Compute an upper bound by taking remaining items greedily (fractional allowed).
    Items are assumed to be pre-sorted by value/weight ratio.
    taken_mask: bitmask of items taken so far (1 = taken).
    fixed_index: next item index to consider branching on.
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


def cpu_branch_and_bound(instance: KnapsackInstance) -> BnBResult:
    """
    Classic depth-first Branch & Bound for 0-1 knapsack on CPU.
    Uses fractional knapsack bound and explores full tree (exact).
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


# ---------- GPU-style level-parallel / batched B&B heuristic ----------

def gpu_style_batched_bnb(instance: KnapsackInstance,
                          beam_width: Optional[int] = None,
                          max_depth: Optional[int] = None,
                          time_limit: Optional[float] = None) -> BnBResult:
    """
    Level-parallel, batched B&B heuristic inspired by GPU design.
    Implemented with NumPy; can be ported to PyTorch on M1/MPS.

    - BFS style expansion
    - Fractional bound using prefix sums
    - Optional beam search (beam_width)
    - Optional depth limit and time limit
    """
    start = time.perf_counter()

    weights = instance.weights.copy()
    values = instance.values.copy()
    cap = instance.capacity
    n = len(weights)

    # Sort by value/weight ratio for better bounding
    ratio = values / weights
    order = np.argsort(-ratio)
    w = weights[order]
    v = values[order]

    # Greedy incumbent (feasible)
    remaining_cap = cap
    best_value = 0
    best_mask = 0
    for i in range(n):
        if w[i] <= remaining_cap:
            remaining_cap -= int(w[i])
            best_value += int(v[i])
            best_mask |= (1 << i)

    best_solution_value = best_value
    best_solution_mask = best_mask

    # Batch state: arrays for current frontier
    curr_weights = np.array([0], dtype=np.int64)
    curr_values = np.array([0], dtype=np.int64)
    curr_masks = np.array([0], dtype=np.int64)
    curr_index = 0
    nodes_explored = 0

    # Precompute prefix sums for fractional bound
    prefix_w = np.cumsum(w)
    prefix_v = np.cumsum(v)

    # Helper: fractional bound for a batch, vectorized as much as possible
    def fractional_bound_batch(node_weights: np.ndarray,
                               node_values: np.ndarray,
                               next_index: int) -> np.ndarray:
        if next_index >= n:
            return node_values.astype(float)
        # remaining capacity per node
        rem_cap = cap - node_weights
        base_weight = prefix_w[next_index - 1] if next_index > 0 else 0.0
        base_value = prefix_v[next_index - 1] if next_index > 0 else 0.0

        # target prefix weight threshold including capacity
        target = base_weight + rem_cap
        # position where prefix_w >= target
        idx = np.searchsorted(prefix_w, target, side="left")

        ub = node_values.astype(float).copy()
        for j in range(node_weights.size):
            if rem_cap[j] <= 0:
                # no remaining capacity
                continue
            if idx[j] >= n:
                # take all remaining items
                ub[j] += float(prefix_v[-1] - base_value)
            else:
                # full items up to prev_index
                prev_index = idx[j] - 1
                if prev_index >= next_index:
                    ub[j] += float(prefix_v[prev_index] - base_value)
                # weight used by full items from next_index..prev_index
                cap_used = 0.0
                if prev_index >= next_index:
                    cap_used = float(prefix_w[prev_index] - base_weight)
                # capacity remaining for fractional item
                remaining = float(rem_cap[j] - cap_used)
                if remaining <= 0:
                    continue
                # check if we exactly hit an item boundary
                if math.isclose(remaining, float(w[idx[j]]), rel_tol=1e-9, abs_tol=1e-9):
                    ub[j] += float(v[idx[j]])
                else:
                    # take fraction of item idx[j]
                    item_w = float(w[idx[j]])
                    item_v = float(v[idx[j]])
                    frac = max(0.0, min(1.0, remaining / item_w))
                    ub[j] += item_v * frac
        return ub

    # Main BFS / batched expansion loop
    while curr_index < n and curr_weights.size > 0:
        # Optional depth limit
        if max_depth is not None and curr_index >= max_depth:
            break
        if time_limit is not None and (time.perf_counter() - start) > time_limit:
            break

        # Branch: 0 = skip item, 1 = take item
        count = curr_weights.size
        new_weights = np.concatenate([curr_weights, curr_weights])
        new_values = np.concatenate([curr_values, curr_values])
        new_masks = np.concatenate([curr_masks, curr_masks])

        # apply "take" to second half
        new_weights[count:] += w[curr_index]
        new_values[count:] += v[curr_index]
        new_masks[count:] |= (1 << curr_index)

        # Feasibility pruning
        feasible = new_weights <= cap
        new_weights = new_weights[feasible]
        new_values = new_values[feasible]
        new_masks = new_masks[feasible]

        if new_weights.size == 0:
            break

        next_index = curr_index + 1

        if next_index < n:
            # Fractional bounds for batch
            ub = fractional_bound_batch(new_weights, new_values, next_index)
            # Prune by incumbent
            keep = ub > best_solution_value
            new_weights = new_weights[keep]
            new_values = new_values[keep]
            new_masks = new_masks[keep]
            ub = ub[keep]

            if new_weights.size == 0:
                break

            # Beam search: keep only top-k by bound
            if beam_width is not None and new_weights.size > beam_width:
                idx_sorted = np.argsort(-ub)
                top_idx = idx_sorted[:beam_width]
                new_weights = new_weights[top_idx]
                new_values = new_values[top_idx]
                new_masks = new_masks[top_idx]
                ub = ub[top_idx]
        else:
            # Leaf nodes: all items decided
            # Update best solution from these complete states
            for j in range(new_values.size):
                if new_values[j] > best_solution_value:
                    best_solution_value = int(new_values[j])
                    best_solution_mask = int(new_masks[j])
            nodes_explored += new_weights.size
            break

        curr_weights = new_weights
        curr_values = new_values
        curr_masks = new_masks
        curr_index = next_index
        nodes_explored += curr_weights.size

        if curr_weights.size == 0:
            break

    # Map mask back to original indices
    taken_sorted = [i for i in range(n) if best_solution_mask & (1 << i)]
    taken_orig = [int(order[i]) for i in taken_sorted]
    taken_orig.sort()

    elapsed = time.perf_counter() - start
    # This method is heuristic if beam_width or depth/time limits are used.
    optimal = (beam_width is None and max_depth is None and time_limit is None)
    return BnBResult(best_value=int(best_solution_value),
                     best_items=taken_orig,
                     nodes_explored=nodes_explored,
                     time_sec=elapsed,
                     optimal=optimal)


# ---------- Optional: PyTorch/MPS variant skeleton ----------

def gpu_style_batched_bnb_torch(instance: KnapsackInstance,
                                beam_width: Optional[int] = None,
                                max_depth: Optional[int] = None,
                                time_limit: Optional[float] = None,
                                device: Optional[str] = None) -> BnBResult:
    """
    PyTorch version that can run on M1 GPU via MPS (if available).
    This is a skeleton using the same logic as the NumPy version, but using torch tensors.
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch not available")
    if device is None:
        if HAS_MPS:
            device = "mps"
        else:
            device = "cpu"

    start = time.perf_counter()

    weights = torch.tensor(instance.weights, dtype=torch.float32, device=device)
    values = torch.tensor(instance.values, dtype=torch.float32, device=device)
    cap = float(instance.capacity)
    n = weights.numel()

    # Sort by ratio
    ratio = values / weights
    order = torch.argsort(-ratio)
    w = weights[order]
    v = values[order]

    # Greedy incumbent on device
    remaining_cap = cap
    best_value = 0.0
    best_mask = 0
    for i in range(n):
        wi = float(w[i])
        if wi <= remaining_cap:
            remaining_cap -= wi
            best_value += float(v[i])
            best_mask |= (1 << int(i))

    best_solution_value = best_value
    best_solution_mask = best_mask

    curr_weights = torch.zeros(1, dtype=torch.float32, device=device)
    curr_values = torch.zeros(1, dtype=torch.float32, device=device)
    curr_masks = torch.zeros(1, dtype=torch.int64, device=device)
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
        # For simplicity, loop in Python over nodes (still OK for demo)
        for j in range(node_weights.numel()):
            if rem_cap[j] <= 0:
                continue
            ij = int(idx[j])
            if ij >= n:
                ub[j] = ub[j] + (prefix_v[-1] - base_value)
            else:
                prev_index = ij - 1
                val_add = 0.0
                if prev_index >= next_index:
                    val_add += float(prefix_v[prev_index] - base_value)
                cap_used = 0.0
                if prev_index >= next_index:
                    cap_used = float(prefix_w[prev_index] - base_weight)
                remaining = float(rem_cap[j] - cap_used)
                if remaining > 0:
                    item_w = float(w[ij])
                    item_v = float(v[ij])
                    frac = max(0.0, min(1.0, remaining / item_w))
                    val_add += item_v * frac
                ub[j] = ub[j] + val_add
        return ub

    while curr_index < n and curr_weights.numel() > 0:
        if max_depth is not None and curr_index >= max_depth:
            break
        if time_limit is not None and (time.perf_counter() - start) > time_limit:
            break

        count = curr_weights.numel()
        new_weights = torch.cat([curr_weights, curr_weights])
        new_values = torch.cat([curr_values, curr_values])
        new_masks = torch.cat([curr_masks, curr_masks])

        new_weights[count:] = new_weights[count:] + w[curr_index]
        new_values[count:] = new_values[count:] + v[curr_index]
        new_masks[count:] = new_masks[count:] | (1 << int(curr_index))

        feasible = new_weights <= cap
        new_weights = new_weights[feasible]
        new_values = new_values[feasible]
        new_masks = new_masks[feasible]

        if new_weights.numel() == 0:
            break

        next_index = curr_index + 1

        if next_index < n:
            ub = fractional_bound_batch_torch(new_weights, new_values, next_index)
            keep = ub > best_solution_value
            new_weights = new_weights[keep]
            new_values = new_values[keep]
            new_masks = new_masks[keep]
            ub = ub[keep]

            if new_weights.numel() == 0:
                break

            if beam_width is not None and new_weights.numel() > beam_width:
                # top-k selection by ub
                ub_sorted, idx_sorted = torch.sort(ub, descending=True)
                top_idx = idx_sorted[:beam_width]
                new_weights = new_weights[top_idx]
                new_values = new_values[top_idx]
                new_masks = new_masks[top_idx]
        else:
            # leaf
            for j in range(new_values.numel()):
                if float(new_values[j]) > best_solution_value:
                    best_solution_value = float(new_values[j])
                    best_solution_mask = int(new_masks[j].item())
            nodes_explored += int(new_weights.numel())
            break

        curr_weights = new_weights
        curr_values = new_values
        curr_masks = new_masks
        curr_index = next_index
        nodes_explored += int(curr_weights.numel())

        if curr_weights.numel() == 0:
            break

    taken_sorted = [i for i in range(n) if best_solution_mask & (1 << i)]
    order_np = order.cpu().numpy()
    taken_orig = [int(order_np[i]) for i in taken_sorted]
    taken_orig.sort()

    elapsed = time.perf_counter() - start
    optimal = (beam_width is None and max_depth is None and time_limit is None)
    return BnBResult(best_value=int(best_solution_value),
                     best_items=taken_orig,
                     nodes_explored=nodes_explored,
                     time_sec=elapsed,
                     optimal=optimal)


# ---------- Benchmark / comparison driver ----------

def compare_solvers(n_items: int = 30,
                    capacity_factor: float = 0.5,
                    beam_width: Optional[int] = None,
                    max_depth: Optional[int] = None,
                    time_limit: Optional[float] = None,
                    seed: Optional[int] = 42,
                    use_torch: bool = False) -> None:
    """
    Generate an instance and compare:
    - exact CPU depth-first B&B
    - batched BFS-style B&B (NumPy or PyTorch)
    """
    inst = generate_knapsack_instance(n_items,
                                      capacity_factor=capacity_factor,
                                      seed=seed)
    print(f"Instance: n={n_items}, capacity={inst.capacity}")
    print("Weights:", inst.weights)
    print("Values :", inst.values)

    print("\nRunning CPU depth-first B&B (exact)...")
    res_cpu = cpu_branch_and_bound(inst)
    print(f"CPU B&B: value={res_cpu.best_value}, items={res_cpu.best_items}")
    print(f"           nodes={res_cpu.nodes_explored}, time={res_cpu.time_sec:.4f} s")

    if use_torch and HAS_TORCH:
        print("\nRunning batched GPU-style B&B (PyTorch, device=mps/cpu)...")
        res_gpu = gpu_style_batched_bnb_torch(inst,
                                              beam_width=beam_width,
                                              max_depth=max_depth,
                                              time_limit=time_limit)
    else:
        print("\nRunning batched GPU-style B&B (NumPy)...")
        res_gpu = gpu_style_batched_bnb(inst,
                                        beam_width=beam_width,
                                        max_depth=max_depth,
                                        time_limit=time_limit)

    print(f"GPU-style B&B: value={res_gpu.best_value}, items={res_gpu.best_items}")
    print(f"               nodes={res_gpu.nodes_explored}, time={res_gpu.time_sec:.4f} s")
    print(f"               optimal_flag={res_gpu.optimal}")

    # Compare solution quality
    gap = (res_cpu.best_value - res_gpu.best_value) / max(1, abs(res_cpu.best_value))
    print(f"\nRelative gap (GPU-style vs CPU-optimal): {gap:.4%}")


if __name__ == "__main__":
    # Example benchmark:
    #  - exact CPU B&B
    #  - GPU-style heuristic with a modest beam width
    compare_solvers(n_items=28,
                    capacity_factor=0.5,
                    beam_width=2048,
                    max_depth=None,
                    time_limit=None,
                    seed=123,
                    use_torch=False)
