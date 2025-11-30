import argparse
import sys
import os
import time

# Ensure src is in path if running directly
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.problems.knapsack import Knapsack
from src.problems.subset_sum import SubsetSum
from src.solvers.cpu_dfs import CpuDfsSolver
from src.solvers.gpu_bfs import GpuBfsSolver
from src.solvers.hybrid import HybridSolver

def main():
    parser = argparse.ArgumentParser(description="Knapsack Branch & Bound Benchmark")
    parser.add_argument("--n", type=int, default=30, help="Number of items")
    parser.add_argument("--problem", type=str, default="knapsack", choices=["knapsack", "subset_sum"], help="Problem type")
    parser.add_argument("--capacity_factor", type=float, default=0.5, help="Capacity factor")
    parser.add_argument("--beam_width", type=int, default=None, help="Beam width for BFS (default: auto)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default=None, help="Device for GPU solver (cpu, cuda, mps, xpu)")
    parser.add_argument("--no_torch", action="store_true", help="Disable PyTorch usage")
    parser.add_argument("--hybrid", action="store_true", help="Use Hybrid Solver (GPU BFS -> CPU DFS)")
    parser.add_argument("--time_limit", type=float, default=None, help="Time limit in seconds")
    parser.add_argument("--switch_depth", type=int, default=12, help="Depth to switch from GPU to CPU in Hybrid mode")
    
    args = parser.parse_args()

    # --- Problem Generation ---
    print(f"Generating {args.problem.replace('_', ' ').title()} Instance (n={args.n})...")
    if args.problem == "knapsack":
        problem = Knapsack()
    else:
        problem = SubsetSum()
        
    instance = problem.generate_instance(n=args.n, capacity_factor=args.capacity_factor, seed=args.seed)
    
    print(f"Capacity: {instance.capacity}")
    if args.n <= 50:
        print(f"Weights: {instance.weights}")
        print(f"Values: {instance.values}")
    else:
        print(f"Weights: [Hidden for large N]")

    # --- Solver Configuration ---
    
    # For large N, we MUST use a beam width for the GPU solver to avoid OOM
    # and to demonstrate the heuristic gap.
    gpu_beam_width = args.beam_width
    if gpu_beam_width is None:
        if args.n > 35:
            gpu_beam_width = 5000
            print(f"\n[NOTE] Large N detected. Auto-setting beam_width={gpu_beam_width} for GPU solver.")
        else:
            gpu_beam_width = None # Exact mode

    # --- CPU DFS (Exact) ---
    # Only run exact CPU DFS if N is small enough, otherwise it takes forever
    if args.n <= 40:
        print("\n--- Running CPU DFS Solver (Exact) ---")
        cpu_solver = CpuDfsSolver()
        res_cpu = cpu_solver.solve(instance)
        print(f"Value: {res_cpu.best_value}")
        print(f"Time:  {res_cpu.time_sec:.4f}s")
        print(f"Nodes: {res_cpu.nodes_explored}")
        exact_value = res_cpu.best_value
    else:
        print("\n[SKIP] CPU DFS skipped for N > 40 (too slow).")
        exact_value = None

    # --- GPU BFS (Heuristic or Exact) ---
    print(f"\n--- Running GPU BFS Solver ({'Heuristic' if gpu_beam_width else 'Exact'}) ---")
    gpu_solver = GpuBfsSolver()
    res_gpu = gpu_solver.solve(instance, 
                               beam_width=gpu_beam_width,
                               time_limit=args.time_limit,
                               device=args.device,
                               use_torch=not args.no_torch)
    print(f"Value: {res_gpu.best_value}")
    print(f"Time:  {res_gpu.time_sec:.4f}s")
    print(f"Nodes: {res_gpu.nodes_explored}")
    print(f"Optimal: {res_gpu.optimal}")

    # --- Hybrid Solver ---
    if args.hybrid:
        print("\n--- Running Hybrid Solver (GPU Warm Start -> CPU DFS) ---")
        hybrid_solver = HybridSolver()
        # Use a small switch depth to get a quick bound
        res_hybrid = hybrid_solver.solve(instance, 
                                         switch_depth=args.switch_depth, 
                                         beam_width=2000, 
                                         time_limit=args.time_limit,
                                         device=args.device, 
                                         use_torch=not args.no_torch)
        print(f"Value: {res_hybrid.best_value}")
        print(f"Time:  {res_hybrid.time_sec:.4f}s")
        print(f"Nodes: {res_hybrid.nodes_explored}")
        
        if exact_value is None:
            exact_value = res_hybrid.best_value # Assume hybrid found best if CPU skipped

    # --- Comparison ---
    if exact_value is not None:
        gap = (exact_value - res_gpu.best_value) / max(1, abs(exact_value))
        print(f"\nRelative Gap (GPU vs Best Known): {gap:.4%}")
        if gap > 0:
            print("-> Gap detected! The heuristic GPU search traded optimality for speed/memory.")
        else:
            print("-> No gap. GPU found the optimal solution.")
    else:
        print("\nCannot calculate gap (no exact solution available).")

if __name__ == "__main__":
    main()
