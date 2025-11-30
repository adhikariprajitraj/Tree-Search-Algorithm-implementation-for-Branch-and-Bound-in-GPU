import argparse
import sys
import os

# Ensure src is in path if running directly
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.problems.knapsack import Knapsack
from src.solvers.cpu_dfs import CpuDfsSolver
from src.solvers.gpu_bfs import GpuBfsSolver

def main():
    parser = argparse.ArgumentParser(description="Knapsack Branch & Bound Benchmark")
    parser.add_argument("--n", type=int, default=30, help="Number of items")
    parser.add_argument("--capacity_factor", type=float, default=0.5, help="Capacity factor")
    parser.add_argument("--beam_width", type=int, default=None, help="Beam width for BFS")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default=None, help="Device for GPU solver (cpu, cuda, mps, xpu)")
    parser.add_argument("--no_torch", action="store_true", help="Disable PyTorch usage")
    
    args = parser.parse_args()

    print(f"Generating Knapsack Instance (n={args.n})...")
    problem = Knapsack()
    instance = problem.generate_instance(n=args.n, capacity_factor=args.capacity_factor, seed=args.seed)
    
    print(f"Capacity: {instance.capacity}")
    print(f"Weights: {instance.weights}")
    print(f"Values: {instance.values}")

    print("\n--- Running CPU DFS Solver ---")
    cpu_solver = CpuDfsSolver()
    res_cpu = cpu_solver.solve(instance)
    print(f"Value: {res_cpu.best_value}")
    print(f"Items: {res_cpu.best_items}")
    print(f"Time:  {res_cpu.time_sec:.4f}s")
    print(f"Nodes: {res_cpu.nodes_explored}")

    print("\n--- Running GPU BFS Solver ---")
    gpu_solver = GpuBfsSolver()
    res_gpu = gpu_solver.solve(instance, 
                               beam_width=args.beam_width,
                               device=args.device,
                               use_torch=not args.no_torch)
    print(f"Value: {res_gpu.best_value}")
    print(f"Items: {res_gpu.best_items}")
    print(f"Time:  {res_gpu.time_sec:.4f}s")
    print(f"Nodes: {res_gpu.nodes_explored}")
    print(f"Optimal: {res_gpu.optimal}")

    gap = (res_cpu.best_value - res_gpu.best_value) / max(1, abs(res_cpu.best_value))
    print(f"\nRelative Gap: {gap:.4%}")

if __name__ == "__main__":
    main()
