"""
Quick test to demonstrate the GPU performance issue
with a single very large problem.
"""

import time
import sys
import os
sys.path.append(os.path.dirname(__file__))

from src.problems.knapsack import Knapsack
from src.solvers.cpu_dfs import CpuDfsSolver
from src.solvers.gpu_bfs import GpuBfsSolver

def test_very_large():
    print("=" * 80)
    print("TESTING VERY LARGE KNAPSACK PROBLEM")
    print("=" * 80)
    
    # Test with progressively larger sizes
    for n in [50, 70, 100]:
        print(f"\n{'='*80}")
        print(f"Problem Size: N = {n} items")
        print(f"{'='*80}\n")
        
        # Generate problem
        problem = Knapsack()
        instance = problem.generate_instance(n=n, capacity_factor=0.5, seed=42)
        
        print(f"Capacity: {instance.capacity:,}")
        print(f"Total Weight: {instance.weights.sum():,}")
        print(f"Total Value: {instance.values.sum():,}")
        
        # GPU BFS with beam search
        print(f"\n--- GPU BFS (Beam Width = 10,000) ---")
        gpu_solver = GpuBfsSolver()
        
        try:
            start = time.perf_counter()
            res_gpu = gpu_solver.solve(
                instance,
                beam_width=10000,
                device="mps",
                use_torch=True
            )
            elapsed = time.perf_counter() - start
            
            print(f"✓ Success!")
            print(f"  Best Value: {res_gpu.best_value:,}")
            print(f"  Time: {res_gpu.time_sec:.4f}s")
            print(f"  Nodes Explored: {res_gpu.nodes_explored:,}")
            
        except Exception as e:
            print(f"❌ FAILED: {e}")
            print(f"   This is the integer overflow bug!")
            print(f"   Bitmask (1 << {n}) exceeds int64 range")
            break
    
    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print("\n⚠️  The GPU implementation has a critical bug:")
    print("   - Uses bitmask to track solution (1 << i)")
    print("   - Fails when N > 60-70 due to integer overflow")
    print("   - Need to switch to list-based solution tracking")
    print("\n💡 For large problems, the implementation needs fixes!")

if __name__ == "__main__":
    test_very_large()
