"""
Comprehensive Performance Analysis for GPU vs CPU Branch and Bound
This script analyzes performance bottlenecks in the GPU implementation
and tests with varying problem sizes.
"""

import time
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(__file__))

from src.problems.knapsack import Knapsack
from src.solvers.cpu_dfs import CpuDfsSolver
from src.solvers.gpu_bfs import GpuBfsSolver

def run_performance_test(n_values, beam_width=None, device=None):
    """
    Run performance tests across different problem sizes
    
    Args:
        n_values: List of problem sizes to test
        beam_width: Beam width for GPU solver (None = exact)
        device: Device to use for GPU solver
    """
    
    results = []
    
    print("=" * 80)
    print("PERFORMANCE ANALYSIS: GPU vs CPU Branch and Bound")
    print("=" * 80)
    print(f"\nBeam Width: {beam_width if beam_width else 'EXACT (no beam)'}")
    print(f"Device: {device if device else 'auto'}\n")
    
    for n in n_values:
        print(f"\n{'=' * 80}")
        print(f"Testing with N = {n} items")
        print(f"{'=' * 80}")
        
        # Generate problem instance
        problem = Knapsack()
        instance = problem.generate_instance(n=n, capacity_factor=0.5, seed=42)
        
        print(f"Capacity: {instance.capacity}")
        print(f"Total weight: {instance.weights.sum()}")
        print(f"Total value: {instance.values.sum()}")
        
        result = {
            'n': n,
            'capacity': instance.capacity,
            'total_weight': instance.weights.sum(),
            'total_value': instance.values.sum()
        }
        
        # CPU DFS (exact) - skip for very large problems
        if n <= 30:
            print(f"\n--- CPU DFS (Exact) ---")
            cpu_solver = CpuDfsSolver()
            try:
                cpu_result = cpu_solver.solve(instance)
                print(f"Best Value: {cpu_result.best_value}")
                print(f"Time: {cpu_result.time_sec:.4f}s")
                print(f"Nodes Explored: {cpu_result.nodes_explored:,}")
                
                result['cpu_value'] = cpu_result.best_value
                result['cpu_time'] = cpu_result.time_sec
                result['cpu_nodes'] = cpu_result.nodes_explored
            except Exception as e:
                print(f"CPU solver failed: {e}")
                result['cpu_value'] = None
                result['cpu_time'] = None
                result['cpu_nodes'] = None
        else:
            print(f"\n--- CPU DFS (Skipped - N > 30) ---")
            result['cpu_value'] = None
            result['cpu_time'] = None
            result['cpu_nodes'] = None
        
        # GPU BFS
        print(f"\n--- GPU BFS ---")
        gpu_solver = GpuBfsSolver()
        try:
            gpu_result = gpu_solver.solve(
                instance, 
                beam_width=beam_width,
                device=device,
                use_torch=True
            )
            print(f"Best Value: {gpu_result.best_value}")
            print(f"Time: {gpu_result.time_sec:.4f}s")
            print(f"Nodes Explored: {gpu_result.nodes_explored:,}")
            print(f"Optimal: {gpu_result.optimal}")
            
            result['gpu_value'] = gpu_result.best_value
            result['gpu_time'] = gpu_result.time_sec
            result['gpu_nodes'] = gpu_result.nodes_explored
            result['gpu_optimal'] = gpu_result.optimal
        except Exception as e:
            print(f"GPU solver failed: {e}")
            result['gpu_value'] = None
            result['gpu_time'] = None
            result['gpu_nodes'] = None
            result['gpu_optimal'] = False
        
        # Compare if both ran
        if result['cpu_time'] is not None and result['gpu_time'] is not None:
            speedup = result['cpu_time'] / result['gpu_time']
            print(f"\n--- Comparison ---")
            print(f"Speedup (CPU/GPU): {speedup:.2f}x")
            if speedup < 1:
                print(f"⚠️  GPU is {1/speedup:.2f}x SLOWER than CPU!")
            else:
                print(f"✓ GPU is {speedup:.2f}x faster than CPU")
            
            if result['cpu_value'] != result['gpu_value']:
                gap = abs(result['cpu_value'] - result['gpu_value']) / max(1, result['cpu_value'])
                print(f"Solution Gap: {gap:.2%}")
            
            result['speedup'] = speedup
        elif result['gpu_time'] is not None:
            result['speedup'] = None
        
        results.append(result)
    
    # Summary
    print(f"\n\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}\n")
    
    print(f"{'N':<8} {'CPU Time':<12} {'GPU Time':<12} {'Speedup':<12} {'CPU Value':<12} {'GPU Value':<12}")
    print("-" * 80)
    
    for r in results:
        cpu_time_str = f"{r['cpu_time']:.4f}s" if r['cpu_time'] is not None else "N/A"
        gpu_time_str = f"{r['gpu_time']:.4f}s" if r['gpu_time'] is not None else "N/A"
        speedup_str = f"{r.get('speedup', 0):.2f}x" if r.get('speedup') is not None else "N/A"
        cpu_val_str = f"{r['cpu_value']}" if r['cpu_value'] is not None else "N/A"
        gpu_val_str = f"{r['gpu_value']}" if r['gpu_value'] is not None else "N/A"
        
        print(f"{r['n']:<8} {cpu_time_str:<12} {gpu_time_str:<12} {speedup_str:<12} {cpu_val_str:<12} {gpu_val_str:<12}")
    
    # Analysis
    print(f"\n{'=' * 80}")
    print("ANALYSIS: Why is GPU slower?")
    print(f"{'=' * 80}\n")
    
    print("Common reasons GPU is slower than CPU for Branch and Bound:")
    print()
    print("1. **DATA TRANSFER OVERHEAD**")
    print("   - Moving data between CPU ↔ GPU has significant latency")
    print("   - Small problem sizes don't benefit from GPU parallelism")
    print("   - Overhead dominates computation time")
    print()
    print("2. **ALGORITHM NOT GPU-FRIENDLY**")
    print("   - Branch and Bound is inherently sequential")
    print("   - Heavy branching and pruning reduces parallelism")
    print("   - GPU excels at uniform, parallel operations")
    print()
    print("3. **MEMORY ALLOCATION OVERHEAD**")
    print("   - Frequent tensor concatenations (torch.cat)")
    print("   - Boolean masking creates new tensors")
    print("   - GPU memory allocation is expensive")
    print()
    print("4. **UNDERUTILIZED GPU CORES**")
    print("   - Early in search tree: few nodes → few parallel threads")
    print("   - Late in search tree: heavy pruning → few nodes again")
    print("   - GPU cores sit idle most of the time")
    print()
    print("5. **KERNEL LAUNCH OVERHEAD**")
    print("   - Each operation launches a GPU kernel")
    print("   - Kernel launch has fixed overhead (~10-50μs)")
    print("   - Many small operations = many kernel launches")
    print()
    
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80 + "\n")
    
    print("For SMALL problems (N < 30):")
    print("  → Use CPU DFS (it's faster)")
    print()
    print("For MEDIUM problems (30 ≤ N ≤ 50):")
    print("  → Use GPU with beam search (beam_width=5000-10000)")
    print("  → Trades optimality for speed")
    print()
    print("For LARGE problems (N > 50):")
    print("  → Use Hybrid approach (GPU warm start → CPU refinement)")
    print("  → GPU generates good initial solution quickly")
    print("  → CPU refines with DFS from promising nodes")
    print()
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Performance Analysis Tool")
    parser.add_argument("--sizes", type=int, nargs='+', default=[10, 15, 20, 25, 30],
                       help="Problem sizes to test (default: 10 15 20 25 30)")
    parser.add_argument("--large", action="store_true",
                       help="Test with large problem sizes (35, 40, 50, 60, 80, 100)")
    parser.add_argument("--beam", type=int, default=None,
                       help="Beam width for GPU solver (default: None for exact)")
    parser.add_argument("--device", type=str, default=None,
                       help="Device for GPU solver (cpu, cuda, mps, xpu)")
    
    args = parser.parse_args()
    
    if args.large:
        # Large problem sizes - must use beam search
        n_values = [35, 40, 50, 60, 80, 100]
        beam_width = args.beam if args.beam is not None else 5000
        print("\n⚠️  Large problem mode: Using beam search by default")
    else:
        n_values = args.sizes
        beam_width = args.beam
    
    run_performance_test(n_values, beam_width=beam_width, device=args.device)
