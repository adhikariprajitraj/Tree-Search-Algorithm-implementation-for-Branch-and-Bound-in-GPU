"""
Comprehensive Algorithm Comparison
Tests CPU DFS, GPU BFS, and Hybrid across N=5 to N=300 with step size of 5
Generates performance comparison graphs
"""

import time
import numpy as np
import sys
import os
import json
from datetime import datetime

sys.path.append(os.path.dirname(__file__))

from src.problems.knapsack import Knapsack
from src.solvers.cpu_dfs import CpuDfsSolver
from src.solvers.gpu_bfs import GpuBfsSolver
from src.solvers.hybrid import HybridSolver

def benchmark_all_algorithms(n_start=5, n_end=300, step=5, beam_width=10000):
    """
    Run comprehensive benchmark comparing all three algorithms
    
    Args:
        n_start: Starting problem size
        n_end: Ending problem size
        step: Step size
        beam_width: Beam width for GPU and Hybrid solvers
    """
    
    results = []
    n_values = list(range(n_start, n_end + 1, step))
    
    print("=" * 80)
    print("COMPREHENSIVE ALGORITHM COMPARISON")
    print("=" * 80)
    print(f"Testing N from {n_start} to {n_end} with step size {step}")
    print(f"GPU Beam Width: {beam_width}")
    print(f"Total tests: {len(n_values)}")
    print("=" * 80)
    
    for i, n in enumerate(n_values, 1):
        print(f"\n[{i}/{len(n_values)}] Testing N = {n}")
        print("-" * 80)
        
        # Generate problem instance
        problem = Knapsack()
        instance = problem.generate_instance(n=n, capacity_factor=0.5, seed=42)
        
        result = {
            'n': n,
            'capacity': instance.capacity,
            'total_weight': int(instance.weights.sum()),
            'total_value': int(instance.values.sum())
        }
        
        # Test CPU DFS (exact)
        print(f"  CPU DFS... ", end='', flush=True)
        try:
            cpu_solver = CpuDfsSolver()
            cpu_result = cpu_solver.solve(instance)
            result['cpu_time'] = cpu_result.time_sec
            result['cpu_value'] = cpu_result.best_value
            result['cpu_nodes'] = cpu_result.nodes_explored
            print(f"✓ {cpu_result.time_sec:.4f}s ({cpu_result.nodes_explored:,} nodes)")
        except KeyboardInterrupt:
            raise
        except Exception as e:
            result['cpu_time'] = None
            result['cpu_value'] = None
            result['cpu_nodes'] = None
            print(f"✗ Failed: {e}")
        
        # Test GPU BFS (with beam search)
        print(f"  GPU BFS... ", end='', flush=True)
        try:
            gpu_solver = GpuBfsSolver()
            gpu_result = gpu_solver.solve(
                instance,
                beam_width=beam_width,
                device="mps",
                use_torch=True
            )
            result['gpu_time'] = gpu_result.time_sec
            result['gpu_value'] = gpu_result.best_value
            result['gpu_nodes'] = gpu_result.nodes_explored
            print(f"✓ {gpu_result.time_sec:.4f}s ({gpu_result.nodes_explored:,} nodes)")
        except KeyboardInterrupt:
            raise
        except Exception as e:
            result['gpu_time'] = None
            result['gpu_value'] = None
            result['gpu_nodes'] = None
            print(f"✗ Failed: {e}")
        
        # Test Hybrid
        print(f"  Hybrid... ", end='', flush=True)
        try:
            hybrid_solver = HybridSolver()
            hybrid_result = hybrid_solver.solve(
                instance,
                switch_depth=15,
                beam_width=beam_width,
                device="mps",
                use_torch=True
            )
            result['hybrid_time'] = hybrid_result.time_sec
            result['hybrid_value'] = hybrid_result.best_value
            result['hybrid_nodes'] = hybrid_result.nodes_explored
            print(f"✓ {hybrid_result.time_sec:.4f}s ({hybrid_result.nodes_explored:,} nodes)")
        except KeyboardInterrupt:
            raise
        except Exception as e:
            result['hybrid_time'] = None
            result['hybrid_value'] = None
            result['hybrid_nodes'] = None
            print(f"✗ Failed: {e}")
        
        results.append(result)
        
        # Print quick comparison
        if all(result.get(f'{algo}_time') is not None for algo in ['cpu', 'gpu', 'hybrid']):
            fastest = min(
                ('CPU', result['cpu_time']),
                ('GPU', result['gpu_time']),
                ('Hybrid', result['hybrid_time']),
                key=lambda x: x[1]
            )
            print(f"  → Fastest: {fastest[0]} ({fastest[1]:.4f}s)")
    
    return results

def save_results(results, filename='benchmark_results.json'):
    """Save results to JSON file"""
    data = {
        'timestamp': datetime.now().isoformat(),
        'results': results
    }
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"\n✓ Results saved to {filename}")

def generate_graphs(results):
    """Generate comparison graphs"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
    except ImportError:
        print("\n⚠️  matplotlib not installed. Skipping graph generation.")
        print("   Install with: pip install matplotlib")
        return
    
    # Extract data
    n_values = [r['n'] for r in results]
    cpu_times = [r.get('cpu_time') for r in results]
    gpu_times = [r.get('gpu_time') for r in results]
    hybrid_times = [r.get('hybrid_time') for r in results]
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Algorithm Performance Comparison: CPU DFS vs GPU BFS vs Hybrid', 
                 fontsize=16, fontweight='bold')
    
    # 1. Execution Time Comparison (Linear Scale)
    ax1 = axes[0, 0]
    if any(cpu_times):
        ax1.plot(n_values, cpu_times, 'b-o', label='CPU DFS', linewidth=2, markersize=4)
    if any(gpu_times):
        ax1.plot(n_values, gpu_times, 'r-s', label='GPU BFS', linewidth=2, markersize=4)
    if any(hybrid_times):
        ax1.plot(n_values, hybrid_times, 'g-^', label='Hybrid', linewidth=2, markersize=4)
    ax1.set_xlabel('Problem Size (N)', fontsize=12)
    ax1.set_ylabel('Time (seconds)', fontsize=12)
    ax1.set_title('Execution Time vs Problem Size', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2. Execution Time Comparison (Log Scale)
    ax2 = axes[0, 1]
    if any(cpu_times):
        cpu_times_clean = [t if t is not None and t > 0 else None for t in cpu_times]
        ax2.semilogy([n for n, t in zip(n_values, cpu_times_clean) if t is not None],
                     [t for t in cpu_times_clean if t is not None],
                     'b-o', label='CPU DFS', linewidth=2, markersize=4)
    if any(gpu_times):
        gpu_times_clean = [t if t is not None and t > 0 else None for t in gpu_times]
        ax2.semilogy([n for n, t in zip(n_values, gpu_times_clean) if t is not None],
                     [t for t in gpu_times_clean if t is not None],
                     'r-s', label='GPU BFS', linewidth=2, markersize=4)
    if any(hybrid_times):
        hybrid_times_clean = [t if t is not None and t > 0 else None for t in hybrid_times]
        ax2.semilogy([n for n, t in zip(n_values, hybrid_times_clean) if t is not None],
                     [t for t in hybrid_times_clean if t is not None],
                     'g-^', label='Hybrid', linewidth=2, markersize=4)
    ax2.set_xlabel('Problem Size (N)', fontsize=12)
    ax2.set_ylabel('Time (seconds, log scale)', fontsize=12)
    ax2.set_title('Execution Time vs Problem Size (Log Scale)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, which='both')
    
    # 3. Speedup Factor (CPU / Others)
    ax3 = axes[1, 0]
    gpu_speedup = [cpu_times[i] / gpu_times[i] if cpu_times[i] and gpu_times[i] else None 
                   for i in range(len(results))]
    hybrid_speedup = [cpu_times[i] / hybrid_times[i] if cpu_times[i] and hybrid_times[i] else None 
                      for i in range(len(results))]
    
    if any(gpu_speedup):
        gpu_speedup_clean = [(n, s) for n, s in zip(n_values, gpu_speedup) if s is not None]
        if gpu_speedup_clean:
            ax3.plot([x[0] for x in gpu_speedup_clean], [x[1] for x in gpu_speedup_clean],
                    'r-s', label='CPU/GPU Speedup', linewidth=2, markersize=4)
    if any(hybrid_speedup):
        hybrid_speedup_clean = [(n, s) for n, s in zip(n_values, hybrid_speedup) if s is not None]
        if hybrid_speedup_clean:
            ax3.plot([x[0] for x in hybrid_speedup_clean], [x[1] for x in hybrid_speedup_clean],
                    'g-^', label='CPU/Hybrid Speedup', linewidth=2, markersize=4)
    
    ax3.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Equal Performance')
    ax3.set_xlabel('Problem Size (N)', fontsize=12)
    ax3.set_ylabel('Speedup Factor (CPU time / Other time)', fontsize=12)
    ax3.set_title('CPU Speedup vs Other Methods', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # 4. Nodes Explored Comparison
    ax4 = axes[1, 1]
    cpu_nodes = [r.get('cpu_nodes') for r in results]
    gpu_nodes = [r.get('gpu_nodes') for r in results]
    hybrid_nodes = [r.get('hybrid_nodes') for r in results]
    
    if any(cpu_nodes):
        cpu_nodes_clean = [(n, nodes) for n, nodes in zip(n_values, cpu_nodes) if nodes is not None]
        if cpu_nodes_clean:
            ax4.semilogy([x[0] for x in cpu_nodes_clean], [x[1] for x in cpu_nodes_clean],
                        'b-o', label='CPU DFS', linewidth=2, markersize=4)
    if any(gpu_nodes):
        gpu_nodes_clean = [(n, nodes) for n, nodes in zip(n_values, gpu_nodes) if nodes is not None]
        if gpu_nodes_clean:
            ax4.semilogy([x[0] for x in gpu_nodes_clean], [x[1] for x in gpu_nodes_clean],
                        'r-s', label='GPU BFS', linewidth=2, markersize=4)
    if any(hybrid_nodes):
        hybrid_nodes_clean = [(n, nodes) for n, nodes in zip(n_values, hybrid_nodes) if nodes is not None]
        if hybrid_nodes_clean:
            ax4.semilogy([x[0] for x in hybrid_nodes_clean], [x[1] for x in hybrid_nodes_clean],
                        'g-^', label='Hybrid', linewidth=2, markersize=4)
    
    ax4.set_xlabel('Problem Size (N)', fontsize=12)
    ax4.set_ylabel('Nodes Explored (log scale)', fontsize=12)
    ax4.set_title('Search Space Efficiency', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    
    # Save figure
    filename = 'algorithm_comparison.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n✓ Graph saved to {filename}")
    
    return filename

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive Algorithm Benchmark")
    parser.add_argument("--start", type=int, default=5, help="Starting N")
    parser.add_argument("--end", type=int, default=300, help="Ending N")
    parser.add_argument("--step", type=int, default=5, help="Step size")
    parser.add_argument("--beam", type=int, default=10000, help="Beam width for GPU/Hybrid")
    parser.add_argument("--no-graph", action="store_true", help="Skip graph generation")
    
    args = parser.parse_args()
    
    print("\n🚀 Starting comprehensive benchmark...")
    print(f"   N range: {args.start} to {args.end} (step {args.step})")
    print(f"   Total tests: {len(range(args.start, args.end + 1, args.step)) * 3}")
    print(f"   Estimated time: ~{len(range(args.start, args.end + 1, args.step)) * 10} seconds\n")
    
    try:
        results = benchmark_all_algorithms(
            n_start=args.start,
            n_end=args.end,
            step=args.step,
            beam_width=args.beam
        )
        
        # Save results
        save_results(results)
        
        # Generate graphs
        if not args.no_graph:
            print("\n📊 Generating comparison graphs...")
            graph_file = generate_graphs(results)
            if graph_file:
                print(f"\n✅ Benchmark complete! Check {graph_file} for visualizations.")
        
        print("\n" + "=" * 80)
        print("SUMMARY STATISTICS")
        print("=" * 80)
        
        cpu_times = [r.get('cpu_time') for r in results if r.get('cpu_time')]
        gpu_times = [r.get('gpu_time') for r in results if r.get('gpu_time')]
        hybrid_times = [r.get('hybrid_time') for r in results if r.get('hybrid_time')]
        
        if cpu_times:
            print(f"CPU DFS:   {len(cpu_times)} tests, avg={np.mean(cpu_times):.4f}s, "
                  f"min={np.min(cpu_times):.4f}s, max={np.max(cpu_times):.4f}s")
        if gpu_times:
            print(f"GPU BFS:   {len(gpu_times)} tests, avg={np.mean(gpu_times):.4f}s, "
                  f"min={np.min(gpu_times):.4f}s, max={np.max(gpu_times):.4f}s")
        if hybrid_times:
            print(f"Hybrid:    {len(hybrid_times)} tests, avg={np.mean(hybrid_times):.4f}s, "
                  f"min={np.min(hybrid_times):.4f}s, max={np.max(hybrid_times):.4f}s")
        
        # Count wins
        wins = {'CPU': 0, 'GPU': 0, 'Hybrid': 0}
        for r in results:
            times = {
                'CPU': r.get('cpu_time'),
                'GPU': r.get('gpu_time'),
                'Hybrid': r.get('hybrid_time')
            }
            valid_times = {k: v for k, v in times.items() if v is not None}
            if valid_times:
                winner = min(valid_times, key=valid_times.get)
                wins[winner] += 1
        
        print(f"\nWins: CPU={wins['CPU']}, GPU={wins['GPU']}, Hybrid={wins['Hybrid']}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Benchmark interrupted by user")
        sys.exit(1)
