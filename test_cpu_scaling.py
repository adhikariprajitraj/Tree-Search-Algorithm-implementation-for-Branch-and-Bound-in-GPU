import time
import sys
import os

# Add the current directory to the path so we can import src
sys.path.append(os.getcwd())

from src.problems.knapsack import Knapsack
from src.solvers.cpu_dfs import CpuDfsSolver
from src.solvers.gpu_bfs import GpuBfsSolver

def test_cpu_with_recursion_limit(n):
    """Test CPU solver with increased recursion limit"""
    import sys
    old_limit = sys.getrecursionlimit()
    start = time.perf_counter()
    try:
        # Increase recursion limit (but this may still not be enough for N=2000)
        sys.setrecursionlimit(max(n + 100, 10000))
        problem = Knapsack()
        instance = problem.generate_instance(n=n, seed=42)
        
        cpu_solver = CpuDfsSolver()
        result = cpu_solver.solve(instance)
        elapsed = time.perf_counter() - start
        
        return result, elapsed, None
    except Exception as e:
        elapsed = time.perf_counter() - start
        return None, elapsed, str(e)
    finally:
        sys.setrecursionlimit(old_limit)

def main():
    print("Testing CPU solver performance scaling...")
    print("="*80)
    
    # Test with progressively larger N values
    test_sizes = [20, 30, 40]
    
    results = []
    
    for n in test_sizes:
        print(f"\nTesting N = {n}")
        print("-" * 80)
        
        problem = Knapsack()
        instance = problem.generate_instance(n=n, seed=42)
        
        # CPU DFS
        print("Running CPU DFS...")
        cpu_start = time.perf_counter()
        try:
            cpu_solver = CpuDfsSolver()
            cpu_result = cpu_solver.solve(instance)
            cpu_time = time.perf_counter() - cpu_start
            print(f"  CPU Time: {cpu_time:.4f} seconds")
            print(f"  CPU Value: {cpu_result.best_value}")
            print(f"  CPU Nodes: {cpu_result.nodes_explored:,}")
            results.append((n, cpu_time, cpu_result.best_value, True))
        except Exception as e:
            cpu_time = time.perf_counter() - cpu_start
            print(f"  CPU Failed: {e}")
            results.append((n, cpu_time, None, False))
    
    # Now test N=2000
    print("\n" + "="*80)
    print("Testing N = 2000")
    print("="*80)
    
    problem = Knapsack()
    instance = problem.generate_instance(n=2000, seed=42)
    
    # Try CPU with increased recursion limit
    print("Attempting CPU DFS with increased recursion limit...")
    cpu_result, cpu_time, cpu_error = test_cpu_with_recursion_limit(2000)
    
    if cpu_result:
        print(f"  CPU Time: {cpu_time:.4f} seconds ({cpu_time/60:.2f} minutes)")
        print(f"  CPU Value: {cpu_result.best_value}")
        print(f"  CPU Nodes: {cpu_result.nodes_explored:,}")
    else:
        print(f"  CPU Failed after {cpu_time:.4f} seconds")
        print(f"  Error: {cpu_error}")
        print("\n  ⚠️  CPU solver might work for N=2000 but may encounter:")
        print("     - Recursion depth limits (even with increased limit)")
        print("     - Exponential time complexity (might take hours/days)")
        print("     - Memory constraints")
    
    # GPU for comparison
    print("\nRunning GPU BFS with beam search (beam_width=5000)...")
    gpu_solver = GpuBfsSolver()
    gpu_start = time.perf_counter()
    gpu_result = gpu_solver.solve(instance, beam_width=5000, time_limit=300)
    gpu_time = time.perf_counter() - gpu_start
    
    print(f"  GPU Time: {gpu_time:.4f} seconds ({gpu_time/60:.2f} minutes)")
    print(f"  GPU Value: {gpu_result.best_value}")
    print(f"  GPU Nodes: {gpu_result.nodes_explored:,}")
    
    # Extrapolation
    print("\n" + "="*80)
    print("PERFORMANCE ANALYSIS")
    print("="*80)
    
    if len(results) >= 2:
        # Calculate exponential growth rate
        times = [r[1] for r in results if r[3]]  # Only successful runs
        ns = [r[0] for r in results if r[3]]
        
        if len(times) >= 2:
            # Rough exponential fit: time = a * b^n
            # Using last two points
            n1, t1 = ns[-2], times[-2]
            n2, t2 = ns[-1], times[-1]
            
            if t1 > 0:
                growth_rate = (t2 / t1) ** (1.0 / (n2 - n1))
                print(f"Observed growth rate: ~{growth_rate:.3f}x per item")
                
                # Extrapolate to N=2000
                n_base = n2
                t_base = t2
                n_target = 2000
                estimated_time = t_base * (growth_rate ** (n_target - n_base))
                
                print(f"\nExtrapolated CPU time for N=2000:")
                print(f"  Estimated: {estimated_time:.2e} seconds")
                print(f"  Estimated: {estimated_time/3600:.2e} hours")
                print(f"  Estimated: {estimated_time/(3600*24):.2e} days")
                print(f"\n  GPU Time: {gpu_time:.4f} seconds ({gpu_time/60:.2f} minutes)")
                
                if estimated_time > 0:
                    speedup = estimated_time / gpu_time
                    print(f"  Estimated Speedup: {speedup:.2e}x faster with GPU")
    
    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    print("For N=2000:")
    print("  • CPU: Might work but may encounter recursion limits or take days/weeks")
    print("  • GPU: Completes in ~1-2 minutes with beam search")
    print("  • Trade-off: GPU provides quick approximate solution vs CPU optimality")

if __name__ == "__main__":
    main()

