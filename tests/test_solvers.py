import unittest
import sys
import os
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.problems.knapsack import Knapsack
from src.problems.subset_sum import SubsetSum
from src.solvers.cpu_dfs import CpuDfsSolver
from src.solvers.gpu_bfs import GpuBfsSolver
from src.solvers.hybrid import HybridSolver

class TestSolvers(unittest.TestCase):
    def setUp(self):
        self.knapsack_problem = Knapsack()
        self.subset_sum_problem = SubsetSum()
        # Small instance for exact testing
        self.knapsack_instance = self.knapsack_problem.generate_instance(n=15, seed=42)
        self.subset_sum_instance = self.subset_sum_problem.generate_instance(n=15, seed=42)
        
    # Knapsack Tests
    def test_cpu_solver_knapsack(self):
        solver = CpuDfsSolver()
        res = solver.solve(self.knapsack_instance)
        self.assertTrue(res.optimal)
        self.assertGreater(res.best_value, 0)

    def test_gpu_solver_numpy_knapsack(self):
        solver = GpuBfsSolver()
        # Force numpy by disabling torch
        res = solver.solve(self.knapsack_instance, use_torch=False)
        
        # CPU result for comparison
        cpu_res = CpuDfsSolver().solve(self.knapsack_instance)
        
        self.assertEqual(res.best_value, cpu_res.best_value)

    def test_hybrid_solver_knapsack(self):
        solver = HybridSolver()
        res = solver.solve(self.knapsack_instance, switch_depth=5)
        
        cpu_res = CpuDfsSolver().solve(self.knapsack_instance)
        self.assertEqual(res.best_value, cpu_res.best_value)
    
    # Subset Sum Tests
    def test_cpu_solver_subset_sum(self):
        solver = CpuDfsSolver()
        res = solver.solve(self.subset_sum_instance)
        self.assertTrue(res.optimal)
        self.assertGreater(res.best_value, 0)
        # For subset sum, values should equal weights
        total_weight = sum(self.subset_sum_instance.weights[i] for i in res.best_items)
        self.assertEqual(res.best_value, total_weight)

    def test_gpu_solver_numpy_subset_sum(self):
        solver = GpuBfsSolver()
        # Force numpy by disabling torch
        res = solver.solve(self.subset_sum_instance, use_torch=False)
        
        # CPU result for comparison
        cpu_res = CpuDfsSolver().solve(self.subset_sum_instance)
        
        self.assertEqual(res.best_value, cpu_res.best_value)
        # Verify subset sum property: value equals weight
        total_weight = sum(self.subset_sum_instance.weights[i] for i in res.best_items)
        self.assertEqual(res.best_value, total_weight)

    def test_hybrid_solver_subset_sum(self):
        solver = HybridSolver()
        res = solver.solve(self.subset_sum_instance, switch_depth=5)
        
        cpu_res = CpuDfsSolver().solve(self.subset_sum_instance)
        self.assertEqual(res.best_value, cpu_res.best_value)
        # Verify subset sum property: value equals weight
        total_weight = sum(self.subset_sum_instance.weights[i] for i in res.best_items)
        self.assertEqual(res.best_value, total_weight)
    
    # Verify subset sum property: values equal weights
    def test_subset_sum_property(self):
        # Check that generated instance has values == weights
        self.assertTrue(np.array_equal(self.subset_sum_instance.values, self.subset_sum_instance.weights))
        
        # Verify solution maintains this property
        solver = CpuDfsSolver()
        res = solver.solve(self.subset_sum_instance)
        total_weight = sum(self.subset_sum_instance.weights[i] for i in res.best_items)
        self.assertEqual(res.best_value, total_weight)

if __name__ == '__main__':
    unittest.main()
