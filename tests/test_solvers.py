import unittest
import sys
import os
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.problems.knapsack import Knapsack
from src.solvers.cpu_dfs import CpuDfsSolver
from src.solvers.gpu_bfs import GpuBfsSolver
from src.solvers.hybrid import HybridSolver

class TestSolvers(unittest.TestCase):
    def setUp(self):
        self.problem = Knapsack()
        # Small instance for exact testing
        self.instance = self.problem.generate_instance(n=15, seed=42)
        
    def test_cpu_solver(self):
        solver = CpuDfsSolver()
        res = solver.solve(self.instance)
        self.assertTrue(res.optimal)
        self.assertGreater(res.best_value, 0)

    def test_gpu_solver_numpy(self):
        solver = GpuBfsSolver()
        # Force numpy by disabling torch
        res = solver.solve(self.instance, use_torch=False)
        
        # CPU result for comparison
        cpu_res = CpuDfsSolver().solve(self.instance)
        
        self.assertEqual(res.best_value, cpu_res.best_value)

    def test_hybrid_solver(self):
        solver = HybridSolver()
        res = solver.solve(self.instance, switch_depth=5)
        
        cpu_res = CpuDfsSolver().solve(self.instance)
        self.assertEqual(res.best_value, cpu_res.best_value)

if __name__ == '__main__':
    unittest.main()
