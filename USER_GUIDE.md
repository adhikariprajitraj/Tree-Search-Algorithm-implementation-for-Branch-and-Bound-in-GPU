# Complete User Guide: Branch & Bound Solvers

## Table of Contents
1. [Overview](#overview)
2. [Available Solvers](#available-solvers)
3. [Command-Line Flags](#command-line-flags)
4. [Usage Examples](#usage-examples)
5. [Benchmarking and Analysis](#benchmarking-and-analysis)
6. [Performance Guidelines](#performance-guidelines)

---

## Overview

This project provides three solvers for the 0-1 Knapsack problem:
1. **CPU DFS** - Exact, depth-first search on CPU
2. **GPU BFS** - Breadth-first search on GPU (exact or heuristic with beam search)
3. **Hybrid** - GPU warm start followed by CPU refinement

---

## Available Solvers

### 1. CPU DFS Solver (Default)
- **Algorithm**: Depth-first search with branch and bound
- **Optimality**: Always finds exact optimal solution
- **Speed**: Best for N < 100
- **Memory**: Very low (uses recursion stack)
- **When to use**: Small to medium problems, when you need guaranteed optimal solution

### 2. GPU BFS Solver
- **Algorithm**: Breadth-first search on GPU
- **Optimality**: Exact if no beam width, approximate with beam search
- **Speed**: Slower than CPU for small N due to data transfer overhead
- **Memory**: High (stores all nodes per level)
- **When to use**: Large problems (N > 100) with beam search for fast approximate solutions (CPU might work but may take longer)

### 3. Hybrid Solver
- **Algorithm**: GPU BFS to depth D, then CPU DFS from best nodes
- **Optimality**: Exact (combines GPU speed with CPU precision)
- **Speed**: Good balance for medium-large problems
- **Memory**: Moderate
- **When to use**: Medium to large problems (N = 40-100) when you want both speed and optimality

---

## Command-Line Flags

### Main Entry Point: `src/main.py`

```bash
python3 src/main.py [OPTIONS]
```

### Required/Basic Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--n` | int | 30 | Number of items in the knapsack problem |
| `--problem` | str | `knapsack` | Problem type: `knapsack` or `subset_sum` |

### Solver Selection Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--hybrid` | flag | False | Use Hybrid solver (GPU → CPU). If not set, runs both CPU and GPU independently |
| `--no_torch` | flag | False | Force NumPy implementation, disable PyTorch GPU acceleration |

### GPU Configuration Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--device` | str | auto | GPU device: `cpu`, `cuda`, `mps` (Mac M1/M2/M3), `xpu` (Intel) |
| `--beam_width` | int | auto | Max nodes per BFS level. Auto-set to 5000 for N>35. Use `None` for exact (no beam) |

### Search Control Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--time_limit` | float | None | Maximum execution time in seconds. Solver stops when limit reached |
| `--switch_depth` | int | 12 | Depth at which Hybrid solver switches from GPU to CPU |

### Problem Generation Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--capacity_factor` | float | 0.5 | Knapsack capacity as fraction of total weight (0.0-1.0) |
| `--seed` | int | 42 | Random seed for reproducible problem generation |

---

## Usage Examples

### Basic Usage

#### 1. Run with Default Settings (N=30, both CPU and GPU)
```bash
python3 src/main.py
```
**Output**: Runs CPU DFS (exact) and GPU BFS (exact), compares results

#### 2. Small Problem (N=20)
```bash
python3 src/main.py --n 20
```
**Expected**: CPU will be ~2000x faster than GPU

#### 3. Large Problem (N=100)
```bash
python3 src/main.py --n 100
```
**Expected**: Auto-enables beam search for GPU, CPU might work but may take longer

### Solver-Specific Usage

#### CPU DFS Only (Fastest for Small N)
```bash
python3 src/main.py --n 25 --no_torch
```
**Effect**: Disables GPU, runs only CPU solver (NumPy implementation)

#### GPU BFS with Beam Search (Fast Approximate)
```bash
python3 src/main.py --n 150 --beam 10000
```
**Effect**: GPU with beam width 10,000 (heuristic, fast but not optimal)

#### GPU BFS Exact (Slow for Large N)
```bash
python3 src/main.py --n 25 --beam 0
```
**Note**: beam=0 or no --beam flag means exact mode (explores all nodes)

#### Hybrid Solver (Best of Both Worlds)
```bash
python3 src/main.py --n 80 --hybrid
```
**Effect**: GPU explores first 12 levels, CPU refines from there

### Advanced Configuration

#### Custom Switch Depth for Hybrid
```bash
python3 src/main.py --n 100 --hybrid --switch_depth 15 --beam 5000
```
**Effect**: GPU explores to depth 15 with beam search, then CPU takes over

#### Time-Limited Search
```bash
python3 src/main.py --n 200 --time_limit 10.0 --beam 20000
```
**Effect**: GPU runs for max 10 seconds, returns best solution found

#### Specific GPU Device
```bash
# Mac M1/M2/M3
python3 src/main.py --n 50 --device mps

# NVIDIA GPU
python3 src/main.py --n 50 --device cuda

# Intel GPU
python3 src/main.py --n 50 --device xpu

# Force CPU (even with PyTorch)
python3 src/main.py --n 50 --device cpu
```

#### Custom Problem Instance
```bash
python3 src/main.py --n 60 --capacity_factor 0.3 --seed 12345
```
**Effect**: 
- 60 items
- Capacity = 30% of total weight (tighter constraint)
- Seed 12345 for reproducibility

#### Subset Sum Problem
```bash
python3 src/main.py --problem subset_sum --n 40
```
**Effect**: Solves subset sum instead of knapsack (all values = weights)

---

## Benchmarking and Analysis

### Performance Analysis Script

```bash
python3 performance_analysis.py [OPTIONS]
```

#### Flags for `performance_analysis.py`

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--sizes` | int[] | [10,15,20,25,30] | List of problem sizes to test |
| `--large` | flag | False | Test large sizes (35,40,50,60,80,100) |
| `--beam` | int | None | Beam width for GPU (auto-set if --large) |
| `--device` | str | auto | GPU device to use |

#### Examples

**Test Small Problem Sizes**
```bash
python3 performance_analysis.py --sizes 10 15 20 25 30
```

**Test Large Problems**
```bash
python3 performance_analysis.py --large
```
Auto-uses beam width 5000

**Custom Size Range**
```bash
python3 performance_analysis.py --sizes 50 100 150 200 --beam 10000
```

### Comprehensive Benchmark Script

```bash
python3 comprehensive_benchmark.py [OPTIONS]
```

#### Flags for `comprehensive_benchmark.py`

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--start` | int | 5 | Starting problem size |
| `--end` | int | 300 | Ending problem size |
| `--step` | int | 5 | Step size between tests |
| `--beam` | int | 10000 | Beam width for GPU and Hybrid |
| `--no-graph` | flag | False | Skip graph generation |

#### Examples

**Full Benchmark (N=5 to N=300)**
```bash
python3 comprehensive_benchmark.py --start 5 --end 300 --step 5
```
**Output**: 
- `benchmark_results.json` (raw data)
- `algorithm_comparison.png` (4 graphs)

**Quick Test (N=10 to N=100)**
```bash
python3 comprehensive_benchmark.py --start 10 --end 100 --step 10
```

**Custom Range**
```bash
python3 comprehensive_benchmark.py --start 50 --end 200 --step 25 --beam 20000
```

### Test Large Problem Script

```bash
python3 test_large_problem.py
```
**Effect**: Tests N=50, 70, 100 with GPU beam search, checks for overflow bugs

---

## Performance Guidelines

### When to Use Each Solver

| Problem Size (N) | Recommended Solver | Command | Expected Time |
|------------------|-------------------|---------|---------------|
| < 30 | CPU DFS | `python3 src/main.py --n 25` | < 0.001s |
| 30-50 | CPU DFS or Hybrid | `python3 src/main.py --n 40 --hybrid` | 0.001-0.1s |
| 50-100 | Hybrid | `python3 src/main.py --n 80 --hybrid --beam 10000` | 0.1-5s |
| 100-200 | GPU BFS | `python3 src/main.py --n 150 --beam 15000` | 5-20s |
| > 200 | GPU BFS + Time Limit | `python3 src/main.py --n 300 --beam 20000 --time_limit 30` | Up to limit |

### Beam Width Guidelines

| Problem Size (N) | Suggested Beam Width | Trade-off |
|------------------|---------------------|-----------|
| < 35 | None (exact) | Optimal solution, slower |
| 35-50 | 5,000 | Good balance |
| 50-100 | 10,000 | Fast, slight quality loss |
| 100-200 | 15,000-20,000 | Very fast, ~95% optimal |
| > 200 | 25,000+ | Quick heuristic solution |

**Higher beam width** = Better solution quality but slower + more memory

### Device Selection

| Device | Flag | Best For | Notes |
|--------|------|----------|-------|
| Auto | (none) | General use | Automatically selects best available |
| CPU | `--device cpu` | Small N < 30 | Fastest for small problems |
| MPS | `--device mps` | Mac M1/M2/M3 | Apple Silicon GPU |
| CUDA | `--device cuda` | NVIDIA GPUs | Best GPU performance |
| XPU | `--device xpu` | Intel Arc/UHD | Requires IPEX |

### Performance Expectations

Based on actual benchmarks:

**CPU Dominance (Small Problems)**
- N=20: CPU=0.0003s, GPU=0.67s → **CPU is 2400x faster**
- N=30: CPU=0.0009s, GPU=1.99s → **CPU is 2200x faster**

**CPU Still Wins (Medium Problems)**
- N=50: CPU=0.0013s, GPU=0.95s → **CPU is 730x faster**
- N=80: CPU=0.0059s, GPU=3.22s → **CPU is 545x faster**

**CPU Remains Competitive (Large Problems)**
- N=100: CPU=0.013s, GPU=4.67s → **CPU is 359x faster**
- N=200: CPU=0.05s, GPU=~8s → **CPU is 160x faster**
- N=500: CPU=0.19s, GPU=~15s → **CPU is 79x faster**

**Conclusion**: CPU is faster for ALL problem sizes tested! GPU is only useful when:
1. You batch many problems together
2. You use time limits on very large N
3. CPU is not available

---

## Common Workflows

### 1. Quick Test
```bash
python3 src/main.py --n 20
```

### 2. Production Optimal Solution
```bash
python3 src/main.py --n 50 --hybrid
```

### 3. Fast Approximate for Large N
```bash
python3 src/main.py --n 300 --beam 25000 --time_limit 30
```

### 4. Benchmark Comparison
```bash
python3 comprehensive_benchmark.py --start 10 --end 100 --step 10
```

### 5. Check GPU Availability
```bash
python3 check_gpu.py
```

---

## Troubleshooting

### GPU Not Detected
```bash
python3 check_gpu.py
```
Check if PyTorch sees your GPU. If not, reinstall PyTorch for your platform.

### Out of Memory (GPU)
Reduce beam width:
```bash
python3 src/main.py --n 100 --beam 5000  # Instead of 10000
```

### Too Slow
For large N, always use beam search:
```bash
python3 src/main.py --n 200 --beam 20000
```

### Need Exact Solution for Large N
CPU DFS might work for large N (be patient) or use Hybrid:
```bash
python3 src/main.py --n 100 --hybrid
```

---

## Summary

**Quick Reference Card:**

```bash
# Small problem, optimal
python3 src/main.py --n 25

# Large problem, fast approximate
python3 src/main.py --n 200 --beam 20000

# Medium problem, balanced
python3 src/main.py --n 80 --hybrid

# Full benchmark
python3 comprehensive_benchmark.py

# Check GPU
python3 check_gpu.py
```

**Remember**: CPU is usually faster! Only use GPU for very large N or when batching multiple problems.
