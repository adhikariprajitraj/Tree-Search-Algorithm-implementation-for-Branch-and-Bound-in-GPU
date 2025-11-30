# GPU-Accelerated Branch & Bound for Knapsack Problems

This project implements a modular **Branch and Bound (B&B)** solver for the **0-1 Knapsack** and **Subset Sum** problems. It features a high-performance **GPU-based BFS solver** (using PyTorch) and a **Hybrid solver** that combines GPU parallelism with CPU depth-first search.

## Features

-   **GPU Acceleration**: Uses PyTorch (and Intel Extension for PyTorch) to parallelize node expansion and bounding.
-   **Vectorized Bounding**: Optimized tensor operations to remove Python loops in the critical path.
-   **Hybrid Search**: "Warm Start" strategy using GPU BFS to find a strong initial bound, followed by exact CPU DFS.
-   **Modular Design**: Easy to add new problem types and solvers.
-   **Intel GPU Support**: Specifically adapted for Intel UHD/Arc Graphics via `ipex`.

## Installation

1.  **Python**: Ensure you have Python 3.8+ installed.
2.  **PyTorch**: Install PyTorch.
    -   **Windows/Linux**:
        ```bash
        pip install torch torchvision torchaudio
        ```
    -   **Mac (M1/M2/M3)**:
        PyTorch comes with MPS support by default on Mac. Just install the standard package:
        ```bash
        pip install torch torchvision torchaudio
        ```
3.  **Intel Extension for PyTorch (Optional)**: For Intel GPU support on Windows.
    ```bash
    pip install intel_extension_for_pytorch
    ```

## Usage

The main entry point is `src/main.py`. You can run it from the terminal with various flags.

### Basic Example
Run the Knapsack solver with default settings ($N=30$):
```bash
python src/main.py
```

### Command Line Arguments

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--n` | `30` | Number of items in the problem instance. |
| `--problem` | `knapsack` | Problem type: `knapsack` or `subset_sum`. |
| `--device` | `None` | Device to use: `cpu`, `cuda`, `mps` (Mac), `xpu` (Intel). |

| `--hybrid` | `False` | Enable Hybrid Solver (GPU Warm Start -> CPU DFS). |
| `--beam_width` | `None` | Max nodes per level for GPU BFS. Auto-set for large $N$. |
| `--time_limit` | `None` | Time limit in seconds. |
| `--switch_depth` | `12` | Depth to switch from GPU to CPU in Hybrid mode. |
| `--seed` | `42` | Random seed for reproducibility. |
| `--no_torch` | `False` | Force NumPy implementation (disable PyTorch). |

### Examples

**1. Large Scale Knapsack on Intel GPU:**
Run a 50-item instance on Intel GPU (`xpu`) using the Hybrid solver:
```bash
python src/main.py --n 50 --hybrid --device xpu
```

**2. Subset Sum Problem:**
Solve a Subset Sum instance:
```bash
python src/main.py --problem subset_sum --n 40
```

**3. Time-Limited Search:**
Run for at most 5 seconds:
```bash
python src/main.py --n 100 --time_limit 5.0
```

**4. Tuning Hybrid Search:**
Switch from GPU to CPU deeper in the tree (depth 15):
```bash
python src/main.py --n 60 --hybrid --switch_depth 15
```

## Project Structure

-   `src/problems/`: Problem definitions (`Knapsack`, `SubsetSum`).
-   `src/solvers/`: Solver implementations (`CpuDfsSolver`, `GpuBfsSolver`, `HybridSolver`).
-   `src/utils/`: Utility functions.
-   `tests/`: Unit tests.

## Scaling Guide
For details on how to scale this project further (multi-GPU, distributed), see [SCALING_GUIDE.md](SCALING_GUIDE.md).



### Relevant Studies:

# GPU-Native Branch-and-Bound / Tree Search & Related Work

## 1. GPU-Based Branch-and-Bound / Tree Search

- **El Baz et al., “GPU Implementation of the Branch and Bound Method for Knapsack Problems” (IPDPSW 2012)**  
  - Focus: Hybrid CPU–GPU B&B for knapsack problems, breadth-first search, Dantzig fractional bound, discussion of irregular data structures on GPU.  
  - Link (PDF): https://homepages.laas.fr/elbaz/4676b763.pdf  

- **Shen et al., “An Out-of-Core Branch and Bound Method for Solving the 0–1 Knapsack Problem on a GPU” (ICA3PP 2017)**  
  - Focus: B&B on GPU with out-of-core management of nodes, stream compaction to reduce sparsity, swapping node pools between GPU and CPU memory.  
  - Link (PDF): https://www-hagi.ist.osaka-u.ac.jp/research/papers/201708_shen_ica3pp.pdf  

- **El Baz et al., “Solving knapsack problems on GPU” (Journal of Computational and Applied Mathematics, 2011)**  
  - Focus: Parallel B&B and dynamic programming for knapsack on CUDA GPUs; breadth-first subtree processing and GPU-friendly bounding.  
  - Link: https://www.sciencedirect.com/science/article/pii/S0305054811000876  

## 2. Design Considerations for GPU-Based MIP / MILP

- **Bertsekas et al. (Sandia et al.), “Design Considerations for GPU-based Mixed Integer Programming on Accelerated Architectures” (Tech Report)**  
  - Focus: High-level analysis of challenges/possibilities for MIP on GPUs: memory layout, node pools, LP relaxation on GPU, communication patterns.  
  - Link (PDF): https://www.osti.gov/servlets/purl/1817473  

- **MIPcc26 Challenge – “GPU-Accelerated Primal Heuristics for MIP” (MixedInteger.org, 2026 competition)**  
  - Focus: Community challenge explicitly about GPU-based heuristics for MIP; lists relevant references and expected problem scales/library ecosystem.  
  - Link: https://www.mixedinteger.org/2026/competition/  

## 3. GPU-Accelerated Primal Heuristics and LP Relaxations

- **Çördük et al., “GPU-Accelerated Primal Heuristics for Mixed Integer Programming” (arXiv 2025)**  
  - Focus: Fusion of GPU-accelerated primal heuristics, using PDLP as an approximate LP solver plus Feasibility Pump/Jump and Fix-and-Propagate on GPU.  
  - PDF: https://arxiv.org/pdf/2510.20499  
  - Abs/metadata: https://arxiv.org/abs/2510.20499  

- **Applegate et al., “PDLP: A Practical First-Order Method for Large-Scale Linear Programming” (arXiv 2025)**  
  - Focus: Primal–dual hybrid gradient–based LP solver; very GPU-friendly structure, used as an approximate LP engine in GPU-based MIP heuristics.  
  - PDF: https://arxiv.org/pdf/2501.07018  
  - PDLP math background (OR-Tools): https://developers.google.com/optimization/lp/pdlp_math  

- **Google Research Blog, “Scaling up linear programming with PDLP” (2024)**  
  - Overview blog on PDLP and its large-scale performance, including notes on GPU implementations.  
  - Link: https://research.google/blog/scaling-up-linear-programming-with-pdlp/  

- **NVIDIA cuOpt Blog, “Accelerate Large Linear Programming Problems with NVIDIA cuOpt” (2024)**  
  - Focus: PDLP-based GPU LP solver (cuOpt) with huge speedups; good reference for GPU LP kernels that could be used as MILP node relaxations.  
  - Link: https://developer.nvidia.com/blog/accelerate-large-linear-programming-problems-with-nvidia-cuopt/  

- **Lu, “GPU-Accelerated Linear Programming and Beyond” (MixedInteger 2025 talk slides)**  
  - Focus: What operations GPUs are good at, PDLP/PDHG kernels, and perspectives on GPU-friendly mathematical programming.  
  - Slides (PDF): https://www.mixedinteger.org/2025/slides/2025-06-04%20-%20Talks%20-%20Haihao%20Lu.pdf  

- **Mexi, “Scylla: A primal heuristic for mixed-integer optimization problems” (web preprint / project)**  
  - Focus: Primal heuristics using matrix-free PDHG for LP relaxations, fix-and-propagate, and feasibility-pump-style updates (not strictly GPU-only but structurally similar to PDLP/PDHG-based designs).  
  - Link: https://gionimexi.com/  

## 4. Learning / RL for Branch-and-Bound Heuristics (Relevance: GPU-Friendly Policies, DFL)

- **Scavuzzo, Aardal, Lodi, Yorke-Smith, “Machine learning augmented branch and bound for mixed integer linear programming” (Mathematical Programming, Series B, 2024)**  
  - Focus: ML-augmented branching and node selection in B&B; provides a reference for learning-based heuristics that could be deployed in a GPU-native tree search.  
  - Link: https://link.springer.com/article/10.1007/s10107-024-02130-y  

- **Lodi, “Machine Learning Augmented Branch and Bound for Mixed Integer Linear Programming” (CO@Work 2024 slides)**  
  - Slides summarizing the above work; useful for high-level picture and design patterns.  
  - PDF: https://co-at-work.zib.de/slides/COatWork2024_Lodi_MLaugmentedBaB.pdf  

- **Strang et al., “Planning in Branch-and-Bound: Model-Based Reinforcement Learning for Exact Combinatorial Optimization” (arXiv 2025)**  
  - Focus: Model-based RL for branching decisions in B&B; shows how to frame branching as a planning problem over a tree, compatible with batched evaluation.  
  - PDF: https://arxiv.org/pdf/2511.09219v2  
  - Abs/metadata: https://arxiv.org/abs/2511.09219  

- **A Markov Decision Process for Variable Selection in Branch and Bound (OpenReview, 2023)**  
  - Focus: RL/MDP formulation of variable selection in B&B for MILPs; relevant to designing GPU-evaluable policies for branching in a batched/level-parallel search.  
  - Link: https://openreview.net/forum?id=ifJFKbSZxS  

## 5. General Background on Branch-and-Bound & Knapsack (Reference for Bounds/Heuristics)

- **0/1 Knapsack using Branch and Bound (GeeksforGeeks, tutorial)**  
  - Focus: Educational reference for classic DFS-style B&B on knapsack, with Dantzig bound and tree representation. Good for sanity checks and baseline CPU implementation.  
  - Link: https://www.geeksforgeeks.org/dsa/0-1-knapsack-using-branch-and-bound/  
