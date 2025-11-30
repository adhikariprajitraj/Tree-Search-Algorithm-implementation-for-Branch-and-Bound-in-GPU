### GPU native tree search heuristics for MILPs

In this project, I am trying to implement a tunable-accuracy approximation method focusing on batching, simplifed bounding, relaxed branching and level-parallel expansion.

The branch and bound algorithm is a standard method for solving MILPs. It is a tree search algorithm where each node represents a subproblem and the search space is pruned using bounding. The bounding is done using a relaxation of the problem, which is solved using a linear programming solver. The relaxation is done by relaxing the integrality constraints of the problem. 

#### Challenges with B and B on GPU
 - Dynamic memory allocation and deallocation is not supported on GPU
 - Strong branching and complex node selection are not easily parallelizable
 - Asynchronous memory transfers are not supported on GPU

#### Strategies we are going to explore in this algorithm 
 - Level-parallel node expansion: Instead of expanding nodes one by one, we can expand nodes in parallel. This can be done on GPU. By processing nodes in large batches, we can hide the memory latency and improve the performance. 
 - Simplified bounding: Instead of using the exact bounding, we can use a simplified bounding. This can be done by using a relaxation of the problem. The relaxation is done by relaxing the integrality constraints of the problem. This can be done on GPU. 
 - Relaxed branching: Instead of using the exact bounding, we can use a simplified bounding. This can be done by using a relaxation of the problem. The relaxation is done by relaxing the integrality constraints of the problem. This can be done on GPU. 
 - No Dynamic Priority Queue: Using algorithms such as BFS and Beam Search.
 - DP like Relaxations: We will use DP like relaxations to solve the problem.
 - Approximate Optimality: Although, the solutions are not optimal, they are close to optimal. 

 #### Hardware used
 - GPU: M1 Pro
 - CPU: M1 Pro
 - RAM: 16GB

#### Key Implementation Concepts
 - Use pytorch arrays to represent batches of node data.
 - Avoid Python loops over nodes.
 - Tunable parameters: setting maximum beam width or depth


 Example: Parallel B and B for 0-1 Knapsack
