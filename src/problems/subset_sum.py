import random
import numpy as np
from typing import Tuple, Optional
from .knapsack import Knapsack, KnapsackInstance

class SubsetSum(Knapsack):
    """
    Subset Sum is a special case of Knapsack where value == weight.
    """
    def generate_instance(self, n: int,
                          weight_range: Tuple[int, int] = (5, 50),
                          capacity_factor: float = 0.5,
                          seed: Optional[int] = None,
                          **kwargs) -> KnapsackInstance:
        """Generate a random Subset Sum instance."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Generate weights
        weights = np.random.randint(weight_range[0], weight_range[1] + 1, size=n)
        
        # Values equal weights
        values = weights.copy()
        
        # Capacity is a fraction of total sum
        capacity = int(capacity_factor * weights.sum())
        
        return KnapsackInstance(weights=weights.astype(int),
                                values=values.astype(int),
                                capacity=capacity)
