from abc import ABC, abstractmethod
from typing import Any

class Solver(ABC):
    """Abstract base class for solvers."""
    
    @abstractmethod
    def solve(self, instance: Any, **kwargs) -> Any:
        """Solve the given problem instance."""
        pass
