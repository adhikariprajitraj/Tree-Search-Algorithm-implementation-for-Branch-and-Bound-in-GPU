from abc import ABC, abstractmethod
from typing import Any, Tuple

class Problem(ABC):
    """Abstract base class for optimization problems."""
    
    @abstractmethod
    def generate_instance(self, *args, **kwargs) -> Any:
        """Generate a random instance of the problem."""
        pass
