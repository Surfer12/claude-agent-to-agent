"""Base class for computer use environments."""

import os
from abc import ABC, abstractmethod

class BaseComputer(ABC):
    """Abstract base class for computer environments."""

    def __init__(self):
        self.isolated = os.getenv("USE_DOCKER", False)
        if not self.isolated:
            raise ValueError("Computer use must be isolated (e.g., in Docker) for privacy.")

    @abstractmethod
    def screenshot(self):
        """Take a screenshot."""
        pass

    @abstractmethod
    def mouse_click(self, x: int, y: int):
        """Perform mouse click at coordinates."""
        pass

    @abstractmethod
    def type_text(self, text: str):
        """Type text into the computer."""
        pass

    # Additional methods inspired by CUA (e.g., from computer-use-demo/tools/computer.py)
    def ensure_isolation(self):
        """Ensure operations are isolated."""
        if not self.isolated:
            raise RuntimeError("Operation blocked: Not in isolated environment.") 