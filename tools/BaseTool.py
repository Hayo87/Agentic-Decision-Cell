from abc import ABC, abstractmethod

class BaseTool(ABC):
    """Base class for all tools."""

    @abstractmethod
    def supports(self, action: str, target: str, agent: str) -> bool:
        """Return True if this tool can handle the given request."""
        ...

    @abstractmethod
    def run(self, content: str) -> str:
        """Execute the tool and return an observation."""
        ...