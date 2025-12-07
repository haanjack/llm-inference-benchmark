"""Base evaluator class."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List


class BaseEvaluator(ABC):
    """Abstract base class for LLM evaluators."""

    def __init__(self, model_name: str, endpoint: str, tasks: List[str]):
        """Initialize evaluator.

        Args:
            model_name: Model identifier
            endpoint: API endpoint URL
            tasks: List of task names to evaluate
        """
        self.model_name = model_name
        self.endpoint = endpoint
        self.tasks = tasks

    @abstractmethod
    def evaluate(self) -> Dict[str, Any]:
        """Run evaluation and return results."""

    @abstractmethod
    def get_supported_tasks(self) -> List[str]:
        """Return list of supported evaluation tasks."""
