from abc import ABC, abstractmethod
from typing import Dict, Any, List
from pathlib import Path

class BenchmarkClientBase(ABC):
    """Abstract base class for benchmark clients."""

    @abstractmethod
    def run_single_benchmark(self, test_args: Dict[str, Any], **kwargs):
        """Run a single benchmark test."""
        pass

    @abstractmethod
    def _extract_metrics(self, log_file: Path) -> Dict[str, float]:
        """Extract metrics from the log file."""
        pass
