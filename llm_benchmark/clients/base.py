from abc import ABC, abstractmethod
import logging
from typing import Dict, Any
from pathlib import Path
import pandas as pd

from llm_benchmark.server import BenchmarkBase
from llm_benchmark.utils.script_generator import ScriptGenerator

logger = logging.getLogger(__name__)

class BenchmarkClientBase(ABC):
    """Abstract base class for benchmark clients."""

    def __init__(self,
                 name: str,
                 server: BenchmarkBase,
                 is_dry_run: bool = False,
                 script_generator: ScriptGenerator = None):
        self.name = name
        self.server = server
        self._is_dry_run = is_dry_run
        self.script_generator = script_generator

        self._log_dir = Path("logs") / self.server.model_name / self.server.image_tag
        self._results_file = self._log_dir / self.server.exp_tag / \
            f"result_{Path(self.server.model_config).stem}_{self.server.name}_{self.name}.csv"
        self._total_results_file = self._log_dir / f"total_results_{self.server.name}_{self.name}.csv"

    @abstractmethod
    def run_single_benchmark(self, test_args: Dict[str, Any], **kwargs):
        """Run a single benchmark test."""
        pass

    @abstractmethod
    def _extract_metrics(self, log_file: Path) -> Dict[str, float]:
        """Extract metrics from the log file."""
        pass

    def _get_log_path(self, **kwargs) -> Path:
        """Constructs the log file path from benchmark parameters."""
        request_rate = kwargs.get("request_rate")
        num_prompts = kwargs.get("num_prompts")
        input_length = kwargs.get("input_length")
        output_length = kwargs.get("output_length")
        concurrency = kwargs.get("concurrency")
        return self._log_dir / self.server.exp_tag / \
            f"r{request_rate}_n{num_prompts}_{input_length}_o{output_length}_c{concurrency}.log"

    def _check_existing_result(self, **kwargs) -> Dict[str, float] | None:
        """Check if a benchmark result already exists."""
        log_file = self._get_log_path(**kwargs)
        if not log_file.exists():
            return None

        if not self.results_file.exists():
            return None

        try:
            df = pd.read_csv(self.results_file)
            # Create a filter condition for all kwargs
            condition = pd.Series([True] * len(df))
            for key, value in kwargs.items():
                if key in df.columns:
                    condition &= (df[key].astype(type(value) if value is not None else str) == value)

            if condition.any():
                # logger.info("Found existing results in %s", self.results_file)
                return df[condition].iloc[0].to_dict()
        except (pd.errors.EmptyDataError, KeyError) as e:
            logger.warning("Could not read or parse existing result file %s: %s", self.results_file, e)
            return None

        return None

    def _save_results(self, metrics: Dict[str, float], **kwargs):
        """Save benchmark results to CSV file."""
        result_line = ( # pyright: ignore
            f"{self.server.parallel_size.get('tp', '1')},"
            f"{kwargs.get('request_rate')},{kwargs.get('num_prompts')},{kwargs.get('concurrency')},"
            f"{kwargs.get('input_length')},{kwargs.get('output_length')},{metrics['test_time_s']:.2f},"
            f"{metrics['ttft_mean_ms']:.2f},{metrics['ttft_median_ms']:.2f},{metrics['ttft_p99_ms']:.2f},"
            f"{metrics['tpot_mean_ms']:.2f},{metrics['tpot_median_ms']:.2f},{metrics['tpot_p99_ms']:.2f},"
            f"{metrics['itl_mean_ms']:.2f},{metrics['itl_median_ms']:.2f},{metrics['itl_p99_ms']:.2f},"
            f"{metrics['e2el_mean_ms']:.2f},{metrics['e2el_median_ms']:.2f},{metrics['e2el_p99_ms']:.2f},"
            f"{metrics['request_throughput_rps']:.2f},{metrics['output_token_throughput_tps']:.2f},"
            f"{metrics['total_token_throughput_tps']:.2f}\n"
        )
        with open(self.results_file, 'a', encoding='utf-8') as f:
            f.write(result_line)
        with open(self._total_results_file, 'a', encoding='utf-8') as f:
            result_line = ( # pyright: ignore
                f"{Path(self.server.model_config).stem},{result_line}")
            f.write(result_line)

    @property
    def results_file(self) -> Path:
        """Get the path to the results CSV file."""
        return self._results_file
