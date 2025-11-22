import logging
import re
import subprocess
import pandas as pd
from pathlib import Path
from typing import Any
from typing import Dict, List

from llm_benchmark.clients.base import BenchmarkClientBase
from llm_benchmark.server.base import BenchmarkBase

logger = logging.getLogger(__name__)


class SGLangClient(BenchmarkClientBase):
    """Client for SGLang benchmarking."""

    def __init__(self, server: BenchmarkBase, is_dry_run: bool = False):
        super().__init__("sglang", server, is_dry_run)

    def run_single_benchmark(self, test_args: Dict[str, Any], **kwargs):
        """Run a single benchmark test."""
        request_rate = kwargs.get("request_rate")
        concurrency = kwargs.get("concurrency")
        input_length = kwargs.get("input_length")
        output_length = kwargs.get("output_length")
        num_prompts = kwargs.get("num_prompts")
        batch_size = kwargs.get("batch_size")
        dataset_name = kwargs.get('dataset_name')

        host, port = self.server.endpoint.split(":")

        cmd = []
        if not self.server.in_container:
            cmd.extend([self.server.container_runtime, "exec", self.server.container_name])
        cmd.extend([
            "python3", "-m", "sglang.bench_serving",
            "--host", host,
            "--port", str(port),
            "--dataset-name", dataset_name,
            "--request-rate", str(request_rate),
            "--max-concurrency", str(concurrency),
            "--random-input-len", str(input_length),
            "--random-output-len", str(output_length),
            "--num-prompt", str(num_prompts),
        ])

        if test_args:
            for key, value in test_args.items():
                if value is None:
                    continue
                if isinstance(value, bool):
                    if value:
                        cmd.append(f"--{key.replace('_', '-')}")
                else:
                    cmd.extend([f"--{key.replace('_', '-')}", str(value)])

        if self._is_dry_run:
            logger.info("Dry run - Benchmark command: %s", " ".join(cmd))
            return None

        existing_results = self._check_existing_result(**kwargs)
        if existing_results:
            return existing_results

        log_file = self._get_log_path(**kwargs)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"=== Benchmark: {kwargs} ===\n")
            f.write(f"Command: {' '.join(cmd)}\n\n")
            f.flush()

            subprocess.run(cmd, stdout=f, stderr=f, check=True)

        return self._extract_metrics(log_file)

    def _extract_metrics(self, log_file: Path) -> Dict[str, float]:
        metrics = {}
        patterns = {
            'test_time': r'Benchmark duration \(s\):\s*([\d.]+)',
            'ttft_mean': r'Mean TTFT \(ms\):\s*([\d.]+)',
            'ttft_median': r'Median TTFT \(ms\):\s*([\d.]+)',
            'ttft_p99': r'P99 TTFT \(ms\):\s*([\d.]+)',
            'tpot_mean': r'Mean TPOT \(ms\):\s*([\d.]+)',
            'tpot_median': r'Median TPOT \(ms\):\s*([\d.]+)',
            'tpot_p99': r'P99 TPOT \(ms\):\s*([\d.]+)',
            'itl_mean': r'Mean ITL \(ms\):\s*([\d.]+)',
            'itl_median': r'Median ITL \(ms\):\s*([\d.]+)',
            'itl_p99': r'P99 ITL \(ms\):\s*([\d.]+)',
            'e2el_mean': r'Mean E2EL \(ms\):\s*([\d.]+)',
            'e2el_median': r'Median E2EL \(ms\):\s*([\d.]+)',
            'e2el_p99': r'P99 E2EL \(ms\):\s*([\d.]+)',
            'request_throughput': r'Request throughput \(req/s\):\s*([\d.]+)',
            'output_token_throughput': r'Output token throughput \(tok/s\):\s*([\d.]+)',
            'total_token_throughput': r'Total Token throughput \(tok/s\):\s*([\d.]+)'
        }
        log_content = log_file.read_text()

        for key, pattern in patterns.items():
            match = re.search(pattern, log_content)
            metrics[key] = float(match.group(1)) if match else 0.0

        return metrics

