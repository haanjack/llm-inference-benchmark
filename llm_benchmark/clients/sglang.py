import logging
import re
import subprocess
from pathlib import Path
from typing import Any
from typing import Dict, List

from llm_benchmark.clients.base import BenchmarkClientBase
from llm_benchmark.server.base import BenchmarkBase

logger = logging.getLogger(__name__)


class SGLangClient(BenchmarkClientBase):
    """Client for SGLang benchmarking."""

    def __init__(self, server: BenchmarkBase, is_dry_run: bool = False):
        super().__init__(server, is_dry_run)
        self.name = "sglang"
        self.server = server
        self._is_dry_run = is_dry_run
        self._log_dir = Path("logs") / self.server.model_name / self.server.image_tag
        self.result_file = self._log_dir / "result_list.csv"

    def run_single_benchmark(self, test_args: Dict[str, Any], **kwargs):
        """Run a single benchmark test."""
        request_rate = kwargs.get('request_rate')
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
            "--random-input-len", str(input_length),
            "--random-output-len", str(output_length),
            "--num-prompt", str(num_prompts),
            "--max-concurrency", str(concurrency),
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

        # TODO: Implement _check_existing_result if needed

        log_file = self._log_dir / self.server.exp_tag / f"r{request_rate}_n{num_prompts}_b{batch_size}_{input_length}_o{output_length}_c{concurrency}.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"=== Benchmark: request_rate: {request_rate}, num_prompts: {num_prompts}, batch_size, {batch_size}, concurrency: {concurrency}, isl: {input_length}, osl: {output_length} ===\n")
            f.write(f"Command: {' '.join(cmd)}\n\n")
            f.flush()

            subprocess.run(cmd, stdout=f, stderr=f, check=True)

        return self._extract_metrics(log_file)

    def _extract_metrics(self, log_file: Path) -> Dict[str, float]:
        """Parses the log file to extract performance metrics."""
        metrics = {}
        patterns = {
            'output_token_throughput': r'Output token throughput \(tok/s\):\s*([\d.]+)',
            'request_throughput': r'Request throughput \(req/s\):\s*([\d.]+)',
            'mean_latency': r'Mean latency \(s\):\s*([\d.]+)',
        }
        log_content = log_file.read_text()
        for key, pattern in patterns.items():
            match = re.search(pattern, log_content)
            metrics[key] = float(match.group(1)) if match else 0.0

        if not metrics:
            logger.warning("Could not extract any metrics from the sglang log file: %s", log_file)

        return metrics
