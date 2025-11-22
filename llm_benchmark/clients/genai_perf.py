import logging
import os
import subprocess
import re
from pathlib import Path
from typing import Dict, Any, List
import tempfile
import json

from llm_benchmark.clients.base import BenchmarkClientBase
from llm_benchmark.server.base import BenchmarkBase

logger = logging.getLogger(__name__)


class GenAIPerfClient(BenchmarkClientBase):
    """GenAI-Perf benchmark client."""

    def __init__(self, server: BenchmarkBase, is_dry_run: bool = False):
        super().__init__("genai-perf", server, is_dry_run)

    def run_single_benchmark(self, test_args: Dict[str, Any], **kwargs):
        """Run a single benchmark test."""
        random_range_ratio = test_args.get('random_range_ratio', 0.0)

        triton_image = "nvcr.io/nvidia/tritonserver:25.10-py3-sdk"
        request_rate = kwargs.get('request_rate')
        concurrency = kwargs.get('concurrency')
        input_length = kwargs.get('input_length')
        output_length = kwargs.get('output_length')
        num_prompts = kwargs.get('num_prompts')
        batch_size = kwargs.get('batch_size')
        dataset_name = kwargs.get('dataset_name')

        dataset_args = []
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            if dataset_name == "random":
                  dataset_args.append(f'--synthetic-input-tokens-mean={input_length}')
                  dataset_args.append(f'--extra-inputs=max_tokens:{int(input_length * random_range_ratio)}')
                  dataset_args.append(f'--extra-inputs=min_tokens:{int(input_length * random_range_ratio)}')
            else:
                raise NotImplementedError(f"Dataset {dataset_name} is not yet implemented for GenAIPerfClient")

        cmd = [
            self.server.container_runtime, "run", "--rm",
            "--network=host",
            triton_image,
            "genai-perf", "profile",
            "-m", self.server.model_name,
            "--tokenizer", self.server.model_name,
            "-i", "http",
            "--service-kind", "openai",
            "--concurrency", f"{concurrency}",
            "--request-rate", f"{request_rate}",
            "--request-count", f"{num_prompts}",
            "--output-format", "json",
            "--profile-export-file", "/tmp/profile_export.json",
            "--streaming",
            "--random-seed", "0",
            "--endpoint-type", "chat",
            "--endpoint", f"http://localhost:{self.server.port}/v1/completions"
        ]
        cmd.extend(dataset_args)

        if self._is_dry_run:
            logger.info("Dry run - Benchmark command: %s", " ".join(cmd))
            return None

        existing_results = self._check_existing_result(**kwargs)
        if existing_results:
            return existing_results

        log_file = self._log_dir / self.server.exp_tag / f"r{request_rate}_n{num_prompts}_b{batch_size}_{input_length}_o{output_length}_c{concurrency}.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"=== Benchmark: {request_rate}, {num_prompts}, {batch_size}, {concurrency}, {input_length}, {output_length} ===\n")
            f.write(f"Command: {' '.join(cmd)}\n\n")
            f.flush()
            subprocess.run(cmd, stdout=f, stderr=f, check=True)

        return self._extract_metrics(log_file)

    def _extract_metrics(self, log_file: Path) -> Dict[str, float]:
        # genai-perf outputs metrics to the profile export file, not stdout.
        # This is a placeholder for now.
        return {}
