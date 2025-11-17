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
        self.server = server
        self._is_dry_run = is_dry_run
        self._log_dir = Path("logs") / self.server.model_name / self.server.image_tag
        self.result_file = self._log_dir / "result_list.csv"

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

        if self._check_existing_result(request_rate, concurrency, input_length, output_length, num_prompts, batch_size):
            return None

        log_file = self._log_dir / self.server.exp_tag / f"r{request_rate}_n{num_prompts}_b{batch_size}_{input_length}_o{output_length}_c{concurrency}.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"=== Benchmark: {request_rate}, {num_prompts}, {batch_size}, {concurrency}, {input_length}, {output_length} ===\n")
            subprocess.run(cmd, stdout=f, stderr=f, check=True)

        return self._extract_metrics(log_file)

    def _check_existing_result(self, request_rate, concurrency, input_length, output_length, num_prompts, batch_size) -> bool:
        if not self.result_file.exists() or self._is_dry_run:
            return False
        search_str = f"{Path(self.server.model_config).stem},{self.server.parallel_size.get('tp', '1')},{request_rate},{num_prompts},{batch_size},{concurrency},{input_length},{output_length}"
        with open(self.result_file, 'r', encoding='utf-8') as f:
            for line in f:
                if search_str in line:
                    logger.info(self._format_result_for_console(line.strip().split(',')))
                    return True
        return False

    def _extract_metrics(self, log_file: Path) -> Dict[str, float]:
        # genai-perf outputs metrics to the profile export file, not stdout.
        # This is a placeholder for now.
        return {}

    def _format_result_for_console(self, values: List[str]) -> str:
        columns = [
            ("Model Config", 16), ("TP", 8), ("Req Rate", 8), ("Num Prompts", 11),
            ("Batch", 8), ("Conc", 8), ("In Len", 8), ("Out Len", 8),
            ("Test Time(s)", 10), ("TTFT Mean(ms)", 10), ("TTFT Med(ms)", 10), ("TTFT P99(ms)", 10),
            ("TPOT Mean(ms)", 10), ("TPOT Med(ms)", 10), ("TPOT P99(ms)", 10),
            ("ITL Mean(ms)", 10), ("ITL Med(ms)", 10), ("ITL P99(ms)", 10),
            ("E2E Mean(ms)", 10), ("E2E Med(ms)", 10), ("E2E P99(ms)", 10),
            ("Req req/s", 10), ("Out Tok/s", 10), ("Total Tok/s", 10)
        ]
        if len(values) != len(columns):
            logger.warning("Mismatch between result values and column definitions.")
            return ' '.join(values)
        formatted_values = [os.path.basename(values[0]).ljust(columns[0][1])]
        formatted_values.extend(val.rjust(width) for val, (_, width) in zip(values[1:], columns[1:]))
        return ' '.join(formatted_values)
