import logging
import os
import subprocess
import re
from pathlib import Path
from typing import Dict, Any, List

from llm_benchmark.clients.base import BenchmarkClientBase
from llm_benchmark.server.vllm import VLLMServer

logger = logging.getLogger(__name__)

class VLLMClient(BenchmarkClientBase):
    """vLLM benchmark client."""

    def __init__(self, server: VLLMServer, is_dry_run: bool = False):
        self.server = server
        self._is_dry_run = is_dry_run
        self._log_dir = Path("logs") / self.server.model_name / self.server.image_tag
        self.result_file = self._log_dir / "result_list.csv"

    def run_single_benchmark(self, test_args: Dict[str, Any], **kwargs):
        """Run a single benchmark test."""
        request_rate = kwargs.get('request_rate')
        concurrency = kwargs.get('concurrency')
        input_length = kwargs.get('input_length')
        output_length = kwargs.get('output_length')
        num_prompts = kwargs.get('num_prompts')
        batch_size = kwargs.get('batch_size')
        dataset_name = kwargs.get('dataset_name')

        # check vllm bench support dataset
        assert dataset_name in ['random', 'sharegpt', 'burstgpt', 'sonnet', 'random-mm', \
                                'rndom-rerank', 'hf', 'custom', 'prefix_repetition', 'spec_bench'], \
                                f"Dataset {dataset_name} is not supported by vLLM benchmark."

        cmd = []
        if not self.server.in_container:
            cmd.extend([self.server.container_runtime, "exec", self.server.container_name])
        cmd.extend([
            "vllm", "bench", "serve",
            "--model", self.server.get_model_path(),
            "--dataset-name", dataset_name,
            "--ignore-eos",
            "--trust-remote-code",
            f"--request-rate={request_rate if request_rate > 0 else 'inf'}",
            f"--max-concurrency={concurrency}",
            f"--num-prompts={num_prompts}",
            f"--random-input-len={input_length}",
            f"--random-output-len={output_length}",
            "--tokenizer", self.server.get_model_path(),
            "--disable-tqdm",
            "--percentile-metrics", "ttft,tpot,itl,e2el"
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

                if key == 'dataset_path':
                    cmd.extend(['--dataset-path', value])

        if self._is_dry_run:
            logger.info("Dry run - Benchmark command: %s", " ".join(cmd))
            return None

        if self._check_existing_result(request_rate, concurrency, input_length, output_length, num_prompts, batch_size):
            return None

        log_file = self._log_dir / self.server.exp_tag / f"r{request_rate}_n{num_prompts}_b{batch_size}_{input_length}_o{output_length}_c{concurrency}.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"=== Benchmark: request_rate: {request_rate}, num_prompts: {num_prompts}, batch_size, {batch_size}, concurrency: {concurrency}, isl: {input_length}, osl: {output_length} ===\n")
            f.write(f"Command: {' '.join(cmd)}\n\n")
            f.flush()

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
