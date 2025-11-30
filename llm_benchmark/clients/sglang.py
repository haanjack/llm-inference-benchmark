import logging
import re
import subprocess
from pathlib import Path
from typing import Any
from typing import Dict

from llm_benchmark.clients.base import BenchmarkClientBase
from llm_benchmark.server.base import BenchmarkBase
from llm_benchmark.utils.script_generator import ScriptGenerator

logger = logging.getLogger(__name__)

SGLANG_IMAGE = "docker.io/rocm/sgl-dev:v0.5.5.post3-rocm700-mi30x-20251123"

class SGLangClient(BenchmarkClientBase):
    """Client for SGLang benchmarking."""

    def __init__(self, server: BenchmarkBase, is_dry_run: bool = False, script_generator: ScriptGenerator = None):
        super().__init__("sglang", server, is_dry_run, script_generator)

    def run_single_benchmark(self,
                             test_args: Dict[str, Any],
                             client_image: str = SGLANG_IMAGE,
                             **kwargs):
        """Run a single benchmark test."""
        request_rate = kwargs.get("request_rate")
        concurrency = kwargs.get("concurrency")
        input_length = kwargs.get("input_length")
        output_length = kwargs.get("output_length")
        num_prompts = kwargs.get("num_prompts")
        dataset_name = kwargs.get('dataset_name')

        use_script_vars = self.script_generator is not None

        # Use shell variables for script generation, otherwise use actual values.
        concurrency_val = f"${{CONCURRENCY}}" if use_script_vars else str(concurrency)
        num_prompts_val = f"${{NUM_PROMPTS}}" if use_script_vars else str(num_prompts)
        input_length_val = f"${{INPUT_LENGTH}}" if use_script_vars else str(input_length)
        output_length_val = f"${{OUTPUT_LENGTH}}" if use_script_vars else str(output_length)
        model_path_val = "$MODEL_PATH" if use_script_vars else self.server.get_model_path()
        request_rate_val = f"${{REQUEST_RATE}}" if use_script_vars else (str(request_rate) if request_rate > 0 else 'inf')

        cmd = []
        if not self.server.in_container:
            if self.server.name != "sglang" or self.server.addr != "0.0.0.0":
                cmd.extend([
                    self.server.container_runtime, "run", "--rm",
                    "--network=host",
                    "-v", f"{Path.cwd()}:{Path.cwd()}",
                    "-v", f"{self.server._model_path}:{model_path_val}",
                    client_image,
                ])
            else:
                cmd.extend([self.server.container_runtime, "exec", self.server.container_name])
        cmd.extend([
            "python3", "-m", "sglang.bench_serving",
            "--host", self.server.addr,
            "--port", str(self.server.port),
            "--backend", self.server.name,
            "--tokenizer", model_path_val,
            "--dataset-name", dataset_name,
            "--request-rate", request_rate_val,
            "--max-concurrency", concurrency_val,
            "--random-input-len", input_length_val,
            "--random-output-len", output_length_val,
            "--num-prompt", num_prompts_val,
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

        if self.script_generator:
            # In script generation mode, we return the command template.
            # The runner will handle the rest.
            return cmd

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
            'test_time_s': r'Benchmark duration \(s\):\s*([\d.]+)',
            'ttft_mean_ms': r'Mean TTFT \(ms\):\s*([\d.]+)',
            'ttft_median_ms': r'Median TTFT \(ms\):\s*([\d.]+)',
            'ttft_p99_ms': r'P99 TTFT \(ms\):\s*([\d.]+)',
            'tpot_mean_ms': r'Mean TPOT \(ms\):\s*([\d.]+)',
            'tpot_median_ms': r'Median TPOT \(ms\):\s*([\d.]+)',
            'tpot_p99_ms': r'P99 TPOT \(ms\):\s*([\d.]+)',
            'itl_mean_ms': r'Mean ITL \(ms\):\s*([\d.]+)',
            'itl_median_ms': r'Median ITL \(ms\):\s*([\d.]+)',
            'itl_p99_ms': r'P99 ITL \(ms\):\s*([\d.]+)',
            'e2el_mean_ms': r'Mean E2EL \(ms\):\s*([\d.]+)',
            'e2el_median_ms': r'Median E2EL \(ms\):\s*([\d.]+)',
            'e2el_p99_ms': r'P99 E2EL \(ms\):\s*([\d.]+)',
            'request_throughput_rps': r'Request throughput \(req/s\):\s*([\d.]+)',
            'output_token_throughput_tps': r'Output token throughput \(tok/s\):\s*([\d.]+)',
            'total_token_throughput_tps': r'Total Token throughput \(tok/s\):\s*([\d.]+)'
        }
        log_content = log_file.read_text()

        for key, pattern in patterns.items():
            match = re.search(pattern, log_content)
            metrics[key] = float(match.group(1)) if match else 0.0

        return metrics
