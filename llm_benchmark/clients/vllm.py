import logging
import subprocess
import os
import re
from pathlib import Path
from typing import Dict, Any

from llm_benchmark.clients.base import BenchmarkClientBase
from llm_benchmark.server.vllm import BenchmarkBase
from llm_benchmark.utils.script_generator import ScriptGenerator

logger = logging.getLogger(__name__)

VLLM_IMAGE = "docker.io/rocm/vllm:rocm7.0.0_vllm_0.11.1_20251103"

class VLLMClient(BenchmarkClientBase):
    """vLLM benchmark client."""

    def __init__(self,
            server: BenchmarkBase,
            is_dry_run: bool = False,
            log_dir: Path = None,
            script_generator: ScriptGenerator = None):
        super().__init__("vllm", server, is_dry_run, log_dir, script_generator)

    def run_single_benchmark(self,
                             test_args: Dict[str, Any],
                             client_image: str = VLLM_IMAGE,
                             **kwargs):
        """Run a single benchmark test."""
        request_rate = kwargs.get('request_rate')
        concurrency = kwargs.get('concurrency')
        input_length = kwargs.get('input_length')
        output_length = kwargs.get('output_length')
        num_prompts = kwargs.get('num_prompts')
        dataset_name = kwargs.get('dataset_name')

        # check vllm bench support dataset
        assert dataset_name in ['random', 'sharegpt', 'burstgpt', 'sonnet', 'random-mm', \
                                'rndom-rerank', 'hf', 'custom', 'prefix_repetition', 'spec_bench'], \
                                f"Dataset {dataset_name} is not supported by vLLM benchmark."

        use_script_vars = self.script_generator is not None

        # Use shell variables for script generation, otherwise use actual values.
        concurrency_val = f"${{CONCURRENCY}}" if use_script_vars else str(concurrency) # pylint: disable=f-string-without-interpolation
        num_prompts_val = f"${{NUM_PROMPTS}}" if use_script_vars else str(num_prompts) # pylint: disable=f-string-without-interpolation
        input_length_val = f"${{INPUT_LENGTH}}" if use_script_vars else str(input_length) # pylint: disable=f-string-without-interpolation
        output_length_val = f"${{OUTPUT_LENGTH}}" if use_script_vars else str(output_length) # pylint: disable=f-string-without-interpolation
        model_path_val = "$MODEL_PATH" if use_script_vars else self.server.get_model_path()
        request_rate_val = f"${{REQUEST_RATE}}" if use_script_vars else (str(request_rate) if request_rate > 0 else 'inf') # pylint: disable=f-string-without-interpolation
        dataset_name_val = f"${{DATASET_NAME}}" if use_script_vars else dataset_name # pylint: disable=f-string-without-interpolation

        cmd = []
        if not self.server.in_container:
            if self.server.addr != "0.0.0.0" or self.server.name != "vllm":
                group_option = "keep-groups" if os.environ.get("SLURM_JOB_ID", None) else "video"
                cmd.extend([
                    self.server.container_runtime, "run", "--rm",
                    "--device", "/dev/kfd", "--device", "/dev/dri", "--device", "/dev/mem",
                    "--group-add", group_option,
                    "--cap-add=CAP_SYS_ADMIN",
                    "--cap-add=SYS_PTRACE",
                    "--network=host",
                    "-v", f"{Path.cwd()}:{Path.cwd()}",
                    "-v", f"{self.server.get_host_model_path()}:{model_path_val}",
                    client_image,
                ])
            else:
                container_name_val = "$CONTAINER_NAME" if use_script_vars else self.server.container_name
                cmd.extend([self.server.container_runtime, "exec"])
                # Environment variables are already set in the container from docker run
                # No need to pass them again with -e flags
                cmd.append(container_name_val)
        cmd.extend([
            "vllm", "bench", "serve",
            "--model", model_path_val,
            "--backend", "openai",
            "--host", self.server.addr, "--port", str(self.server.port),
            "--dataset-name", dataset_name_val,
            "--ignore-eos",
            "--trust-remote-code",
            "--request-rate", request_rate_val,
            "--max-concurrency", concurrency_val,
            "--num-prompts", num_prompts_val,
            "--random-input-len", input_length_val,
            "--random-output-len", output_length_val,
            "--tokenizer", model_path_val,
            "--disable-tqdm",
            "--percentile-metrics", "ttft,tpot,itl,e2el",
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

        if self.script_generator:
            # In script generation mode, return the command template for the loop
            return cmd

        if self._is_dry_run:
            logger.info("Dry run - Benchmark command: %s", " ".join(cmd))
            return None

        # Check for existing results to avoid redundant runs
        existing_results = self._check_existing_result(**kwargs)
        if existing_results:
            return existing_results

        # Run the benchmark and log output
        log_file = self._log_dir / self.server.exp_tag / f"r{request_rate}_n{num_prompts}_{input_length}_o{output_length}_c{concurrency}.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"=== Benchmark: request_rate: {request_rate}, num_prompts: {num_prompts}, concurrency: {concurrency}, isl: {input_length}, osl: {output_length} ===\n")
            f.write(f"Command: {' '.join(cmd)}\n\n")
            f.flush()

            subprocess.run(cmd, stdout=f, stderr=f, check=True)

        # Extract metrics and save results
        metrics = self._extract_metrics(log_file)
        self._save_results(metrics, **kwargs)
        return metrics

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
