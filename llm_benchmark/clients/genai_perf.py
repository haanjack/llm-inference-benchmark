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
from llm_benchmark.utils.script_generator import ScriptGenerator

logger = logging.getLogger(__name__)

GENAI_PERF_IMAGE = "nvcr.io/nvidia/tritonserver:25.10-py3-sdk"

class GenAIPerfClient(BenchmarkClientBase):
    """GenAI-Perf benchmark client."""

    def __init__(self, server: BenchmarkBase, is_dry_run: bool = False, script_generator: ScriptGenerator = None):
        super().__init__("genai-perf", server, is_dry_run, script_generator)

    def run_single_benchmark(self,
                             test_args: Dict[str, Any],
                             client_image: str = GENAI_PERF_IMAGE,
                             **kwargs):
        """Run a single benchmark test."""
        random_range_ratio = test_args.get('random_range_ratio', 0.0)


        request_rate = kwargs.get('request_rate')
        concurrency = kwargs.get('concurrency')
        input_length = kwargs.get('input_length')
        output_length = kwargs.get('output_length')
        num_prompts = kwargs.get('num_prompts')
        batch_size = kwargs.get('batch_size')
        dataset_name = kwargs.get('dataset_name')

        log_file = self._get_log_path(**kwargs)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        json_output_file_host = log_file.with_suffix('.json')
        json_output_file_container = json_output_file_host.name

        use_script_vars = self.script_generator is not None

        # Use shell variables for script generation, otherwise use actual values.
        concurrency_val = f"${{CONCURRENCY}}" if use_script_vars else str(concurrency)
        num_prompts_val = f"${{NUM_PROMPTS}}" if use_script_vars else str(num_prompts)
        input_length_val = f"${{INPUT_LENGTH}}" if use_script_vars else str(input_length)
        output_length_val = f"${{OUTPUT_LENGTH}}" if use_script_vars else str(output_length)
        model_path_val = "$MODEL_PATH" if use_script_vars else self.server.get_model_path()
        request_rate_val = f"${{REQUEST_RATE}}" if use_script_vars else (str(request_rate) if request_rate > 0 else 'inf')

        cmd = [
            self.server.container_runtime, "run", "--rm",
            "--network=host",
            "-v", f"{log_file.parent.resolve()}:/tmp/artifacts",
            "-v", f"{self.server._model_path}:{model_path_val}",
            client_image,
            "genai-perf", "profile",
            "-m", model_path_val,
            "--tokenizer", model_path_val,
            "--endpoint-type", "chat",
            "--url", f"{self.server.addr}:{self.server.port}",
            "--streaming",
            "--random-seed", "0",
            "--concurrency", concurrency_val,
            "--request-count", num_prompts_val,
            "--synthetic-input-tokens-mean", input_length_val,
            "--output-tokens-mean", output_length_val,
            "--extra-inputs", f"max_tokens:{input_length_val}",
            "--extra-inputs", f"min_tokens:{input_length_val}",
            "--profile-export-file", "profile_export.json",
            "--artifact-dir", "/tmp/artifacts",
            "--warmup-request-count", concurrency_val,
        ]

        if self.script_generator:
            # Return the command template for script generation
            return cmd

        if self._is_dry_run:
            logger.info("Dry run - Benchmark command: %s", " ".join(cmd))
            return None

        existing_results = self._check_existing_result(**kwargs)
        if existing_results:
            return existing_results

        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"=== Benchmark: {request_rate}, {num_prompts}, {batch_size}, {concurrency}, {input_length}, {output_length} ===\n")
            f.write(f"Command: {' '.join(cmd)}\n\n")
            f.flush()
            subprocess.run(cmd, stdout=f, stderr=f, check=True)

        # Extract metrics and save results
        result_file = self.server._log_dir / self.server.exp_tag / f"{self.server.get_model_path().replace('/', '_')}-openai-chat-concurrency{concurrency}" / "profile_export_genai_perf.json"
        metrics = self._extract_metrics(result_file)
        self._save_results(metrics, **kwargs)
        return metrics

    def _extract_metrics(self, log_file: Path) -> Dict[str, float]:
        # In genai-perf, log_file is the path to the JSON output file.
        if not log_file.exists():
            logger.error("GenAI-Perf output file not found: %s", log_file)
            return {}

        with open(log_file, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                logger.error("Failed to parse GenAI-Perf output file: %s", log_file)
                return {}

        # Mapping from genai-perf metric names to our internal names
        METRIC_MAPPING = {
            # TTFT
            'ttft_mean_ms': ('time_to_first_token', 'avg'),
            'ttft_median_ms': ('time_to_first_token', 'p50'),
            'ttft_p99_ms': ('time_to_first_token', 'p99'),

            # TPOT (Mapped from inter_token_latency)
            'tpot_mean_ms': ('inter_token_latency', 'avg'),
            'tpot_median_ms': ('inter_token_latency', 'p50'),
            'tpot_p99_ms': ('inter_token_latency', 'p99'),

            # ITL
            'itl_mean_ms': ('inter_token_latency', 'avg'),
            'itl_median_ms': ('inter_token_latency', 'p50'),
            'itl_p99_ms': ('inter_token_latency', 'p99'),

            # E2EL
            'e2el_mean_ms': ('request_latency', 'avg'),
            'e2el_median_ms': ('request_latency', 'p50'),
            'e2el_p99_ms': ('request_latency', 'p99'),

            # Throughput
            'request_throughput_rps': ('request_throughput', 'avg'),
            'output_token_throughput_tps': ('output_token_throughput', 'avg'),
        }

        # Extract metrics using Dictionary Comprehension (The concise part)
        metrics = {
            key: data[source][stat]
            for key, (source, stat) in METRIC_MAPPING.items()
        }
        metrics['test_time_s'] = data.get('duration', 0.0)

        # Calculate Total Throughput separately (Calculation logic)
        # Total = Output + (Request * Input_Avg_Len)
        metrics['total_token_throughput_tps'] = (
            metrics['output_token_throughput_tps'] +
            (metrics['request_throughput_rps'] * data['input_sequence_length']['avg'])
        )

        return metrics

    def parse_genai_perf_metrics(json_path: Path) -> Dict[str, float]:
        """
        Parses GenAI-Perf 'profile_export.json' and converts it to vLLM benchmark format.

        Args:
            json_path: Path to the GenAI-Perf export JSON file.

        Returns:
            Dict containing vLLM-style metrics (in ms for latencies).
        """
        if not json_path.exists():
            logger.error(f"GenAI-Perf result file not found: {json_path}")
            return {}

        try:
            with open(json_path, 'r') as f:
                data = json.load(f)

            # GenAI-Perf JSON structure usually contains 'experiments' list.
            # We assume there is one experiment per file or we take the first one.
            if "experiments" in data:
                stats = data["experiments"][0]["statistics"]
            else:
                # Fallback if structure is flat or different version
                stats = data

            # Helper to extract value safely (handling nested dicts if necessary)
            # GenAI-Perf metrics are often in nanoseconds (ns).
            # We need to convert to milliseconds (ms) for vLLM consistency.
            # ns_to_ms = 1e-6

            def get_stat(key: str, metric_type: str = "avg") -> float:
                # metric_type: avg, p99, min, max, etc.
                if key not in stats:
                    return 0.0
                return float(stats[key].get(metric_type, 0.0))

            # Latency Conversion Factor (GenAI-Perf creates output in nanoseconds)
            NS_TO_MS = 1.0 / 1_000_000.0

            metrics = {}

            # 1. Time To First Token (TTFT)
            metrics['ttft_mean_ms']   = get_stat("time_to_first_token", "avg") * NS_TO_MS
            metrics['ttft_median_ms'] = get_stat("time_to_first_token", "p50") * NS_TO_MS
            metrics['ttft_p99_ms']    = get_stat("time_to_first_token", "p99") * NS_TO_MS

            # 2. Inter-Token Latency (ITL) -> Map to TPOT/ITL
            # GenAI-Perf uses ITL. For vLLM comparison, we can map this to TPOT/ITL.
            metrics['tpot_mean_ms']   = get_stat("inter_token_latency", "avg") * NS_TO_MS
            metrics['tpot_median_ms'] = get_stat("inter_token_latency", "p50") * NS_TO_MS
            metrics['tpot_p99_ms']    = get_stat("inter_token_latency", "p99") * NS_TO_MS

            metrics['itl_mean_ms']    = metrics['tpot_mean_ms']
            metrics['itl_median_ms']  = metrics['tpot_median_ms']
            metrics['itl_p99_ms']     = metrics['tpot_p99_ms']

            # 3. End-to-End Latency (E2EL)
            metrics['e2el_mean_ms']   = get_stat("request_latency", "avg") * NS_TO_MS
            metrics['e2el_median_ms'] = get_stat("request_latency", "p50") * NS_TO_MS
            metrics['e2el_p99_ms']    = get_stat("request_latency", "p99") * NS_TO_MS

            # 4. Throughput (No conversion needed usually, assuming req/s and tok/s)
            # Sometimes GenAI-Perf puts these at the top level or calculates them differently.
            # If they are in the statistics block:
            metrics['request_throughput_rps'] = get_stat("request_throughput", "avg")
            metrics['output_token_throughput_tps'] = get_stat("output_token_throughput", "avg")

            # Total token throughput might need calculation if not explicitly provided
            # But strictly speaking, output_token_throughput is what vLLM usually compares in decoding speed.
            # If you need (Input + Output) / Time:
            # metrics['total_token_throughput'] = ... (requires input token count data)
            # For now, mapping output throughput is safer.
            metrics['total_token_throughput_tps'] = metrics['output_token_throughput_tps']

            # 5. Test Time (Duration)
            # Often represented as 'duration' in ns or s
            # If not present, can be inferred or passed from arguments
            metrics['test_time_s'] = 0.0

            return metrics

        except Exception as e:
            logger.error(f"Failed to parse GenAI-Perf metrics: {e}")
            return {}
