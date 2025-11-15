import logging
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Any
import itertools
import yaml

from llm_benchmark.server.vllm import VLLMServer
from llm_benchmark.clients.base import BenchmarkClientBase

logger = logging.getLogger(__name__)

class BenchmarkRunner:
    """Benchmark runner."""
    def __init__(self,
                server: VLLMServer,
                client: BenchmarkClientBase,
                test_plan: str,
                sub_tasks: List[str] = None,
                is_dry_run: bool = False):
        self.server = server
        self.client = client
        self._test_plan = test_plan
        self._sub_tasks = sub_tasks
        self._is_dry_run = server.is_dry_run or is_dry_run
        self._test_plan_path = Path(f"configs/benchmark_plans/{test_plan}.yaml")
        self._columns = [
            ("Model Config", 16), ("TP", 8), ("Req Rate", 8), ("Num Prompts", 11),
            ("Batch", 8), ("Conc", 8), ("In Len", 8), ("Out Len", 8),
            ("Test Time(s)", 10), ("TTFT Mean(ms)", 10), ("TTFT Med(ms)", 10), ("TTFT P99(ms)", 10),
            ("TPOT Mean(ms)", 10), ("TPOT Med(ms)", 10), ("TPOT P99(ms)", 10),
            ("ITL Mean(ms)", 10), ("ITL Med(ms)", 10), ("ITL P99(ms)", 10),
            ("E2E Mean(ms)", 10), ("E2E Med(ms)", 10), ("E2E P99(ms)", 10),
            ("Req req/s", 10), ("Out Tok/s", 10), ("Total Tok/s", 10)
        ]
        self._csv_headers = [
            "Model Config", "TP Size", "Request Rate", "Num. Prompts", "Batch Size", "Concurrency",
            "Input Length", "Output Length", "Test Time(s)", "Mean TTFT(ms)", "Median TTFT(ms)",
            "P99 TTFT(ms)", "Mean TPOT(ms)", "Median TPOT(ms)", "P99 TPOT(ms)", "Mean ITL(ms)",
            "Median ITL(ms)", "P99 ITL(ms)", "Mean E2EL(ms)", "Median E2EL(ms)", "P99 E2EL(ms)",
            "Request Tput(req/s)", "Output Tput(tok/s)", "Total Tput(tok/s)"
        ]
        self._setup_logging_dirs()
        if not self._test_plan_path.exists() and not self._is_dry_run:
            raise FileNotFoundError(f"Could not find test plan: {self._test_plan_path}.")

        self._print_benchmark_info()

    def _setup_logging_dirs(self):
        """Setup benchmark result logging directories."""
        self._log_dir = Path("logs") / self.server.model_name / self.server.image_tag
        self.result_file = self._log_dir / "result_list.csv"

        if self._is_dry_run:
            return
        self.result_file.parent.mkdir(parents=True, exist_ok=True)
        if not self.result_file.exists():
            with open(self.result_file, 'w', encoding='utf-8') as f:
                f.write(','.join(self._csv_headers) + '\n')

    def _print_benchmark_info(self):
        logger.info("Start vLLM benchmark")
        logger.info("Model Name: %s", self.server.model_name)
        logger.info("vLLM docker image: %s", self.server.vllm_image)
        logger.info("GPU devices: %s", self.server.gpu_devices)
        logger.info("Benchmark plan: %s", self._test_plan)
        logger.info("Benchmark test plan:")
        try:
            with open(self._test_plan_path, mode="r", encoding="utf-8") as f:
                plan_content = f.read()
                indented_content = '\n'.join('    ' + line for line in plan_content.splitlines())
                logger.info("\n%s", indented_content)
        except Exception as e:
            logger.warning("Could not read test plan file '%s': %s", self._test_plan_path, e)

    def _load_test_plan(self):
        with open(self._test_plan_path, mode="r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        dataset_name = config.get('dataset_name', 'random')

        def ensure_list(value, default):
            if value is None:
                return default
            if isinstance(value, (int, float)):
                return [value]
            return value

        # load benchmark arguments
        test_args = {}
        raw_test_args = config.get('test_args', [])
        # Validate test_args structure
        if raw_test_args is None:
            raw_test_args = []
        elif isinstance(raw_test_args, dict):
            raw_test_args = [raw_test_args]
        elif not isinstance(raw_test_args, list):
            raise ValueError(
                f"test_args must be a list of dictionaries, or a single dictionary. Got: {type(raw_test_args).__name__}"
            )
        for idx, loaded_args in enumerate(raw_test_args):
            if not isinstance(loaded_args, dict):
                raise ValueError(
                    f"Each item in test_args must be a dictionary. Item {idx} is of type {type(loaded_args).__name__}: {loaded_args}"
                )
            test_args.update(loaded_args)

        # benchmark sweep
        test_plans = []
        for scenario in config.get('test_scenarios', []):
            if self._sub_tasks and scenario.get('name') not in self._sub_tasks:
                continue
            if self._sub_tasks:
                logger.info("Sub task selected: %s", scenario.get('name'))

            if 'num_iteration' in scenario and 'num_prompts' in scenario:
                raise AssertionError("num_iteration and num_prompts are exclusive.")

            params = {
                'request_rates': ensure_list(scenario.get('request_rate'), [0]),
                'concurrencies': ensure_list(scenario.get('concurrency'), [1]),
                'input_lengths': ensure_list(scenario.get('input_length'), [512]),
                'output_lengths': ensure_list(scenario.get('output_length'), [128]),
                'num_iterations': ensure_list(scenario.get('num_iteration'), [8 if 'num_prompts' not in scenario else 1]),
                'num_prompts': ensure_list(scenario.get('num_prompts'), [1000 if 'num_iteration' not in scenario else 1]),
                'batch_sizes': ensure_list(scenario.get('batch_size'), [256])
            }
            dataset_name_ = scenario.get('dataset_name', dataset_name)
            if dataset_name == 'random' and dataset_name_ != 'random':
                logger.warning('Benchmark with non-random dataset with no-enable-prefix-caching.')

            for rate, batch, num_iter, num_prompts_val, in_len, out_len, conc in itertools.product(
                    params['request_rates'], params['batch_sizes'], params['num_iterations'],
                    params['num_prompts'], params['input_lengths'], params['output_lengths'],
                    params['concurrencies']):
                num_prompts_final = conc * num_iter if 'num_iteration' in scenario else num_prompts_val
                test_plans.append({
                    'request_rate': rate, 'concurrency': conc, 'input_length': in_len,
                    'output_length': out_len, 'num_prompts': num_prompts_final,
                    'batch_size': batch, 'dataset_name': dataset_name_
                })
        if not test_plans:
            raise ValueError("No test scenarios loaded.")

        return test_args, test_plans

    def run(self):
        if self.server.num_gpus == 0:
            raise ValueError("No GPU is allocated")

        test_args, test_plans = self._load_test_plan()

        try:
            self._print_header()
            self._run_benchmark(test_args, test_plans)
        finally:
            self.server.cleanup()
            if not self._is_dry_run and not self.server.in_container:
                logger.info("Benchmarking complete. Results saved to %s", self.result_file)

    def _print_header(self):
        """Print result's table header."""
        if self._is_dry_run:
            return

        header_line1 = []
        header_line2 = []
        for header, width in self._columns:
            parts = header.split(' ', 1)
            header_line1.append(parts[0].rjust(width))
            header_line2.append(parts[1].rjust(width) if len(parts) > 1 else ' '.rjust(width))
        logger.info(' '.join(header_line1))
        logger.info(' '.join(header_line2))

    def _run_benchmark(self, test_args: Dict[str, Any], test_plans: List[Dict]):
        for test_plan in test_plans:
            try:
                metrics = self.client.run_single_benchmark(test_args, **test_plan)
                if metrics:
                    self._print_result(metrics=metrics, **test_plan)
                    self._save_results(metrics=metrics, **test_plan)

                if self._is_dry_run:
                    if input("Continue? (Y/n) ").lower() in ['n', 'no']:
                        break
            except subprocess.CalledProcessError as e:
                logger.error("Benchmark failed for plan: %s", test_plan)
                logger.error("%s", str(e).rsplit('\n', maxsplit=1)[-1])
                return

    def _save_results(self, metrics: Dict[str, float], **kwargs):
        result_line = (
            f"{Path(self.server.model_config).stem},{self.server.parallel_size.get('tp', '1')},"
            f"{kwargs.get('request_rate')},{kwargs.get('num_prompts')},{kwargs.get('batch_size')},{kwargs.get('concurrency')},{kwargs.get('input_length')},{kwargs.get('output_length')},{metrics['test_time']:.2f},"
            f"{metrics['ttft_mean']:.2f},{metrics['ttft_median']:.2f},{metrics['ttft_p99']:.2f},"
            f"{metrics['tpot_mean']:.2f},{metrics['tpot_median']:.2f},{metrics['tpot_p99']:.2f},"
            f"{metrics['itl_mean']:.2f},{metrics['itl_median']:.2f},{metrics['itl_p99']:.2f},"
            f"{metrics['e2el_mean']:.2f},{metrics['e2el_median']:.2f},{metrics['e2el_p99']:.2f},"
            f"{metrics['request_throughput']:.2f},{metrics['output_token_throughput']:.2f},{metrics['total_token_throughput']:.2f}\n"
        )
        with open(self.result_file, 'a', encoding='utf-8') as f:
            f.write(result_line)

    def _format_result_for_console(self, values: List[str]) -> str:
        if len(values) != len(self._columns):
            logger.warning("Mismatch between result values and column definitions.")
            return ' '.join(values)
        formatted_values = [os.path.basename(values[0]).ljust(self._columns[0][1])]
        formatted_values.extend(val.rjust(width) for val, (_, width) in zip(values[1:], self._columns[1:]))
        return ' '.join(formatted_values)

    def _print_result(self, metrics: Dict[str, float], **kwargs):
        values = [
            Path(self.server.model_config).stem, str(self.server.parallel_size.get('tp', '1')),
            str(kwargs.get('request_rate')), str(kwargs.get('num_prompts')), str(kwargs.get('batch_size')), str(kwargs.get('concurrency')), str(kwargs.get('input_length')), str(kwargs.get('output_length')), f"{metrics['test_time']:.2f}",
            f"{metrics['ttft_mean']:.2f}", f"{metrics['ttft_median']:.2f}", f"{metrics['ttft_p99']:.2f}",
            f"{metrics['tpot_mean']:.2f}", f"{metrics['tpot_median']:.2f}", f"{metrics['tpot_p99']:.2f}",
            f"{metrics['itl_mean']:.2f}", f"{metrics['itl_median']:.2f}", f"{metrics['itl_p99']:.2f}",
            f"{metrics['e2el_mean']:.2f}", f"{metrics['e2el_median']:.2f}", f"{metrics['e2el_p99']:.2f}",
            f"{metrics['request_throughput']:.2f}", f"{metrics['output_token_throughput']:.2f}", f"{metrics['total_token_throughput']:.2f}"
        ]
        logger.info(self._format_result_for_console(values))
