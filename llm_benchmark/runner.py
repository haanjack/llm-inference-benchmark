import logging
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Any
import itertools
import yaml

from llm_benchmark.server.base import BenchmarkBase
from llm_benchmark.clients.base import BenchmarkClientBase
from llm_benchmark.utils.script_generator import ScriptGenerator

logger = logging.getLogger(__name__)

class BenchmarkRunner:
    """Benchmark runner."""
    def __init__(self,
                server: BenchmarkBase,
                client: BenchmarkClientBase,
                test_plan: str,
                sub_tasks: List[str] = None,
                is_dry_run: bool = False,
                script_generator: ScriptGenerator = None):
        self.server = server
        self.client = client
        self._test_plan = test_plan
        self._sub_tasks = sub_tasks
        self._is_dry_run = server.is_dry_run or is_dry_run
        self.script_generator = script_generator
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
            "model_config", "tp_size", "request_rate", "num_prompts", "concurrency",
            "input_length", "output_length", "test_time_s", "ttft_mean_ms", "ttft_median_ms",
            "ttft_p99_ms", "tpot_mean_ms", "tpot_median_ms", "tpot_p99_ms", "itl_mean_ms",
            "itl_median_ms", "itl_p99_ms", "e2el_mean_ms", "e2el_median_ms", "e2el_p99_ms",
            "request_throughput_rps", "output_token_throughput_tps", "total_token_throughput_tps"
        ]
        self._setup_logging_dirs()
        if not self._test_plan_path.exists() and not self._is_dry_run:
            raise FileNotFoundError(f"Could not find test plan: {self._test_plan_path}.")

        self._print_benchmark_info()

    def _setup_logging_dirs(self):
        """Setup benchmark result logging directories."""
        self._log_dir = Path("logs") / self.server.model_name / self.server.image_tag

        if self._is_dry_run:
            return
        self.client.results_file.parent.mkdir(parents=True, exist_ok=True)

        # create result saving csv file
        if not self.client.results_file.exists():
            with open(self.client.results_file, 'w', encoding='utf-8') as f:
                f.write(','.join(self._csv_headers) + '\n')

    def _print_benchmark_info(self):
        logger.info("Benchmark plan: %s", self._test_plan)
        logger.info("Benchmark test plan:")
        try:
            with open(self._test_plan_path, mode="r", encoding="utf-8") as f:
                plan_content = f.read()
                indented_content = '\n'.join('    ' + line for line in plan_content.splitlines())
                logger.info("\n%s", indented_content)
        except Exception as e:
            logger.warning("Could not read test plan file '%s': %s", self._test_plan_path, e)

    def _load_test_plan(self, for_script_gen: bool = False):
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
        scenario_params_for_script_gen = []
        for scenario in config.get('test_scenarios', []):
            if self._sub_tasks and scenario.get('name') not in self._sub_tasks:
                continue
            if self._sub_tasks:
                logger.info("Sub task selected: %s", scenario.get('name'))

            if 'num_iteration' in scenario and 'num_prompts' in scenario:
                raise AssertionError("num_iteration and num_prompts are exclusive.")

            # These are the parameters we want to create loops for in the script
            params = {
                'request_rates': ensure_list(scenario.get('request_rate'), [0]),
                'concurrencies': ensure_list(scenario.get('concurrency'), [1]),
                'input_lengths': ensure_list(scenario.get('input_length'), [512]),
                'output_lengths': ensure_list(scenario.get('output_length'), [128]),
                'num_iterations': ensure_list(scenario.get('num_iteration'), [8 if 'num_prompts' not in scenario else 1]),
                'num_prompts': ensure_list(scenario.get('num_prompts'), [1000 if 'num_iteration' not in scenario else 1]),
            }
            if for_script_gen:
                scenario_params_for_script_gen.append(params)
                continue

            dataset_name_ = scenario.get('dataset_name', dataset_name)

            if dataset_name == 'random' and dataset_name_ != 'random':
                logger.warning('Benchmark with non-random dataset might have issues with prefix-caching settings.')

            for rate, num_iter, num_prompts_val, in_len, out_len, conc in itertools.product(
                    params['request_rates'], params['num_iterations'],
                    params['num_prompts'], params['input_lengths'], params['output_lengths'],
                    params['concurrencies']):
                num_prompts_final = conc * num_iter if 'num_iteration' in scenario else num_prompts_val
                test_plans.append({
                    'request_rate': rate, 'concurrency': conc, 'input_length': in_len,
                    'output_length': out_len, 'num_prompts': num_prompts_final,
                    'dataset_name': dataset_name_
                })
        if for_script_gen:
            return test_args, scenario_params_for_script_gen

        if not test_plans:
            raise ValueError("No test scenarios loaded.")

        return test_args, test_plans

    def run(self):
        test_args, test_plans = self._load_test_plan()

        if self.script_generator:
            # Get scenario information for script generation
            test_args_gen, scenario_params = self._load_test_plan(for_script_gen=True)

            # Load scenarios with their names
            with open(self._test_plan_path, 'r') as f:
                config = yaml.safe_load(f)

            scenarios_with_names = []
            for scenario in config.get('test_scenarios', []):
                if self._sub_tasks and scenario.get('name') not in self._sub_tasks:
                    continue
                scenarios_with_names.append(scenario)

            # Generate separate script for each scenario
            for idx, scenario in enumerate(scenarios_with_names):
                scenario_name = scenario.get('name', f'scenario_{idx}')
                logger.info(f"Generating script for scenario: {scenario_name}")

                # Create a new ScriptGenerator for this scenario with scenario name in filename
                base_path = self.script_generator.output_path
                scenario_script_path = base_path.parent / f"{base_path.stem}-{scenario_name}{base_path.suffix}"

                from llm_benchmark.utils.script_generator import ScriptGenerator
                scenario_generator = ScriptGenerator(
                    output_path=scenario_script_path,
                    in_container=self.script_generator.in_container
                )

                # Generate server command for this script
                self.server.generate_script(scenario_generator)

                # Get parameters for just this scenario
                params = scenario_params[idx]

                # Create loop parameters from this single scenario
                loop_params = {
                    'request_rates': params.get('request_rates', []),
                    'concurrencies': params.get('concurrencies', []),
                    'input_lengths': params.get('input_lengths', []),
                    'output_lengths': params.get('output_lengths', []),
                    'num_prompts': [],
                    'dataset_name': [scenario.get('dataset_name', 'random')]
                }

                # Calculate num_prompts from num_iterations and concurrencies
                num_iterations = params.get('num_iterations', [1])
                concurrencies = params.get('concurrencies', [1])
                num_prompts_raw = params.get('num_prompts', [1])

                if len(num_iterations) > 1 or num_iterations[0] != 1:
                    for conc in concurrencies:
                        for num_iter in num_iterations:
                            val = conc * num_iter
                            if val not in loop_params['num_prompts']:
                                loop_params['num_prompts'].append(val)
                else:
                    for val in num_prompts_raw:
                        if val not in loop_params['num_prompts']:
                            loop_params['num_prompts'].append(val)

                # Generate template client command using first test plan for this scenario
                command_template = None
                scenario_test_plans = [p for p in test_plans
                                     if (p.get('input_length') in loop_params['input_lengths'] and
                                         p.get('output_length') in loop_params['output_lengths'])]
                if scenario_test_plans:
                    command_template = self.client.run_single_benchmark(test_args, **scenario_test_plans[0])

                if command_template:
                    scenario_generator.set_client_loop(loop_params, command_template)

                scenario_generator.generate()

            logger.info("Script generation complete. Exiting.")
            return

        # Normal execution mode (not generating scripts)
        try:
            self._print_header()
            self._run_benchmark(test_args, test_plans)
        finally:
            self.server.cleanup()
            if not self._is_dry_run:
                logger.info("Benchmarking complete. Results saved to %s", self.client.results_file)

    def _print_header(self):
        """Print result's table header."""
        if self._is_dry_run:
            return

        header_line1 = []
        header_line2 = []
        for header, width in self._columns: # pyright: ignore
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

                if self._is_dry_run:
                    if input("Continue? (Y/n) ").lower() in ['n', 'no']:
                        break
            except subprocess.CalledProcessError as e:
                if not self._is_dry_run:
                    logger.error("Benchmark failed for plan: %s", test_plan)
                    logger.error("%s", str(e).rsplit('\n', maxsplit=1)[-1])
                    return

    def _format_result_for_console(self, values: List[str]) -> str:
        if len(values) != len(self._columns): # pyright: ignore
            logger.warning("Mismatch between result values and column definitions.")
            return ' '.join(values)
        formatted_values = [os.path.basename(values[0]).ljust(self._columns[0][1])] # pyright: ignore
        formatted_values.extend(val.rjust(width) for val, (_, width) in zip(values[1:], self._columns[1:])) # pyright: ignore
        return ' '.join(formatted_values)

    def _print_result(self, metrics: Dict[str, float], **kwargs): # pyright: ignore
        values = [
            Path(self.server.model_config).stem, str(self.server.parallel_size.get('tp', '1')),
            str(kwargs.get('request_rate')), str(kwargs.get('num_prompts')), str(kwargs.get('concurrency')), str(kwargs.get('input_length')), str(kwargs.get('output_length')), f"{metrics['test_time_s']:.2f}",
            f"{metrics['ttft_mean_ms']:.2f}", f"{metrics['ttft_median_ms']:.2f}", f"{metrics['ttft_p99_ms']:.2f}",
            f"{metrics['tpot_mean_ms']:.2f}", f"{metrics['tpot_median_ms']:.2f}", f"{metrics['tpot_p99_ms']:.2f}",
            f"{metrics['itl_mean_ms']:.2f}", f"{metrics['itl_median_ms']:.2f}", f"{metrics['itl_p99_ms']:.2f}",
            f"{metrics['e2el_mean_ms']:.2f}", f"{metrics['e2el_median_ms']:.2f}", f"{metrics['e2el_p99_ms']:.2f}",
            f"{metrics['request_throughput_rps']:.2f}", f"{metrics['output_token_throughput_tps']:.2f}", f"{metrics['total_token_throughput_tps']:.2f}"
        ]
        logger.info(self._format_result_for_console(values))
