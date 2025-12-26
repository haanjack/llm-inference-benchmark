import logging
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional
import itertools
import yaml
from datetime import datetime

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
            ("Model Config", 16), ("TP", 8), ("Req Rate", 8), ("Num Prompts", 11), # 4
            ("Conc", 8), ("In Len", 8), ("Out Len", 8), # 7
            ("Test Time(s)", 10), ("TTFT Mean(ms)", 10), ("TTFT Med(ms)", 10), ("TTFT P99(ms)", 10), # 11
            ("TPOT Mean(ms)", 10), ("TPOT Med(ms)", 10), ("TPOT P99(ms)", 10), # 14
            ("ITL Mean(ms)", 10), ("ITL Med(ms)", 10), ("ITL P99(ms)", 10), # 17
            ("E2E Mean(ms)", 10), ("E2E Med(ms)", 10), ("E2E P99(ms)", 10), # 20
            ("Req req/s", 10), ("Out Tok/s", 10), ("Total Tok/s", 10) # 23
        ]
        self._csv_headers = [
            "tp_size", "request_rate", "num_prompts", "concurrency", # 5
            "input_length", "output_length", "test_time_s", "ttft_mean_ms", "ttft_median_ms", # 10
            "ttft_p99_ms", "tpot_mean_ms", "tpot_median_ms", "tpot_p99_ms", "itl_mean_ms", # 15
            "itl_median_ms", "itl_p99_ms", "e2el_mean_ms", "e2el_median_ms", "e2el_p99_ms", # 20
            "request_throughput_rps", "output_token_throughput_tps", "total_token_throughput_tps" # 23
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
        if not self.client.total_results_file.exists():
            with open(self.client.total_results_file, 'w', encoding='utf-8') as f:
                f.write('model_config,' + ','.join(self._csv_headers) + '\n')

        # Global "test set" dashboard (one row per BenchmarkRunner execution)
        self._global_dashboard_file = Path("logs") / "test_results.tsv"
        self._global_dashboard_file.parent.mkdir(parents=True, exist_ok=True)
        if not self._global_dashboard_file.exists():
            with open(self._global_dashboard_file, "w", encoding="utf-8") as f:
                f.write("\t".join([
                    "timestamp",
                    "model",
                    "image_tag",
                    "model_config",
                    "test_plan",
                    "sub_task",     # joined if multiple; empty if none
                    "result"        # total_results CSV path if success; otherwise "failure"
                ]) + "\n")

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

        # default benchmark dataset is random
        if 'dataset_name' not in test_args:
            test_args['dataset_name'] = 'random'

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
                'dataset_names': ensure_list(scenario.get('dataset_name', [test_args['dataset_name']]), [test_args['dataset_name']])
            }
            if for_script_gen:
                scenario_params_for_script_gen.append(params)
                continue

            # warn if mixing with random dataset and non-random dataset
            if test_args['dataset_name'] == 'random' and params['dataset_names'] != ['random']:
                logger.warning('Benchmark with non-random dataset might have issues with prefix-caching settings.')

            for rate, num_iter, num_prompts_val, in_len, out_len, conc, dataset_name in itertools.product(
                    params['request_rates'], params['num_iterations'],
                    params['num_prompts'], params['input_lengths'], params['output_lengths'],
                    params['concurrencies'], params['dataset_names']):
                num_prompts_final = conc * num_iter if 'num_iteration' in scenario else num_prompts_val
                test_plans.append({
                    'request_rate': rate, 'concurrency': conc, 'input_length': in_len,
                    'output_length': out_len, 'num_prompts': num_prompts_final,
                    'dataset_name': dataset_name
                })
        if for_script_gen:
            return test_args, scenario_params_for_script_gen

        if not test_plans:
            raise ValueError("No test scenarios loaded.")

        # Remove dataset_name from test_args to avoid duplication in benchmark clients
        # It's already included in each test_plan
        if 'dataset_name' in test_args:
            del test_args['dataset_name']

        return test_args, test_plans

    def run(self):
        test_args, test_plans = self._load_test_plan()

        if self.script_generator:
            # Get scenario information for script generation
            _, scenario_params = self._load_test_plan(for_script_gen=True)

            # Load scenarios with their names
            with open(self._test_plan_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            scenarios_with_names = []
            for scenario in config.get('test_scenarios', []):
                if self._sub_tasks and scenario.get('name') not in self._sub_tasks:
                    continue
                scenarios_with_names.append(scenario)

            # Generate separate script for each scenario
            for idx, scenario in enumerate(scenarios_with_names):
                scenario_name = scenario.get('name', f'scenario_{idx}')
                logger.info("Generating script for scenario: %s", scenario_name)

                # Create a new ScriptGenerator for this scenario with scenario name in filename
                base_path = self.script_generator.output_path
                scenario_script_path = base_path.parent / f"{base_path.stem}-{scenario_name}{base_path.suffix}"

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
            success = self._run_benchmark(test_args, test_plans)
        finally:
            self.server.cleanup()
            if not self._is_dry_run:
                logger.info("Benchmarking complete. Results saved to %s", self.client.results_file)

            self._write_test_set_dashboard_entry(
                    success=success,
                    result_path=self.client.total_results_file if success else None
                )

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
        """Run benchmark for all test plans."""
        success = True
        for test_plan in test_plans:
            try:
                metrics = self.client.run_single_benchmark(test_args, **test_plan)
                if metrics:
                    self._print_result(metrics=metrics, **test_plan)
                else:
                    success = False
                    break

                if self._is_dry_run:
                    if input("Continue? (Y/n) ").lower() in ['n', 'no']:
                        break

            except subprocess.CalledProcessError as e:
                success = False
                if not self._is_dry_run:
                    logger.error("Benchmark failed for plan: %s", test_plan)
                    # Format command as a simple string if available
                    cmd_str = ' '.join(e.cmd) if isinstance(e.cmd, list) else str(e.cmd)
                    logger.error("Command: %s", cmd_str)
                break
        return success

    def _format_result_for_console(self, values: List[str]) -> str:
        if len(values) != len(self._columns): # pyright: ignore
            logger.warning(f"Mismatch between result values and column definitions. {len(values)} values vs {len(self._columns)} columns.")
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

    # test dashboard helper
    def _write_test_set_dashboard_entry(self, success: bool, result_path: Optional[Path] = None):
        """Append a single row summarizing this test set execution."""
        if self._is_dry_run:
            return
        model = self.server.model_name
        image_tag = self.server.image_tag
        model_config = Path(self.server.model_config).stem
        test_plan = f"{self._test_plan}"
        sub_task_str = ""
        if self._sub_tasks:
            sub_task_str = "+".join(self._sub_tasks)
            test_plan += f"+{sub_task_str}"
        with open(self._global_dashboard_file, "a", encoding="utf-8") as f:
            f.write("\t".join([
                datetime.now().isoformat(timespec="seconds"),
                str(model),
                str(image_tag),
                str(model_config),
                test_plan,
                str(result_path) if success and result_path else "failure"
            ]) + "\n")
