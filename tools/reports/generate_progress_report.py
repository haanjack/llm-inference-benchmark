#!/usr/bin/env python3
"""
Generate test_results_generated.tsv from a run_list file.

This tool scans a run_list file and checks if each test has results in the logs,
allowing you to track benchmark progress without re-running tests.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import itertools
import yaml
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TestResultsGenerator:
    """Generate test_results_generated.tsv from a run_list file."""

    def __init__(self, logs_dir: Path):
        """Initialize the generator.

        Args:
            logs_dir: Root directory containing benchmark logs
        """
        self.logs_dir = Path(logs_dir)

    def parse_run_list_line(self, line: str) -> Optional[Dict]:
        """Parse a single line from run_list file.

        Format: backend docker_image model_path model_config test_plan benchmark_client gpu_devices [sub_task] [async]

        Args:
            line: A single line from run_list file

        Returns:
            Dictionary with parsed values or None if parsing fails
        """
        line = line.strip()

        # Skip comments and empty lines
        if not line or line.startswith('#'):
            return None

        parts = line.split()
        if len(parts) < 7:
            logger.warning(f"Invalid run_list line (too few columns): {line}")
            return None

        backend = parts[0]
        docker_image = parts[1]
        model_path = parts[2]
        model_config = parts[3]
        test_plan = parts[4]
        benchmark_client = parts[5]
        gpu_devices = parts[6]
        sub_task = parts[7] if len(parts) > 7 else ""

        # Extract image_tag from docker image (everything after last ':')
        if ':' in docker_image:
            image_tag = docker_image.split(':')[-1]
        else:
            image_tag = 'latest'

        # Extract tp_size from gpu_devices
        gpu_list = gpu_devices.split(',')
        tp_size = len(gpu_list)

        return {
            'backend': backend,
            'docker_image': docker_image,
            'model': model_path,
            'image_tag': image_tag,
            'model_config': model_config,
            'test_plan': test_plan,
            'client': benchmark_client,
            'gpu_devices': gpu_devices,
            'tp_size': tp_size,
            'sub_task': sub_task
        }

    def load_test_plan_yaml(self, test_plan_path: Path) -> Optional[List[Dict]]:
        """Load test plan yaml and get expected test scenarios.

        Args:
            test_plan_path: Path to the yaml file

        Returns:
            List of scenario dicts or None if file doesn't exist
        """
        if not test_plan_path.exists():
            logger.warning(f"Test plan file not found: {test_plan_path}")
            return None

        try:
            with open(test_plan_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            scenarios = config.get('test_scenarios', [])
            return scenarios
        except Exception as e:
            logger.error(f"Error loading test plan yaml {test_plan_path}: {e}")
            return None

    def count_expected_tests(self, scenarios: List[Dict]) -> int:
        """Count total expected test cases from scenarios.

        Args:
            scenarios: List of scenario configurations

        Returns:
            Total number of expected test combinations
        """
        def ensure_list(value, default):
            if value is None:
                return default
            if isinstance(value, (int, float)):
                return [value]
            return value

        total = 0
        for scenario in scenarios:
            request_rates = ensure_list(scenario.get('request_rate'), [0])
            concurrencies = ensure_list(scenario.get('concurrency'), [1])
            input_lengths = ensure_list(scenario.get('input_length'), [512])
            output_lengths = ensure_list(scenario.get('output_length'), [128])

            # Handle num_iteration vs num_prompts
            if 'num_iteration' in scenario:
                num_iterations = ensure_list(scenario.get('num_iteration'), [1])
                num_prompts_list = [1]  # Only 1 value when using iterations
            else:
                num_iterations = [1]  # Only 1 iteration when using num_prompts
                num_prompts_list = ensure_list(scenario.get('num_prompts'), [1000])

            dataset_names = ensure_list(scenario.get('dataset_name'), ['random'])

            # Count combinations
            count = len(request_rates) * len(concurrencies) * len(input_lengths) * \
                   len(output_lengths) * len(num_iterations) * len(num_prompts_list) * len(dataset_names)
            total += count

        return total

    def check_csv_results(self, csv_path: Path, expected_count: int, tp_size: int, model_config: str = None) -> Tuple[bool, int]:
        """Check if CSV file exists and has expected results for this TP size and model_config.

        Args:
            csv_path: Path to the CSV file
            expected_count: Expected number of test results
            tp_size: Tensor parallelism size to filter by
            model_config: Optional model configuration to filter by (e.g., configs/models/kimi-k2-vllm-0.yaml)

        Returns:
            Tuple of (has_all_results, actual_count)
        """
        if not csv_path.exists():
            return False, 0

        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            if len(lines) < 2:  # Only header or empty
                return False, 0

            # Parse header to find tp_size and model_config columns
            header = lines[0].strip().split(',')
            if 'tp_size' not in header:
                return False, 0

            tp_size_idx = header.index('tp_size')
            model_config_idx = header.index('model_config') if 'model_config' in header else None

            # Extract config basename if model_config provided
            config_basename = None
            if model_config and model_config_idx is not None:
                config_basename = Path(model_config).stem

            # Count lines matching this tp_size and model_config
            count = 0
            for line in lines[1:]:
                parts = line.strip().split(',')
                if len(parts) > tp_size_idx and parts[tp_size_idx].strip() == str(tp_size):
                    # If model_config filtering is needed, also check that
                    if config_basename and model_config_idx is not None:
                        if len(parts) > model_config_idx and parts[model_config_idx].strip() == config_basename:
                            count += 1
                    else:
                        count += 1

            return count >= expected_count, count
        except Exception as e:
            logger.error(f"Error reading CSV {csv_path}: {e}")
            return False, 0

    def get_csv_modification_time(self, csv_path: Path) -> str:
        """Get CSV file modification time as ISO format timestamp.

        Args:
            csv_path: Path to the CSV file

        Returns:
            ISO format timestamp string
        """
        if not csv_path.exists():
            return ""

        try:
            mtime = csv_path.stat().st_mtime
            dt = datetime.fromtimestamp(mtime)
            return dt.isoformat(timespec='seconds')
        except Exception as e:
            logger.error(f"Error getting mtime for {csv_path}: {e}")
            return ""

    def check_server_logs_for_errors(self, logs_base: Path, model: str, image_tag: str, model_config: str = None, tp_size: int = None) -> bool:
        """Check the most recent server log for critical error patterns.

        Args:
            logs_base: Base logs directory
            model: Model path (e.g., amd/Llama-3.1-70B-Instruct-FP8-KV)
            image_tag: Image tag (e.g., rocm7.0.0_vllm_0.11.2_20251210)
            model_config: Optional model config to search for specific configuration errors
            tp_size: Optional tp_size to search for specific configuration errors

        Returns:
            True if critical errors found, False otherwise
        """
        # Look for server_logs in any subdirectory under model/image_tag
        search_dir = logs_base / model / image_tag
        if not search_dir.exists():
            return False

        # More specific error patterns - looking for actual failures, not warnings
        error_patterns = [
            'oom', 'out of memory', 'out-of-memory',
            'cuda error', 'cublas error', 'triton error',
            'segmentation fault', 'segfault',
            'assertion error', 'assert failed',
            'exception', 'traceback',
            '[error]', 'error:', 'fatal:',  # case-insensitive patterns
            'killed', 'crashed', 'crashed:',
        ]

        try:
            # If model_config and tp_size provided, look for errors in specific configuration directory
            if model_config and tp_size:
                # Extract config base name (e.g., "kimi-k2-vllm-0" from "configs/models/kimi-k2-vllm-0.yaml")
                config_basename = Path(model_config).stem

                # Look for directory matching both config name and tp_size
                specific_dir = search_dir / f"{config_basename}-tp{tp_size}"

                if specific_dir.exists():
                    server_logs_dir = specific_dir / "server_logs"
                    if server_logs_dir.exists():
                        # Find the most recently created log file
                        log_files = list(server_logs_dir.glob("*.txt"))
                        if log_files:
                            # Get the most recent file by modification time
                            latest_log = max(log_files, key=lambda f: f.stat().st_mtime)

                            try:
                                content = latest_log.read_text(errors='ignore').lower()
                                for pattern in error_patterns:
                                    if pattern in content:
                                        logger.debug(f"Found error pattern '{pattern}' in {specific_dir.name} latest log {latest_log.name}")
                                        return True
                            except Exception as e:
                                logger.debug(f"Error reading log file {latest_log}: {e}")

                # No errors found in the specific configuration directory
                return False

            # Otherwise, look for most recent server_logs file recursively (backward compatibility)
            all_log_files = list(search_dir.rglob("server_logs/*.txt"))
            if not all_log_files:
                return False

            # Get the most recent file
            latest_log = max(all_log_files, key=lambda f: f.stat().st_mtime)

            try:
                content = latest_log.read_text(errors='ignore').lower()
                for pattern in error_patterns:
                    if pattern in content:
                        logger.debug(f"Found error pattern '{pattern}' in latest log {latest_log}")
                        return True
            except Exception as e:
                logger.debug(f"Error reading log file {latest_log}: {e}")
        except Exception as e:
            logger.debug(f"Error checking server logs in {search_dir}: {e}")

        return False

    def process_run_list(self, run_list_path: Path) -> List[Dict]:
        """Process run_list file and check status of each test.

        Args:
            run_list_path: Path to the run_list file

        Returns:
            List of result dictionaries for each test
        """
        if not run_list_path.exists():
            raise FileNotFoundError(f"Run list file not found: {run_list_path}")

        results = []

        with open(run_list_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                parsed = self.parse_run_list_line(line)
                if not parsed:
                    continue

                # Load test plan to get expected count
                test_plan_path = Path('configs/benchmark_plans') / f"{parsed['test_plan']}.yaml"
                scenarios = self.load_test_plan_yaml(test_plan_path)

                if not scenarios:
                    logger.warning(f"Could not load test plan for {parsed['test_plan']}")
                    expected_count = 0
                else:
                    expected_count = self.count_expected_tests(scenarios)

                # Check for CSV results filtered by tp_size and model_config
                csv_path = self.logs_dir / parsed['model'] / parsed['image_tag'] / \
                          f"total_results_{parsed['backend']}_{parsed['client']}.csv"

                has_all, actual_count = self.check_csv_results(csv_path, expected_count, parsed['tp_size'], parsed['model_config'])
                has_errors = self.check_server_logs_for_errors(self.logs_dir, parsed['model'], parsed['image_tag'],
                                                               parsed['model_config'], parsed['tp_size'])

                # Determine status
                if not csv_path.exists():
                    # CSV doesn't exist at all
                    if has_errors:
                        status = 'failure'
                    else:
                        status = 'not_tested'
                    log_path = 'N/A'
                    timestamp = 'N/A'
                elif has_all:
                    # CSV exists and has all results for this TP size
                    # Test completed successfully (has all expected results)
                    status = 'success'
                    log_path = str(csv_path)
                    timestamp = self.get_csv_modification_time(csv_path)
                else:
                    # CSV exists but incomplete results for this TP size
                    if has_errors:
                        status = 'failure'
                    else:
                        status = 'incomplete'
                    log_path = str(csv_path)
                    timestamp = self.get_csv_modification_time(csv_path)

                result = {
                    'timestamp': timestamp,
                    'model': parsed['model'],
                    'image_tag': parsed['image_tag'],
                    'model_config': parsed['model_config'],
                    'tp_size': parsed['tp_size'],
                    'test_plan': parsed['test_plan'],
                    'sub_task': parsed['sub_task'],
                    'result': status,
                    'log_path': log_path,
                    'expected_tests': expected_count,
                    'actual_tests': actual_count,
                    # Store original parsed data for shell script generation
                    'backend': parsed['backend'],
                    'docker_image': parsed['docker_image'],
                    'client': parsed['client'],
                    'gpu_devices': parsed['gpu_devices']
                }

                results.append(result)
                logger.debug(f"Line {line_num}: {parsed['model']} tp{parsed['tp_size']} -> {status} ({actual_count}/{expected_count})")

        return results

    def generate_report(self, run_list_path: Path, output_path: Path, generate_scripts: bool = False) -> int:
        """Generate test_results_generated.tsv file.

        Args:
            run_list_path: Path to the run_list file
            output_path: Path to output TSV file
            generate_scripts: Whether to generate incomplete.sh and not_tested.sh files

        Returns:
            Number of test entries processed
        """
        results = self.process_run_list(run_list_path)

        if not results:
            logger.warning("No test entries processed")
            return 0

        # Write TSV file
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            # Write header
            header = ['timestamp', 'model', 'image_tag', 'model_config', 'tp_size',
                     'test_plan', 'sub_task', 'result', 'log_path']
            f.write('\t'.join(header) + '\n')

            # Write results
            for result in results:
                row = [
                    result['timestamp'],
                    result['model'],
                    result['image_tag'],
                    result['model_config'],
                    str(result['tp_size']),
                    result['test_plan'],
                    result['sub_task'],
                    result['result'],
                    result['log_path']
                ]
                f.write('\t'.join(row) + '\n')

        logger.info(f"Generated report: {output_path} ({len(results)} entries)")

        # Generate run_list files for incomplete and not_tested entries if requested
        if generate_scripts:
            self.generate_run_lists(results, run_list_path)

        # Print summary
        success_count = sum(1 for r in results if r['result'] == 'success')
        incomplete_count = sum(1 for r in results if r['result'] == 'incomplete')
        not_tested_count = sum(1 for r in results if r['result'] == 'not_tested')
        failure_count = sum(1 for r in results if r['result'] == 'failure')

        logger.info(f"\nSummary:")
        logger.info(f"  Success:     {success_count}")
        logger.info(f"  Incomplete:  {incomplete_count}")
        logger.info(f"  Failure:     {failure_count}")
        logger.info(f"  Not tested:  {not_tested_count}")
        logger.info(f"  Total:       {len(results)}")

        return len(results)

    def generate_run_lists(self, results: List[Dict], run_list_path: Path) -> None:
        """Generate incomplete.sh and not_tested.sh run_list files from results.

        Args:
            results: List of result dictionaries
            run_list_path: Path to the original run_list file (to get directory)
        """
        run_list_dir = run_list_path.parent

        # Reconstruct lines for incomplete and not_tested entries
        incomplete_lines = []
        not_tested_lines = []

        for result in results:
            if result['result'] == 'incomplete':
                # Reconstruct the run_list line format
                line = f"{result['backend']} {result['docker_image']} {result['model']} {result['model_config']} " \
                       f"{result['test_plan']} {result['client']} {result['gpu_devices']}"
                if result['sub_task']:
                    line += f" {result['sub_task']}"
                incomplete_lines.append(line)

            elif result['result'] == 'not_tested':
                # Reconstruct the run_list line format
                line = f"{result['backend']} {result['docker_image']} {result['model']} {result['model_config']} " \
                       f"{result['test_plan']} {result['client']} {result['gpu_devices']}"
                if result['sub_task']:
                    line += f" {result['sub_task']}"
                not_tested_lines.append(line)

        # Write incomplete.sh
        if incomplete_lines:
            incomplete_path = run_list_dir / "incomplete.sh"
            with open(incomplete_path, 'w', encoding='utf-8') as f:
                f.write("#!/bin/bash\n")
                f.write("# Auto-generated incomplete tests - tests with partial results\n\n")
                for line in incomplete_lines:
                    f.write(line + "\n")

            import os
            os.chmod(incomplete_path, 0o755)
            logger.info(f"Generated incomplete.sh: {incomplete_path} ({len(incomplete_lines)} entries)")

        # Write not_tested.sh
        if not_tested_lines:
            not_tested_path = run_list_dir / "not_tested.sh"
            with open(not_tested_path, 'w', encoding='utf-8') as f:
                f.write("#!/bin/bash\n")
                f.write("# Auto-generated not tested - tests that have not been run yet\n\n")
                for line in not_tested_lines:
                    f.write(line + "\n")

            import os
            os.chmod(not_tested_path, 0o755)
            logger.info(f"Generated not_tested.sh: {not_tested_path} ({len(not_tested_lines)} entries)")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate test_results_generated.tsv from a run_list file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate report from tests/run_list/total.sh
  python tools/reports/generate_progress_report.py \\
    --run-list tests/run_list/total.sh

  # Generate report with run_list files for incomplete and not_tested
  python tools/reports/generate_progress_report.py \\
    --run-list tests/run_list/total.sh --generate-run-list

  # Generate report from custom run_list with custom output
  python tools/reports/generate_progress_report.py \\
    --run-list tests/run_list/custom.sh \\
    --output logs/my_results.tsv

  # Verbose output
  python tools/reports/generate_progress_report.py \\
    --run-list tests/run_list/total.sh --verbose
        """
    )

    parser.add_argument(
        '--run-list',
        type=Path,
        required=True,
        help='Path to the run_list file (e.g., tests/run_list/total.sh)'
    )

    parser.add_argument(
        '--logs-dir',
        type=Path,
        default=Path('logs'),
        help='Root directory containing benchmark logs (default: logs)'
    )

    parser.add_argument(
        '--output',
        type=Path,
        default=Path('logs/test_results.tsv'),
        help='Output file path (default: logs/test_results.tsv)'
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    parser.add_argument(
        '--generate-run-list',
        action='store_true',
        help='Generate incomplete.sh and not_tested.sh run_list files'
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        generator = TestResultsGenerator(args.logs_dir)
        count = generator.generate_report(args.run_list, args.output, args.generate_run_list)

        if count > 0:
            logger.info("Report generation completed successfully")
            return 0
        else:
            logger.error("Report generation failed")
            return 1

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=args.verbose)
        return 1


if __name__ == '__main__':
    sys.exit(main())
