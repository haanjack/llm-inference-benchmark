#!/usr/bin/env python3
"""
Generate comprehensive markdown reports from benchmark results.

This tool reads a run_list file, updates test_results.tsv, generates comparison plots,
and produces markdown reports with progress tables and embedded visualizations.

Workflow:
1. Update test_results.tsv by running progress report
2. Load data and group by model
3. Generate TP comparison and configuration comparison plots
4. Generate individual model reports with progress tables and plots
5. Generate master index report with summary
"""

import argparse
import logging
import shutil
import subprocess
import sys
import re
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict

import pandas as pd

# Add tools/plotter to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "plotter"))

try:
    from plot_benchmark_results import BenchmarkDataLoader, BenchmarkPlotter
except ImportError as e:
    print(f"Error: Could not import plotter modules. Make sure plot_benchmark_results.py exists.")
    print(f"Details: {e}")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MarkdownReportGenerator:
    """Generate markdown reports from benchmark results."""

    def __init__(self, logs_dir: Path, reports_dir: Path, throughput: str = 'tokens_per_sec_per_gpu',
                 latencies: str = 'e2e,itl,ttft,interactivity'):
        """Initialize report generator.

        Args:
            logs_dir: Path to logs directory
            reports_dir: Path to reports output directory
            throughput: Y-axis throughput metric
            latencies: Comma-separated latency metrics to plot
        """
        self.logs_dir = Path(logs_dir)
        self.reports_dir = Path(reports_dir)
        self.plots_dir = self.reports_dir / "plots"
        self.throughput = throughput
        self.latencies = [l.strip() for l in latencies.split(',')]

        # Ensure directories exist
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        self.data_loader = BenchmarkDataLoader(str(logs_dir))
        self.plotter = BenchmarkPlotter()
        self.data = None
        self.test_results = None

    def sanitize_model_name(self, model: str) -> str:
        """Sanitize model name for use in filenames.

        Args:
            model: Model name (e.g., 'amd/Llama-3.1-70B-Instruct-FP8-KV')

        Returns:
            Sanitized name (e.g., 'amd_Llama-3.1-70B-Instruct-FP8-KV')
        """
        return model.replace('/', '_')

    def sanitize_path(self, path: str) -> str:
        """Sanitize a path for use in filenames.

        Args:
            path: Path string (e.g., 'configs/models/llama-vllm.yaml')

        Returns:
            Sanitized name (e.g., 'configs_models_llama-vllm.yaml')
        """
        return path.replace('/', '_')

    def update_test_results(self, run_list_path: Path) -> pd.DataFrame:
        """Update test_results.tsv by running generate_progress_report.py.

        Args:
            run_list_path: Path to run_list file

        Returns:
            DataFrame with test results
        """
        logger.info("Updating test_results.tsv from run_list...")

        # Import here to avoid circular dependency
        sys.path.insert(0, str(Path(__file__).parent))
        from generate_progress_report import TestResultsGenerator

        try:
            generator = TestResultsGenerator(self.logs_dir)
            output_path = self.logs_dir / "test_results.tsv"
            count = generator.generate_report(run_list_path, output_path, generate_scripts=False)
            logger.info(f"✓ Updated test_results.tsv with {count} entries")

            # Copy to reports directory
            reports_test_results = self.reports_dir / "test_results.tsv"
            shutil.copy(output_path, reports_test_results)
            logger.info(f"✓ Copied test_results.tsv to {reports_test_results}")

            # Load and return the data
            return pd.read_csv(output_path, sep='\t')
        except Exception as e:
            logger.error(f"Error updating test results: {e}")
            raise

    def load_benchmark_data(self) -> pd.DataFrame:
        """Load benchmark data for plotting.

        Returns:
            DataFrame with benchmark data (only success entries)
        """
        logger.info("Loading benchmark data...")
        self.data = self.data_loader.load_data()

        if self.data is None or self.data.empty:
            logger.warning("No benchmark data loaded")
            return pd.DataFrame()

        logger.info(f"✓ Loaded {len(self.data)} benchmark data points")
        return self.data

    def get_models_from_test_results(self, test_results: pd.DataFrame) -> Set[str]:
        """Extract unique models from test_results.

        Args:
            test_results: DataFrame from test_results.tsv

        Returns:
            Set of unique model names
        """
        return set(test_results['model'].unique()) if 'model' in test_results.columns else set()

    def load_config_details(self, config_path: str) -> Dict:
        """Load configuration details from YAML file.

        Args:
            config_path: Path to config file (e.g., 'configs/models/kimi-k2-vllm-0.yaml')

        Returns:
            Dictionary with envs, server_args, and compilation_config sections
        """
        try:
            full_path = Path(config_path)
            if not full_path.exists():
                logger.warning(f"Config file not found: {config_path}")
                return {}

            with open(full_path, 'r') as f:
                config = yaml.safe_load(f)

            return {
                'envs': config.get('envs', {}),
                'server_args': config.get('server_args', {}),
                'compilation_config': config.get('compilation_config', {})
            }
        except Exception as e:
            logger.error(f"Error loading config {config_path}: {e}")
            return {}

    def format_config_value(self, value) -> str:
        """Format a config value for display in table.

        Args:
            value: Config value (can be dict, list, or scalar)

        Returns:
            Formatted string
        """
        if isinstance(value, dict):
            if not value:
                return "-"
            # Format as key: value pairs, but handle nested structures
            items = []
            for k, v in value.items():
                if isinstance(v, list):
                    if len(v) > 10:
                        items.append(f"{k}: [{len(v)} items]")
                    else:
                        items.append(f"{k}: {v}")
                elif isinstance(v, dict):
                    items.append(f"{k}: {{{len(v)} keys}}")
                else:
                    items.append(f"{k}: {v}")
            return "<br>".join(items[:10])  # Limit to first 10 items
        elif isinstance(value, list):
            if not value:
                return "-"
            if len(value) > 10:
                # Show first few and count
                return f"[{len(value)} items: {', '.join(map(str, value[:3]))}...]"
            return f"[{', '.join(map(str, value))}]"
        elif value is None or value == "":
            return "-"
        else:
            return str(value)

    def generate_tp_comparison_plots(self, model: str) -> Dict[Tuple, Optional[Path]]:
        """Generate TP comparison plots for a model.

        For each (image_tag, model_config, request_rate, input_length, output_length) combination,
        create a 1x4 plot comparing different TP sizes.
        Uses plot_tp_comparison_for_config which shows TP sizes as series across latency metrics.

        Args:
            model: Model name

        Returns:
            Dictionary mapping (tag, config, request_rate, isl, osl) to plot file path or None if no data
        """
        plots = {}

        if self.data is None or self.data.empty:
            return plots

        # Filter data for this model
        model_data = self.data[self.data['model'] == model].copy()

        if model_data.empty:
            logger.warning(f"No data found for model {model}")
            return plots

        # Ensure required columns exist
        if 'request_rate' not in model_data.columns:
            model_data['request_rate'] = 0
        if 'input_length' not in model_data.columns:
            model_data['input_length'] = 0
        if 'output_length' not in model_data.columns:
            model_data['output_length'] = 0

        # Get unique combinations of image_tag, model_config, request_rate, ISL, OSL
        grouping_cols = ['image_tag', 'model_config', 'request_rate', 'input_length', 'output_length']
        config_combinations = model_data[grouping_cols].drop_duplicates()

        model_dir = self.plots_dir / self.sanitize_model_name(model)
        model_dir.mkdir(parents=True, exist_ok=True)

        for _, row in config_combinations.iterrows():
            tag = row['image_tag']
            config = row['model_config']
            request_rate = row['request_rate']
            isl = row['input_length']
            osl = row['output_length']

            # Filter for this exact combination
            filtered = model_data[
                (model_data['image_tag'] == tag) &
                (model_data['model_config'] == config) &
                (model_data['request_rate'] == request_rate) &
                (model_data['input_length'] == isl) &
                (model_data['output_length'] == osl)
            ]

            if filtered.empty:
                continue

            # Generate TP comparison plot using plot_tp_comparison_for_config
            try:
                config_sanitized = self.sanitize_path(config)
                request_rate_str = f"rr{int(request_rate)}" if request_rate == int(request_rate) else f"rr{request_rate}"
                plot_filename = f"{self.sanitize_model_name(model)}_tp_compare_{tag}_{config_sanitized}_{request_rate_str}_isl{isl}_osl{osl}.png"
                plot_path = model_dir / plot_filename

                title = f"TP Comparison | {tag} | {config} | RR: {request_rate} | ISL: {isl} / OSL: {osl}"
                fig = self.plotter.plot_tp_comparison_for_config(
                    filtered,
                    title_prefix=title,
                    output_file=str(plot_path),
                    throughput=self.throughput,
                    latencies=self.latencies
                )
                plots[(tag, config, request_rate, isl, osl)] = plot_path
                logger.info(f"  ✓ Generated TP comparison plot: {plot_filename}")
            except Exception as e:
                logger.error(f"  ✗ Error generating TP comparison plot for {tag}/{config}/RR:{request_rate}/ISL:{isl}/OSL:{osl}: {e}")
                plots[(tag, config, request_rate, isl, osl)] = None

        return plots

    def generate_config_comparison_plots(self, model: str) -> Dict[Tuple, Optional[Path]]:
        """Generate config comparison plots for a model.

        For each (tp_size, request_rate, input_length, output_length) combination,
        create a 1x4 plot comparing different image_tags/configs.
        Uses plot_config_comparison_for_tp which shows configs as series across latency metrics.

        Args:
            model: Model name

        Returns:
            Dictionary mapping (tp_size, request_rate, isl, osl) to plot file path or None if no data
        """
        plots = {}

        if self.data is None or self.data.empty:
            return plots

        # Filter data for this model
        model_data = self.data[self.data['model'] == model].copy()

        if model_data.empty:
            return plots

        # Ensure required columns exist
        if 'request_rate' not in model_data.columns:
            model_data['request_rate'] = 0
        if 'input_length' not in model_data.columns:
            model_data['input_length'] = 0
        if 'output_length' not in model_data.columns:
            model_data['output_length'] = 0

        # Get unique combinations of tp_size, request_rate, ISL, OSL
        grouping_cols = ['tp_size', 'request_rate', 'input_length', 'output_length']
        tp_combinations = model_data[grouping_cols].drop_duplicates()

        model_dir = self.plots_dir / self.sanitize_model_name(model)
        model_dir.mkdir(parents=True, exist_ok=True)

        for _, row in tp_combinations.iterrows():
            tp_size = row['tp_size']
            request_rate = row['request_rate']
            isl = row['input_length']
            osl = row['output_length']

            # Filter for this exact combination
            filtered = model_data[
                (model_data['tp_size'] == tp_size) &
                (model_data['request_rate'] == request_rate) &
                (model_data['input_length'] == isl) &
                (model_data['output_length'] == osl)
            ]

            if filtered.empty:
                continue

            # Generate config comparison plot using plot_config_comparison_for_tp
            try:
                request_rate_str = f"rr{int(request_rate)}" if request_rate == int(request_rate) else f"rr{request_rate}"
                plot_filename = f"{self.sanitize_model_name(model)}_config_compare_tp{tp_size}_{request_rate_str}_isl{isl}_osl{osl}.png"
                plot_path = model_dir / plot_filename

                title = f"Configuration Comparison | TP: {tp_size} | RR: {request_rate} | ISL: {isl} / OSL: {osl}"
                fig = self.plotter.plot_config_comparison_for_tp(
                    filtered,
                    title_prefix=title,
                    output_file=str(plot_path),
                    throughput=self.throughput,
                    latencies=self.latencies
                )
                plots[(tp_size, request_rate, isl, osl)] = plot_path
                logger.info(f"  ✓ Generated config comparison plot: {plot_filename}")
            except Exception as e:
                logger.error(f"  ✗ Error generating config comparison plot for TP:{tp_size}/RR:{request_rate}/ISL:{isl}/OSL:{osl}: {e}")
                plots[(tp_size, request_rate, isl, osl)] = None

        return plots

    def generate_model_report(self, model: str, test_results: pd.DataFrame,
                             tp_plots: Dict[Tuple, Optional[Path]],
                             config_plots: Dict[Tuple, Optional[Path]]) -> Path:
        """Generate markdown report for a model.

        Args:
            model: Model name
            test_results: DataFrame with all test results
            tp_plots: Dictionary mapping (tag, config, request_rate, isl, osl) to plot paths
            config_plots: Dictionary mapping (tp_size, request_rate, isl, osl) to plot paths

        Returns:
            Path to generated report file
        """
        model_sanitized = self.sanitize_model_name(model)
        report_path = self.reports_dir / f"{model_sanitized}_report.md"

        # Filter test results for this model
        model_results = test_results[test_results['model'] == model]

        with open(report_path, 'w', encoding='utf-8') as f:
            # Header
            f.write(f"# Benchmark Report: {model}\n\n")

            # Progress Summary Table
            f.write("## Progress Summary\n\n")
            f.write("| model_config | image_tag | tp_size | test_plan | result |\n")
            f.write("|---|---|---|---|---|\n")

            for _, row in model_results.iterrows():
                config = row.get('model_config', 'N/A')
                tag = row.get('image_tag', 'N/A')
                tp = row.get('tp_size', 'N/A')
                plan = row.get('test_plan', 'N/A')
                result = row.get('result', 'N/A')

                # Color code results
                result_emoji = {
                    'success': '✓',
                    'incomplete': '⚠',
                    'not_tested': '◯',
                    'failure': '✗'
                }.get(result, '?')

                f.write(f"| {config} | {tag} | {tp} | {plan} | {result_emoji} {result} |\n")

            # Configuration Details Section
            f.write("\n## Configuration Details\n\n")

            # Get unique config files for this model
            unique_configs = sorted(model_results['model_config'].unique())

            if unique_configs:
                # Load all configs
                config_details = {}
                for config_path in unique_configs:
                    details = self.load_config_details(config_path)
                    if details:
                        config_details[config_path] = details

                if config_details:
                    # Create table header
                    f.write("| Config File | Environment Variables | Server Arguments | Compilation Config |\n")
                    f.write("|---|---|---|---|\n")

                    for config_path in unique_configs:
                        if config_path not in config_details:
                            continue

                        details = config_details[config_path]
                        config_name = config_path.replace('configs/models/', '')

                        # Format each section
                        envs_str = self.format_config_value(details.get('envs', {}))
                        server_args_str = self.format_config_value(details.get('server_args', {}))
                        comp_config_str = self.format_config_value(details.get('compilation_config', {}))

                        f.write(f"| `{config_name}` | {envs_str} | {server_args_str} | {comp_config_str} |\n")
                else:
                    f.write("_No configuration details available._\n\n")
            else:
                f.write("_No configurations found._\n\n")

            # TP Comparison Section
            if tp_plots:
                f.write("\n## TP Comparison Plots\n\n")
                f.write("Comparing different tensor parallel sizes for the same configuration setup.\n\n")

                # Group plots by (tag, config, request_rate, isl, osl) and sort
                for (tag, config, request_rate, isl, osl), plot_path in sorted(tp_plots.items()):
                    if plot_path is None:
                        continue

                    request_rate_str = f"RR: {int(request_rate)}" if request_rate == int(request_rate) else f"RR: {request_rate}"
                    f.write(f"### {tag} | {config} | {request_rate_str} | ISL: {isl} / OSL: {osl}\n\n")
                    rel_path = plot_path.relative_to(self.reports_dir)
                    f.write(f'<img src="{rel_path}" width="1200" alt="TP Comparison - {tag} - {config} - {request_rate_str} - ISL{isl}/OSL{osl}">\n\n')

            # Config Comparison Section
            if config_plots:
                f.write("\n## Configuration Comparison Plots\n\n")
                f.write("Comparing different configurations for the same setup (tp_size, request_rate, ISL, OSL).\n\n")

                for (tp_size, request_rate, isl, osl), plot_path in sorted(config_plots.items()):
                    if plot_path is None:
                        continue

                    request_rate_str = f"RR: {int(request_rate)}" if request_rate == int(request_rate) else f"RR: {request_rate}"
                    f.write(f"### TP: {tp_size} | {request_rate_str} | ISL: {isl} / OSL: {osl}\n\n")
                    rel_path = plot_path.relative_to(self.reports_dir)
                    f.write(f'<img src="{rel_path}" width="1200" alt="Config Comparison - TP{tp_size} - {request_rate_str} - ISL{isl}/OSL{osl}">\n\n')

        logger.info(f"✓ Generated report: {report_path}")
        return report_path

    def generate_index_report(self, test_results: pd.DataFrame, model_reports: Dict[str, Path]) -> Path:
        """Generate master index report.

        Args:
            test_results: DataFrame with all test results
            model_reports: Dictionary mapping model to report path

        Returns:
            Path to generated index file
        """
        index_path = self.reports_dir / "README.md"

        with open(index_path, 'w', encoding='utf-8') as f:
            f.write("# Benchmark Reports\n\n")
            f.write("Complete benchmark progress and analysis reports.\n\n")

            f.write("## Summary\n\n")
            f.write("| Model | Tests | Status | Report |\n")
            f.write("|-------|-------|--------|--------|\n")

            for model in sorted(model_reports.keys()):
                model_results = test_results[test_results['model'] == model]
                total = len(model_results)
                success = len(model_results[model_results['result'] == 'success'])
                failures = len(model_results[model_results['result'] == 'failure'])
                incomplete = len(model_results[model_results['result'] == 'incomplete'])

                # Determine status icon and text
                if success == total:
                    status_icon = "✓"
                    status_text = "Complete"
                elif failures > 0:
                    status_icon = "✗"
                    status_text = "Has Failures"
                else:
                    status_icon = "⚠"
                    status_text = "In Progress"

                # Create relative link to report
                report_path = model_reports[model]
                report_name = report_path.name

                f.write(f"| {model} | {success}/{total} | [{status_icon}] {status_text} | [View]({report_name}) |\n")

            f.write("\n## Detailed Reports\n\n")

            for model, report_path in sorted(model_reports.items()):
                report_name = report_path.name
                f.write(f"- [{model}]({report_name})\n")

        logger.info(f"✓ Generated index report: {index_path}")
        return index_path

    def generate_all_reports(self, run_list_path: Path) -> int:
        """Generate all reports.

        Args:
            run_list_path: Path to run_list file

        Returns:
            Number of models processed
        """
        logger.info("="*70)
        logger.info("MARKDOWN REPORT GENERATOR")
        logger.info("="*70)

        # Step 1: Update test results
        try:
            test_results = self.update_test_results(run_list_path)
        except Exception as e:
            logger.error(f"Failed to update test results: {e}")
            return 0

        if test_results.empty:
            logger.error("No test results to report")
            return 0

        # Step 2: Load benchmark data for plotting
        self.load_benchmark_data()

        # Step 3: Extract unique models
        models = sorted(self.get_models_from_test_results(test_results))

        if not models:
            logger.error("No models found in test results")
            return 0

        logger.info(f"Processing {len(models)} models...\n")

        model_reports = {}

        # Step 4: Generate reports for each model
        for model in models:
            logger.info(f"Processing model: {model}")

            # Generate plots
            logger.info("  Generating plots...")
            tp_plots = self.generate_tp_comparison_plots(model)
            config_plots = self.generate_config_comparison_plots(model)

            # Generate model report
            report_path = self.generate_model_report(model, test_results, tp_plots, config_plots)
            model_reports[model] = report_path

        # Step 5: Generate index report
        logger.info("Generating index report...")
        self.generate_index_report(test_results, model_reports)

        logger.info("="*70)
        logger.info(f"✓ Successfully generated reports for {len(models)} models")
        logger.info(f"  Output directory: {self.reports_dir}")
        logger.info("="*70)

        return len(models)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate comprehensive markdown reports from benchmark results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate reports from run_list
  python tools/reports/generate_markdown_report.py \\
    --run-list tests/run_list/run_list_example.sh

  # Custom output and plots directories
  python tools/reports/generate_markdown_report.py \\
    --run-list tests/run_list/run_list_example.sh \\
    --reports-dir my_reports \\
    --logs-dir logs

  # Custom plot metrics
  python tools/reports/generate_markdown_report.py \\
    --run-list tests/run_list/run_list_example.sh \\
    --throughput tokens_per_sec \\
    --latencies e2e,itl \\
    --verbose
        """
    )

    parser.add_argument(
        '--run-list',
        type=Path,
        required=True,
        help='Path to run_list file (e.g., tests/run_list/run_list_example.sh)'
    )

    parser.add_argument(
        '--reports-dir',
        type=Path,
        default=Path('reports'),
        help='Output directory for reports (default: reports)'
    )

    parser.add_argument(
        '--logs-dir',
        type=Path,
        default=Path('logs'),
        help='Path to logs directory containing benchmark results (default: logs)'
    )

    parser.add_argument(
        '--throughput',
        choices=[
            'tokens_per_sec_per_gpu',
            'tokens_per_sec',
            'output_tokens_per_sec_per_gpu',
            'output_tokens_per_sec'
        ],
        default='tokens_per_sec_per_gpu',
        help='Y-axis throughput metric (default: tokens_per_sec_per_gpu)'
    )

    parser.add_argument(
        '--latencies',
        default='e2e,itl,ttft,interactivity',
        help='Comma-separated latency metrics to plot (default: e2e,itl,ttft,interactivity)'
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        generator = MarkdownReportGenerator(
            args.logs_dir,
            args.reports_dir,
            throughput=args.throughput,
            latencies=args.latencies
        )

        count = generator.generate_all_reports(args.run_list)

        if count > 0:
            return 0
        else:
            return 1

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=args.verbose)
        return 1


if __name__ == '__main__':
    sys.exit(main())
