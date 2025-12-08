"""Evaluation client for model quality/correctness validation."""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from llm_benchmark.clients.base import BenchmarkClientBase
from llm_benchmark.evaluators import EvaluationRunner
from llm_benchmark.server import BenchmarkBase
from llm_benchmark.utils.script_generator import ScriptGenerator

logger = logging.getLogger(__name__)


class EvaluationClient(BenchmarkClientBase):
    """Evaluation client for quality/correctness metrics using lm-eval-harness.

    This client runs after performance benchmarks to validate model outputs
    and measure accuracy, reasoning capability, and other quality metrics.

    Requirements:
        - lm-eval-harness must be installed: pip install lm-eval>=0.4.0
    """

    def __init__(
        self,
        server: BenchmarkBase,
        is_dry_run: bool = False,
        script_generator: Optional[ScriptGenerator] = None,
        evaluation_plan: str = "default",
        cache_dir: Optional[str] = None,
    ):
        """Initialize evaluation client.

        Args:
            server: Server instance to evaluate
            is_dry_run: If True, only show commands without executing
            script_generator: Optional script generator for generating bash scripts
            evaluation_plan: Name of evaluation plan YAML file (without .yaml)
            cache_dir: Optional cache directory for datasets
        """
        super().__init__(
            name="evaluation",
            server=server,
            is_dry_run=is_dry_run,
            script_generator=script_generator
        )
        self.evaluation_plan = evaluation_plan
        self.cache_dir = cache_dir
        self.evaluator_runner = None        # Override log paths for evaluation results
        self._results_file = self._log_dir / self.server.exp_tag / \
            f"evaluation_{Path(self.server.model_config).stem}_{self.evaluation_plan}.json"

    def run_single_benchmark(self, test_args: Dict[str, Any], **kwargs):
        """Run evaluation tasks - not used for evaluation client.

        Evaluation client doesn't run individual benchmark tests,
        it runs a comprehensive evaluation suite.
        """
        logger.warning("run_single_benchmark not supported for EvaluationClient")
        return {}

    def _extract_metrics(self, log_file: Path) -> Dict[str, float]:
        """Extract metrics from evaluation results - not used for JSON results."""
        logger.warning("_extract_metrics not used for EvaluationClient (uses JSON)")
        return {}

    def run(self) -> Dict[str, Any]:
        """Run model evaluation tasks.

        Returns:
            Dictionary with evaluation results

        Raises:
            RuntimeError: If lm-eval-harness is not installed
        """
        logger.info("=" * 60)
        logger.info("Starting Model Evaluation")
        logger.info("Plan: %s", self.evaluation_plan)
        if self.cache_dir:
            logger.info("Cache Directory: %s", self.cache_dir)
        logger.info("=" * 60)

        if self._is_dry_run:
            logger.info("[DRY RUN] Would run evaluation with plan: %s", self.evaluation_plan)
            return {
                "status": "dry_run",
                "evaluation_plan": self.evaluation_plan,
                "model": self.server.model_name
            }

        try:
            endpoint = self.server.get_endpoint()
            model_name = self.server.model_name

            logger.info("Evaluation Endpoint: %s", endpoint)
            logger.info("Model: %s", model_name)

            # EvaluationRunner will check if lm_eval is installed
            self.evaluator_runner = EvaluationRunner(
                model_name=model_name,
                endpoint=endpoint,
                evaluation_plan=self.evaluation_plan,
                cache_dir=self.cache_dir,
            )

            results = self.evaluator_runner.run()

            # Save evaluation results
            self._save_results(results)

            logger.info("=" * 60)
            logger.info("Evaluation Completed")
            self._print_summary(results)
            logger.info("Results saved to: %s", self._results_file)
            logger.info("=" * 60)

            return results

        except RuntimeError as e:
            # Handle lm-eval not installed error
            logger.error("=" * 60)
            logger.error("EVALUATION FAILED: %s", str(e))
            logger.error("=" * 60)
            return {
                "status": "error",
                "error_type": "installation_error",
                "error": str(e),
                "evaluation_plan": self.evaluation_plan,
                "model": self.server.model_name
            }
        except Exception as e:
            logger.error("Evaluation failed: %s", str(e), exc_info=True)
            return {
                "status": "error",
                "error": str(e),
                "evaluation_plan": self.evaluation_plan,
                "model": self.server.model_name
            }

    def _save_results(self, metrics: Dict[str, Any], **kwargs):
        """Save evaluation results to JSON file.

        Args:
            metrics: Evaluation results dictionary (overrides base class signature)
            **kwargs: Additional arguments (unused for evaluation client)
        """
        # For evaluation, we save complete results as JSON, not CSV
        results = metrics
        self._results_file.parent.mkdir(parents=True, exist_ok=True)

        with open(self._results_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        logger.info("Evaluation results saved to %s", self._results_file)

    def _print_summary(self, results: Dict[str, Any]):
        """Print evaluation summary.

        Args:
            results: Evaluation results dictionary
        """
        if "evaluation_results" not in results:
            logger.warning("No evaluation results found")
            return

        for eval_result in results["evaluation_results"]:
            tasks = eval_result.get("tasks", {})
            logger.info("Tasks Evaluated: %d", len(tasks))

            success_count = sum(
                1 for task_result in tasks.values()
                if task_result.get("status") == "success"
            )
            failed_count = sum(
                1 for task_result in tasks.values()
                if task_result.get("status") == "failed"
            )
            timeout_count = sum(
                1 for task_result in tasks.values()
                if task_result.get("status") == "timeout"
            )

            logger.info("  Success: %d", success_count)
            logger.info("  Failed: %d", failed_count)
            logger.info("  Timeout: %d", timeout_count)

            # Show failed tasks with error types
            if failed_count > 0:
                logger.warning("Failed Tasks:")
                for task_name, task_result in tasks.items():
                    if task_result.get("status") == "failed":
                        error_type = task_result.get("error_type", "unknown")
                        logger.warning("  - %s: %s", task_name, error_type)