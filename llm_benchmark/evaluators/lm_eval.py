"""LM-Eval-Harness evaluator implementation."""

import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional

from .base import BaseEvaluator

logger = logging.getLogger(__name__)


class LMEvalEvaluator(BaseEvaluator):
    """Evaluator using lm-eval-harness with per-task configuration."""

    SUPPORTED_TASKS = [
        "mmlu",
        "arc_challenge",
        "truthfulqa",
        "hellaswag",
        "gsm8k",
        "humaneval",
        "winogrande",
        "strategyqa",
    ]

    def __init__(
        self,
        model_name: str,
        endpoint: str,
        tasks: Dict[str, Dict[str, Any]],
        cache_dir: Optional[str] = None,
    ):
        """Initialize LMEvalEvaluator.

        Args:
            model_name: Model identifier
            endpoint: API endpoint URL
            tasks: Dict of task_name -> {num_fewshot, batch_size, timeout, etc.}
                   Example: {"mmlu": {"num_fewshot": 5, "batch_size": 8}}
            cache_dir: Optional cache directory for datasets (default: ~/.cache/huggingface)
        """
        # Extract task names from the config
        task_names = list(tasks.keys())
        super().__init__(model_name, endpoint, task_names)
        self.task_configs = tasks
        self.cache_dir = cache_dir

    def evaluate(self) -> Dict[str, Any]:
        """Run lm-eval-harness evaluation with per-task configuration."""
        results = {
            "model": self.model_name,
            "endpoint": self.endpoint,
            "evaluator": "lm-eval-harness",
            "tasks": {},
        }

        for task, config in self.task_configs.items():
            if task not in self.SUPPORTED_TASKS:
                logger.warning("Task %s not supported, skipping", task)
                continue

            logger.info("Running evaluation task: %s with config: %s", task, config)
            task_result = self._run_task(task, config)
            results["tasks"][task] = task_result

        return results

    def _run_task(self, task: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single evaluation task with specific configuration.

        Args:
            task: Task name
            config: Task configuration with num_fewshot, batch_size, timeout, etc.

        Returns:
            Task results dictionary
        """
        num_fewshot = config.get("num_fewshot", 0)
        batch_size = config.get("batch_size", 1)
        timeout = config.get("timeout", 3600)

        cmd = [
            "lm_eval",
            "--model",
            "openai-chat-completions",
            "--model-args",
            f"model={self.model_name},base_url={self.endpoint}",
            "--tasks",
            task,
            "--num-fewshot",
            str(num_fewshot),
            "--batch-size",
            str(batch_size),
            "--output-path",
            "/tmp/lm_eval_results",
            "--log-samples",
        ]

        # Add cache directory if specified
        if self.cache_dir:
            cmd.extend(["--cache-dir", self.cache_dir])

        try:
            logger.debug("Executing: %s", " ".join(cmd))
            logger.info("Dataset will be downloaded if not cached (may take time on first run)")

            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout, check=False
            )

            if result.returncode == 0:
                logger.info("Task %s completed successfully (fewshot=%s)", task, num_fewshot)
                return self._parse_results(task, config)
            else:
                # Check for specific error types
                error_type = self._classify_error(result.stderr)
                logger.error("Task %s failed with error type: %s", task, error_type)

                return {
                    "status": "failed",
                    "error_type": error_type,
                    "error": result.stderr,
                    "config": config,
                }

        except subprocess.TimeoutExpired:
            logger.error("Task %s timed out after %ss", task, timeout)
            logger.warning(
                "Timeout may be due to large dataset download. "
                "Consider increasing timeout or pre-downloading datasets."
            )
            return {
                "status": "timeout",
                "timeout_seconds": timeout,
                "config": config,
                "note": "May be caused by dataset download",
            }
        except Exception as e:
            logger.error("Task %s error: %s", task, str(e))
            return {"status": "error", "error": str(e), "config": config}

    def _classify_error(self, stderr: str) -> str:
        """Classify error type from stderr output.

        Args:
            stderr: Standard error output

        Returns:
            Error type classification
        """
        stderr_lower = stderr.lower()

        # Check for common error patterns
        if any(pattern in stderr_lower for pattern in [
            "connection", "network", "timeout", "timed out"
        ]):
            return "network_error"

        if any(pattern in stderr_lower for pattern in [
            "dataset", "download", "cache", "hub"
        ]):
            return "dataset_error"

        if any(pattern in stderr_lower for pattern in [
            "authentication", "token", "unauthorized"
        ]):
            return "auth_error"

        if any(pattern in stderr_lower for pattern in [
            "not found", "404", "no such file"
        ]):
            return "not_found_error"

        if any(pattern in stderr_lower for pattern in [
            "out of memory", "oom", "cuda out of memory"
        ]):
            return "oom_error"

        return "unknown_error"

    def _parse_results(self, task: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Parse evaluation results from output.

        Args:
            task: Task name
            config: Task configuration

        Returns:
            Parsed results with config included
        """
        results_file = Path("/tmp/lm_eval_results") / f"{task}_results.json"
        if results_file.exists():
            with open(results_file, encoding="utf-8") as f:
                parsed = json.load(f)
                # Add the config used for this task to results
                parsed["config"] = config
                parsed["status"] = "success"
                return parsed

        logger.warning("Results file not found for task %s: %s", task, results_file)
        return {
            "status": "no_results_file",
            "expected_path": str(results_file),
            "config": config,
        }

    def get_supported_tasks(self) -> List[str]:
        """Return list of supported evaluation tasks."""
        return self.SUPPORTED_TASKS
