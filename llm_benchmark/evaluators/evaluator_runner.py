"""Evaluation runner orchestrator."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .base import BaseEvaluator
from .lm_eval import LMEvalEvaluator

logger = logging.getLogger(__name__)


class EvaluationRunner:
    """Orchestrates model evaluation with task-specific configurations."""

    def __init__(
        self,
        model_name: str,
        endpoint: str,
        evaluation_plan: str = "default",
        cache_dir: Optional[str] = None,
    ):
        """Initialize evaluation runner.

        Args:
            model_name: Model identifier
            endpoint: API endpoint URL
            evaluation_plan: Name of evaluation plan YAML file (without .yaml)
            cache_dir: Optional cache directory for datasets
        """
        self.model_name = model_name
        self.endpoint = endpoint
        self.evaluation_plan = evaluation_plan
        self.cache_dir = cache_dir
        self.evaluators: List[BaseEvaluator] = []

    def load_evaluation_plan(self) -> Dict[str, Any]:
        """Load evaluation plan from YAML.

        Returns:
            Evaluation plan dictionary
        """
        plan_path = (
            Path(__file__).parent.parent.parent
            / "configs"
            / "evaluation_plans"
            / f"{self.evaluation_plan}.yaml"
        )

        if not plan_path.exists():
            logger.warning("Evaluation plan %s not found, using minimal defaults", plan_path)
            return {
                "tasks": {
                    "mmlu": {"num_fewshot": 5, "batch_size": 8},
                    "arc_challenge": {"num_fewshot": 25, "batch_size": 8},
                }
            }

        with open(plan_path, encoding="utf-8") as f:
            return yaml.safe_load(f)

    def setup_evaluators(self, plan: Dict[str, Any]):
        """Initialize evaluators based on plan with task-specific configs.

        Args:
            plan: Evaluation plan dictionary
        """
        tasks_config = plan.get("tasks", {})

        if not isinstance(tasks_config, dict):
            raise ValueError("'tasks' in evaluation plan must be a dictionary with task configs")

        # Add lm-eval-harness evaluator with per-task configuration
        self.evaluators.append(
            LMEvalEvaluator(
                model_name=self.model_name,
                endpoint=self.endpoint,
                tasks=tasks_config,  # Pass the full task config dict
                cache_dir=self.cache_dir,
            )
        )

        logger.info("Configured %d evaluation tasks", len(tasks_config))
        if self.cache_dir:
            logger.info("Cache directory: %s", self.cache_dir)
        else:
            logger.info("Cache directory: default (~/.cache/huggingface)")
        for task, config in tasks_config.items():
            logger.info("  - %s: %d-shot", task, config.get('num_fewshot', 0))

    def run(self) -> Dict[str, Any]:
        """Run all evaluators and collect results.

        Returns:
            Aggregated evaluation results
        """
        plan = self.load_evaluation_plan()
        self.setup_evaluators(plan)

        all_results = {
            "model": self.model_name,
            "evaluation_plan": self.evaluation_plan,
            "cache_dir": self.cache_dir,
            "evaluation_results": [],
        }

        for evaluator in self.evaluators:
            logger.info("Running %s", evaluator.__class__.__name__)
            results = evaluator.evaluate()
            all_results["evaluation_results"].append(results)

        return all_results
