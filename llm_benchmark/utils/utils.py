import os
from typing import Dict
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def parse_env_file(file_path: str) -> Dict[str, str]:
    """
    Parses a shell-like environment file, resolving variables.
    Ignores comments and empty lines.
    """
    if not os.path.exists(file_path):
        return {}

    envs = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()

            # Expand environment variables like ${HOME} or $HOME
            value = os.path.expandvars(value)

            envs[key] = value
    return envs


def setup_global_dashboard(output_dir: str) -> Path:
    """Setup the global test set dashboard file."""
    global_dashboard_file = Path(output_dir) / "test_results.tsv"
    global_dashboard_file.parent.mkdir(parents=True, exist_ok=True)
    if not global_dashboard_file.exists():
        with open(global_dashboard_file, "w", encoding="utf-8") as f:
            f.write("\t".join([
                "timestamp",
                "model",
                "image_tag",
                "model_config",
                "tp_size",
                "test_plan",
                "sub_task",
                "result",
                "log_path"
            ]) + "\n")
    return global_dashboard_file


def write_test_set_dashboard_entry(
    global_dashboard_file: Path,
    server,
    test_plan: str,
    sub_tasks: list,
    success: bool,
    result_path: Path = None,
    is_dry_run: bool = False
):
    """Append a single row summarizing this test set execution.

    Handles cases where server or client initialization failed by extracting
    available information from server and test_plan parameters.
    """
    if is_dry_run:
        return

    try:
        model = server.model_name if server else "unknown"
        image_tag = server.image_tag if server else "unknown"
        model_config = Path(server.model_config).stem if server else "unknown"
        tp_size = str(server.parallel_size.get('tp', '1')) if server else "unknown"
    except Exception as e:
        logger.warning("Could not extract server info for dashboard entry: %s", e)
        model = "unknown"
        image_tag = "unknown"
        model_config = "unknown"
        tp_size = "unknown"

    sub_task_str = ""
    if sub_tasks:
        sub_task_str = "+".join(sub_tasks)

    try:
        with open(global_dashboard_file, "a", encoding="utf-8") as f:
            f.write("\t".join([
                datetime.now().isoformat(timespec="seconds"),
                str(model),
                str(image_tag),
                str(model_config),
                str(tp_size),
                str(test_plan),
                sub_task_str,
                "success" if success else "failure",
                str(result_path) if (success and result_path) else "N/A"
            ]) + "\n")
    except Exception as e:
        logger.error("Failed to write test set dashboard entry: %s", e)