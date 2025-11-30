import logging
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class ScriptGenerator:
    """Generates a bash script for running the benchmark."""

    def __init__(self, output_path: Path, in_container: bool = False):
        self.output_path = output_path
        self.in_container = in_container
        self.script_parts = {
            "header": ["#!/bin/bash", "# Auto-generated benchmark script", ""],
            "variables": ["# Benchmark Variables"],
            "server_cmd": ["# Start Server Command"],
            "wait_cmd": ["# Wait for Server to be Ready"],
            "server_env_vars": ["# Server Environment Variables"] if in_container else [],
            "client_loop": [],
            "client_cmds": [],
            "client_loop_end": [],
            "footer": ["", "echo 'Benchmark script finished.'"],
        }

    def add_variable(self, name: str, value: Any):
        """Adds a variable to the script."""
        self.script_parts["variables"].append(f'{name}="{value}"')

    def add_env_variable(self, name: str, value: Any):
        """Adds an environment variable to the script (only for in_container mode)."""
        if self.in_container:
            self.script_parts["server_env_vars"].append(f'export {name}="{value}"')

    def set_server_command(self, command: List[str]):
        """Sets the server start command."""
        cmd_str = " \\\n    ".join(self._format_command(command))
        self.script_parts["server_cmd"].append(f"{cmd_str} &")
        self.script_parts["server_cmd"].append("SERVER_PID=$!")
        self.script_parts["server_cmd"].append("trap 'kill $SERVER_PID' EXIT") # Cleanup on exit

    def set_wait_command(self, port: int):
        """Sets the command to wait for the server port with progress reporting."""
        wait_script = [
            f"echo 'Waiting for server on port {port}...'",
            "START_TIME=$SECONDS",
            "LAST_REPORT=0",
            f"while ! nc -z localhost {port}; do",
            "  sleep 5",
            "  ELAPSED=$((SECONDS - START_TIME))",
            "  if [ $((ELAPSED - LAST_REPORT)) -ge 60 ]; then",
            "    echo \"Still waiting... $((ELAPSED / 60)) minute(s) $((ELAPSED % 60)) second(s) elapsed\"",
            "    LAST_REPORT=$ELAPSED",
            "  fi",
            "done",
            "TOTAL_TIME=$((SECONDS - START_TIME))",
            "echo \"Server is ready! (took $((TOTAL_TIME / 60))m $((TOTAL_TIME % 60))s)\"",
            ""
        ]
        self.script_parts["wait_cmd"].extend(wait_script)

    def _format_command(self, command: List[str]) -> List[str]:
        """Formats a command list to group arguments with their values."""
        formatted_command = []
        i = 0
        while i < len(command):
            current = command[i]
            if current.startswith('-') and not current.startswith('--') and len(current) > 2:
                # Handle combined short options like -e, -v
                formatted_command.append(current)
                i += 1
            elif current.startswith('--') and i + 1 < len(command) and not command[i+1].startswith('-'):
                # Argument with value
                formatted_command.append(f"{current} {command[i+1]}")
                i += 2
            elif current.startswith('-') and i + 1 < len(command) and not command[i+1].startswith('-'):
                 # short argument with value
                formatted_command.append(f"{current} {command[i+1]}")
                i += 2
            else:
                # Standalone argument or flag
                formatted_command.append(current)
                i += 1
        return formatted_command

    def set_client_loop(self, loop_params: Dict[str, List[Any]], command_template: List[str]):
        """Builds nested client benchmark loops with the command."""
        self.script_parts["client_loop"].append("\n# Benchmark Loop")

        # Mapping from plural parameter names to singular variable names used in commands
        param_to_var_map = {
            'request_rates': 'REQUEST_RATE',
            'concurrencies': 'CONCURRENCY',
            'input_lengths': 'INPUT_LENGTH',
            'output_lengths': 'OUTPUT_LENGTH',
            'num_prompts': 'NUM_PROMPTS',
            'dataset_name': 'DATASET_NAME'
        }

        # Define the order for nested loops (most to least significant)
        param_order = ['dataset_name', 'num_prompts', 'request_rates', 'input_lengths', 'output_lengths', 'concurrencies']

        # Filter to only include params that exist in loop_params
        params_to_loop = [p for p in param_order if p in loop_params]

        # Build nested for loops
        indent_level = 0
        for param in params_to_loop:
            values = loop_params[param]
            var_name = param_to_var_map.get(param, param.upper())

            # Convert values to strings, handling strings specially
            str_values = [f'"{v}"' if isinstance(v, str) else str(v) for v in values]

            indent = "    " * indent_level
            self.script_parts["client_loop"].append(f"{indent}for {var_name} in {' '.join(str_values)}; do")
            indent_level += 1

        # Add the client command inside all the loops
        indent = "    " * indent_level
        formatted_cmd = self._format_command(command_template)
        self.script_parts["client_loop"].append("")
        self.script_parts["client_loop"].append(indent + (" \\\n" + indent + "    ").join(formatted_cmd))

        # Close all loops
        for _ in range(len(params_to_loop)):
            indent_level -= 1
            indent = "    " * indent_level
            self.script_parts["client_loop_end"].append(f"{indent}done")

    def generate(self):
        """Writes the generated script to the output file."""
        script_content = []
        for part in self.script_parts.values():
            script_content.extend(part)

        if not self.output_path.parent.exists():
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(script_content))

        self.output_path.chmod(0o755)
        logger.info("Generated benchmark script: %s", self.output_path.resolve())


def prettify_generated_scripts(
    tmp_dir: Path,
    output_dir: Path,
    model_config_name: str,
    test_plan: str
):
    """
    Prettify generated scripts from tmp directory and save to output directory.
    
    Args:
        tmp_dir: Temporary directory containing generated scripts
        output_dir: Output directory for prettified scripts
        model_config_name: Model configuration name (stem)
        test_plan: Test plan name
    """
    import sys
    import subprocess
    import shutil
    
    logger.info("Prettifying generated scripts...")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all generated scripts in tmp directory
    pattern = f"run-{model_config_name}-{test_plan}*.sh"
    
    prettifier_script = Path("scripts/generated_script_prettifier.py")
    if not prettifier_script.exists():
        logger.warning("Prettifier script not found at %s. Skipping prettification.", prettifier_script)
        return
    
    for tmp_script in tmp_dir.glob(pattern):
        output_file = output_dir / tmp_script.name
        logger.info("Prettifying: %s -> %s", tmp_script, output_file)
        try:
            subprocess.run(
                [sys.executable, str(prettifier_script), "-i", str(tmp_script), "-o", str(output_file)],
                check=True,
                capture_output=True,
                text=True
            )
        except subprocess.CalledProcessError as e:
            logger.error("Failed to prettify %s: %s", tmp_script, e.stderr)
            # Copy the original script if prettification fails
            shutil.copy2(tmp_script, output_file)
            logger.info("Copied unprettified script to %s", output_file)
    
    logger.info("All scripts prettified and saved to %s", output_dir)