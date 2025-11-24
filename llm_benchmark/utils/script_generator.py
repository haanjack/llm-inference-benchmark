import logging
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class ScriptGenerator:
    """Generates a bash script for running the benchmark."""

    def __init__(self, output_path: Path):
        self.output_path = output_path
        self.script_parts = {
            "header": ["#!/bin/bash", "# Auto-generated benchmark script", ""],
            "variables": ["# Benchmark Variables"],
            "server_cmd": ["# Start Server Command"],
            "wait_cmd": ["# Wait for Server to be Ready"],
            "server_env_vars": ["# Server Environment Variables"],
            "client_loop": [],
            "client_cmds": [],
            "client_loop_end": [],
            "footer": ["", "echo 'Benchmark script finished.'"],
        }

    def add_variable(self, name: str, value: Any):
        """Adds a variable to the script."""
        self.script_parts["variables"].append(f'{name}="{value}"')

    def add_env_variable(self, name: str, value: Any):
        """Adds an environment variable to the script."""
        self.script_parts["server_env_vars"].append(f'export {name}="{value}"')

    def set_server_command(self, command: List[str]):
        """Sets the server start command."""
        cmd_str = " \\\n    ".join(self._format_command(command))
        self.script_parts["server_cmd"].append(f"{cmd_str} &")
        self.script_parts["server_cmd"].append("SERVER_PID=$!")
        self.script_parts["server_cmd"].append("trap 'kill $SERVER_PID' EXIT") # Cleanup on exit

    def set_wait_command(self, port: int):
        """Sets the command to wait for the server port."""
        wait_script = [
            f"echo 'Waiting for server on port {port}...'",
            f"while ! nc -z localhost {port}; do",
            "  sleep 1",
            "done",
            "echo 'Server is ready.'",
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
        """Builds the nested client benchmark loop with the command."""
        self.script_parts["client_loop"].append("\n# Benchmark Loop")

        for param, values in loop_params.items():
            var_name = param.upper() + "_LIST"
            self.script_parts["client_loop"].append(f'{var_name}=({" ".join(map(str, values))})')

        self.script_parts["client_loop"].append("for i in ${!REQUEST_RATE_LIST[@]}; do")
        for param in loop_params.keys():
            self.script_parts["client_loop"].append(f"    {param.upper()}=${{{param.upper()}_LIST[i]}}")

        # The client command will be added separately
        self.script_parts["client_loop_end"].append("done")

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
        logger.info(f"Generated benchmark script: {self.output_path.resolve()}")