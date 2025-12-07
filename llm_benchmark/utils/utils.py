import os
from typing import Dict


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