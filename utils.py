import subprocess
import json
from typing import Optional, Union

GFX_CLK_IDLE_THRESHOLD = 1_000 # Example threshold value in MHz

def get_gfx_clk_value(gpu_index: int = 0, 
                      amd_smi_path: str = "amd-smi", 
                      timeout: float = 5.0) -> int:
    """
    Run 'amd-smi monitor -g <gpu_index> --gfx --json' and return only the gfx_clk value.

    Parameters:
      - gpu_index: Target GPU index (default: 0)
      - amd_smi_path: Path to the amd-smi binary (default: "amd-smi")
      - timeout: Command timeout in seconds (default: 5.0)

    Returns:
      - gfx_clk value as int/float if available
      - None on failure or if value is not found
    """
    cmd = [amd_smi_path, "monitor", "-g", str(gpu_index), "--gfx", "--json"]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout
        )
        combined = (proc.stdout or "") + (proc.stderr or "")
        if not combined:
            return 0

        # Try to extract a JSON array block first, then a JSON object block if needed.
        def extract_json_block(text: str) -> Optional[str]:
            # Prefer array (most amd-smi monitor outputs are arrays)
            start_br = text.find("[")
            end_br = text.rfind("]")
            if start_br != -1 and end_br != -1 and end_br > start_br:
                return text[start_br:end_br + 1].strip()

            # Fallback to object
            start_curly = text.find("{")
            end_curly = text.rfind("}")
            if start_curly != -1 and end_curly != -1 and end_curly > start_curly:
                return text[start_curly:end_curly + 1].strip()

            return 0

        json_text = extract_json_block(combined)
        if not json_text:
            return 0

        data = json.loads(json_text)

        # Handle both array and object forms
        if isinstance(data, list):
            # Assume the first entry contains the metrics for the specified GPU
            if not data:
                return 0
            entry = data[0]
        elif isinstance(data, dict):
            entry = data
        else:
            return 0

        gfx_clk = entry.get("gfx_clk")
        if isinstance(gfx_clk, dict):
            # Typical: {"value": 2103, "unit": "MHz"}
            return gfx_clk.get("value")

        # In case structure differs, attempt a shallow search
        # e.g., some versions might nest the clock info differently
        if isinstance(entry, dict):
            for k, v in entry.items():
                if k.lower() == "gfx_clk" and isinstance(v, (int, float)):
                    return v

        # partitioned GPU does not report gfx_clk, try to get lead GPU's gfx_clk
        if gfx_clk == "N/A" and entry.get("xcp") > 0:
            lead_gpu_id = int(entry.get("gpu")) - int(entry.get("xcp"))
            return get_gfx_clk_value(lead_gpu_id)

        return 0

    except (json.JSONDecodeError, subprocess.TimeoutExpired, FileNotFoundError):
        return 0
    except Exception:
        return 0


if __name__ == "__main__":
    value = get_gfx_clk_value(gpu_index=0)
    print(value)  # Example output: 2103
