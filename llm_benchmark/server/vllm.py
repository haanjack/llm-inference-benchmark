import logging
import os
import subprocess
import time
import json
from pathlib import Path
from typing import List, Optional
import tempfile
import requests
import yaml

from llm_benchmark.server.base import BenchmarkBase
from llm_benchmark.utils.script_generator import ScriptGenerator

logger = logging.getLogger(__name__)

BENCHMARK_BASE_PORT=23400

class VLLMServer(BenchmarkBase):
    """vLLM Server management."""

    def __init__(self, test_plan: str, no_warmup: bool = False, **kwargs):
        super().__init__(name="vllm", **kwargs)
        self._test_plan_path = Path(f"configs/benchmark_plans/{test_plan}.yaml")
        self._is_no_warmup = no_warmup

        self.temp_compile_config_file = None

        self._cache_dir()

    def _cache_dir(self, cache_name: str = "vllm_cache"):
        """Configure vllm cache directories to reduce compilation overhead."""
        super()._cache_dir(cache_name)
        self._aiter_cache_dir = self._host_cache_dir / "aiter"
        self._compile_cache_dir = self._host_cache_dir / "compile_config"
        self._aiter_cache_dir.mkdir(parents=True, exist_ok=True)
        self._compile_cache_dir.mkdir(parents=True, exist_ok=True)

    def _build_vllm_args(self) -> List[str]:
        args = []
        for key, value in self._server_args.items():
            if value is None:
                continue
            if key == "quantization" and value == "auto":
                continue
            if isinstance(value, bool):
                if value:
                    args.append(f"--{key.replace('_', '-')}")
            else:
                args.extend([f"--{key.replace('_', '-')}", str(value)])

        # When generating scripts, inline the compilation-config JSON for portability
        # Only add --compilation-config if the dict is non-empty
        if self.script_generator is not None or self._is_dry_run:
            if self._compilation_config:
                dict_config_str = json.dumps(self._compilation_config, separators=(',', ':'))
                args.extend(["--compilation-config", f"'{dict_config_str}'"])
        else:
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".yaml",
                dir=self._compile_cache_dir,
                encoding="utf-8",
                delete=False,
            ) as f:
                dict_config_str = json.dumps(
                    self._compilation_config, separators=(",", ":")
                )
                f.write(f"compilation_config: '{dict_config_str}'")
                config_path = f.name
                self.temp_compile_config_file = f.name

            if not self._in_container:
                config_path = str(
                    Path("/root/.cache/compile_config") / Path(config_path).name
                )
                args.extend(["--config", config_path])
        return args

    def get_server_run_cmd(self, no_enable_prefix_caching: bool) -> List[str]:
        """Build server run command with container execution"""
        cmd = self._get_docker_run_common_command()

        use_script_vars = self.script_generator is not None
        image_val = "$IMAGE" if use_script_vars else self.image
        model_path_val = "$MODEL_PATH" if use_script_vars else self.get_model_path()
        tp_size_val = (
            "$TP_SIZE" if use_script_vars else str(self._parallel_size.get("tp", "1"))
        )
        port_val = "$PORT" if use_script_vars else str(self._port)

        # set volume mounts and run server command
        cmd.extend([
            "-v", f"{self._host_cache_dir}:/root/.cache",
            "-v", f"{self._compile_cache_dir}:/root/.cache/compile_config",
            "-v", f"{self._aiter_cache_dir}:/root/.aiter",
            image_val,
            "vllm", "serve",
            model_path_val,
            "--host", "0.0.0.0",
            "--no-enable-log-requests",
            "--trust-remote-code",
            "--tensor-parallel-size", tp_size_val,
            "--port", port_val,
        ])
        if no_enable_prefix_caching:
            cmd.append("--no-enable-prefix-caching")
        cmd.extend(self._build_vllm_args())
        return cmd

    def get_server_run_cmd_direct(self, no_enable_prefix_caching: bool) -> List[str]:
        """Build server run command"""

        use_script_vars = self.script_generator is not None
        model_path_val = "$MODEL_PATH" if use_script_vars else self.get_model_path()
        tp_size_val = (
            "$TP_SIZE" if use_script_vars else str(self._parallel_size.get("tp", "1"))
        )
        port_val = "$PORT" if use_script_vars else str(self._port)

        cmd = [
            "vllm",
            "serve",
            model_path_val,
            "--host",
            "0.0.0.0",
            "--no-enable-log-requests",
            "--trust-remote-code",
            "--tensor-parallel-size",
            tp_size_val,
            "--port",
            port_val,
        ]
        if no_enable_prefix_caching:
            cmd.append("--no-enable-prefix-caching")
        cmd.extend(self._build_vllm_args())
        return cmd

    def _load_test_plan(self):
        with open(self._test_plan_path, mode="r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # vllm server prefix caching ops determined which dataset to test
        dataset_name = config.get("dataset_name", "random")
        no_enable_prefix_caching = dataset_name == "random"

        return no_enable_prefix_caching

    def start(self):
        """Start vLLM server."""
        no_enable_prefix_caching = self._load_test_plan()

        if self._in_container:
            self._start_server_direct(no_enable_prefix_caching)
        else:
            self._start_server_container(no_enable_prefix_caching)

        if not self._is_dry_run:
            if not self._wait_for_server():
                raise RuntimeError("Server failed to start")
            logger.info("Server is up and running")
            self._warmup_server()

    def _start_server_container(self, no_enable_prefix_caching: bool):
        """Start vLLM server container"""
        self.cleanup_container()
        cmd = self.get_server_run_cmd(no_enable_prefix_caching)
        if self._is_dry_run:
            logger.info("Dry run - Docker server command:")
            logger.info(" ".join(cmd))
            logger.info("config file content:")
            dict_config_str = json.dumps(
                self._compilation_config, separators=(",", ":")
            )
            logger.info("compilation_config: %s", dict_config_str)
            return

        logger.info("Started to initialize vllm server ...")
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
        logger.info("Environment variables:")
        for key, value in self._env_vars.items():
            logger.info(" - %s: %s", key, value)
        with open(self.server_log_path, "a", encoding="utf-8") as f:
            self._log_process = subprocess.Popen(
                [self._container_runtime, "logs", "-f", self._container_name],
                stdout=f,
                stderr=f,
            )

    def _start_server_direct(self, no_enable_prefix_caching: bool):
        cmd = self.get_server_run_cmd_direct(no_enable_prefix_caching)
        if self._is_dry_run:
            logger.info("Dry run - Direct server command:")
            logger.info(" ".join(cmd))
            return

        logger.info("Starting vLLM server as a direct process...")
        server_env = os.environ.copy()
        server_env["CUDA_VISIBLE_DEVICES"] = self._gpu_devices
        for key, value in self._env_vars.items():
            server_env[key] = str(value)
        if "BENCHMARK_BASE_PORT" in server_env:
            global BENCHMARK_BASE_PORT  # pylint: disable=global-statement
            BENCHMARK_BASE_PORT = int(server_env["BENCHMARK_BASE_PORT"])
            del server_env["BENCHMARK_BASE_PORT"]

        self.server_log_path.parent.mkdir(parents=True, exist_ok=True)
        for key, value in server_env.items():
            logger.info("> Server env var: %s=%s", key, value)
        with open(self.server_log_path, "w", encoding="utf-8") as f:
            self.server_process = subprocess.Popen(
                cmd, stdout=f, stderr=subprocess.STDOUT, env=server_env
            )

    def _wait_for_server(self, timeout: int = 2 * 60 * 60) -> bool:
        if super()._wait_for_server(timeout):
            # vLLM can return 200 with an empty list before it's truly ready
            response = requests.get(
                f"http://localhost:{self._port}/v1/models", timeout=10
            )
            if response.json():
                return True
        return False

    def _warmup_server(self):
        if self._is_dry_run or self._is_no_warmup:
            logger.info("Skipping warmup.")
            return

        logger.info("Warming up the server...")
        warmup_cmd = ["curl", f"http://localhost:{self.port}/v1/models"]
        start_time = time.time()
        subprocess.run(
            warmup_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
        )
        logger.info("Warmup complete in %.2f seconds.", time.time() - start_time)

    def cleanup(self):
        """Cleanup resources."""
        if self._in_container:
            self.cleanup_server_process()
        else:
            self.cleanup_container()
        self._cleanup_temp_files()

    def _cleanup_temp_files(self):
        """Cleanup temporary files."""
        if not self.temp_compile_config_file:
            return

        if not os.path.exists(self.temp_compile_config_file):
            return

        logger.info(
            "Cleaning up temporary compile config file: %s",
            self.temp_compile_config_file,
        )
        os.remove(self.temp_compile_config_file)
        self.temp_compile_config_file = None

    @property
    def image_tag(self) -> str:
        """Returns the vLLM Docker image tag."""
        return self.image.split(":")[-1]

    @property
    def container_runtime(self) -> Optional[str]:
        """Returns the container runtime ('docker' or 'podman')."""
        return self._container_runtime

    @property
    def container_name(self) -> str:
        """Returns the name of the container."""
        return self._container_name

    @property
    def port(self) -> int:
        """Returns the vLLM server port."""
        return self._port

    @property
    def exp_tag(self) -> str:
        """Returns the experiment tag."""
        return self._exp_tag

    def generate_script(self, generator: ScriptGenerator):
        """Generates the vLLM server start command for the script."""
        super().generate_script(generator)
        no_enable_prefix_caching = self._load_test_plan()
        if not self._in_container:
            server_cmd = self.get_server_run_cmd(no_enable_prefix_caching)
        else:
            server_cmd = self.get_server_run_cmd_direct(no_enable_prefix_caching)
        generator.set_server_command(server_cmd)
        generator.set_wait_command(self.port)
