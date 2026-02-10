import logging
import os
import subprocess
from pathlib import Path
from typing import List, Optional
import yaml


from llm_benchmark.server.base import BenchmarkBase
from llm_benchmark.utils.script_generator import ScriptGenerator

logger = logging.getLogger(__name__)


class SGLangServer(BenchmarkBase):
    """SGLang Server management."""

    def __init__(self, test_plan: str, no_warmup: bool = False, **kwargs):
        super().__init__(name="sglang", **kwargs)
        self._test_plan_path = Path(f"configs/benchmark_plans/{test_plan}.yaml")
        self._is_no_warmup = no_warmup

        self._cache_dir()

    def _cache_dir(self, cache_name: str = "sglang_cache"):
        super()._cache_dir(cache_name)

    def _build_sglang_args(self) -> List[str]:
        args = []
        for key, value in self._server_args.items():
            if value is None:
                continue
            if isinstance(value, bool):
                if value:
                    args.append(f"--{key.replace('_', '-')}")
            else:
                args.extend([f"--{key.replace('_', '-')}", str(value)])
        return args

    def get_server_run_cmd(self) -> List[str]:
        """Build server run command with container execution"""
        cmd = self._get_docker_run_common_command()

        use_script_vars = self.script_generator is not None
        image_val = "$IMAGE" if use_script_vars else self.image
        model_path_val = "$MODEL_PATH" if use_script_vars else self.get_model_path()
        tp_size_val = (
            "$TP_SIZE" if use_script_vars else str(self._parallel_size.get("tp", "1"))
        )
        port_val = "$PORT" if use_script_vars else str(self._port)

        cmd.extend(
            [
                "-v",
                f"{self._host_cache_dir}:/root/.cache",
                image_val,
            ]
        )

        # Build sglang launch command
        sglang_cmd = [
            "python",
            "-m",
            "sglang.launch_server",
            "--model-path",
            model_path_val,
            "--host",
            "0.0.0.0",
            "--port",
            port_val,
            "--tensor-parallel-size",
            tp_size_val,
        ]
        sglang_cmd.extend(self._build_sglang_args())

        # Build startup command: install pip packages first, then start sglang
        pip_cmd = self._get_pip_install_cmd_prefix()
        if pip_cmd:
            shell_cmd = f"{pip_cmd} && {' '.join(sglang_cmd)}"
            cmd.extend(["/bin/bash", "-c", shell_cmd])
        else:
            cmd.extend(sglang_cmd)

        return cmd

    def get_server_run_cmd_direct(self) -> List[str]:
        """Build server run command"""

        use_script_vars = self.script_generator is not None
        model_path_val = "$MODEL_PATH" if use_script_vars else self.get_model_path()
        tp_size_val = (
            "$TP_SIZE" if use_script_vars else str(self._parallel_size.get("tp", "1"))
        )
        port_val = "$PORT" if use_script_vars else str(self._port)

        cmd = [
            "python",
            "-m",
            "sglang.launch_server",
            "--model-path",
            model_path_val,
            "--host",
            "0.0.0.0",
            "--tensor-parallel-size",
            tp_size_val,
            "--port",
            port_val,
        ]
        cmd.extend(self._build_sglang_args())
        return cmd

    def _load_test_plan(self):
        with open(self._test_plan_path, mode="r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # vllm server prefix caching ops determined which dataset to test
        dataset_name = config.get("dataset_name", "random")

    def start(self):
        """Start SGLang server."""
        self._load_test_plan()

        if self._in_container:
            self._start_server_direct()
        else:
            self._start_server_container()

        if not self._is_dry_run:
            if not self._wait_for_server():
                raise RuntimeError("Server failed to start")
            logger.info("Server is up and running")
            self._warmup_server()

    def _start_server_container(self):
        """Start SGLang server container"""
        self.cleanup_container()

        cmd = self.get_server_run_cmd()
        logger.info("SGLang server command: %s", " ".join(cmd))

        if self._is_dry_run:
            logger.info("Dry run - Container command would execute")
            return

        logger.info("Started to initialize sglang server ...")
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

    def _start_server_direct(self):
        # Update pip packages if configured
        if self._pip_packages and not self._is_dry_run:
            self._update_pip_packages_direct()

        cmd = self.get_server_run_cmd_direct()
        if self._is_dry_run:
            cmd_str = " ".join(cmd)
            cmd_str = cmd_str.replace("$MODEL_PATH", str(self._model_path))
            cmd_str = cmd_str.replace(
                "$TP_SIZE", str(self._parallel_size.get("tp", "1"))
            )
            cmd_str = cmd_str.replace("$PORT", str(self._port))

            logger.info("Dry run - Direct server command:")
            logger.info(cmd_str)
            return

        logger.info("Starting SGLang server as a direct process...")
        server_env = os.environ.copy()
        server_env["CUDA_VISIBLE_DEVICES"] = self._gpu_devices
        for key, value in self._env_vars.items():
            server_env[key] = str(value)

        self.server_log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.server_log_path, "w", encoding="utf-8") as f:
            self.server_process = subprocess.Popen(
                cmd, stdout=f, stderr=subprocess.STDOUT, env=server_env
            )

    def cleanup(self):
        """Cleanup resources."""
        if self._in_container:
            self.cleanup_server_process()
        else:
            self.cleanup_container()

    @property
    def image_tag(self) -> str:
        """Returns the SGLang Docker image tag."""
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
    def model_config(self) -> str:
        """Returns the model config."""
        return self._model_config

    @property
    def parallel_size(self) -> dict:
        """Returns the parallel size."""
        return self._parallel_size

    @property
    def port(self) -> int:
        """Returns the server port."""
        return self._port

    @property
    def exp_tag(self) -> str:
        """Returns the experiment tag."""
        return self._exp_tag

    def generate_script(self, generator: ScriptGenerator):
        """Generates the SGLang server start command for the script."""
        super().generate_script(generator)
        self._load_test_plan()
        server_cmd = self.get_server_run_cmd()
        generator.set_server_command(server_cmd)
        generator.set_wait_command(self.port)
