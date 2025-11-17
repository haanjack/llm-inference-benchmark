
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import List, Optional
import requests
import datetime
import yaml
import dotenv

import random
import string

from llm_benchmark.server.base import BenchmarkBase

logger = logging.getLogger(__name__)

class SGLangServer(BenchmarkBase):
    """SGLang Server management."""
    def __init__(self,
                 sglang_image: str,
                 test_plan: str,
                 no_warmup: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.image = sglang_image
        self._test_plan_path = Path(f"configs/benchmark_plans/{test_plan}.yaml")
        self._is_no_warmup = no_warmup

        self._setup_container_name()
        self._setup_logging_dirs()
        self._cache_dir()

    def _cache_dir(self):
        super()._cache_dir("sglang_cache")

    def _build_sglang_args(self) -> List[str]:
        args = []
        for key, value in self._vllm_args.items():
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
        group_option = "keep-groups" if os.environ.get("SLURM_JOB_ID", None) else "video"
        cmd = [
            self._container_runtime, "run", "-d",
            "--name", self.container_name,
            "-v", f"{os.environ.get('HF_HOME')}:/root/.cache/huggingface",
            "--device", "/dev/kfd", "--device", "/dev/dri", "--device", "/dev/mem",
            "--group-add", group_option,
            "--network=host",
            "--cap-add=CAP_SYS_ADMIN",
            "--cap-add=SYS_PTRACE",
            "--shm-size=16gb",
            "--security-opt", "seccomp=unconfined",
            "-e", f"CUDA_VISIBLE_DEVICES={self._gpu_devices}",
            "--env-file", self._common_env_file,
        ]

        for key, value in self._env_vars.items():
            cmd.extend(["-e", f"{key}={value}"])

        cmd.extend([
            "-v", f"{self._model_path}:{self.get_model_path()}:ro",
            "-v", f"{self._host_cache_dir}:/root/.cache",
            self.image,
            "python", "-m", "sglang.launch_server",
            "--model-path", self.get_model_path(),
            "--host", "0.0.0.0",
            "--port", str(self._vllm_port),
            "--tensor-parallel-size", str(self._parallel_size.get('tp', '1')),
        ])
        cmd.extend(self._build_sglang_args())
        return cmd

    def get_server_run_cmd_direct(self) -> List[str]:
        """Build server run command"""
        cmd = [
            "python", "-m", "sglang.launch_server",
            "--model-path", str(self._model_path),
            "--host", "0.0.0.0",
            "--port", str(self._vllm_port),
            "--tensor-parallel-size", str(self._parallel_size.get('tp', '1')),
        ]
        cmd.extend(self._build_sglang_args())
        return cmd

    def start(self):
        """Start SGLang server."""
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
        if self._is_dry_run:
            logger.info("Dry run - Docker server command:")
            logger.info(" ".join(cmd))
            return

        logger.info("Started to initialize sglang server ...")
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)

        with open(self.server_log, 'a', encoding='utf-8') as f:
            self._log_process = subprocess.Popen(
                [self._container_runtime, "logs", "-f", self._container_name],
                stdout=f,
                stderr=f
            )

    def _start_server_direct(self):
        cmd = self.get_server_run_cmd_direct()
        if self._is_dry_run:
            logger.info("Dry run - Direct server command:")
            logger.info(" ".join(cmd))
            return

        logger.info("Starting SGLang server as a direct process...")
        server_env = os.environ.copy()
        server_env["CUDA_VISIBLE_DEVICES"] = self._gpu_devices
        with open(self._common_env_file, "r", encoding="utf-8") as f:
            common_env = dotenv.dotenv_values(stream=f)
            server_env.update(common_env)
        for key, value in self._env_vars.items():
            server_env[key] = str(value)

        self.server_log.parent.mkdir(parents=True, exist_ok=True)
        with open(self.server_log, 'w', encoding='utf-8') as f:
            self.server_process = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=server_env)

    def _warmup_server(self, num_warmup_requests=5, prompt_length=16):
        if self._is_dry_run or self._is_no_warmup:
            logger.info("Skipping warmup.")
            return

        logger.info("Warming up the SGLang server with %d requests...", num_warmup_requests)
        for i in range(num_warmup_requests):
            # Generate a random prompt
            prompt = ''.join(random.choices(string.ascii_letters + string.digits, k=prompt_length))
            payload = {
                "prompt": prompt,
                "max_tokens": 16,
            }
            try:
                response = requests.post(f"http://localhost:{self._vllm_port}/v1/completions", json=payload, timeout=60)
                response.raise_for_status()
                logger.info("Warmup request %d successful.", i + 1)
            except requests.exceptions.RequestException as e:
                logger.warning("Warmup request %d failed: %s", i + 1, e)
        logger.info("Warmup complete.")

    def cleanup(self):
        """Cleanup resources."""
        if self._in_container:
            self.cleanup_server_process()
        else:
            self.cleanup_container()

    @property
    def image_tag(self) -> str:
        """Returns the SGLang Docker image tag."""
        return self.image.split(':')[-1]

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
    def vllm_port(self) -> int:
        """Returns the server port."""
        return self._vllm_port

    @property
    def exp_tag(self) -> str:
        """Returns the experiment tag."""
        return self._exp_tag
