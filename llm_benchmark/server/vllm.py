import logging
import os
import subprocess
import time
import json
from pathlib import Path
from typing import Dict, List, Optional
import tempfile
import requests
import yaml
import dotenv

from llm_benchmark.server.base import BenchmarkBase

logger = logging.getLogger(__name__)

class VLLMServer(BenchmarkBase):
    """vLLM Server management."""
    def __init__(self,
                vllm_image: str,
                test_plan: str,
                no_warmup: bool = False,
                **kwargs):
        super().__init__(**kwargs)
        self._vllm_image = vllm_image
        self._test_plan_path = Path(f"configs/benchmark_plans/{test_plan}.yaml")
        self._is_no_warmup = no_warmup

        self._image_tag = self._vllm_image.split(':')[-1]
        self.server_process = None
        self.temp_compile_config_file = None
        self._log_process: Optional[subprocess.Popen] = None

        self._setup_container_name()
        self._setup_logging_dirs()
        self._cache_dir()

    def _setup_container_name(self):

        slurm_job_id = os.environ.get("SLURM_JOB_ID", None)
        self._container_name = ""
        if slurm_job_id:
            self._container_name = f"{slurm_job_id}-"
        self._container_name += f"{os.path.basename(self._model_name)}-{self._image_tag}-g{self._gpu_devices.replace(',', '_')}"

    def _setup_logging_dirs(self):
        self._log_dir = Path("logs") / self._model_name / self._image_tag
        self.server_log = self._log_dir / "server_logs" / f"{os.path.basename(self._model_name)}-{self._image_tag}-t{self._parallel_size.get('tp', '1')}.txt"
        self._exp_tag = f"{Path(self._model_config).stem}_tp{self._parallel_size.get('tp', '1')}"
        if not self._is_dry_run:
            self.server_log.parent.mkdir(parents=True, exist_ok=True)

    def _cache_dir(self):
        """Configure vllm cache directories to reduce compilation overhead."""
        self._host_cache_dir = Path.cwd() / "vllm_cache" / self._exp_tag
        self._host_cache_dir.mkdir(parents=True, exist_ok=True)
        self._aiter_cache_dir = self._host_cache_dir / "aiter"
        self._aiter_cache_dir.mkdir(parents=True, exist_ok=True)
        self._compile_cache_dir = self._host_cache_dir / "compile_config"
        self._compile_cache_dir.mkdir(parents=True, exist_ok=True)

    def _build_vllm_args(self) -> List[str]:
        args = []
        for key, value in self._vllm_args.items():
            if value is None:
                continue
            if key == "quantization" and value == "auto":
                continue
            if isinstance(value, bool):
                if value:
                    args.append(f"--{key.replace('_', '-')}")
            else:
                args.extend([f"--{key.replace('_', '-')}", str(value)])

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", dir=self._compile_cache_dir, encoding="utf-8", delete=False) as f:
            dict_config_str = json.dumps(self._compilation_config, separators=(',', ':'))
            f.write(f"compilation_config: '{dict_config_str}'")

            config_path = f.name
            if not self._in_container:
                config_path = str(Path("/root/.cache/compile_config") / Path(f.name).name)

            args.extend(["--config", config_path])
            self.temp_compile_config_file = f.name
        return args

    def get_server_run_cmd(self, no_enable_prefix_caching: bool) -> List[str]:
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

        # add inferencing control environment variables
        for key, value in self._env_vars.items():
            cmd.extend(["-e", f"{key}={value}"])

        # set volume mounts and run server command
        cmd.extend([
            "-v", f"{self._model_path}:{self.get_model_path()}:ro",
            "-v", f"{self._host_cache_dir}:/root/.cache",
            "-v", f"{self._compile_cache_dir}:/root/.cache/compile_config",
            "-v", f"{self._aiter_cache_dir}:/root/.aiter",
            "-v", f"{os.environ.get('HOME')}:{os.environ.get('HOME')}",
            "-w", f"{os.environ.get('HOME')}",
            self._vllm_image,
            "vllm", "serve",
            self.get_model_path(),
            "--host", "0.0.0.0",
            "--no-enable-log-requests",
            "--trust-remote-code",
            "--tensor-parallel-size", str(self._parallel_size.get('tp', '1')),
            "--port", str(self._vllm_port),
        ])
        if no_enable_prefix_caching:
            cmd.append("--no-enable-prefix-caching")
        cmd.extend(self._build_vllm_args())
        return cmd

    def get_server_run_cmd_direct(self, no_enable_prefix_caching: bool) -> List[str]:
        """Build server run command"""
        cmd = [
            "vllm", "serve",
            str(self._model_path),
            "--host", "0.0.0.0",
            "--no-enable-log-requests",
            "--trust-remote-code",
            "--tensor-parallel-size", str(self._parallel_size.get('tp', '1')),
            "--port", str(self._vllm_port),
        ]
        if no_enable_prefix_caching:
            cmd.append("--no-enable-prefix-caching")
        cmd.extend(self._build_vllm_args())
        return cmd

    def _load_test_plan(self):
        with open(self._test_plan_path, mode="r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # vllm server prefix caching ops determined which dataset to test
        dataset_name = config.get('dataset_name', 'random')
        no_enable_prefix_caching = (dataset_name == 'random')

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
            with open(self.temp_compile_config_file, "r", encoding="utf-8") as f:
                compile_config = yaml.safe_load(f)
                logger.info(compile_config)
        else:
            logger.info("Started to initialize vllm server ...")
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)

        if self._is_dry_run:
            return
        with open(self.server_log, 'a', encoding='utf-8') as f:
            self._log_process = subprocess.Popen(
                [self._container_runtime, "logs", "-f", self._container_name],
                stdout=f,
                stderr=f
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
        with open(self._common_env_file, "r", encoding="utf-8") as f:
            common_env = dotenv.dotenv_values(stream=f)
            server_env.update(common_env)
        for key, value in self._env_vars.items():
            server_env[key] = str(value)
        if 'BENCHMARK_BASE_PORT' in server_env:
            global BENCHMARK_BASE_PORT # pylint: disable=global-statement
            BENCHMARK_BASE_PORT = int(server_env['BENCHMARK_BASE_PORT'])
            del server_env['BENCHMARK_BASE_PORT']

        self.server_log.parent.mkdir(parents=True, exist_ok=True)
        with open(self.server_log, 'w', encoding='utf-8') as f:
            self.server_process = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=server_env)

    def _is_server_process_alive(self) -> bool:
        if self._is_dry_run:
            return True
        if self._in_container:
            return self.server_process and self.server_process.poll() is None
        else:
            try:
                cmd = [self._container_runtime, "ps", "-q", "--filter", f"name=^{self._container_name}$"]
                result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=5)
                return bool(result.stdout.strip())
            except (subprocess.SubprocessError, FileNotFoundError):
                return False

    def _wait_for_server(self, timeout: int = 2 * 60 * 60) -> bool:
        start_time = time.time()
        last_log_time = start_time
        while True:
            try:
                response = requests.get(f"http://localhost:{self._vllm_port}/v1/models", timeout=10)
                if response.status_code == 200 and response.json():
                    return True
            except requests.exceptions.RequestException:
                pass

            if not self._is_server_process_alive():
                logger.error("vLLM server process is not running. Check server log: %s", self.server_log)
                return False

            elapsed_time = time.time() - start_time
            if elapsed_time > timeout:
                logger.error("Timeout waiting for vLLM server. Check log: %s", self.server_log)
                return False

            if time.time() - last_log_time > 60:
                last_log_time = time.time()
                logger.info("Waiting for vLLM server... %s seconds elapsed", int(elapsed_time))
            time.sleep(5)

    def _warmup_server(self):
        if self._is_dry_run or self._is_no_warmup:
            logger.info("Skipping warmup.")
            return

        logger.info("Warming up the server...")
        warmup_cmd = []
        if not self._in_container:
            warmup_cmd.extend([self._container_runtime, "exec", self._container_name])
        warmup_cmd.extend([
            "vllm", "bench", "serve", "--model", self.get_model_path(),
            "--backend", "vllm", "--host", "localhost", f"--port={self.vllm_port}",
            "--dataset-name", "random", "--ignore-eos", "--trust-remote-code",
            "--request-rate=10", "--max-concurrency=1", "--num-prompts=4",
            "--random-input-len=16", "--random-output-len=16",
            "--tokenizer", self.get_model_path(), "--disable-tqdm"
        ])
        start_time = time.time()
        subprocess.run(warmup_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        logger.info("Warmup complete in %.2f seconds.", time.time() - start_time)

    def cleanup(self):
        """Cleanup resources."""
        if self._in_container:
            self.cleanup_server_process()
        else:
            self.cleanup_container()
        self._cleanup_temp_files()

    def cleanup_server_process(self):
        """Cleanup server process."""
        if self._is_dry_run:
            return

        if self.server_process:
            logger.info("Shutting down vLLM server process...")
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.server_process.kill()

    def cleanup_container(self):
        """Cleanup container."""
        if self._is_dry_run:
            return

        self._cleanup_log_processes()
        if not self._container_runtime or not self._container_name:
            logger.error("Container runtime or name not defined.")
            return
        try:
            subprocess.run([self._container_runtime, "rm", "-f", self._container_name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        except subprocess.CalledProcessError as e:
            logger.warning("Failed to remove container %s: %s", self._container_name, e)

    def _cleanup_log_processes(self):
        """Cleanup log processes."""
        if self._log_process is None:
            return

        try:
            self._log_process.terminate()
            self._log_process.wait(timeout=5)
        except (subprocess.TimeoutExpired, ProcessLookupError):
            self._log_process.kill()

    def _cleanup_temp_files(self):
        """Cleanup temporary files."""
        if not self.temp_compile_config_file:
            return

        if not os.path.exists(self.temp_compile_config_file):
            return

        logger.info("Cleaning up temporary compile config file: %s", self.temp_compile_config_file)
        os.remove(self.temp_compile_config_file)
        self.temp_compile_config_file = None

    @property
    def vllm_image(self) -> str:
        """Returns the vLLM Docker image."""
        return self._vllm_image

    @property
    def image_tag(self) -> str:
        """Returns the vLLM Docker image tag."""
        return self._image_tag

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
    def parallel_size(self) -> Dict[str, str]:
        """Returns the parallel size."""
        return self._parallel_size

    @property
    def vllm_port(self) -> int:
        """Returns the vLLM server port."""
        return self._vllm_port

    @property
    def exp_tag(self) -> str:
        """Returns the experiment tag."""
        return self._exp_tag
