import logging
import os
import subprocess
import time
import sys
from pathlib import Path
from llm_benchmark.utils.script_generator import ScriptGenerator
from typing import Dict, List, Optional, Union
import yaml
import requests
from urllib.parse import urlparse
from huggingface_hub import snapshot_download

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

BENCHMARK_BASE_PORT = 23400

class BenchmarkBase:
    """Base class for benchmark components."""
    def __init__(self,
                 name: str,
                 image: str = None,
                 envs: Dict[str, str] = None,
                 model_path_or_id: str = None,
                 model_root_dir: str = None,
                 model_config: str = None,
                 gpu_devices: str = None,
                 num_gpus: int = None,
                 arch: str = None,
                 dry_run: bool = False,
                 in_container: bool = False,
                 endpoint: str = None,
                 script_generator: ScriptGenerator = None):

        self.name = name
        self.image = image
        self.server_process: Optional[subprocess.Popen] = None
        self._common_envs = envs
        self._env_vars = {}
        self._server_args = {}
        self._compilation_config = {}
        self._arch = arch
        self._model_config = model_config
        self._is_dry_run = dry_run
        self._in_container = in_container
        self._log_process: Optional[subprocess.Popen] = None
        self._model_path_or_id = model_path_or_id
        self._remote_server_endpoint = endpoint if endpoint else None
        self.script_generator = script_generator

        self._model_path = self._load_model_from_path_or_hub(model_path_or_id, model_root_dir)
        self._model_name = f"{self._model_path.parent.name}/{self._model_path.name}"
        self._container_model_path = Path(f"/models/{self._model_name}")

        # Initialize container runtime
        if endpoint is None:
            self._gpu_devices, self._num_gpus, self._port = self._get_system_config(gpu_devices, num_gpus)
            self._parallel_size = {'tp': str(self.num_gpus)}
            self._load_model_config()

        self._container_runtime = None
        if not self.in_container:
            self._container_runtime = "docker" if self._is_docker_available() else "podman"
            self._container_name = self._setup_container_name()
        self._setup_logging_dirs()

        if not self._model_path.exists() and not self._is_dry_run:
            raise FileNotFoundError(f"Could not find model at {self._model_name} in {self.get_model_path()}.")

    def _setup_container_name(self):
        slurm_job_id = os.environ.get("SLURM_JOB_ID", None)
        container_name = ""
        if slurm_job_id:
            container_name = f"{slurm_job_id}-"
        container_name += f"{os.path.basename(self._model_name)}-{self.image_tag}"
        if hasattr(self, '_gpu_devices'):
            container_name += f"-g{self._gpu_devices.replace(',', '_')}"

        return container_name

    def _setup_logging_dirs(self):
        current_time = time.strftime("%Y%m%d-%H%M%S")
        self._log_dir = Path("logs") / self._model_name / self.image_tag
        self._exp_tag = f"{Path(self._model_config).stem}"
        if not self._remote_server_endpoint:
            self._exp_tag += f"-tp{self._parallel_size.get('tp', '1')}"
            self.server_log_path = self._log_dir / self._exp_tag / "server_logs" / f"{self._parallel_size.get('tp', '1')}-{current_time}.txt"
        if not self._is_dry_run:
            self.server_log_path.parent.mkdir(parents=True, exist_ok=True)

    def _cache_dir(self, cache_name: str):
        self._host_cache_dir = Path.cwd() / cache_name / self._exp_tag
        self._host_cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_system_config(self, gpu_devices: Union[str, None], num_gpus: Union[int, None]):
        """Required benchmark system configurations."""
        if gpu_devices is None and num_gpus is None:
            raise AssertionError("GPU devices or number of GPUs must be specified.")
        if gpu_devices is not None and num_gpus is not None:
            raise ValueError("Only one of 'gpu_devices' or 'num_gpus' can be specified.")

        lead_gpu = 0
        if gpu_devices is not None:
            gpu_array = [dev.strip() for dev in gpu_devices.split(',') if dev.strip()]
            if not gpu_array:
                raise ValueError("gpu_devices string is invalid or empty.")
            gpu_devices = ",".join(gpu_array)
            num_gpus = len(gpu_array)
            lead_gpu = int(gpu_array[0])
        elif num_gpus is not None:
            num_gpus = int(num_gpus)
            if num_gpus <= 0:
                raise ValueError("num_gpus must be a positive integer.")
            gpu_devices = ",".join(map(str, range(num_gpus)))

        port = BENCHMARK_BASE_PORT + lead_gpu

        return  gpu_devices, num_gpus, port

    def _get_docker_run_common_command(self) -> List[str]:
        group_option = "keep-groups" if os.environ.get("SLURM_JOB_ID", None) else "video"
        cmd = [
            self._container_runtime, "run", "-d",
            "--name", self._container_name,
            "--device", "/dev/kfd", "--device", "/dev/dri", "--device", "/dev/mem",
            "--group-add", group_option,
            "--network=host",
            "--cap-add=CAP_SYS_ADMIN",
            "--cap-add=SYS_PTRACE",
            "--shm-size=16gb",
            "--security-opt", "seccomp=unconfined",
            "-e", f"CUDA_VISIBLE_DEVICES={self._gpu_devices}",
            "-v", f"{self.get_host_model_path()}:{self.get_model_path()}:ro",
        ]

        if os.environ.get('HF_HOME'):
            cmd.extend(["-v", f"{os.environ.get('HF_HOME')}:/root/.cache/huggingface"])

        for key in self._common_envs:
            cmd.extend(["-e", f"{key}={self._common_envs[key]}"])
        for key, value in self._env_vars.items():
            cmd.extend(["-e", f"{key}={value}"])

        return cmd

    def _is_docker_available(self) -> bool:
        """Check if Docker is installed on the system."""
        try:
            subprocess.run(["docker", "images"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            return True
        except (FileNotFoundError, subprocess.CalledProcessError):
            return False

    def _load_model_config(self) -> None:
        """Load model configuration from the specified config file."""
        config_path = Path(self._model_config)
        if not config_path.exists():
            raise FileNotFoundError(f"Model config file not found: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            config_content = f.read()
            model_config = yaml.safe_load(config_content)

        self._env_vars.update(model_config.get('envs', {}))

        if self._arch:
            arch_params = model_config.get('arch_specific_params', {})
            if self._arch in arch_params:
                self._env_vars.update(arch_params.get(self._arch, {}))
            else:
                logger.warning("Architecture '%s' not found in model config arch_specific_params.", self._arch)
        else:
            logger.info("No architecture specified. Skipping architecture-specific environment variables.")

        parallel_dict = model_config.get('parallel', {})
        if self._num_gpus in parallel_dict:
            if parallel_dict[self._num_gpus]:
                self._server_args.update(parallel_dict[self._num_gpus])

        self._server_args.update(model_config.get('server_args', {}))
        self._compilation_config = model_config.get('compilation_config', {})

    def _load_model_from_path_or_hub(self, model_path_or_id: str,
                                     model_root_dir: Optional[Union[str, Path]] = None) -> Path:
        def download_model(model_id: str, model_root_dir: Optional[Union[str, Path]] = None) -> str:
            cache_dir = os.environ.get("HF_HOME", None)
            token = os.environ.get("HF_TOKEN", None)
            if token is None:
                logger.warning("HF_TOKEN is not defined. Model may not be unavailable to download")
            if model_root_dir:
                model_save_dir = Path(model_root_dir) / model_id
                if not model_save_dir.exists():
                    model_save_dir.mkdir(parents=True, exist_ok=True)
                return snapshot_download(repo_id=model_id, local_dir=model_save_dir, cache_dir=cache_dir, token=token)
            return snapshot_download(repo_id=model_id, cache_dir=cache_dir, token=token)

        if model_root_dir is None:
            model_root_dir = Path.home()
        else:
            model_root_dir = Path(model_root_dir)
            if not model_root_dir.is_absolute():
                model_root_dir = Path.home() / model_root_dir

        if Path(model_path_or_id).is_absolute():
            if Path(model_path_or_id).exists():
                return Path(model_path_or_id)
            else:
                model_id = Path(model_path_or_id).relative_to(model_root_dir)
                if not self._is_dry_run:
                    download_model(str(model_id), model_root_dir)
                return Path(model_root_dir) / model_id

        if (Path.cwd() / model_path_or_id).exists():
            return (Path.cwd() / model_path_or_id).resolve()

        if (model_root_dir / model_path_or_id).exists():
            return model_root_dir / model_path_or_id

        if model_path_or_id.count('/') != 1:
            raise ValueError("Model id should be in the format of 'namespace/model_name'")
        if not self._is_dry_run:
            download_model(model_path_or_id, model_root_dir)
        return Path(model_root_dir) / model_path_or_id

    def get_model_path(self) -> str:
        """Select proper model path following execution mode"""
        if self.in_container:
            # model path is directly accessible path in container
            return str(self._model_path)
        else:
            # model path is translated path in container
            return str(self._container_model_path)

    def get_host_model_path(self) -> str:
        """Get the host model path. Used for host mounts for container execution."""
        return str(self._model_path)

    def _is_server_process_alive(self) -> bool:
        if self._is_dry_run:
            return True
        if self.in_container:
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
                response = requests.get(f"http://localhost:{self._port}/v1/models", timeout=10)
                if response.status_code == 200:
                    return True
            except requests.exceptions.RequestException:
                pass

            if not self._is_server_process_alive():
                logger.error("Server process is not running. Check server log: %s", self.server_log_path)
                return False

            elapsed_time = time.time() - start_time
            if elapsed_time > timeout:
                logger.error("Timeout waiting for server. Check log: %s", self.server_log_path)
                return False

            if time.time() - last_log_time > 60:
                last_log_time = time.time()
                logger.info("Waiting for server... %s seconds elapsed", int(elapsed_time))
            time.sleep(5)

    def cleanup_server_process(self):
        """Cleanup server process."""
        if self._is_dry_run:
            return

        if self.server_process:
            logger.info("Shutting down server process...")
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

    @property
    def in_container(self) -> bool:
        """Returns True if running in container mode."""
        return self._in_container

    @property
    def is_dry_run(self) -> bool:
        """Returns True if dry run mode is enabled."""
        return self._is_dry_run

    @property
    def model_name(self) -> str:
        """Returns the model name."""
        return self._model_name

    @property
    def model_path_or_id(self) -> str:
        """Returns the model path or ID."""
        return self.model_path_or_id

    @property
    def exp_tag(self) -> str:
        """Returns the experiment tag."""
        return self._exp_tag

    @property
    def model_config(self) -> str:
        """Returns the model config."""
        return self._model_config

    @property
    def parallel_size(self) -> Dict[str, str]:
        """Returns the parallel size."""
        return self._parallel_size

    @property
    def gpu_devices(self) -> str:
        """Returns the GPU devices string."""
        if not hasattr(self, '_gpu_devices'):
            return -1
        return self._gpu_devices

    @property
    def num_gpus(self) -> int:
        """Returns the number of GPUs."""
        return self._num_gpus

    @property
    def image_tag(self) -> str:
        """Returns the Docker image tag."""
        return self.image.split(':')[-1]

    @property
    def container_runtime(self) -> str:
        """Returns the container runtime."""
        return self._container_runtime

    @property
    def container_name(self) -> str:
        """Returns the container name."""
        return self._container_name

    @property
    def addr(self) -> str:
        """Returns the server address"""
        if not self._remote_server_endpoint:
            return "0.0.0.0"
        parsed_url = urlparse(self._remote_server_endpoint)
        return parsed_url.hostname

    @property
    def port(self) -> int:
        """Returns the port number."""
        if not self._remote_server_endpoint:
            return self._port
        parsed_url = urlparse(self._remote_server_endpoint)
        return parsed_url.port

    @property
    def endpoint(self) -> str:
        """returns server endpoint"""
        return f"{self.addr}:{self.port}"

    def generate_script(self, generator: ScriptGenerator):
        """Generates the server-side portion of the benchmark script."""
        generator.add_variable("MODEL_NAME", self.model_name)
        generator.add_variable("MODEL_PATH", self.get_model_path())
        generator.add_variable("IMAGE", self.image)
        if hasattr(self, '_parallel_size'):
            generator.add_variable("TP_SIZE", self.parallel_size.get("tp", 1))
        generator.add_variable("PORT", self.port)
        if hasattr(self, '_container_name'):
            generator.add_variable("CONTAINER_NAME", self._container_name)

        for key, value in self._env_vars.items():
            generator.add_env_variable(key, value)

        # This method should be overridden by subclasses to add specific commands
        logger.warning("generate_script() not fully implemented for %s", self.__class__.__name__)