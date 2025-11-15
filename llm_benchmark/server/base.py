import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import yaml
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
                 env_file: str = None,
                 model_path_or_id: str = None,
                 model_root_dir: str = None,
                 model_config: str = None,
                 gpu_devices: str = None,
                 num_gpus: int = None,
                 arch: str = None,
                 dry_run: bool = False,
                 in_container: bool = False):

        self._common_env_file = env_file
        self._env_vars = {}
        self._vllm_args = {}
        self._compilation_config = {}
        self._arch = arch
        self._model_config = model_config
        self._is_dry_run = dry_run
        self._in_container = in_container

        self._system_config(gpu_devices, num_gpus)
        self._parallel_size = {'tp': str(self._num_gpus)}

        self._load_model_config()

        self._model_path = self._load_model_from_path_or_hub(model_path_or_id, model_root_dir)
        self._model_name = self._model_path.name
        self._container_model_path = Path(f"/models/{self._model_name}")

        if not self._model_path.exists() and not self._is_dry_run:
            raise FileNotFoundError(f"Could not find model at {self._model_name} in {self.get_model_path()}.")

        self._container_runtime = None
        if not self._in_container:
            self._container_runtime = "docker" if self._is_docker_available() else "podman"

    def _system_config(self, gpu_devices: Union[str, None], num_gpus: Union[int, None]):
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
            self._gpu_devices = ",".join(gpu_array)
            self._num_gpus = len(gpu_array)
            lead_gpu = int(gpu_array[0])
        elif num_gpus is not None:
            self._num_gpus = int(num_gpus)
            if self._num_gpus <= 0:
                raise ValueError("num_gpus must be a positive integer.")
            self._gpu_devices = ",".join(map(str, range(self._num_gpus)))

        self._vllm_port = BENCHMARK_BASE_PORT + lead_gpu

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
                self._vllm_args.update(parallel_dict[self._num_gpus])

        self._vllm_args.update(model_config.get('vllm_server_args', {}))
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
        if self._in_container:
            # model path is directly accessible path in container
            return str(self._model_path)
        else:
            # model path is translated path in container
            return str(self._container_model_path)

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
    def gpu_devices(self) -> str:
        """Returns the GPU devices string."""
        return self._gpu_devices

    @property
    def num_gpus(self) -> int:
        """Returns the number of GPUs."""
        return self._num_gpus
