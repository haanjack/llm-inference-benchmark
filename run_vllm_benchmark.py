#!/usr/bin/env python3

import argparse
import logging
import os
import subprocess
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Union
import re
import tempfile
import itertools
import requests
import yaml
import dotenv

from huggingface_hub import snapshot_download

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

BENCHMARK_BASE_PORT = 23400

# pylint: disable=line-too-long

def get_args():
    """Benchmark arguments"""
    parser = argparse.ArgumentParser(description='Run vLLM benchmarks')

    # benchmark configuration
    parser.add_argument('--model-config', required=True,
                        help='Model config file path')
    parser.add_argument('--model-path-or-id', required=True,
                        help='Model checkpoint path or model id in huggingface hub')
    parser.add_argument('--vllm-image', required=True,
                        help='vLLM Docker image.')
    parser.add_argument('--test-plan', default='test',
                        help='Benchmark test plan YAML file in configs/benchmark_plans/ \
                            (without .yaml extension)')
    parser.add_argument('--sub-tasks', default=None, type=str, nargs='+',
                        help='Testing sub-tasks in test-plan')
    parser.add_argument('--env-file', default="configs/envs/common",
                        help='Environment file name')
    parser.add_argument('--model-root-dir', default="models",
                        help='Model root directory')
    parser.add_argument('--gpu-devices', default=None,
                        help='Comma-separated GPU device IDs')
    parser.add_argument('--num-gpus', default=None,
                        help='Number of GPUs')
    parser.add_argument('--arch', default=None,
                        help='Target GPU architecture for model config')

    # test control
    parser.add_argument('--no-warmup', action='store_true',
                        help='no warmup at benchmark start')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show commands without executing them')
    parser.add_argument('--in-container', action='store_true',
                        help='Run benchmark directly without launching a new container')

    args = parser.parse_args()

    return args


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


class VLLMServer(BenchmarkBase):
    """vLLM Server management."""
    def __init__(self, vllm_image: str, **kwargs):
        super().__init__(**kwargs)
        self._vllm_image = vllm_image
        self.server_process = None
        self.temp_compile_config_file = None
        self._log_process: Optional[subprocess.Popen] = None

        self._setup_container_name()
        self._setup_logging_dirs()
        self._cache_dir()

    def _setup_container_name(self):
        image_tag = self._vllm_image.split(':')[-1]
        slurm_job_id = os.environ.get("SLURM_JOB_ID", None)
        self._container_name = ""
        if slurm_job_id:
            self._container_name = f"{slurm_job_id}-"
        self._container_name += f"{os.path.basename(self._model_name)}-{image_tag}-g{self._gpu_devices.replace(',', '_')}"

    def _setup_logging_dirs(self):
        image_tag = self._vllm_image.split(':')[-1]
        self._log_dir = Path("logs") / self._model_name / image_tag
        self.server_log = self._log_dir / "server_logs" / f"{os.path.basename(self._model_name)}-{image_tag}-t{self._parallel_size.get('tp', '1')}.txt"
        self.server_log.parent.mkdir(parents=True, exist_ok=True)
        self._exp_tag = f"{Path(self._model_config).stem}_tp{self._parallel_size.get('tp', '1')}"

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
            "--name", self._container_name,
            "-v", f"{os.environ.get('HF_HOME')}:/root/.cache/huggingface",
            "--device", "/dev/kfd", "--device", "/dev/dri", "--device", "/dev/mem",
            "--group-add", group_option,
            "--network=host",
            "--cap-add=CAP_SYS_ADMIN", "--cap-add=SYS_PTRACE",
            "--shm-size=16gb",
            "--security-opt", "seccomp=unconfined",
            "-e", f"CUDA_VISIBLE_DEVICES={self._gpu_devices}",
            "--env-file", self._common_env_file,
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
        ]
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

    def start(self, no_enable_prefix_caching: bool):
        """Start vLLM server."""
        if self._in_container:
            self._start_server_direct(no_enable_prefix_caching)
        else:
            self._start_server_container(no_enable_prefix_caching)

        if not self._is_dry_run:
            if not self._wait_for_server():
                raise RuntimeError("Server failed to start")
            logger.info("Server is up and running")

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

        with open(self.server_log, 'a', encoding='utf-8') as f:
            if self._is_dry_run:
                return
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


class BenchmarkRunner:
    """Benchmark runner."""
    def __init__(self, server: VLLMServer, test_plan: str, sub_tasks: List[str] = None, no_warmup: bool = False):
        self.server = server
        self._test_plan = test_plan
        self._sub_tasks = sub_tasks
        self._is_no_warmup = no_warmup
        self._test_plan_path = Path(f"configs/benchmark_plans/{test_plan}.yaml")
        self._columns = [
            ("Model Config", 16), ("TP", 8), ("Req Rate", 8), ("Num Prompts", 11),
            ("Batch", 8), ("Conc", 8), ("In Len", 8), ("Out Len", 8),
            ("Test Time(s)", 10), ("TTFT Mean(ms)", 10), ("TTFT Med(ms)", 10), ("TTFT P99(ms)", 10),
            ("TPOT Mean(ms)", 10), ("TPOT Med(ms)", 10), ("TPOT P99(ms)", 10),
            ("ITL Mean(ms)", 10), ("ITL Med(ms)", 10), ("ITL P99(ms)", 10),
            ("E2E Mean(ms)", 10), ("E2E Med(ms)", 10), ("E2E P99(ms)", 10),
            ("Req req/s", 10), ("Out Tok/s", 10), ("Total Tok/s", 10)
        ]
        self._csv_headers = [
            "Model Config", "TP Size", "Request Rate", "Num. Prompts", "Batch Size", "Concurrency",
            "Input Length", "Output Length", "Test Time(s)", "Mean TTFT(ms)", "Median TTFT(ms)",
            "P99 TTFT(ms)", "Mean TPOT(ms)", "Median TPOT(ms)", "P99 TPOT(ms)", "Mean ITL(ms)",
            "Median ITL(ms)", "P99 ITL(ms)", "Mean E2EL(ms)", "Median E2EL(ms)", "P99 E2EL(ms)",
            "Request Tput(req/s)", "Output Tput(tok/s)", "Total Tput(tok/s)"
        ]
        self._setup_logging_dirs()
        if not self._test_plan_path.exists() and not self.server.is_dry_run:
            raise FileNotFoundError(f"Could not find test plan: {self._test_plan_path}.")

        self._print_benchmark_info()

    def _setup_logging_dirs(self):
        self._log_dir = Path("logs") / self.server.model_name / self.server.vllm_image
        self.result_file = self._log_dir / "result_list.csv"
        self.result_file.parent.mkdir(parents=True, exist_ok=True)
        if not self.server.is_dry_run and not self.result_file.exists():
            with open(self.result_file, 'w', encoding='utf-8') as f:
                f.write(','.join(self._csv_headers) + '\n')

    def _print_benchmark_info(self):
        logger.info("Start vLLM benchmark")
        logger.info("Model Name: %s", self.server.model_name)
        logger.info("vLLM docker image: %s", self.server.vllm_image)
        logger.info("GPU devices: %s", self.server.gpu_devices)
        logger.info("Benchmark plan: %s", self._test_plan)
        logger.info("Benchmark test plan:")
        try:
            with open(self._test_plan_path, mode="r", encoding="utf-8") as f:
                plan_content = f.read()
                indented_content = '\n'.join('    ' + line for line in plan_content.splitlines())
                logger.info("\n%s", indented_content)
        except Exception as e:
            logger.warning("Could not read test plan file '%s': %s", self._test_plan_path, e)

    def _load_test_plan(self):
        with open(self._test_plan_path, mode="r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        dataset_name = config.get('dataset_name', 'random')
        no_enable_prefix_caching = (dataset_name == 'random')

        def ensure_list(value, default):
            if value is None:
                return default
            if isinstance(value, (int, float)):
                return [value]
            return value

        test_plans = []
        for scenario in config.get('test_scenarios', []):
            if self._sub_tasks and scenario.get('name') not in self._sub_tasks:
                continue
            if self._sub_tasks:
                logger.info("Sub task selected: %s", scenario.get('name'))

            if 'num_iteration' in scenario and 'num_prompts' in scenario:
                raise AssertionError("num_iteration and num_prompts are exclusive.")

            params = {
                'request_rates': ensure_list(scenario.get('request_rate'), [0]),
                'concurrencies': ensure_list(scenario.get('concurrency'), [1]),
                'input_lengths': ensure_list(scenario.get('input_length'), [512]),
                'output_lengths': ensure_list(scenario.get('output_length'), [128]),
                'num_iterations': ensure_list(scenario.get('num_iteration'), [8 if 'num_prompts' not in scenario else 1]),
                'num_prompts': ensure_list(scenario.get('num_prompts'), [1000 if 'num_iteration' not in scenario else 1]),
                'batch_sizes': ensure_list(scenario.get('batch_size'), [256])
            }
            dataset_name_ = scenario.get('dataset_name', dataset_name)
            if dataset_name == 'random' and dataset_name_ != 'random':
                logger.warning('Benchmark with non-random dataset with no-enable-prefix-caching.')

            for rate, batch, num_iter, num_prompts_val, in_len, out_len, conc in itertools.product(
                    params['request_rates'], params['batch_sizes'], params['num_iterations'],
                    params['num_prompts'], params['input_lengths'], params['output_lengths'],
                    params['concurrencies']):
                num_prompts_final = conc * num_iter if 'num_iteration' in scenario else num_prompts_val
                test_plans.append({
                    'request_rate': rate, 'concurrency': conc, 'input_length': in_len,
                    'output_length': out_len, 'num_prompts': num_prompts_final,
                    'batch_size': batch, 'dataset_name': dataset_name_
                })
        if not test_plans:
            raise ValueError("No test scenarios loaded.")
        return test_plans, no_enable_prefix_caching

    def run(self):
        if self.server.num_gpus == 0:
            raise ValueError("No GPU is allocated")

        test_plans, no_enable_prefix_caching = self._load_test_plan()

        try:
            self.server.start(no_enable_prefix_caching)
            self._warmup_server()
            self._print_header()
            self._run_vllm_benchmark(test_plans)
        finally:
            self.server.cleanup()
            if not self.server.is_dry_run and not self.server.in_container:
                logger.info("Benchmarking complete. Results saved to %s", self.result_file)

    def _warmup_server(self):
        if self.server.is_dry_run or self._is_no_warmup:
            logger.info("Skipping warmup.")
            return

        logger.info("Warming up the server...")
        warmup_cmd = []
        if not self.server.in_container:
            warmup_cmd.extend([self.server.container_runtime, "exec", self.server.container_name])
        warmup_cmd.extend([
            "vllm", "bench", "serve", "--model", self.server.get_model_path(),
            "--backend", "vllm", "--host", "localhost", f"--port={self.server.vllm_port}",
            "--dataset-name", "random", "--ignore-eos", "--trust-remote-code",
            "--request-rate=10", "--max-concurrency=1", "--num-prompts=4",
            "--random-input-len=16", "--random-output-len=16",
            "--tokenizer", self.server.get_model_path(), "--disable-tqdm"
        ])
        start_time = time.time()
        subprocess.run(warmup_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        logger.info("Warmup complete in %.2f seconds.", time.time() - start_time)

    def _print_header(self):
        """Print result's table header."""
        if self.server.is_dry_run:
            return

        header_line1 = []
        header_line2 = []
        for header, width in self._columns:
            parts = header.split(' ', 1)
            header_line1.append(parts[0].rjust(width))
            header_line2.append(parts[1].rjust(width) if len(parts) > 1 else ' '.rjust(width))
        logger.info(' '.join(header_line1))
        logger.info(' '.join(header_line2))

    def _run_vllm_benchmark(self, test_plans: List[Dict]):
        for test_plan in test_plans:
            try:
                self.run_single_benchmark(**test_plan)
                if self.server.is_dry_run:
                    if input("Continue? (Y/n) ").lower() in ['n', 'no']:
                        break
            except subprocess.CalledProcessError as e:
                logger.error("Benchmark failed for plan: %s", test_plan)
                logger.error("%s", str(e).rsplit('\n', maxsplit=1)[-1])
                return

    def run_single_benchmark(self, request_rate: int, concurrency: int, input_length: int,
                             output_length: int, num_prompts: int, batch_size: int, dataset_name: str):
        """Run a benchmark."""
        cmd = []
        if not self.server.in_container:
            cmd.extend([self.server.container_runtime, "exec", self.server.container_name])
        cmd.extend([
            "vllm", "bench", "serve", "--model", self.server.get_model_path(),
            "--backend", "vllm", "--host", "localhost", f"--port={self.server.vllm_port}",
            f"--dataset-name={dataset_name}", "--ignore-eos", "--trust-remote-code",
            f"--request-rate={request_rate if request_rate > 0 else 'inf'}",
            f"--max-concurrency={concurrency}", f"--num-prompts={num_prompts}",
            f"--random-input-len={input_length}", f"--random-output-len={output_length}",
            "--tokenizer", self.server.get_model_path(), "--disable-tqdm",
            "--percentile-metrics", "ttft,tpot,itl,e2el"
        ])

        if self.server.is_dry_run:
            logger.info("Dry run - Benchmark command: %s", " ".join(cmd))
            return

        if self._check_existing_result(request_rate, concurrency, input_length, output_length, num_prompts, batch_size):
            return

        log_file = self._log_dir / self.server.exp_tag / f"r{request_rate}_n{num_prompts}_b{batch_size}_{input_length}_o{output_length}_c{concurrency}.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"=== Benchmark: {request_rate}, {num_prompts}, {batch_size}, {concurrency}, {input_length}, {output_length} ===\n")
            subprocess.run(cmd, stdout=f, stderr=f, check=True)

        metrics = self._extract_metrics(log_file)
        self._print_result(request_rate, num_prompts, batch_size, concurrency, input_length, output_length, metrics)
        self._save_results(request_rate, num_prompts, batch_size, concurrency, input_length, output_length, metrics)

    def _check_existing_result(self, request_rate, concurrency, input_length, output_length, num_prompts, batch_size) -> bool:
        if not self.result_file.exists() or self.server.is_dry_run:
            return False
        search_str = f"{Path(self.server.model_config).stem},{self.server.parallel_size.get('tp', '1')},{request_rate},{num_prompts},{batch_size},{concurrency},{input_length},{output_length}"
        with open(self.result_file, 'r', encoding='utf-8') as f:
            for line in f:
                if search_str in line:
                    logger.info(self._format_result_for_console(line.strip().split(',')))
                    return True
        return False

    def _extract_metrics(self, log_file: Path) -> Dict[str, float]:
        metrics = {}
        patterns = {
            'test_time': r'Benchmark duration \(s\):\s*([\d.]+)',
            'ttft_mean': r'Mean TTFT \(ms\):\s*([\d.]+)',
            'ttft_median': r'Median TTFT \(ms\):\s*([\d.]+)',
            'ttft_p99': r'P99 TTFT \(ms\):\s*([\d.]+)',
            'tpot_mean': r'Mean TPOT \(ms\):\s*([\d.]+)',
            'tpot_median': r'Median TPOT \(ms\):\s*([\d.]+)',
            'tpot_p99': r'P99 TPOT \(ms\):\s*([\d.]+)',
            'itl_mean': r'Mean ITL \(ms\):\s*([\d.]+)',
            'itl_median': r'Median ITL \(ms\):\s*([\d.]+)',
            'itl_p99': r'P99 ITL \(ms\):\s*([\d.]+)',
            'e2el_mean': r'Mean E2EL \(ms\):\s*([\d.]+)',
            'e2el_median': r'Median E2EL \(ms\):\s*([\d.]+)',
            'e2el_p99': r'P99 E2EL \(ms\):\s*([\d.]+)',
            'request_throughput': r'Request throughput \(req/s\):\s*([\d.]+)',
            'output_token_throughput': r'Output token throughput \(tok/s\):\s*([\d.]+)',
            'total_token_throughput': r'Total Token throughput \(tok/s\):\s*([\d.]+)'
        }
        log_content = log_file.read_text()
        for key, pattern in patterns.items():
            match = re.search(pattern, log_content)
            metrics[key] = float(match.group(1)) if match else 0.0
        return metrics

    def _save_results(self, request_rate: int, num_prompts: int, batch_size: int, concurrency: int, input_length: int, output_length: int, metrics: Dict[str, float]):
        result_line = (
            f"{Path(self.server.model_config).stem},{self.server.parallel_size.get('tp', '1')},"
            f"{request_rate},{num_prompts},{batch_size},{concurrency},{input_length},{output_length},{metrics['test_time']:.2f},"
            f"{metrics['ttft_mean']:.2f},{metrics['ttft_median']:.2f},{metrics['ttft_p99']:.2f},"
            f"{metrics['tpot_mean']:.2f},{metrics['tpot_median']:.2f},{metrics['tpot_p99']:.2f},"
            f"{metrics['itl_mean']:.2f},{metrics['itl_median']:.2f},{metrics['itl_p99']:.2f},"
            f"{metrics['e2el_mean']:.2f},{metrics['e2el_median']:.2f},{metrics['e2el_p99']:.2f},"
            f"{metrics['request_throughput']:.2f},{metrics['output_token_throughput']:.2f},{metrics['total_token_throughput']:.2f}\n"
        )
        with open(self.result_file, 'a', encoding='utf-8') as f:
            f.write(result_line)

    def _format_result_for_console(self, values: List[str]) -> str:
        if len(values) != len(self._columns):
            logger.warning("Mismatch between result values and column definitions.")
            return ' '.join(values)
        formatted_values = [os.path.basename(values[0]).ljust(self._columns[0][1])]
        formatted_values.extend(val.rjust(width) for val, (_, width) in zip(values[1:], self._columns[1:]))
        return ' '.join(formatted_values)

    def _print_result(self, request_rate: int, num_prompts: int, batch_size: int, concurrency: int, input_length: int, output_length: int, metrics: Dict[str, float]):
        values = [
            Path(self.server.model_config).stem, str(self.server.parallel_size.get('tp', '1')),
            str(request_rate), str(num_prompts), str(batch_size), str(concurrency), str(input_length), str(output_length), f"{metrics['test_time']:.2f}",
            f"{metrics['ttft_mean']:.2f}", f"{metrics['ttft_median']:.2f}", f"{metrics['ttft_p99']:.2f}",
            f"{metrics['tpot_mean']:.2f}", f"{metrics['tpot_median']:.2f}", f"{metrics['tpot_p99']:.2f}",
            f"{metrics['itl_mean']:.2f}", f"{metrics['itl_median']:.2f}", f"{metrics['itl_p99']:.2f}",
            f"{metrics['e2el_mean']:.2f}", f"{metrics['e2el_median']:.2f}", f"{metrics['e2el_p99']:.2f}",
            f"{metrics['request_throughput']:.2f}", f"{metrics['output_token_throughput']:.2f}", f"{metrics['total_token_throughput']:.2f}"
        ]
        logger.info(self._format_result_for_console(values))


def main():
    """Main function to run the benchmark."""
    try:
        args = get_args()

        server = VLLMServer(
            env_file=args.env_file,
            model_config=args.model_config,
            model_path_or_id=args.model_path_or_id,
            model_root_dir=args.model_root_dir,
            vllm_image=args.vllm_image,
            gpu_devices=args.gpu_devices,
            num_gpus=args.num_gpus,
            arch=args.arch,
            dry_run=args.dry_run,
            in_container=args.in_container
        )

        runner = BenchmarkRunner(
            server=server,
            test_plan=args.test_plan,
            sub_tasks=args.sub_tasks, no_warmup=args.no_warmup
        )

        runner.run()
    except Exception as e:
        logger.exception("Benchmark failed: %s", str(e))
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Benchmark interrupted by user.")
        sys.exit(1)

if __name__ == "__main__":
    main()
