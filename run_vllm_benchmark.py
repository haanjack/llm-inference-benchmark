#!/usr/bin/env python3

import argparse
import logging
import os
import subprocess
import sys
import time
import datetime
import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import re
import requests
import tempfile

from huggingface_hub import snapshot_download

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


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
                        help='Benchmark test plan YAML file in configs/benchmark_plans/ (without .yaml extension)')
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


class VLLMBenchmark:
    def __init__(self,
                 env_file: str = None,
                 vllm_image: str = None,
                 model_path_or_id: str = None,
                 model_root_dir: str = None,
                 model_config: str = None,
                 test_plan: str = "test",
                 gpu_devices: str = None,
                 num_gpus: int = None,
                 arch: str = None,
                 no_warmup: bool = False,
                 dry_run: bool = False,
                 in_container: bool = False):

        # benchmark configuration
        self._common_env_file = env_file
        self._env_vars = {}
        self._vllm_args = {}
        self._compilation_config = {}
        self._arch = arch
        self._model_config = model_config
        self._is_no_warmup = no_warmup
        self._is_dry_run = dry_run
        self._in_container = in_container

        # GPU configuration
        self._system_config(gpu_devices, num_gpus)
        # TODO: currently only support tp. apply dp, pp.
        self._parallel_size = {
            'tp': str(self._num_gpus)
        }

        self._load_model_config()

        self._model_path = self._load_model_from_path_or_hub(model_path_or_id, model_root_dir)
        self._model_name = self._model_path.name
        self._container_model_path = Path(f"/models/{self._model_name}")
        self._vllm_image = vllm_image
        self._test_plan = test_plan
        self._test_plan_path = Path(f"configs/benchmark_plans/{test_plan}.yaml")

        # Sanity Check
        if not self._test_plan_path.exists() and not self._is_dry_run:
            raise FileNotFoundError(f"Could not find test plan: {self._test_plan_path}. Please check the plan name")
        # check host model path
        if not self._model_path.exists() and not self._is_dry_run:
            raise FileNotFoundError(f"Could not find model at {self._model_name} in {self._get_model_path()}.")

        # Column definitions (headers and widths) for console and CSV
        self._columns = [
            # Configs
            ("Model Config", 16), ("TP", 8), ("Req Rate", 8), ("Num Iter", 8),
            ("Batch", 8), ("Conc", 8), ("In Len", 8), ("Out Len", 8),
            ("Test Time", 8),
            # TTFT
            ("TTFT Mean", 10), ("TTFT Med", 10), ("TTFT P99", 10),
            # TPOT
            ("TPOT Mean", 10), ("TPOT Med", 10), ("TPOT P99", 10),
            # ITL
            ("ITL Mean", 10), ("ITL Med", 10), ("ITL P99", 10),
            # E2E Latency
            ("E2E Mean", 10), ("E2E Med", 10), ("E2E P99", 10),
            # Throughput
            ("Req/s", 10), ("Out Tok/s", 10), ("Total Tok/s", 10)
        ]
        self._csv_headers = [
            "Model Config", "TP Size", "Request Rate", "Num. Iter", "Batch Size", "Concurrency",
            "Input Length", "Output Length", "Test Time(s)", "Mean TTFT(ms)", "Median TTFT(ms)",
            "P99 TTFT(ms)", "Mean TPOT(ms)", "Median TPOT(ms)", "P99 TPOT(ms)", "Mean ITL(ms)",
            "Median ITL(ms)", "P99 ITL(ms)", "Mean E2EL(ms)", "Median E2EL(ms)", "P99 E2EL(ms)",
            "Request Tput(req/s)", "Output Tput(tok/s)", "Total Tput(tok/s)"
        ]

        # determine docker or podman
        if not self._in_container:
            self._container_runtime = "docker" if self._is_docker_available() else "podman"

        # Container name setup
        self._setup_container_name()

        # Setup logging directories
        self._setup_logging_dirs()
        self._cache_dir()

        self._print_benchmark_info()

        # For direct subprocess management
        self.server_process = None

    def _system_config(self, gpu_devices: Union[str, None], num_gpus: Union[int, None]):
        if gpu_devices is None and num_gpus is None:
            raise AssertionError("GPU devices or number of GPUs must be specified.")
        if gpu_devices is not None and num_gpus is not None:
            raise AssertionError("Only one of 'gpu_devices' or 'num_gpus' can be specified.")

        if gpu_devices is not None:
            # TODO: select based on os.environ.get("SLURM_JOB_GPUS", "")
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
            lead_gpu = 0

        # VLLM port setup
        self.vllm_port = 23400 + lead_gpu

    def _cache_dir(self):
        """Configure vllm cache directory which to reduce compilation overhead."""
        self._host_cache_dir = Path.cwd() / "vllm_cache" / self._exp_tag
        self._host_cache_dir.mkdir(parents=True, exist_ok=True)

        self._aiter_cache_dir = self._host_cache_dir / "aiter"
        self._aiter_cache_dir.mkdir(parents=True, exist_ok=True)

        self._compile_cache_dir = self._host_cache_dir / "compile_config"
        self._compile_cache_dir.mkdir(parents=True, exist_ok=True)

    def _print_benchmark_info(self):
        """Print benchmark configuration and test plan information."""
        logger.info("Start vLLM benchmark")
        logger.info(f"Model Name: {self._model_name}")
        logger.info(f"vLLM docker image: {self._vllm_image}")
        logger.info(f"Benchmark plan: {self._test_plan}")

        # Print the test plan YAML content
        logger.info("Benchmark test plan:")
        try:
            with open(self._test_plan_path) as f:
                plan_content = f.read()
                # Add indentation to make the YAML content more readable in logs
                indented_content = '\n'.join('    ' + line for line in plan_content.splitlines())
                logger.info(f"\n{indented_content}")
        except FileNotFoundError:
            logger.warning(f"Could not find test plan file: {self._test_plan_path}")
        except Exception as e:
            logger.warning(f"Error reading test plan: {str(e)}")

    def _is_docker_available(self) -> bool:
        """Check if Docker is installed on the system."""
        try:
            return subprocess.run(["docker", "images"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
        except FileNotFoundError:
            return False

    def _load_model_config(self) -> None:
        """Load model configuration from the specified config file."""
        config_path = Path(self._model_config)
        if not config_path.exists():
            raise FileNotFoundError(f"Model config file not found: {config_path}")

        with open(config_path) as f:
            config_content = f.read()
            # Parse the YAML content
            model_config = yaml.safe_load(config_content)

        env_vars = model_config.get('envs', {})
        self._env_vars.update(env_vars)

        # apply arch specific params
        if self._arch:
            arch_params = model_config.get('arch_specific_params', {})
            if self._arch in arch_params:
                self._env_vars.update(arch_params.get(self._arch, {}))
            else:
                logger.warning(f"Architecture '{self._arch}' not found in model config arch_specific_params. Skipping architecture-specific environment variables.")
        else:
            logger.info("No architecture specified. Skipping architecture-specific environment variables.")

        # apply tp specific arguments
        parallel_dict = model_config.get('parallel', {})
        if self._num_gpus in parallel_dict:
            if parallel_dict[self._num_gpus]:
                self._vllm_args.update(parallel_dict[self._num_gpus])

        vllm_server_args = model_config.get('vllm_server_args', {})
        self._vllm_args.update(vllm_server_args)

        compilation_config = model_config.get('compilation_config', {})
        self._compilation_config = compilation_config

    def _load_model_from_path_or_hub(self, model_path_or_id: str,
                                     model_root_dir: Optional[Union[str, Path]] = None
        ) -> Path:

        # download model under save_root_dir or cache
        def download_model(model_id: str,
                           model_root_dir: Optional[Union[str, Path]] = None) -> str:
            cache_dir = os.environ.get("HF_HOME", None)
            token = os.environ.get("HF_TOKEN", None)
            if token is None:
                logger.warning("HF_TOKEN is not defined. Model may not be unavailable to download")
            if model_root_dir:
                model_save_dir = Path(model_root_dir) / model_id
                if not model_save_dir.exists():
                    model_save_dir.mkdir(parents=True, exist_ok=True)

                return snapshot_download(
                    repo_id=model_id,
                    local_dir=model_save_dir,
                    cache_dir=cache_dir,
                    token=token
                )
            return snapshot_download(
                repo_id=model_id,
                cache_dir=cache_dir,
                token=token
            )

        # set model root dir
        model_root_dir = model_root_dir if Path(model_root_dir).is_absolute() else Path.home() / model_root_dir

        # absolute path
        if Path(model_path_or_id).is_absolute():
            if Path(model_path_or_id).exists():
                return Path(model_path_or_id)
            else:
                model_id = Path(model_path_or_id).relative_to(model_root_dir)
                if not self._is_dry_run:
                    download_model(str(model_id), model_root_dir)
                return Path(model_root_dir) / model_id

        # relative path
        if (Path.cwd() / model_path_or_id).exists():
            return str((Path.cwd() / model_path_or_id).resolve())

        # model id from huggingface hub
        # check if it is in model_root_dir
        if (model_root_dir / model_path_or_id).exists():
            return model_root_dir / model_path_or_id

        # download from huggingface hub (now model_path_or_id is model_id)
        assert model_path_or_id.count('/') == 1, "Model id should be in the format of 'namespace/model_name'"
        if not self._is_dry_run:
            download_model(model_path_or_id, model_root_dir)
        return Path(model_root_dir) / model_path_or_id

    def _setup_container_name(self):
        """Setup container name based on environment and GPU configuration."""
        image_tag = self._vllm_image.split(':')[-1]
        slurm_job_id = os.environ.get("SLURM_JOB_ID", None)

        self.container_name = ""
        if slurm_job_id:
            self.container_name = f"{slurm_job_id}-"
        self.container_name += f"{os.path.basename(self._model_name)}-{image_tag}-g{self._gpu_devices.replace(',', '_')}"

    def _setup_logging_dirs(self):
        """Setup logging directories for the benchmark."""
        image_tag = self._vllm_image.split(':')[-1]
        self._log_dir = Path("logs") / self._model_name / image_tag
        self._log_dir.mkdir(parents=True, exist_ok=True)

        self.result_file = self._log_dir / "result_list.csv"
        self.result_file.parent.mkdir(parents=True, exist_ok=True)

        self.server_log = self._log_dir / "server_logs" / f"{os.path.basename(self._model_name)}-{image_tag}-t{self._parallel_size.get('tp', '1')}.txt"
        self.server_log.parent.mkdir(parents=True, exist_ok=True)

        self._exp_tag = f"{Path(self._model_config).stem}_tp{self._parallel_size.get('tp', '1')}"

        # Initialize result file if it doesn't exist
        if not self.result_file.exists():
            self._init_result_file()

    def _init_result_file(self):
        """Initialize the result file with headers."""
        with open(self.result_file, 'w') as f:
            f.write(','.join(self._csv_headers) + '\n')

    def _get_model_path(self) -> str:
        """Select proper model path following execution mode"""
        return self._model_path if self._in_container else self._container_model_path

    def _print_header(self):
        """Print the header line to console."""
        if self._is_dry_run:
            return

        # print header line to console
        header_line1 = []
        header_line2 = []
        for header, width in self._columns:
            parts = header.split(' ', 1)
            header_line1.append(parts[0].rjust(width))
            header_line2.append(parts[1].rjust(width) if len(parts) > 1 else ' '.rjust(width))

        logger.info(' '.join(header_line1))
        logger.info(' '.join(header_line2))

    def _build_vllm_args(self) -> str:
        """Construct VLLM arguments based on environment variables."""
        # vllm args
        args = []
        for key, value in self._vllm_args.items():
            if key == "quantization" and value == "auto":
                continue
            if isinstance(value, bool):
                if value:
                    args.append(f"--{key.replace('_', '-')}")
            else:
                args.extend([f"--{key.replace('_', '-')}", str(value)])

        # compilation config
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml",
                                         dir=self._compile_cache_dir,
                                         encoding="utf-8", delete=False) as f:
            dict_config_str = json.dumps(self._compilation_config, separators=(',', ':'))
            f.write(f"compilation_config: '{dict_config_str}'\n")
            args.extend(["--config", str(Path("root") / ".cache" / "compile_config" / f.name)])
            self.temp_compile_config_file = f.name

        return " ".join(args)

    def _cleanup_log_processes(self):
        """Terminate log collection processes if they exist."""
        if hasattr(self, 'log_processes') and self.log_processes:
            for process in self.log_processes:
                try:
                    process.terminate()
                    process.wait(timeout=5)  # Wait for graceful termination
                except (subprocess.TimeoutExpired, ProcessLookupError):
                    process.kill()  # Force kill if graceful termination fails

    def _cleanup_server_process(self):
        """Terminate the direct server process if it exists."""
        if self.server_process:
            logger.info("Shutting down vLLM server process...")
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.server_process.kill()

    def _cleanup_container(self, container_name):
        """Remove the Docker container if it exists."""
        self._cleanup_log_processes()
        subprocess.run([self._container_runtime, "rm", "-f", container_name],
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def get_server_run_cmd(self) -> str:
        """Construct the Docker run command for the vLLM server."""
        group_option="keep-groups" if os.environ.get("SLURM_JOB_ID", None) else "video"
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

        # set volume mounts and run server command
        cmd.extend([
            "-v", f"{self._model_path}:{self._get_model_path()}:ro",
            "-v", f"{self._host_cache_dir}:/root/.cache",
            "-v", f"{self._compile_cache_dir}:/root/.cache/compile_config",
            "-v", f"{self._aiter_cache_dir}:/root/.aiter",
            "-v", f"{os.environ.get('HOME')}:{os.environ.get('HOME')}",
            "-w", f"{os.environ.get('HOME')}",
            self._vllm_image,
            "vllm", "serve",
            f"{self._get_model_path()}",
            "--host", "0.0.0.0",
            "--no-enable-log-requests",
            "--trust-remote-code",
            "--tensor-parallel-size", f"{self._parallel_size.get('tp', '1')}",
            "--port", f"{self.vllm_port}",
        ])

        # get extra vLLM args
        cmd.extend(self._build_vllm_args().split())

        return cmd

    def get_server_run_cmd_direct(self) -> str:
        """Construct the direct run command for the vLLM server."""
        cmd = [
            "vllm", "serve",
            f"{self._model_path}",
            "--host", "0.0.0.0",
            "--no-enable-log-requests",
            "--trust-remote-code",
            "--tensor-parallel-size", f"{self._parallel_size.get('tp', '1')}",
            "--port", f"{self.vllm_port}",
        ]

        # get extra vLLM args
        cmd.extend(self._build_vllm_args().split())

        return cmd

    def _start_server(self):
        """Start the vLLM server, either in a container or as a direct process."""
        if self._in_container:
            self._start_server_direct()
        else:
            self._start_server_container()

    def _start_server_container(self):
        """Start the vLLM server in a container."""
        if not self._is_dry_run:
            self._cleanup_container(self.container_name)

        cmd = self.get_server_run_cmd()

        if self._is_dry_run:
            logger.info("Dry run - Docker server command:")
            logger.info(" ".join(cmd))

            logger.info("config file content:")
            with open(self.temp_compile_config_file, "r", encoding="utf-8") as f:
                import yaml
                compile_config = yaml.load(f, yaml.FullLoader)
                logger.info(compile_config)
        else:
            logger.info("Started to initialize vllm server ...")
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)

        # Start log collection
        with open(self.server_log, 'a') as f:
            # Create two processes to capture both stdout and stderr
            stdout_process = subprocess.Popen(
                [self._container_runtime, "logs", "-f", self.container_name],
                stdout=f,
                stderr=subprocess.PIPE
            )
            stderr_process = subprocess.Popen(
                [self._container_runtime, "logs", "-f", self.container_name],
                stdout=subprocess.PIPE,
                stderr=f
            )
            # Store process IDs for later cleanup if needed
            self.log_processes = [stdout_process, stderr_process]

    def _start_server_direct(self):
        """Start the vLLM server as a direct subprocess."""
        cmd = self.get_server_run_cmd_direct()

        if self._is_dry_run:
            logger.info("Dry run - Direct server command:")
            logger.info(" ".join(cmd))
            return

        logger.info("Starting vLLM server as a direct process...")
        server_env = os.environ.copy()
        server_env["CUDA_VISIBLE_DEVICES"] = self._gpu_devices

        # add common environment vars
        with open(self._common_env_file, "r", encoding="utf-8") as f:
            import dotenv
            common_env = dotenv.dotenv_values(stream=f)
            server_env.update(common_env)

        # add vllm environment vars
        for key, value in self._env_vars.items():
            server_env[key] = str(value)

        self.server_log.parent.mkdir(parents=True, exist_ok=True)
        with open(self.server_log, 'w') as f:
            self.server_process = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=server_env)

    def _is_server_process_alive(self) -> bool:
        """Check if the server process (container or native) is still running."""
        if self._is_dry_run:
            return True

        if self._in_container:
            # Check process status
            if not self.server_process:
                return False

            return self.server_process.poll() is None

        else:
            # check container status
            try:
                cmd = [self._container_runtime, "ps", "-q", "--filter", f"name=^{self.container_name}$"]
                result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=5)
                return bool(result.stdout.strip())
            except (subprocess.SubprocessError, FileNotFoundError):
                return False

    def _wait_for_server(self, timeout: int = 2 * 60 * 60) -> bool:
        """Wait for the server to become available."""
        start_time = time.time()
        last_log_time = start_time

        while True:
            # check if the server is ready
            try:
                response = requests.get(f"http://localhost:{self.vllm_port}/v1/models")
                if response.status_code == 200:
                    model_info = response.json()
                    if model_info:
                        return True
            except requests.exceptions.RequestException:
                pass

            # check if the server is alive
            if not self._is_server_process_alive():
                logger.error("vLLM server process is not running.")
                logger.error("Check server log for more details. %s", self.server_log)
                return False

            # check timeout
            elapsed_time = time.time() - start_time
            if elapsed_time > timeout:
                logger.error("Timeout waiting for vLLM server to start.")
                logger.error("Server process is still alive, but endpoint is not responding.")
                logger.error("Check server log for more details. %s", self.server_log)
                return False

            if time.time() - last_log_time > 60:
                last_log_time = time.time()
                logger.info("Waiting for vLLM server to start... %s seconds elapsed", int(elapsed_time))

            time.sleep(5)

    def _check_existing_result(self,
                               request_rate: int,
                               concurrency: int,
                               input_length: int,
                               output_length: int,
                               num_iteration: int,
                               batch_size: int) -> bool:
        """Check if results already exist for this configuration."""
        if not self.result_file.exists() or self._is_dry_run:
            return False

        search_str = f"{Path(self._model_config).stem},{self._parallel_size.get('tp', '1')},{request_rate},{num_iteration},{batch_size},{concurrency},{input_length},{output_length}"
        search_result = any(search_str in line for line in self.result_file.read_text().splitlines())

        # print previous benchmark result if exists
        if search_result:
            with open(self.result_file, 'r') as f:
                for line in f:
                    if search_str in line:
                        logger.info(self._format_result_for_console(line.strip().split(',')))
        return search_result

    def _extract_metrics(self, log_file: Path) -> Dict[str, float]:
        """Extract metrics from a benchmark log file."""
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

    def _save_results(self, request_rate: int, num_iteration: int, batch_size: int,
                      concurrency: int, input_length: int, output_length: int, metrics: Dict[str, float]):

        """Save benchmark results to the result file."""
        result_line = (
            f"{Path(self._model_config).stem},{self._parallel_size.get('tp', '1')},"
            f"{request_rate},{num_iteration},{batch_size},{concurrency},{input_length},{output_length},{metrics['test_time']},"
            f"{metrics['ttft_mean']:.2f},{metrics['ttft_median']:.2f},{metrics['ttft_p99']:.2f},"
            f"{metrics['tpot_mean']:.2f},{metrics['tpot_median']:.2f},{metrics['tpot_p99']:.2f},"
            f"{metrics['itl_mean']:.2f},{metrics['itl_median']:.2f},{metrics['itl_p99']:.2f},"
            f"{metrics['e2el_mean']:.2f},{metrics['e2el_median']:.2f},{metrics['e2el_p99']:.2f},"
            f"{metrics['request_throughput']:.2f},"
            f"{metrics['output_token_throughput']:.2f},{metrics['total_token_throughput']:.2f}\n"
        )

        with open(self.result_file, 'a') as f:
            f.write(result_line)

    def _format_result_for_console(self, values: List[str]) -> str:
        """Formats a list of result values for console output."""
        if len(values) != len(self._columns):
            logger.warning("Mismatch between result values and column definitions.")
            return ' '.join(values)

        formatted_values = [val.rjust(width) for val, (header, width) in zip(values, self._columns)]
        formatted_values[0] = os.path.basename(values[0]).ljust(self._columns[0][1])
        return ' '.join(formatted_values)

    def _print_result(self, request_rate: int, num_iteration: int, batch_size: int,
                      concurrency: int, input_length: int, output_length: int, metrics: Dict[str, float]):
        """Print the result to console."""
        values = [
            Path(self._model_config).stem, str(self._parallel_size.get('tp', '1')),
            str(request_rate), str(num_iteration), str(batch_size), str(concurrency), str(input_length), str(output_length), f"{metrics['test_time']:.2f}",
            f"{metrics['ttft_mean']:.2f}", f"{metrics['ttft_median']:.2f}", f"{metrics['ttft_p99']:.2f}",
            f"{metrics['tpot_mean']:.2f}", f"{metrics['tpot_median']:.2f}", f"{metrics['tpot_p99']:.2f}",
            f"{metrics['itl_mean']:.2f}", f"{metrics['itl_median']:.2f}", f"{metrics['itl_p99']:.2f}",
            f"{metrics['e2el_mean']:.2f}", f"{metrics['e2el_median']:.2f}", f"{metrics['e2el_p99']:.2f}",
            f"{metrics['request_throughput']:.2f}", f"{metrics['output_token_throughput']:.2f}", f"{metrics['total_token_throughput']:.2f}"
        ]
        logger.info(self._format_result_for_console(values))

    def run_single_benchmark(self,
                             request_rate: int,
                             concurrency: int,
                             input_length: int,
                             output_length: int,
                             num_iteration: int,
                             batch_size: int):
        """Run a single benchmark iteration."""

        # if required iteration is not given, use default value
        # which enables to iterate following benchmark plan
        num_prompts = concurrency * num_iteration

        base_cmd = []
        if not self._in_container:
            base_cmd.extend([self._container_runtime, "exec", self.container_name])

        base_cmd.extend([
            "vllm", "bench", "serve",
            "--model", f"{self._get_model_path()}",
            "--backend", "vllm",
            "--host", "localhost",
            f"--port={self.vllm_port}",
            "--dataset-name", "random",
            "--ignore-eos",
            "--trust-remote-code",
            f"--request-rate={request_rate if request_rate > 0 else 'inf'}",
            f"--max-concurrency={concurrency}",
            f"--num-prompts={num_prompts}",
            f"--random-input-len={input_length}",
            f"--random-output-len={output_length}",
            "--tokenizer", f"{self._get_model_path()}",
            "--disable-tqdm",
            "--percentile-metrics", "ttft,tpot,itl,e2el"
        ])

        cmd = base_cmd

        # Check if this configuration has already been tested
        if self._check_existing_result(request_rate, concurrency, input_length, output_length, num_iteration, batch_size):
            # logger.info(f"Skipping existing configuration: c{client_count}_i{input_length}_o{output_length}")
            return

        if self._is_dry_run:
            logger.info(f"Dry run - Benchmark command for r{request_rate}_n{num_iteration}_c{concurrency}_i{input_length}_o{output_length}")
            logger.info(" ".join(cmd))
            return

        # TODO: env directory will have more parallelism size info
        log_file = self._log_dir / self._exp_tag / f"r{request_rate}_n{num_iteration}_i{input_length}_o{output_length}_c{concurrency}.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, 'w') as f:
            f.write(f"=== Benchmark: request_rate={request_rate}, num_iteration={num_iteration}, concurrency={concurrency}, input_len={input_length}, output_len={output_length} ===\n")

        # Run the benchmark command and redirect output to log file
        with open(log_file, 'w') as f:
            subprocess.run(cmd, stdout=f, stderr=f, check=True)

        # Extract metrics from log file
        metrics = self._extract_metrics(log_file)

        # Save and print results
        self._print_result(
            request_rate, num_iteration, batch_size, concurrency, input_length, output_length, metrics)
        self._save_results(
            request_rate, num_iteration, batch_size, concurrency, input_length, output_length, metrics)

    def _load_test_plan(self):
        """Load test configuration from YAML file."""
        yaml_path = Path("configs/benchmark_plans") / f"{self._test_plan}.yaml"
        if not yaml_path.exists():
            raise FileNotFoundError(f"Test plan not found: {yaml_path}")

        with open(yaml_path) as f:
            config = yaml.safe_load(f)

        test_plans = []
        for scenario in config.get('test_scenarios', []):
            # Convert all parameters to lists if they're not already
            request_rates = [scenario.get('request_rate')] if isinstance(scenario.get('request_rate'), (int, float)) \
                else scenario.get('request_rate', [0])
            concurrencies = [scenario.get('concurrency')] if isinstance(scenario.get('concurrency'), int) \
                else scenario.get('concurrency', [1])
            input_lengths = [scenario.get('input_length')] if isinstance(scenario.get('input_length'), int) \
                else scenario.get('input_length', [512])
            output_lengths = [scenario.get('output_length')] if isinstance(scenario.get('output_length'), int) \
                else scenario.get('output_length', [128])
            num_iterations = [scenario.get('num_iteration')] if isinstance(scenario.get('num_iteration'), int) \
                else scenario.get('num_iteration', [8])
            batch_sizes = [scenario.get('batch_size')] if isinstance(scenario.get('batch_size'), int) \
                else scenario.get('batch_size', [256])

            # Generate all combinations
            for rate in request_rates:
                for batch_size in batch_sizes:
                    for num_iter in num_iterations:
                        for in_len in input_lengths:
                            for out_len in output_lengths:
                                for concurrency in concurrencies:
                                    test_plan = {
                                        'request_rate': rate,
                                        'concurrency': concurrency,
                                        'input_length': in_len,
                                        'output_length': out_len,
                                        'num_iteration': num_iter,
                                        'batch_size': batch_size,
                                    }
                                    test_plans.append(test_plan)

        if not test_plans:
            raise ValueError("No test scenarios found in test plan")

        return test_plans

    def _warmup_server(self):
        """Warmup the server before benchmarking."""
        if self._is_dry_run:
            return

        if self._is_no_warmup:
            logger.info("Skipping warmup as per user request")
            return

        logger.info("Warming up the server...")
        warmup_cmd = []

        if not self._in_container:
            warmup_cmd.extend(
                [self._container_runtime, "exec", self.container_name])

        warmup_cmd.extend([
            "vllm", "bench", "serve",
            "--model", f"{self._get_model_path()}",
            "--backend", "vllm",
            "--host", "localhost",
            f"--port={self.vllm_port}",
            "--dataset-name", "random",
            "--ignore-eos",
            "--trust-remote-code",
            f"--request-rate=10",
            f"--max-concurrency=1",
            f"--num-prompts=4",
            f"--random-input-len=16",
            f"--random-output-len=16",
            "--tokenizer", f"{self._get_model_path()}",
            "--disable-tqdm"
        ])

        if self._is_dry_run:
            logger.info("Dry run - Warmup command:")
            logger.info(" ".join(warmup_cmd))
            return

        start_time = time.time()
        subprocess.run(warmup_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        elapsed_time = time.time() - start_time
        logger.info(f"Warmup complete in {elapsed_time:.2f} seconds.")

    def _run_vllm_benchmark(self):
        """Run benchmarks using vLLM test plans."""
        for test_plan in self._load_test_plan():
            try:
                self.run_single_benchmark(**test_plan)

                if self._is_dry_run:
                    ans = input("Continue to generate benchmark command? (Y/n) ")
                    if ans.lower() == 'n' or ans.lower() == 'no':
                        break
            except subprocess.CalledProcessError as e:
                logger.error(f"Single benchmark failed for \
                             r{test_plan['request_rate']}_n{test_plan['num_iteration']}_c{test_plan['concurrency']}_i{test_plan['input_length']}_o{test_plan['output_length']}")
                logger.error(f"{str(e)}")
                return

    def run(self):
        """Run the full benchmark suite."""
        if self._num_gpus == 0:
            raise ValueError("No GPU is allocated")

        # Start server for this configuration
        self._start_server()
        if not self._is_dry_run:
            if not self._wait_for_server():
                raise RuntimeError("Server failed to start")

            logger.info("Server is up and running")

        self._warmup_server()
        self._print_header()

        self._run_vllm_benchmark()

        # Cleanup container after this configuration
        if self._in_container:
            self._cleanup_server_process()
        else:
            self._cleanup_container(self.container_name)
        if not self._is_dry_run and not self._in_container:
            logger.info(f"Benchmarking complete. Results saved to {self.result_file}")

def main():
    try:
        args = get_args()

        benchmark = VLLMBenchmark(
            env_file=args.env_file,
            model_config=args.model_config,
            model_path_or_id=args.model_path_or_id,
            model_root_dir=args.model_root_dir,
            vllm_image=args.vllm_image,
            test_plan=args.test_plan,
            gpu_devices=args.gpu_devices,
            num_gpus=args.num_gpus,
            arch=args.arch,
            no_warmup=args.no_warmup,
            dry_run=args.dry_run,
            in_container=args.in_container
        )

        benchmark.run()
    except Exception as e:
        logger.error(f"Benchmark failed: {str(e)}")
        sys.exit(1)
    finally:
        # Clean up temporary compile config file if it exists
        if "benchmark" in locals():
            if hasattr(benchmark, "temp_compile_config_file") and \
                os.path.exists(benchmark.temp_compile_config_file):
                    os.remove(benchmark.temp_compile_config_file)

            # Ensure server process is killed on error
            if benchmark._in_container:
                benchmark._cleanup_server_process()
            else:
                benchmark._cleanup_container(benchmark.container_name)


if __name__ == "__main__":
    main()
