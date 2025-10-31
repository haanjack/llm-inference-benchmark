#!/usr/bin/env python3

import argparse
import logging
import os
import subprocess
import sys
import time
import datetime
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import re
import requests
import utils
import tempfile

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_args():
    """Benchmark arguments"""
    parser = argparse.ArgumentParser(description='Run vLLM benchmarks')

    # benchmark configuration
    parser.add_argument('--env-file', default='baseline', help='Environment file name')
    parser.add_argument('--model-path', help='Model checkpoint path')
    parser.add_argument('--vllm-image', help='vLLM Docker image')
    parser.add_argument('--test-plan', default='test',
                        help='Benchmark test plan YAML file in configs/benchmark_plans/ (without .yaml extension)')
    parser.add_argument('--gpu-devices', default="0", help='Comma-separated GPU device IDs')

    # server control arguments
    parser.add_argument('--request-rate', type=int, default=None,
                       help='Request rate for the benchmark')
    parser.add_argument('--max-num-seqs', type=int, default=None,
                        help='Max num sequence for vllm serving benchmark (a.k.a batch-size)')
    parser.add_argument('--num-iteration', type=int, default=None,
                        help='Number of batch iterations')

    # test control
    parser.add_argument('--no-warmup', action='store_true',
                        help='no warmup at benchmark start')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show commands without executing them')

    args = parser.parse_args()

    return args


class VLLMBenchmark:
    def __init__(self,
                 env_file: str = "baseline",
                 model_path: str = None,
                 vllm_image: str = None,
                 test_plan: str = "test",
                 gpu_devices: str = "0",
                 num_iteration: int = None,
                 request_rate: int = None,
                 max_num_seqs: int = None,
                 no_warmup: bool = False,
                 dry_run: bool = False):

        # benchmark configuration
        self._env_file = env_file
        self._env_vars = self._load_env_file()
        self._model_path = model_path or self._env_vars.get('MODEL_PATH')
        if not self._model_path:
            raise ValueError("Model path must be provided via --model-path or MODEL_PATH in env file.")
        self._model_path = (Path().cwd() / self._model_path) if not Path(self._model_path).is_absolute() else Path(self._model_path)
        self._model_name = self._model_path.name
        self._container_model_path = Path(f"/models/{self._model_name}")
        self._vllm_image = vllm_image or self._env_vars.get("VLLM_IMAGE", "docker.io/rocm/vllm:latest")
        self._test_plan = test_plan
        self._test_plan_path = Path(f"configs/benchmark_plans/{test_plan}.yaml")
        self._gpu_devices = gpu_devices # TODO: select based on os.environ.get("SLURM_JOB_GPUS", "")

        # Set benchmark parameters based on scope
        self._num_iteration = int(self._env_vars.get('NUM_ITERATION', 1)) if num_iteration is None else num_iteration
        self._request_rate = self._env_vars.get('REQUEST_RATE', 1) if request_rate is None else request_rate
        if self._request_rate == 'inf':
            self._request_rate = 0
        self._max_num_seqs = int(self._env_vars.get('MAX_NUM_SEQS', '256')) if max_num_seqs is None else max_num_seqs

        self._is_no_warmup = no_warmup
        self._is_dry_run = dry_run

        # Sanity Check
        if not self._test_plan_path.exists() and not self._is_dry_run:
            raise FileNotFoundError(f"Could not find test plan: {self._test_plan_path}. Please check the plan name")
        if not self._model_path.exists() and not self._is_dry_run:
            raise FileNotFoundError(f"Could not find model path {self._model_name}. Please check model path.")

        # GPU configuration
        gpu_array = self._gpu_devices.split(',')
        if len(gpu_array) == 0:
            raise AssertionError("No GPU is specified. Please specify at least one GPU.")
        self._num_gpus = len(gpu_array)

        # Result file headers
        self._headers = [
            "env,TP Size,",
            "Request Rate,Num. Iter,Client Count,MaxNumSeqs,Input Length,Output Length,Test Time,",
            "Mean TTFT (ms),Median TTFT (ms),P99 TTFT (ms),",
            "Mean TPOT (ms),Median TPOT (ms),P99 TPOT (ms),",
            "Mean ITL (ms),Median ITL (ms),P99 ITL (ms),",
            "Mean E2EL (ms),Median E2EL (ms),P99 E2EL (ms),",
            "Request Throughput (req/s),Output token throughput (tok/s),",
            "Total Token throughput (tok/s)"
        ]

        # determine docker or podman
        self._container_runtime = "docker" if self._is_docker_available() else "podman"

        # Container name setup
        self._setup_container_name()

        # Setup logging directories
        self._setup_logging_dirs()
        self._cache_dir()

        # VLLM port setup
        self.vllm_port = 23400 + int(gpu_array[0]) if gpu_array else 23400

        self._metric_start_column_idx = 9
        self._print_benchmark_info()

    def _cache_dir(self):
        """Configure vllm cache directory which to reduce compilation overhead."""
        self._host_cache_dir = Path.cwd() / "vllm_cache"
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

    def _load_env_file(self) -> Dict[str, str]:
        """Load environment variables from the specified env file."""
        env_file_path = Path.cwd() / self._env_file # env only accepts relative path
        env_vars = {}

        if not env_file_path.exists():
            raise FileNotFoundError(f"Environment file not found: {env_file_path}. It only accepts relative path from current directory")

        with open(env_file_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    try:
                        key, value = line.split('=', 1)
                        env_vars[key.strip()] = value.strip().strip('"')
                    except ValueError:
                        continue
        return env_vars

    def _setup_container_name(self):
        """Setup container name based on environment and GPU configuration."""
        image_tag = self._vllm_image.split(':')[-1]
        slurm_job_id = os.environ.get("SLURM_JOB_ID", None)

        self.container_name = ""
        if slurm_job_id:
            self.container_name = f"{slurm_job_id}-"
        self.container_name += f"{os.path.basename(self._model_name)}-{image_tag}-{os.path.basename(self._env_file)}-g{self._gpu_devices.replace(',', '_')}"

    def _setup_logging_dirs(self):
        """Setup logging directories for the benchmark."""
        image_tag = self._vllm_image.split(':')[-1]
        self.log_dir = Path("logs") / self._model_name / image_tag
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.result_file = self.log_dir / "result_list.csv"
        self.result_file.parent.mkdir(parents=True, exist_ok=True)

        self.server_log = self.log_dir / "server_logs" / f"{os.path.basename(self._model_name)}-{image_tag}-{os.path.basename(self._env_file)}-t{self._num_gpus}.txt"
        self.server_log.parent.mkdir(parents=True, exist_ok=True)

        self._env_tag = "-".join(Path(os.path.basename(self._env_file)).parts)

        # Initialize result file if it doesn't exist
        if not self.result_file.exists():
            self._init_result_file()

    def _init_result_file(self):
        """Initialize the result file with headers."""
        with open(self.result_file, 'w') as f:
            f.write(''.join(self._headers) + '\n')

    def _print_header(self):
        """Print the header line to console."""
        if self._is_dry_run:
            return

        # print header line to console
        headers_split = ''.join(self._headers).split(',')
        headers_line = [os.path.basename(headers_split[0]).ljust(16)]
        headers_line += [h.rjust(8) for h in headers_split[1:self._metric_start_column_idx]]
        headers_line += [h.rjust(10) for h in headers_split[self._metric_start_column_idx:]]
        logger.info('\t'.join(headers_line))

    def _get_vllm_args(self) -> str:
        """Construct VLLM arguments based on environment variables."""
        args = [
            "--kv-cache-dtype", f"{self._env_vars.get('KV_CACHE_DTYPE', '')}",
            "--gpu_memory_utilization", f"{self._env_vars.get('GPU_MEMORY_UTILIZATION', '0.9')}",
            "--max-num-batched-token", f"{self._env_vars.get('MAX_NUM_BATCHED_TOKENS', '4096')}",
            "--max-num-seqs", f"{self._max_num_seqs}",
            "--swap-space", "16",
            "--no-enable-prefix-caching",
        ]
        if self._env_vars.get('QUANTIZATION', 'auto') != 'auto':
            args.extend(["--quantization", f"{self._env_vars.get('QUANTIZATION')}"])

        # ROCM version handling
        # rocm version format should be like "6.4.0" or "7.0.1"
        # load rocm version from pytorch installed in container
        if not self._is_dry_run:
            container_rocm_version = subprocess.run(
                [self._container_runtime, "run", "--rm", self._vllm_image, "python3", "-c",
                    "import torch; print(torch.version.hip)"],
                capture_output=True, text=True
            ).stdout.strip()
            if container_rocm_version is None or container_rocm_version == "":
                logger.warning("Failed to get ROCM version from container")

            rocm_version_nums = [int(x) for x in re.findall(r'\d+', container_rocm_version)]
            if len(rocm_version_nums) >= 2:
                if (rocm_version_nums[0] >= 7):
                    args.append("--async-scheduling")

        vllm_use_v1 = self._env_vars.get("VLLM_USE_V1", "1") # V1 is default
        if vllm_use_v1 == "0":
            self._env_vars["VLLM_USE_TRITON_FLASH_ATTN"] = "0"
        else:
            if self._env_vars.get("VLLM_ROCM_USE_AITER") == "1":
                cudagraph_mode = self._env_vars.get("VLLM_CUDAGRAPH_MODE", "FULL_AND_PIECEWISE")
                if cudagraph_mode:
                    modes = {
                        "NONE":                 "{\"cudagraph_mode\": null}",
                        "PIECEWISE":            "{\"cudagraph_mode\": \"PIECEWISE\"}",
                        "FULL":                 "{\"cudagraph_mode\": \"FULL\"}",
                        "FULL_DECODE_ONLY":     "{\"cudagraph_mode\": \"FULL_DECODE_ONLY\"}",
                        "FULL_AND_PIECEWISE":   "{\"cudagraph_mode\": \"FULL_AND_PIECEWISE\"}",
                        "MOE": "{\"compile_sizes\":[1,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,84,86,88,90,92,94,96,98,100,102,104,106,108,110,112,114,116,118,120,122,124,126,128,256,512,1024,2048,8192], "\
                                "\"cudagraph_capture_sizes\":[1,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,84,86,88,90,92,94,96,98,100,102,104,106,108,110,112,114,116,118,120,122,124,126,128,136,144,152,160,168,176,184,192,200,208,216,224,232,240,248,256,264,272,280,288,296,304,312,320,328,336,344,352,360,368,376,384,392,400,408,416,424,432,440,448,456,464,472,480,488,496,504,512,520,528,536,544,552,560,568,576,584,592,600,608,616,624,632,640,648,656,664,672,680,688,696,704,712,720,728,736,744,752,760,768,776,784,792,800,808,816,824,832,840,848,856,864,872,880,888,896,904,912,920,928,936,944,952,960,968,976,984,992,1000,1008,1016,1024,2048,4096,8192], "\
                                "\"cudagraph_mode\": \"FULL\"}"
                    }
                    if cudagraph_mode in modes:
                        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml",
                                                         dir=self._compile_cache_dir,
                                                         encoding="utf-8", delete=False) as f:
                            f.write(f"compilation_config: '{modes[cudagraph_mode]}'\n")
                            args.extend(["--config", str(Path("root") / ".cache" / "compile_config" / f.name)])
                            self.temp_compile_config_file = f.name

                            print(str(Path("root") / ".cache" / "compile_config" / f.name))

        return " ".join(args)

    def _cleanup_log_processes(self):
        """Terminate log collection processes if they exist."""
        if hasattr(self, 'log_processes'):
            for process in self.log_processes:
                try:
                    process.terminate()
                    process.wait(timeout=5)  # Wait for graceful termination
                except (subprocess.TimeoutExpired, ProcessLookupError):
                    process.kill()  # Force kill if graceful termination fails

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
            "--shm-size=8g",
            "--security-opt", "seccomp=unconfined",
            "--env-file", str(Path.cwd() / self._env_file),
            "-e", f"VLLM_USE_TRITON_FLASH_ATTN={self._env_vars.get('VLLM_USE_TRITON_FLASH_ATTN', '0')}",
            "-e", f"CUDA_VISIBLE_DEVICES={self._gpu_devices}",
            "-v", f"{os.environ.get('HOME')}:{os.environ.get('HOME')}",
            "-v", f"{self._model_path}:{self._container_model_path}:ro",
            "-v", f"{self._host_cache_dir}:/root/.cache",
            "-v", f"{self._compile_cache_dir}:/root/.cache/compile_config",
            "-v", f"{self._aiter_cache_dir}:/root/.aiter",
            "-w", f"{os.environ.get('HOME')}",
            self._vllm_image,
            "vllm", "serve",
            f"{self._container_model_path}",
            "--no-enable-log-requests",
            "--trust-remote-code",
            "--tensor-parallel-size", f"{self._num_gpus}",
            "--distributed-executor-backend", "mp",
            "--port", f"{self.vllm_port}",
            "--host", "0.0.0.0"
        ]

        # get extra vLLM args
        cmd.extend(self._get_vllm_args().split())

        return cmd

    def _start_server(self):
        """Start the vLLM server in a Docker container."""
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

    def _wait_for_server(self, timeout: int = 2400) -> bool:
        """Wait for the server to become available."""
        start_time = time.time()

        while True:
            gpu_active = utils.get_gfx_clk_value(int(self._gpu_devices.split(',')[0])) > utils.GFX_CLK_IDLE_THRESHOLD

            # check if the server is ready
            try:
                response = requests.get(f"http://localhost:{self.vllm_port}/v1/models")
                if response.status_code == 200:
                    model_info = response.json()
                    if model_info:
                        return True
            except requests.exceptions.RequestException:
                pass

            # check timeout
            if time.time() - start_time > timeout:
                if not gpu_active:
                    logger.error("vLLM server GPU is idle. Server failed to start.")
                    return False
                else:
                    logger.info("vLLM server is still starting...")
                    time.sleep(30)
                    continue

            time.sleep(5)

        logger.error("Timeout waiting for vLLM server to start.")
        return False

    def _check_existing_result(self,
                               request_rate: int,
                               concurrency: int,
                               input_length: int,
                               output_length: int,
                               num_iteration: Optional[int]) -> bool:
        """Check if results already exist for this configuration."""
        if not self.result_file.exists():
            return False

        if num_iteration is None:
            num_iteration = int(self._env_vars.get('NUM_ITERATION', self._num_iteration))
        search_str = f"{self._env_file},{self._num_gpus},{request_rate},{num_iteration},{self._max_num_seqs},{concurrency},{input_length},{output_length}"
        search_result = any(search_str in line for line in self.result_file.read_text().splitlines())

        # print previous benchmark result if exists
        if search_result:
            with open(self.result_file, 'r') as f:
                for line in f:
                    if search_str in line:
                        line = line.strip()
                        s_line = [os.path.basename(line.split(',')[0]).ljust(16)]
                        s_line += [h.rjust(8) for h in line.split(',')[1:self._metric_start_column_idx]]
                        s_line += [h.rjust(10) for h in line.split(',')[self._metric_start_column_idx:]]
                        logger.info(f"{''.join(s_line)}")
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

    def _save_results(self, request_rate: int, num_iteration: int, concurrency: int, input_length: int, output_length: int, metrics: Dict[str, float]):

        """Save benchmark results to the result file."""
        result_line = (
            f"{os.path.basename(self._env_file)},{self._num_gpus},"
            f"{request_rate},{num_iteration},{self._max_num_seqs},{concurrency},{input_length},{output_length},{metrics['test_time']},"
            f"{metrics['ttft_mean']:.2f},{metrics['ttft_median']:.2f},{metrics['ttft_p99']:.2f},"
            f"{metrics['tpot_mean']:.2f},{metrics['tpot_median']:.2f},{metrics['tpot_p99']:.2f},"
            f"{metrics['itl_mean']:.2f},{metrics['itl_median']:.2f},{metrics['itl_p99']:.2f},"
            f"{metrics['e2el_mean']:.2f},{metrics['e2el_median']:.2f},{metrics['e2el_p99']:.2f},"
            f"{metrics['request_throughput']:.2f},"
            f"{metrics['output_token_throughput']:.2f},{metrics['total_token_throughput']:.2f}\n"
        )

        with open(self.result_file, 'a') as f:
            f.write(result_line)

    def _print_result(self, request_rate: int, num_iteration: int, concurrency: int, input_length: int, output_length: int, metrics: Dict[str, float]):
        """Print the result to console."""
        result_line = (
            f"{os.path.basename(self._env_file).ljust(16)}\t{self._num_gpus:>6d}"
            f"{request_rate}{num_iteration:>6d}{self._max_num_seqs:>6d}{concurrency:>6d}{input_length:>6d}{output_length:>6d}{metrics['test_time']:>6.2f}"
            f"{metrics['ttft_mean']:10.2f}{metrics['ttft_median']:10.2f}{metrics['ttft_p99']:10.2f}"
            f"{metrics['tpot_mean']:10.2f}{metrics['tpot_median']:10.2f}{metrics['tpot_p99']:10.2f}"
            f"{metrics['itl_mean']:10.2f}{metrics['itl_median']:10.2f}{metrics['itl_p99']:10.2f}"
            f"{metrics['e2el_mean']:10.2f}{metrics['request_throughput']:10.2f}"
            f"{metrics['output_token_throughput']:10.2f}{metrics['total_token_throughput']:10.2f}"
        )

        logger.info(result_line)

    def run_single_benchmark(self,
                             request_rate: int,
                             concurrency: int,
                             input_length: int,
                             output_length: int,
                             num_iteration: Optional[int] = None):
        """Run a single benchmark iteration."""

        # if required iteration is not given, use default value
        # which enables to iterate following benchmark plan
        num_iteration = int(self._num_iteration if num_iteration is None else num_iteration)

        num_prompts = concurrency * num_iteration
        cmd = [
            self._container_runtime, "exec", self.container_name,
            "vllm", "bench", "serve",
            "--model", f"{self._container_model_path}",
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
            "--tokenizer", f"{self._container_model_path}",
            "--disable-tqdm",
            "--percentile-metrics", "ttft,tpot,itl,e2el"
        ]

        # Check if this configuration has already been tested
        if self._check_existing_result(request_rate, concurrency, input_length, output_length, num_iteration):
            # logger.info(f"Skipping existing configuration: c{client_count}_i{input_length}_o{output_length}")
            return

        if self._is_dry_run:
            logger.info(f"Dry run - Benchmark command for r{request_rate}_n{num_iteration}_c{concurrency}_i{input_length}_o{output_length}")
            logger.info(" ".join(cmd))
            return

        # TODO: env directory will have more parallelism size info
        log_file = self.log_dir / f"{self._env_tag}_tp{self._env_vars.get('TENSOR_PARALLEL_SIZE', '1')}" / f"r{request_rate}_n{num_iteration}_i{input_length}_o{output_length}_c{concurrency}.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, 'w') as f:
            f.write(f"=== Benchmark: request_rate={request_rate}, num_iteration={num_iteration}, concurrency={concurrency}, input_len={input_length}, output_len={output_length} ===\n")

        # Run the benchmark command and redirect output to log file
        with open(log_file, 'w') as f:
            subprocess.run(cmd, stdout=f, stderr=f, check=True)

        # Extract metrics from log file
        metrics = self._extract_metrics(log_file)

        # Save and print results
        self._save_results(
            request_rate, num_iteration, concurrency, input_length, output_length, metrics)
        self._print_result(
            request_rate, num_iteration, concurrency, input_length, output_length, metrics)

    def _get_test_plans(self):
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
                else scenario.get('num_iteration', [self._num_iteration])

            # Generate all combinations
            for rate in request_rates:
                for concurrency in concurrencies:
                    for in_len in input_lengths:
                        for out_len in output_lengths:
                            for num_iter in num_iterations:
                                test_plan = {
                                    'request_rate': rate,
                                    'concurrency': concurrency,
                                    'input_length': in_len,
                                    'output_length': out_len,
                                    'num_iteration': num_iter
                                }
                                test_plans.append(test_plan)

        if not test_plans:
            raise ValueError("No test scenarios found in test plan")

        return test_plans

    def _warmup_server(self):
        """Warmup the server before benchmarking."""
        if self._is_no_warmup:
            logger.info("Skipping warmup as per user request")
            return

        logger.info("Warming up the server...")
        warmup_cmd = [
            self._container_runtime, "exec", self.container_name,
            "vllm", "bench", "serve",
            "--model", f"{self._container_model_path}",
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
            "--tokenizer", f"{self._container_model_path}",
            "--disable-tqdm"
        ]

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
        for test_plan in self._get_test_plans():
            try:
                # use specified argument in cli
                if self._request_rate is not None:
                    test_plan["request_rate"] = self._request_rate
                if self._num_iteration is not None:
                    test_plan["num_iteration"] = self._num_iteration

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
        self._cleanup_container(self.container_name)
        if not self._is_dry_run:
            logger.info(f"Benchmarking complete. Results saved to {self.result_file}")

def main():
    try:
        args = get_args()

        benchmark = VLLMBenchmark(
            env_file=args.env_file,
            model_path=args.model_path,
            vllm_image=args.vllm_image,
            test_plan=args.test_plan,
            gpu_devices=args.gpu_devices,
            request_rate=args.request_rate,
            num_iteration=args.num_iteration,
            max_num_seqs=args.max_num_seqs,
            no_warmup=args.no_warmup,
            dry_run=args.dry_run
        )

        benchmark.run()
    except Exception as e:
        logger.error(f"Benchmark failed: {str(e)}")

        if "benchmark" in locals():
            if hasattr(benchmark, "temp_compile_config_file") and \
                os.path.exists(benchmark.temp_compile_config_file):
                    os.remove(benchmark.temp_compile_config_file)

        sys.exit(1)

if __name__ == "__main__":
    main()
