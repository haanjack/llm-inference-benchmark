#!/usr/bin/env python3

import argparse
import logging
import os
import subprocess
import sys
import time
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import re
import requests
import utils

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
                        help='Benchmark test plan which should match with one of file in configs/plans/')
    parser.add_argument('--gpu-devices', help='Comma-separated GPU device IDs')
    
    # server control arguments
    parser.add_argument('--request-rate', type=int, default=None,
                       help='Request rate for the benchmark')
    parser.add_argument('--max-num-seq', type=int, default=None,
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
                 max_seq_num: int = None,
                 no_warmup: bool = False,
                 dry_run: bool = False):
        
        # benchmark configuration
        self._env_file = env_file
        self._env_vars = self._load_env_file()
        self._model_path = self.env_vars.get('MODEL_PATH', self._model_name) if model_path is None else model_path
        self._model_path = (Path().cwd() / self._model_path) if not Path(self._model_path).is_absolute() else Path(self._model_path)
        self._model_name = self._model_path.name
        self._container_model_path = Path(f"/models/{self._model_name}")
        self._vllm_image = vllm_image or self.env_vars.get("VLLM_IMAGE", "docker.io/rocm/vllm:latest")
        self._test_plan = test_plan
        self._test_plan_path = Path(f"configs/plans/{test_plan}.txt")
        self._gpu_devices = gpu_devices # TODO: select based on os.environ.get("SLURM_JOB_GPUS", "")

        self._num_iteration = num_iteration
        self._request_rate = request_rate
        self._max_seq_num = max_seq_num

        self._is_no_warmup = no_warmup
        self._is_dry_run = dry_run
 
        # Sanity Check
        if not self._test_plan_path.exists():
            raise FileNotFoundError(f"Could not find test plan in configs/plans directory. Please check the plan name")
        if not self._container_model_path.exists():
            raise FileNotFoundError(f"Could not find model path {self._model_name}. Please check model path.")
        
        # Set benchmark parameters based on scope
        self._num_iteration = self.env_vars.get('NUM_ITERATION', 1)
        if num_iteration is not None:
            self._num_iteration = num_iteration
        self._request_rate = self.env_vars.get('REQUEST_RATE', 1)
        if request_rate is not None:
            self._request_rate = request_rate
        self.max_num_seqs = int(self.env_vars.get('MAX_NUM_SEQS', '256'))
        
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

        # VLLM port setup
        self.vllm_port = 23400 + int(gpu_array[0]) if gpu_array else 23400

        self._metric_start_column_idx = 9
        self._print_benchmark_info()

    def _print_benchmark_info(self):
        logger.info("Start vLLM benchmark")
        logger.info(f"Model Name: {self._model_name}")
        logger.info(f"vLLM docker image: {self._vllm_image}")
        logger.info(f"Benchmark plan: {self._test_plan}")
        if self._test_plan == "test":
            # reports test plan from test plan file in configs/plans/{self._test_plan}.txt
            logger.info("Benchmark test plan::")
            logger.info("request_rate client_count input_length output_length num_iteration")
            with open(f"configs/plans/{self._test_plan}.txt") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):  # Skip empty lines and comments
                        continue
                    try:
                        r, c, i, o, n = map(int, line.split())
                        logger.info(f"{r:>8d}, {c:>8d}, {i:>8d}, {o:>8d}, {n:>8d}")
                    except ValueError as e:
                        logger.warning(f"Skipping invalid line: {line} - {str(e)}")
                        continue

    def _is_docker_available(self) -> bool:
        """Check if Docker is installed on the system."""
        try:
            return subprocess.run(["docker", "images"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
        except FileNotFoundError:
            return False
        
    def _load_env_file(self) -> Dict[str, str]:
        """Load environment variables from the specified env file."""
        env_file_path = Path.cwd() / self._env_file
        env_vars = {}

        if not env_file_path.exists():
            raise FileNotFoundError(f"Environment file not found: {env_file_path}")
            
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
        
        self.server_log = self.log_dir / "server_logs" / f"{os.path.basename(self._model_name)}-{image_tag}-{os.path.basename(self._env_file)}-t{self._num_gpus}.txt"
        self.result_file = self.log_dir / "result_list.csv"
        
        self.server_log.parent.mkdir(parents=True, exist_ok=True)
        self.result_file.parent.mkdir(parents=True, exist_ok=True)

        # Initialize result file if it doesn't exist
        if not self.result_file.exists():
            self._init_result_file()

    def _init_result_file(self):
        """Initialize the result file with headers."""
        with open(self.result_file, 'w') as f:
            f.write(''.join(self._headers) + '\n')

    def _print_header(self):
        """Print the header line to console."""
        # print header line to console
        headers_split = ''.join(self._headers).split(',')
        headers_line = [headers_split[0].ljust(30)]
        headers_line += [h.rjust(8) for h in headers_split[1:self._metric_start_column_idx]]
        headers_line += [h.rjust(10) for h in headers_split[self._metric_start_column_idx:]]
        logger.info('\t'.join(headers_line))

    def _get_vllm_args(self) -> str:
        """Construct VLLM arguments based on environment variables."""
        max_model_len = self.input_lengths[-1] + self.output_lengths[-1] + 256
        args = [
            "--kv-cache-dtype", f"{self.env_vars.get('KV_CACHE_DTYPE', '')}",
            "--gpu_memory_utilization", f"{self.env_vars.get('GPU_MEMORY_UTILIZATION', '0.9')}",
            "--max-num-batched-token", f"{self.env_vars.get('MAX_NUM_BATCHED_TOKENS', '4096')}",
            "--max-num-seqs", f"{self.max_num_seqs}",
            "--swap-space", "16",
            "--no-enable-prefix-caching",
        ]
        if self.env_vars.get('QUANTIZATION', 'auto') != 'auto':
            args.extend(["--quantization", f"{self.env_vars.get('QUANTIZATION')}"])
        
        # ROCM version handling
        # rocm version format should be like "6.4.0" or "7.0.1"
        # load rocm version from pytorch installed in container
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

        vllm_use_v1 = self.env_vars.get("VLLM_USE_V1", "0")
        if vllm_use_v1 == "0":
            self.env_vars["VLLM_USE_TRITON_FLASH_ATTN"] = "0"
        else:
            if self.env_vars.get("VLLM_ROCM_USE_AITER") == "1":
                cudagraph_mode = self.env_vars.get("VLLM_CUDAGRAPH_MODE", "")
                if cudagraph_mode:
                    modes = {
                        "NONE": "null",
                        "PIECEWISE": "PIECEWISE",
                        "FULL": "FULL",
                        "FULL_DECODE_ONLY": "FULL_DECODE_ONLY",
                        "FULL_AND_PIECEWISE": "FULL_AND_PIECEWISE",
                        "MOE": "{\"compile_sizes\":[1, 2, 4, 8, 16, 24, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192], "
                                "\"cudagraph_capture_sizes\":[8192,4096,2048,1024,1008,992,976,960,944,928,912,896,880,864,848,832,816,800,784,768,752,736,720,704,688,672,656,640,624,608,592,576,560,544,528,512,496,480,464,448,432,416,400,384,368,352,336,320,304,288,272,256,248,240,232,224,216,208,200,192,184,176,168,160,152,144,136,128,120,112,104,96,88,80,72,64,56,48,40,32,24,16,8,4,2,1], "
                                "\"cudagraph_mode\": \"FULL\"}"
                    }
                    if cudagraph_mode in modes:
                        if cudagraph_mode != "MOE":
                            args.append(f'--compilation-config {{"cudagraph_mode": "{modes[cudagraph_mode]}"}}')
                        else:
                            args.append(f'--compilation-config {modes[cudagraph_mode]}')


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

    def _cleanup_container(self):
        """Remove the Docker container if it exists."""
        self._cleanup_log_processes()
        subprocess.run([self._container_runtime, "rm", "-f", self.container_name], 
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def start_server(self):
        """Start the vLLM server in a Docker container."""
        if not self._is_dry_run:
            self._cleanup_container()
        
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
            "-e", f"VLLM_USE_TRITON_FLASH_ATTN={self.env_vars.get('VLLM_USE_TRITON_FLASH_ATTN', '0')}",
            "-e", f"CUDA_VISIBLE_DEVICES={self._gpu_devices}",
            "-v", f"{os.environ.get('HOME')}:/workspace/",
            "-v", f"{self._model_path}:{self._container_model_path}:ro",
            self._vllm_image,
            "vllm", "serve",
            self._container_model_path,
            "--no-enable-log-requests",
            "--trust-remote-code",
            "--tensor-parallel-size", f"{self._num_gpus}",
            "--distributed-executor-backend", "mp",
            "--port", f"{self.vllm_port}",
            "--host", "0.0.0.0"
        ]
        
        cmd.extend(self._get_vllm_args().split())
        
        if self._is_dry_run:
            logger.info("Dry run - Docker server command:")
            logger.info(" ".join(cmd))
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
        
        while time.time() - start_time < timeout or utils.get_gfx_clk_value(self._gpu_devices[0]) > 1_000:
            try:
                response = requests.get(f"http://localhost:{self.vllm_port}/v1/models")
                return True
            except requests.exceptions.RequestException:
                time.sleep(5)
        return False

    def _check_existing_result(self, 
                               request_rate: int, 
                               client_count: int, 
                               input_length: int, 
                               output_length: int, 
                               num_iteration: Optional[int]) -> bool:
        """Check if results already exist for this configuration."""
        if not self.result_file.exists():
            return False

        if num_iteration is None:
            num_iteration = int(self.env_vars.get('NUM_ITERATION', self._num_iteration))
        search_str = f"{self._env_file},{self._num_gpus},{request_rate},{num_iteration},{self.max_num_seqs},{client_count},{input_length},{output_length}"
        search_result = any(search_str in line for line in self.result_file.read_text().splitlines())

        # print previous benchmark result if exists
        if search_result:
            with open(self.result_file, 'r') as f:
                for line in f:
                    if search_str in line:
                        line = line.strip()
                        s_line = [line.split(',')[0].ljust(30)]
                        s_line += [h.rjust(8) for h in line.split(',')[1:self._metric_start_column_idx]] 
                        s_line += [h.rjust(10) for h in line.split(',')[self._metric_start_column_idx:]]
                        logger.info(f"{''.join(s_line)}")
        return search_result

    def _extract_metrics(self, log_file: Path) -> Dict[str, float]:
        """Extract metrics from a benchmark log file."""
        metrics = {}
        patterns = {
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

    def _save_results(self, request_rate: int, num_iteration: int, client_count: int, input_length: int, output_length: int, metrics: Dict[str, float]):

        """Save benchmark results to the result file."""
        result_line = (
            f"{self._env_file},{self._num_gpus},"
            f"{request_rate},{num_iteration},{self.max_num_seqs},{client_count},{input_length},{output_length},{metrics['test_time']},"
            f"{metrics['ttft_mean']:.2f},{metrics['ttft_median']:.2f},{metrics['ttft_p99']:.2f},"
            f"{metrics['tpot_mean']:.2f},{metrics['tpot_median']:.2f},{metrics['tpot_p99']:.2f},"
            f"{metrics['itl_mean']:.2f},{metrics['itl_median']:.2f},{metrics['itl_p99']:.2f},"
            f"{metrics['e2el_mean']:.2f},{metrics['e2el_median']:.2f},{metrics['e2el_p99']:.2f},"
            f"{metrics['request_throughput']:.2f},"
            f"{metrics['output_token_throughput']:.2f},{metrics['total_token_throughput']:.2f}\n"
        )
        
        with open(self.result_file, 'a') as f:
            f.write(result_line)

    def _print_result(self, request_rate: int, num_iteration: int, client_count: int, input_length: int, output_length: int, metrics: Dict[str, float]):
        """Print the result to console."""
        if request_rate == 0:
            request_rate = 'inf'
        result_line = (
            f"{self._env_file.ljust(30)}\t{self._num_gpus:>8d}\t"
            f"{request_rate}\t{num_iteration:>4d}\t{self.max_num_seqs:>4d}\t{client_count:>4d}\t{input_length:>8d}\t{output_length:>8d}\t{metrics['test_time']}\t"
            f"{metrics['ttft_mean']:10.2f}\t{metrics['ttft_median']:10.2f}\t{metrics['ttft_p99']:10.2f}\t"
            f"{metrics['tpot_mean']:10.2f}\t{metrics['tpot_median']:10.2f}\t{metrics['tpot_p99']:10.2f}\t"
            f"{metrics['itl_mean']:10.2f}\t{metrics['itl_median']:10.2f}\t{metrics['itl_p99']:10.2f}\t"
            f"{metrics['e2el_mean']:10.2f}\t{metrics['request_throughput']:10.2f}\t"
            f"{metrics['output_token_throughput']:10.2f}\t{metrics['total_token_throughput']:10.2f}"
        )

        logger.info(result_line)

    def run_single_benchmark(self, 
                             request_rate: int, 
                             client_count: int, 
                             input_length: int, 
                             output_length: int, 
                             num_iteration: Optional[int] = None):
        """Run a single benchmark iteration."""
        
        # if required iteration is not given, use default value
        # which enables to iterate following benchmark plan
        num_iteration = int(self._num_iteration if num_iteration is None else num_iteration)

        num_prompts = client_count * num_iteration
        if request_rate == 0:
            request_rate = 'inf'
        cmd = [
            self._container_runtime, "exec", self.container_name,
            "vllm", "bench", "serve",
            "--model", self._container_model_path,
            "--backend", "vllm",
            "--host", "localhost",
            f"--port={self.vllm_port}",
            "--dataset-name", "random",
            "--ignore-eos",
            "--trust-remote-code",
            f"--num-prompts={num_prompts}",
            f"--request-rate={request_rate}",
            f"--max-concurrency={client_count}",
            f"--random-input-len={input_length}",
            f"--random-output-len={output_length}",
            "--tokenizer", self._container_model_path,
            "--disable-tqdm",
            "--percentile-metrics", "ttft,tpot,itl,e2el"
        ]

        # Check if this configuration has already been tested
        if self._check_existing_result(request_rate, client_count, input_length, output_length, num_iteration):
            # logger.info(f"Skipping existing configuration: c{client_count}_i{input_length}_o{output_length}")
            return

        if self._is_dry_run:
            logger.info(f"Dry run - Benchmark command for r{request_rate}_n{num_iteration}_c{client_count}_i{input_length}_o{output_length}")
            logger.info(" ".join(cmd))
            return
        
        log_file = self.log_dir / f"vllm_tp{self.env_vars.get('TENSOR_PARALLEL_SIZE', '1')}_r{request_rate}_n{num_iteration}_i{input_length}_o{output_length}_c{client_count}.log"
        with open(log_file, 'w') as f:
            f.write(f"=== Benchmark: request_rate={request_rate}, num_iteration={num_iteration}, clients={client_count}, input_len={input_length}, output_len={output_length} ===\n")
        
        start_time = time.time()

        # Run the benchmark command and redirect output to log file
        with open(log_file, 'w') as f:
            subprocess.run(cmd, stdout=f, stderr=f, check=True)

        # Extract metrics from log file
        metrics = self._extract_metrics(log_file)

        # calculate test time
        metrics['test_time'] = str(datetime.timedelta(seconds=(time.time() - start_time))).split(".")[0]

        # Save and print results
        self._save_results(
            request_rate, num_iteration, client_count, input_length, output_length, metrics)
        self._print_result(
            request_rate, num_iteration, client_count, input_length, output_length, metrics)

    def _get_benchmark_scope(self):
        """Set benchmark parameters based on scope."""
        # load from test combinations in self._test_plan_path
        # combinations should be in the format of "request_rate client_count input_length output_length num_iteration"
        # one combination per line
        if not self._test_plan_path:
            raise ValueError("Custom scope file must be provided for custom scope")
        
        request_rates = []
        num_iterations = []
        client_counts = []
        input_lengths = []
        output_lengths = []

        with open(self._test_plan_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):  # Skip empty lines and comments
                    continue
                try:
                    r, c, i, o, n = map(int, line.split())
                    request_rates.append(r)
                    client_counts.append(c)
                    input_lengths.append(i)
                    output_lengths.append(o)
                    num_iterations.append(n)
                except ValueError as e:
                    logger.warning(f"Skipping invalid line: {line} - {str(e)}")
                    continue
        
        if not self.client_counts:  # Check if we have any valid combinations
            raise ValueError("No valid test combinations found in custom scope file")
        
        return request_rates, client_counts, input_lengths, output_lengths, num_iterations

    def run_benchmark(self):
        """Run the full benchmark suite."""
        if self._num_gpus == 0:
            raise ValueError("No GPU is allocated")

        # Start server for this configuration
        self.start_server()
        if not self._is_dry_run:
            if not self._wait_for_server():
                raise RuntimeError("Server failed to start")
        
            logger.info("Server is up and running")
        
        # Warmup
        if not self._is_no_warmup:
            warmup_cmd = [
                self._container_runtime, "exec", self.container_name,
                "vllm", "bench", "serve",
                "--model", self._container_model_path,
                "--backend", "vllm",
                "--host", "localhost",
                "--port", f"{self.vllm_port}",
                "--dataset-name", "random",
                "--ignore-eos",
                "--trust-remote-code",
                "--num-prompts", "16",
                "--max-concurrency", "4",
                "--random-input-len", "256",
                "--random-output-len", "256",
                "--tokenizer", self._container_model_path
            ]
            if not self._is_dry_run:
                logger.info("Started vLLM server warmup. Will have small tests ahead of real benchmarks")
                subprocess.run(warmup_cmd, stdout=subprocess.DEVNULL)
                logger.info("Warmup complete")

        self._print_header()

        # Run benchmarks for all configurations
        request_rates, num_iterations, client_counts, input_lengths, output_lengths = \
            self._get_benchmark_scope()
        for r, c, i, o, n in zip(request_rates, num_iterations, client_counts, input_lengths, output_lengths):
            try:
                # use specified argument in cli
                if self._request_rate is not None:
                    r = self._request_rate
                if self._num_iteration is not None:
                    r = self._num_iteration

                self.run_single_benchmark(r, c, i, o, n)
            except subprocess.CalledProcessError as e:
                logger.error(f"Benchmark failed for r{r}_n{n}_c{c}_i{i}_o{o}: {str(e)}")
                return
    
        # Cleanup container after this configuration
        self._cleanup_container()
        if not self._is_dry_run:
            logger.info(f"Benchmarking complete. Results saved to {self.result_file}")


def main():
    args = get_args()    
    
    try:
        benchmark = VLLMBenchmark(
            env_file=args.env_file,
            model_path=args.model_path,
            vllm_image=args.vllm_image,
            test_plan=args.test_plan,
            gpu_devices=args.gpu_devices,
            request_rate=args.request_rate,
            num_iteration=args.num_iteration,
            max_seq_num=args.max_seq_num,
            no_warmup=args.no_warmup,
            dry_run=args.dry_run,
        )
        benchmark.run_benchmark()
    except Exception as e:
        logger.error(f"Benchmark failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
