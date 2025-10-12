#!/usr/bin/env python3

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import re
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VLLMBenchmark:
    def __init__(self, 
                 env_file: str = "baseline", 
                 model_name: Optional[str] = None, 
                 vllm_image: Optional[str] = None, 
                 bench_scope: str = "test",
                 custom_visible_devices: Optional[str] = None,
                 num_iteration: int = None,
                 request_rate: int = None,
                 no_warmup: bool = False,
                 custom_scope_file: Optional[str] = None,
                 dry_run: bool = False):
        self.env_file = env_file
        self.env_vars = self._load_env_file()
        self.dry_run = dry_run
        self._bench_scope = bench_scope
        self.no_warmup = no_warmup
 
        # Initialize configuration
        self.model_name = model_name or self.env_vars.get("MODEL_NAME", "Meta-Llama-3-8B-Instruct-FP8")
        self.model_path = f"/workspace/{self.model_name}"
        self.vllm_image = vllm_image or self.env_vars.get("VLLM_IMAGE", "docker.io/rocm/vllm:latest")
        
        # Set benchmark parameters based on scope
        self._custom_scope_file = custom_scope_file
        self._set_benchmark_scope()
        self._num_iteration = self.env_vars.get('NUM_ITERATION', 1)
        if num_iteration is not None:
            self._num_iteration = num_iteration
        self._request_rate = self.env_vars.get('REQUEST_RATE', 1)
        if request_rate is not None:
            self._request_rate = request_rate
        
        # GPU configuration
        self.gpu_devices = custom_visible_devices # TODO: select based on os.environ.get("SLURM_JOB_GPUS", "")
        self.gpu_array = self.gpu_devices.split(',')
        self.first_gpu_id = self.gpu_array[0] if self.gpu_array else None
        self.num_gpus = len(self.gpu_array) if self.gpu_array[0] else 0
        
        # Result file headers
        self._headers = [
            "env,TP Size,",
            "Request Rate,Num. Iter,Client Count,Input Length,Output Length,Test Time,",
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
        self.vllm_port = 23400 + int(self.first_gpu_id) if self.first_gpu_id else 23400

        self._print_benchmark_info()

    def _print_benchmark_info(self):
        logger.info("Start vLLM benchmark")
        logger.info(f"Model Name: {self.model_name}")
        logger.info(f"vLLM docker image: {self.vllm_image}")
        logger.info(f"Benchmark scenario: {self._bench_scope}")
        if self._bench_scope == "custom":
            logger.info(f"{self._custom_scope_file}")

    def _is_docker_available(self) -> bool:
        """Check if Docker is installed on the system."""
        try:
            return subprocess.run(["docker", "images"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
        except FileNotFoundError:
            return False
        
    def _load_env_file(self) -> Dict[str, str]:
        """Load environment variables from the specified env file."""
        env_file_path = Path.cwd() / self.env_file
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

    def _set_benchmark_scope(self):
        """Set benchmark parameters based on scope."""
        if self._bench_scope == "test":
            self.client_counts = [4]
            self.input_lengths = [2048]
            self.output_lengths = [2048]
        elif self._bench_scope == "prefill":
            self.client_counts = [1, 2, 4, 8, 16, 32, 64, 128]
            self.input_lengths = [256, 512, 1024, 2048, 4096, 8192]
            self.output_lengths = [128]
        elif self._bench_scope == "decode":
            self.client_counts = [1, 2, 4, 8, 16, 32, 64, 128]
            self.input_lengths = [128]
            self.output_lengths = [128, 1024, 2048, 4096]
        elif self._bench_scope == "middle":
            self.client_counts = [1, 2, 4, 8, 16, 32, 64, 128]
            self.input_lengths = [1024, 2048, 4096, 8192]
            self.output_lengths = [1024, 2048, 4096]
        elif self._bench_scope == "custom":
            # load from test combinations in self._custom_scope_file
            # combinations should be in the format of "request_rate client_count input_length output_length num_iteration"
            # one combination per line
            if not self._custom_scope_file:
                raise ValueError("Custom scope file must be provided for custom scope")
            custom_scope_path = Path(self._custom_scope_file)
            if not custom_scope_path.exists():
                raise FileNotFoundError(f"Custom scope file not found: {custom_scope_path}")
            
            self.request_rates = []
            self.client_counts = []
            self.input_lengths = []
            self.output_lengths = []
            self.num_iterations = []

            with open(custom_scope_path) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):  # Skip empty lines and comments
                        continue
                    try:
                        r, c, i, o, n = map(int, line.split())
                        self.request_rates.append(r)
                        self.client_counts.append(c)
                        self.input_lengths.append(i)
                        self.output_lengths.append(o)
                        self.num_iterations.append(n)
                    except ValueError as e:
                        logger.warning(f"Skipping invalid line: {line} - {str(e)}")
                        continue
            
            if not self.client_counts:  # Check if we have any valid combinations
                raise ValueError("No valid test combinations found in custom scope file")

        else:
            raise ValueError(f"Invalid benchmark scope: {self._bench_scope}")
        

    def _setup_container_name(self):
        """Setup container name based on environment and GPU configuration."""
        process_name = self.env_file
        image_tag = self.vllm_image.split(':')[-1]
        slurm_job_id = os.environ.get("SLURM_JOB_ID", None)
        
        self.container_name = ""
        if slurm_job_id:
            self.container_name = f"{slurm_job_id}-"
        self.container_name += f"{os.path.basename(self.model_name)}-{image_tag}-{os.path.basename(process_name)}-g{self.gpu_devices.replace(',', '_')}"

    def _setup_logging_dirs(self):
        """Setup logging directories for the benchmark."""
        image_tag = self.vllm_image.split(':')[-1]
        self.log_dir = Path("logs") / self.model_name / image_tag
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.server_log = self.log_dir / f"server_log_t{self.num_gpus}.txt"
        self.result_file = self.log_dir / "result_list.csv"
        
        # Initialize result file if it doesn't exist
        if not self.result_file.exists():
            self._init_result_file()

    def _init_result_file(self):
        """Initialize the result file with headers."""
        with open(self.result_file, 'w') as f:
            f.write(''.join(self._headers) + '\n')

    def _print_header(self):
        """Print the header line to console."""
        metric_start_column_idx = 8
        # print header line to console
        headers_split = ''.join(self._headers).split(',')
        headers_line = [headers_split[0].ljust(30)]
        headers_line += [h.rjust(8) for h in headers_split[1:metric_start_column_idx]]
        headers_line += [h.rjust(10) for h in headers_split[metric_start_column_idx:]]
        logger.info('\t'.join(headers_line))

    def _get_vllm_args(self) -> str:
        """Construct VLLM arguments based on environment variables."""
        max_model_len = self.input_lengths[-1] + self.output_lengths[-1] + 256
        args = [
            "--kv-cache-dtype", f"{self.env_vars.get('KV_CACHE_DTYPE', '')}",
            "--gpu_memory_utilization", f"{self.env_vars.get('GPU_MEMORY_UTILIZATION', '0.9')}",
            "--max-num-batched-token", f"{self.env_vars.get('MAX_NUM_BATCHED_TOKENS', '4096')}",
            "--max-num-seqs", f"{self.env_vars.get('MAX_NUM_SEQS', '128')}",
            "--swap-space", "64",
            "--no-enable-prefix-caching",
        ]
        if self.env_vars.get('QUANTIZATION', 'auto') != 'auto':
            args.extend(["--quantization", f"{self.env_vars.get('QUANTIZATION')}"])
        
        # ROCM version handling
        # rocm version format should be like "6.4.0" or "7.0.1"
        # load rocm version from pytorch installed in container
        container_rocm_version = subprocess.run(
            [self._container_runtime, "run", "--rm", self.vllm_image, "python3", "-c", 
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
        if not self.dry_run:
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
            "--env-file", str(Path.cwd() / self.env_file),
            "-e", f"VLLM_USE_TRITON_FLASH_ATTN={self.env_vars.get('VLLM_USE_TRITON_FLASH_ATTN', '0')}",
            "-e", f"CUDA_VISIBLE_DEVICES={self.gpu_devices}",
            "-v", f"{os.environ.get('HOME')}:/workspace/",
            self.vllm_image,
            "vllm", "serve",
            self.model_path,
            "--no-enable-log-requests",
            "--trust-remote-code",
            "--tensor-parallel-size", f"{self.num_gpus}",
            "--distributed-executor-backend", "mp",
            "--port", f"{self.vllm_port}",
            "--host", "0.0.0.0"
        ]
        
        cmd.extend(self._get_vllm_args().split())
        
        if self.dry_run:
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

    def _wait_for_server(self, timeout: int = 1200) -> bool:
        """Wait for the server to become available."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"http://localhost:{self.vllm_port}/v1/models")
                print(response)
                return True
            except requests.exceptions.RequestException:
                time.sleep(5)
        return False

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

    def run_single_benchmark(self, 
                             request_rate: int, 
                             client_count: int, 
                             input_length: int, 
                             output_length: int, 
                             num_iteration: Optional[int] = None):
        """Run a single benchmark iteration."""
        log_file = self.log_dir / f"vllm_tp{self.env_vars.get('TENSOR_PARALLEL_SIZE', '1')}_i{input_length}_o{output_length}_c{client_count}.log"
        
        # Check if this configuration has already been tested
        num_iteration = self._num_iteration if num_iteration is None else num_iteration
        if not self.dry_run and self._check_existing_result(request_rate, client_count, input_length, output_length, num_iteration):
            # logger.info(f"Skipping existing configuration: c{client_count}_i{input_length}_o{output_length}")
            return

        num_prompts = client_count * num_iteration
        if request_rate == 0:
            request_rate = 'inf'
        cmd = [
            self._container_runtime, "exec", self.container_name,
            "vllm", "bench", "serve",
            "--model", self.model_path,
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
            "--tokenizer", self.model_path,
            "--disable-tqdm",
            "--percentile-metrics", "ttft,tpot,itl,e2el"
        ]

        if self.dry_run:
            logger.info(f"Dry run - Benchmark command for c{client_count}_i{input_length}_o{output_length}:")
            logger.info(" ".join(cmd))
            return
        
        with open(log_file, 'a') as f:
            f.write(f"=== Benchmark: clients={client_count}, input_len={input_length}, output_len={output_length}, request_rate={request_rate}, num_iteration={num_iteration or self.env_vars.get('NUM_ITERATION', self._num_iteration)} ===\n")
        start_time = time.time()
        with open(log_file, 'a') as f:
            subprocess.run(cmd, stdout=f, stderr=f, check=True)
        elapsed_time_seconds = time.time() - start_time
        hours = int(elapsed_time_seconds // 3600)
        minutes = int((elapsed_time_seconds % 3600) // 60)
        seconds = int(elapsed_time_seconds % 60)
        test_time=f"{hours:02}:{minutes:02}:{seconds:02}"
        metrics = self._extract_metrics(log_file)
        metrics['test_time'] = test_time

        self._save_results(
            request_rate, num_iteration, client_count, input_length, output_length, metrics)
        self._print_result(
            request_rate, num_iteration, client_count, input_length, output_length, metrics)

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
            num_iteration = self.env_vars.get('NUM_ITERATION', self._num_iteration)
        search_str = f"{self.env_file},{self.num_gpus},{request_rate},{num_iteration},{client_count},{input_length},{output_length}"
        search_result = any(search_str in line for line in self.result_file.read_text().splitlines())

        # print previous benchmark result if exists
        if search_result:
            with open(self.result_file, 'r') as f:
                for line in f:
                    if search_str in line:
                        line = line.strip()
                        s_line = [line.split(',')[0].ljust(30)]
                        s_line += [h.rjust(8) for h in line.split(',')[1:5]] 
                        s_line += [h.rjust(10) for h in line.split(',')[5:]]
                        logger.info(f"{''.join(s_line)}")
        return search_result
    def _save_results(self, request_rate: int, num_iteration: int, client_count: int, input_length: int, output_length: int, metrics: Dict[str, float]):

        """Save benchmark results to the result file."""
        result_line = (
            f"{self.env_file},{self.num_gpus},"
            f"{request_rate},{num_iteration},{client_count},{input_length},{output_length},{metrics['test_time']},"
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
            f"{self.env_file.ljust(30)}\t{self.num_gpus:>8d}\t"
            f"{request_rate:>4d}\t{num_iteration:>4d}\t{client_count:>8d}\t{input_length:>8d}\t{output_length:>8d}\t{metrics['test_time']}\t"
            f"{metrics['ttft_mean']:10.2f}\t{metrics['ttft_median']:10.2f}\t{metrics['ttft_p99']:10.2f}\t"
            f"{metrics['tpot_mean']:10.2f}\t{metrics['tpot_median']:10.2f}\t{metrics['tpot_p99']:10.2f}\t"
            f"{metrics['itl_mean']:10.2f}\t{metrics['itl_median']:10.2f}\t{metrics['itl_p99']:10.2f}\t"
            f"{metrics['e2el_mean']:10.2f}\t{metrics['request_throughput']:10.2f}\t"
            f"{metrics['output_token_throughput']:10.2f}\t{metrics['total_token_throughput']:10.2f}"
        )

        logger.info(result_line)

    def run_benchmark(self):
        """Run the full benchmark suite."""
        if self.num_gpus == 0:
            raise ValueError("No GPU is allocated")

        # Start server for this configuration
        self.start_server()
        if not self.dry_run:
            if not self._wait_for_server():
                raise RuntimeError("Server failed to start")
        
            logger.info("Server is up and running")
        
        # Warmup
        if not self.no_warmup:
            warmup_cmd = [
                self._container_runtime, "exec", self.container_name,
                "vllm", "bench", "serve",
                "--model", self.model_path,
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
                "--tokenizer", self.model_path
            ]
            if not self.dry_run:
                logger.info("Started vLLM server warmup. Will have small tests ahead of real benchmarks")
                subprocess.run(warmup_cmd, stdout=subprocess.DEVNULL)
                logger.info("Warmup complete")

        self._print_header()

        # Run benchmarks for all configurations
        if self._bench_scope == "custom":
            for r, c, i, o, n in zip(self.request_rates, self.client_counts, self.input_lengths, self.output_lengths, self.num_iterations):
                try:
                    self.run_single_benchmark(r, c, i, o, n)
                except subprocess.CalledProcessError as e:
                    logger.error(f"Benchmark failed for c{c}_i{i}_o{o}_n{n}: {str(e)}")
                    continue
            # Cleanup container after this configuration
            self._cleanup_container()
            if not self.dry_run:
                logger.info(f"Benchmarking complete. Results saved to {self.result_file}")
            return
        else:
            request_rate = self._request_rate if self._request_rate is not None else 'inf'
            for output_length in self.output_lengths:
                for input_length in self.input_lengths:
                    # Run benchmarks for all client counts
                    for client_count in self.client_counts:
                        try:
                            self.run_single_benchmark(request_rate, client_count, input_length, output_length, self._num_iteration)
                        except subprocess.CalledProcessError as e:
                            logger.error(f"Benchmark failed for c{client_count}_i{input_length}_o{output_length}: {str(e)}")
                            continue
                
        # Cleanup container after this configuration
        self._cleanup_container()

        if not self.dry_run:
            logger.info(f"Benchmarking complete. Results saved to {self.result_file}")

def main():
    parser = argparse.ArgumentParser(description='Run vLLM benchmarks')
    parser.add_argument('--env-file', default='baseline', help='Environment file name')
    parser.add_argument('--model-name', help='Model name')
    parser.add_argument('--vllm-image', help='vLLM Docker image')
    parser.add_argument('--bench-scope', default='test', 
                       choices=['test', 'custom', 'prefill', 'decode', 'middle'],
                       help='Benchmark scope')
    parser.add_argument('--gpu-devices', help='Comma-separated GPU device IDs')
    parser.add_argument('--num-iteration', type=int, default=None,
                        help='Number of batch iterations')
    parser.add_argument('--request-rate', type=int, default=None,
                       help='Request rate for the benchmark')
    parser.add_argument('--custom-scope-file', type=str, default=None,
                        help='Path to the custom scope file (if bench-scope is custom)')
    parser.add_argument('--no-warmup', action='store_true', 
                        help='no warmup at benchmark start')
    parser.add_argument('--dry-run', action='store_true', 
                       help='Show commands without executing them')
    
    args = parser.parse_args()
    
    try:
        benchmark = VLLMBenchmark(
            env_file=args.env_file,
            model_name=args.model_name,
            vllm_image=args.vllm_image,
            bench_scope=args.bench_scope,
            custom_visible_devices=args.gpu_devices,
            request_rate=args.request_rate,
            custom_scope_file=args.custom_scope_file,
            num_iteration=args.num_iteration,
            no_warmup=args.no_warmup,
            dry_run=args.dry_run,
        )
        benchmark.run_benchmark()
    except Exception as e:
        logger.error(f"Benchmark failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
