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
                 num_iterations: int = 8,
                 dry_run: bool = False):
        self.env_file = env_file
        self.env_vars = self._load_env_file()
        self.dry_run = dry_run
        
        # Initialize configuration
        self.model_name = model_name or self.env_vars.get("MODEL_NAME", "Meta-Llama-3-8B-Instruct-FP8")
        self.model_path = f"/workspace/models/{self.model_name}"
        self.vllm_image = vllm_image or self.env_vars.get("VLLM_IMAGE", "docker.io/rocm/vllm:latest")
        
        # Set benchmark parameters based on scope
        self._set_benchmark_scope(bench_scope)
        self._num_iterations = num_iterations
        
        # GPU configuration
        self.gpu_devices = custom_visible_devices # TODO: select based on os.environ.get("SLURM_JOB_GPUS", "")
        self.gpu_array = self.gpu_devices.split(',')
        self.first_gpu_id = self.gpu_array[0] if self.gpu_array else None
        self.num_gpus = len(self.gpu_array) if self.gpu_array[0] else 0
        
        # Result file headers
        self._headers = [
            "env,TP Size,Client Count,Input Length,Output Length,",
            "Mean TTFT (ms),Median TTFT (ms),P99 TTFT (ms),",
            "Mean TPOT (ms),Median TPOT (ms),P99 TPOT (ms),",
            "Mean ITL (ms),Median ITL (ms),P99 ITL (ms),",
            "Mean E2EL (ms),Median E2EL (ms),P99 E2EL (ms),",
            "Request Throughput (req/s),Output token throughput (tok/s),",
            "Total Token throughput (tok/s)"
        ]

        # Container name setup
        self._setup_container_name()
        
        # Setup logging directories
        self._setup_logging_dirs()

        # VLLM port setup
        self.vllm_port = 23400 + int(self.first_gpu_id) if self.first_gpu_id else 23400

    def _load_env_file(self) -> Dict[str, str]:
        """Load environment variables from the specified env file."""
        env_file_path = Path.cwd() / "envs" / self.env_file
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

    def _set_benchmark_scope(self, scope: str):
        """Set benchmark parameters based on scope."""
        if scope == "test":
            self.client_counts = [4]
            self.input_lengths = [2048]
            self.output_lengths = [2048]
        elif scope == "prefill":
            self.client_counts = [1, 2, 4, 8, 16, 32, 64, 128]
            self.input_lengths = [256, 512, 1024, 2048, 4096, 8192]
            self.output_lengths = [128]
        elif scope == "decode":
            self.client_counts = [1, 2, 4, 8, 16, 32, 64, 128]
            self.input_lengths = [128]
            self.output_lengths = [128, 1024, 2048, 4096]
        elif scope == "middle":
            self.client_counts = [1, 2, 4, 8, 16, 32, 64, 128]
            self.input_lengths = [1024, 2048, 4096, 8192]
            self.output_lengths = [1024, 2048, 4096]
        

    def _setup_container_name(self):
        """Setup container name based on environment and GPU configuration."""
        process_name = self.env_file
        slurm_job_id = os.environ.get("SLURM_JOB_ID", "")
        
        if slurm_job_id:
            self.container_name = f"{process_name}-{slurm_job_id}-g{self.gpu_devices.replace(',', '_')}"
        else:
            self.container_name = f"{process_name}-g{self.gpu_devices.replace(',', '_')}"

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
        # print header line to console
        headers_split = ''.join(self._headers).split(',')
        headers_line = [headers_split[0].ljust(30)]
        headers_line += [h.rjust(8) for h in headers_split[1:5]]
        headers_line += [h.rjust(10) for h in headers_split[5:]]
        logger.info('\t'.join(headers_line))

    def _get_vllm_args(self) -> str:
        """Construct VLLM arguments based on environment variables."""
        args = [
            "--kv-cache-dtype", f"{self.env_vars.get('KV_CACHE_DTYPE', '')}",
            "--gpu_memory_utilization", f"{self.env_vars.get('GPU_MEMORY_UTILIZATION', '0.9')}",
            "--max-seq-len-to-capture", f"{self.env_vars.get('MAX_SEQ_LEN_TO_CAPTURE', '8192')}",
            "--max-num-batched-token", f"{self.env_vars.get('MAX_NUM_BATCHED_TOKENS', '8192')}",
            "--swap-space", "64",
            "--no-enable-prefix-caching",
            "--async-scheduling"
        ]
        if self.env_vars.get('QUANTIZATION', 'auto') != 'auto':
            args.extend(["--quantization", f"{self.env_vars.get('QUANTIZATION')}"])

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
        subprocess.run(["docker", "rm", "-f", self.container_name], 
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def start_server(self, max_model_len: int):
        """Start the vLLM server in a Docker container."""
        if not self.dry_run:
            self._cleanup_container()
        
        cmd = [
            "docker", "run", "-d",
            "--name", self.container_name,
            "-v", f"{os.environ.get('HF_HOME')}:/root/.cache/huggingface",
            "--device", "/dev/kfd", "--device", "/dev/dri", "--device", "/dev/mem",
            "--group-add", "video",
            "--ipc=host", "--network=host",
            "--cap-add=CAP_SYS_ADMIN",
            "--cap-add=SYS_PTRACE",
            "--shm-size=2g",
            "--security-opt", "seccomp=unconfined",
            "--env-file", str(Path.cwd() / "envs" / self.env_file),
            "-e", f"VLLM_USE_TRITON_FLASH_ATTN={self.env_vars.get('VLLM_USE_TRITON_FLASH_ATTN', '0')}",
            "-e", f"CUDA_VISIBLE_DEVICES={self.gpu_devices}",
            "-v", f"{os.environ.get('HOME')}:/workspace/",
            self.vllm_image,
            "vllm", "serve",
            self.model_path,
            "--no-enable-log-requests",
            "--trust-remote-code",
            "--max-model-len", f"{max_model_len}",
            "--tensor-parallel-size", f"{self.num_gpus}",
            "--distributed-executor-backend", "mp",
            "--block-size", "64",
            "--port", f"{self.vllm_port}",
            "--host", "0.0.0.0"
        ]
        
        cmd.extend(self._get_vllm_args().split())
        
        if self.dry_run:
            logger.info("Dry run - Docker server command:")
            logger.info(" ".join(cmd))
        else:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
        
        # Start log collection
        with open(self.server_log, 'a') as f:
            # Create two processes to capture both stdout and stderr
            stdout_process = subprocess.Popen(
                ["docker", "logs", "-f", self.container_name],
                stdout=f,
                stderr=subprocess.PIPE
            )
            stderr_process = subprocess.Popen(
                ["docker", "logs", "-f", self.container_name],
                stdout=subprocess.PIPE,
                stderr=f
            )
            # Store process IDs for later cleanup if needed
            self.log_processes = [stdout_process, stderr_process]

    def _wait_for_server(self, timeout: int = 600) -> bool:
        """Wait for the server to become available."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"http://localhost:{self.vllm_port}/v1/models")
                if response.status_code == 200:
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

    def run_single_benchmark(self, client_count: int, input_length: int, output_length: int):
        """Run a single benchmark iteration."""
        log_file = self.log_dir / f"vllm_tp{self.env_vars.get('TENSOR_PARALLEL_SIZE', '1')}_i{input_length}_o{output_length}_c{client_count}.log"
        
        # Check if this configuration has already been tested
        if not self.dry_run and self._check_existing_result(client_count, input_length, output_length):
            # logger.info(f"Skipping existing configuration: c{client_count}_i{input_length}_o{output_length}")
            return

        num_prompts = client_count + self._num_iterations
        cmd = [
            "docker", "exec", self.container_name,
            "vllm", "bench", "serve",
            "--model", self.model_path,
            "--backend", "vllm",
            "--host", "localhost",
            f"--port={self.vllm_port}",
            "--dataset-name", "random",
            "--ignore-eos",
            "--trust-remote-code",
            f"--num-prompts={num_prompts}",
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
        else:
            with open(log_file, 'w') as f:
                subprocess.run(cmd, stdout=f, stderr=f, check=True)

        # Process and save results
        if not self.dry_run:
            metrics = self._extract_metrics(log_file)
            self._save_results(client_count, input_length, output_length, metrics)
            self._print_result(client_count, input_length, output_length, metrics)

    def _check_existing_result(self, client_count: int, input_length: int, output_length: int) -> bool:
        """Check if results already exist for this configuration."""
        if not self.result_file.exists():
            return False

        search_str = f"{self.env_file},{self.num_gpus},{client_count},{input_length},{output_length}"
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

    def _save_results(self, client_count: int, input_length: int, output_length: int, metrics: Dict[str, float]):
        """Save benchmark results to the result file."""
        result_line = (
            f"{self.env_file},{self.num_gpus},"
            f"{client_count},{input_length},{output_length},"
            f"{metrics['ttft_mean']:.2f},{metrics['ttft_median']:.2f},{metrics['ttft_p99']:.2f},"
            f"{metrics['tpot_mean']:.2f},{metrics['tpot_median']:.2f},{metrics['tpot_p99']:.2f},"
            f"{metrics['itl_mean']:.2f},{metrics['itl_median']:.2f},{metrics['itl_p99']:.2f},"
            f"{metrics['e2el_mean']:.2f},{metrics['e2el_median']:.2f},{metrics['e2el_p99']:.2f},"
            f"{metrics['request_throughput']:.2f},"
            f"{metrics['output_token_throughput']:.2f},{metrics['total_token_throughput']:.2f}\n"
        )
        
        with open(self.result_file, 'a') as f:
            f.write(result_line)

    def _print_result(self, client_count: int, input_length: int, output_length: int, metrics: Dict[str, float]):
        """Print the result to console."""
        result_line = (
            f"{self.env_file.ljust(30)}\t{self.num_gpus:>8d}\t"
            f"{client_count:>8d}\t{input_length:>8d}\t{output_length:>8d}\t"
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
        max_model_len = self.input_lengths[-1] + self.output_lengths[-1] + 256
        self.start_server(max_model_len)
        if not self.dry_run:
            if not self._wait_for_server():
                raise RuntimeError("Server failed to start")
        
            logger.info("Server is up and running")
        
        # Warmup
        warmup_cmd = [
            "docker", "exec", self.container_name,
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
            subprocess.run(warmup_cmd, stdout=subprocess.DEVNULL)
            logger.info("Warmup complete")

            self._print_header()

        # Run benchmarks for all configurations
        for output_length in self.output_lengths:
            for input_length in self.input_lengths:
                # Run benchmarks for all client counts
                for client_count in self.client_counts:
                    try:
                        self.run_single_benchmark(client_count, input_length, output_length)
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
                       choices=['test', 'prefill', 'decode', 'middle'],
                       help='Benchmark scope')
    parser.add_argument('--gpu-devices', help='Comma-separated GPU device IDs')
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
            dry_run=args.dry_run
        )
        benchmark.run_benchmark()
    except Exception as e:
        logger.error(f"Benchmark failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()