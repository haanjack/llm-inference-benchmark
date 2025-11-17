#!/usr/bin/env python3

import argparse
import logging
import sys

from llm_benchmark.server.vllm import VLLMServer
from llm_benchmark.server.sglang import SGLangServer
from llm_benchmark.runner import BenchmarkRunner
from llm_benchmark.clients.vllm import VLLMClient
from llm_benchmark.clients.genai_perf import GenAIPerfClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

def get_args():
    """Benchmark arguments"""
    parser = argparse.ArgumentParser(description='Run LLM benchmarks')

    # benchmark configuration
    parser.add_argument('--model-config', required=True,
                        help='Model config file name')
    parser.add_argument('--model-path-or-id', required=True,
                        help='Model checkpoint path or model id in huggingface hub')
    parser.add_argument('--backend', default='vllm', choices=['vllm', 'sglang'],
                        help='LLM serving backend to use')
    parser.add_argument('--vllm-image',
                        help='vLLM Docker image.')
    parser.add_argument('--sglang-image',
                        help='SGLang Docker image.')
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
    parser.add_argument('--benchmark-client', default='genai-perf',
                        choices=['vllm', 'genai-perf'],
                        help='Benchmark client to use')

    # test control
    parser.add_argument('--no-warmup', action='store_true',
                        help='no warmup at benchmark start')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show commands without executing them')
    parser.add_argument('--in-container', action='store_true',
                        help='Run benchmark directly without launching a new container')
    parser.add_argument('--server-test', action='store_true',
                        help='Initialize server only, and show benchmark commands for test')

    args = parser.parse_args()

    if args.backend == 'vllm' and not args.vllm_image:
        parser.error("--vllm-image is required when backend is 'vllm'")
    if args.backend == 'sglang' and not args.sglang_image:
        parser.error("--sglang-image is required when backend is 'sglang'")

    return args


def main():
    """Main function to run the benchmark."""
    try:
        args = get_args()

        model_config_path = f"configs/models/{args.model_config}-{args.backend}"

        server_kwargs = {
            "env_file": args.env_file,
            "model_config": model_config_path,
            "model_path_or_id": args.model_path_or_id,
            "model_root_dir": args.model_root_dir,
            "gpu_devices": args.gpu_devices,
            "num_gpus": args.num_gpus,
            "arch": args.arch,
            "dry_run": args.dry_run,
            "no_warmup": args.no_warmup,
            "in_container": args.in_container,
            "test_plan": args.test_plan,
        }

        if args.backend == 'vllm':
            server = VLLMServer(vllm_image=args.vllm_image, **server_kwargs)
        elif args.backend == 'sglang':
            server = SGLangServer(sglang_image=args.sglang_image, **server_kwargs)
        else:
            raise ValueError(f"Unknown backend: {args.backend}")

        server.start()

        if args.benchmark_client == 'vllm':
            client = VLLMClient(server=server, is_dry_run=args.server_test)
        elif args.benchmark_client == 'genai-perf':
            client = GenAIPerfClient(server=server, is_dry_run=args.server_test)
        else:
            raise ValueError(f"Unknown benchmark client: {args.benchmark_client}")

        runner = BenchmarkRunner(
            server=server,
            client=client,
            test_plan=args.test_plan,
            sub_tasks=args.sub_tasks,
            is_dry_run=args.server_test,
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