#!/usr/bin/env python3

import argparse
import logging
import sys
from pathlib import Path

from llm_benchmark.server import VLLMServer, SGLangServer, RemoteServer
from llm_benchmark.clients import VLLMClient, SGLangClient, GenAIPerfClient, EvaluationClient
from llm_benchmark.runner import BenchmarkRunner
from llm_benchmark.utils.script_generator import ScriptGenerator, prettify_generated_scripts
from llm_benchmark.utils.utils import parse_env_file

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

TMP_SCRIPT_DIR = Path("/tmp/generated_benchmark_scripts")

def get_args():
    """Benchmark arguments"""
    parser = argparse.ArgumentParser(description="Run LLM benchmarks")

    # benchmark configuration
    parser.add_argument("--model-config", required=True, help="Model config file name")
    parser.add_argument(
        "--model-path-or-id",
        required=True,
        help="Model checkpoint path or model id in huggingface hub",
    )
    parser.add_argument(
        "--backend",
        default="vllm",
        choices=["vllm", "sglang"],
        help="LLM serving backend to use",
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Docker image for the selected backend (vLLM or SGLang).",
    )
    parser.add_argument(
        "--test-plan",
        default="test",
        help="Benchmark test plan YAML file in configs/benchmark_plans/ \
                            (without .yaml extension)",
    )
    parser.add_argument(
        "--sub-tasks",
        default=None,
        type=str,
        nargs="+",
        help="Testing sub-tasks in test-plan",
    )
    parser.add_argument(
        "--env-file", default="configs/envs/common", help="Environment file name"
    )
    parser.add_argument(
        "--model-root-dir", default="models", help="Model root directory"
    )
    parser.add_argument(
        "--gpu-devices", default=None, help="Comma-separated GPU device IDs"
    )
    parser.add_argument("--num-gpus", default=None, help="Number of GPUs")
    parser.add_argument(
        "--arch", default=None, help="Target GPU architecture for model config"
    )
    parser.add_argument(
        "--benchmark-client",
        default="vllm",
        choices=["vllm", "genai-perf", "sglang"],
        help="Benchmark client to use",
    )
    parser.add_argument(
        "--endpoint",
        default=None,
        help="Specify a remote endpoint URL to benchmark against. "
        "If provided, the script will not start a local server.",
    )

    # test control
    parser.add_argument('--no-warmup', action='store_true',
                        help='no warmup at benchmark start')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show commands without executing them')
    parser.add_argument('--in-container', action='store_true',
                        help='Run benchmark directly without launching a new container')
    parser.add_argument('--server-test', action='store_true',
                        help='Initialize server only, and show benchmark commands for test')
    parser.add_argument('--generate-script', action='store_true',
                        help='Generate a bash script for the benchmark run.')
    parser.add_argument('--generated-script-output-dir', default='scripts/generated',
                        help='Output directory for generated scripts (default: scripts/generated)')

    # evaluation configuration - chain of clients
    parser.add_argument(
        '--run-evaluation',
        action='store_true',
        help='Run model evaluation after benchmark (chain of clients)'
    )
    parser.add_argument(
        '--evaluation-plan',
        default='default',
        help='Evaluation plan YAML file in configs/evaluation_plans/ (without .yaml)'
    )
    parser.add_argument(
        '--evaluation-cache-dir',
        default=None,
        help='Cache directory for evaluation datasets (default: ~/.cache/huggingface)'
    )

    args = parser.parse_args()

    if args.endpoint:
        # When using an endpoint, some arguments related to server startup are not needed
        # but we still need model and image for logging.
        # Let's make them optional if endpoint is provided.
        # We can achieve this by relaxing the 'required' constraint post-parsing.
        pass
    return args


def main():
    """Main function to run the benchmark."""
    try:
        args = get_args()

        script_generator = None
        if args.generate_script:
            args.dry_run = True  # --generate-script implies --dry-run
            model_config_name = Path(args.model_config).stem
            # Use /tmp directory for intermediate script generation
            TMP_SCRIPT_DIR.mkdir(parents=True, exist_ok=True)
            tmp_script_path = TMP_SCRIPT_DIR / f"run-{model_config_name}_with_{args.benchmark_client}-{args.test_plan}.sh"
            script_generator = ScriptGenerator(output_path=tmp_script_path, in_container=args.in_container)

        envs = parse_env_file(args.env_file) if args.env_file else {}

        # Common arguments for all server types
        server_kwargs = {
            "image": args.image,
            "model_config": args.model_config,
            "model_path_or_id": args.model_path_or_id,
            "model_root_dir": args.model_root_dir,
            "gpu_devices": args.gpu_devices,
            "num_gpus": args.num_gpus,
            "arch": args.arch,
            "dry_run": args.dry_run,
            "no_warmup": args.no_warmup,
            "in_container": args.in_container,
            "test_plan": args.test_plan,
            "envs": envs,
            "script_generator": script_generator,
        }

        if args.endpoint:
            server_kwargs["endpoint"] = args.endpoint

        logger.info("Model Name: %s", args.model_path_or_id)
        logger.info("GPU devices: %s", args.gpu_devices)
        logger.info("Backend: %s", args.backend)
        logger.info("Image: %s", args.image)
        logger.info("Benchmark Client: %s", args.benchmark_client)

        if args.endpoint:
            server = RemoteServer(**server_kwargs)
        else:
            if args.backend == "vllm":
                server = VLLMServer(**server_kwargs)
            elif args.backend == "sglang":
                server = SGLangServer(**server_kwargs)
            else:
                raise ValueError(f"Unknown backend: {args.backend}")

            server.start()

        if args.benchmark_client == "vllm":
            client = VLLMClient(
                server=server,
                is_dry_run=args.server_test or args.dry_run,
                script_generator=script_generator,
            )
        elif args.benchmark_client == "genai-perf":
            client = GenAIPerfClient(
                server=server,
                is_dry_run=args.server_test or args.dry_run,
                script_generator=script_generator,
            )
        elif args.benchmark_client == "sglang":
            client = SGLangClient(
                server=server,
                is_dry_run=args.server_test or args.dry_run,
                script_generator=script_generator,
            )
        else:
            raise ValueError(f"Unknown benchmark client: {args.benchmark_client}")

        # Create evaluation client if requested (chain of clients pattern)
        evaluation_client = None
        if args.run_evaluation:
            logger.info("Evaluation: ENABLED (chained after benchmarks)")
            logger.info("Evaluation Plan: %s", args.evaluation_plan)
            if args.evaluation_cache_dir:
                logger.info("Evaluation Cache Dir: %s", args.evaluation_cache_dir)
            evaluation_client = EvaluationClient(
                server=server,
                is_dry_run=args.server_test or args.dry_run,
                script_generator=script_generator,
                evaluation_plan=args.evaluation_plan,
                cache_dir=args.evaluation_cache_dir,
            )

        runner = BenchmarkRunner(
            server=server,
            client=client,
            evaluation_client=evaluation_client,  # ← Chained evaluation client
            test_plan=args.test_plan,
            sub_tasks=args.sub_tasks,
            is_dry_run=args.server_test,
            script_generator=script_generator,
        )

        runner.run()

        # After script generation, prettify the scripts
        if args.generate_script and script_generator:
            output_dir = Path(args.generated_script_output_dir)
            model_config_name = Path(args.model_config).stem
            prettify_generated_scripts(TMP_SCRIPT_DIR, output_dir, model_config_name, args.benchmark_client, args.test_plan)

    except Exception as e:
        logger.exception("Benchmark failed: %s", str(e))
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Benchmark interrupted by user.")
        sys.exit(1)


if __name__ == "__main__":
    main()
