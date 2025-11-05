# LLM Inference Benchmark Suite

This is benchmark script for extensive inference tests for various setups. This benchmark script help users to test with the following variants.
 - various environment variable sets
 - input/output sequence length
 - num concurrency
 - iteration based num prompts control
 - request rate control
 - `docker` and `podman` agnostic

In addition, this benchmark test tries to obey [AMD's vLLM V1 performance optimization](https://rocm.docs.amd.com/en/develop/how-to/rocm-for-ai/inference-optimization/vllm-optimization.html) guide and validation.

This benchmark script generate vLLM server command, execute, and perform benchmark.

## Basic usage of test

The following command shows how to use this.

```bash
python run_vllm_benchmark.py \
    --env-file configs/envs/mi300x/vllm/baseline \
    --model-path ~/models/Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 \
    --vllm-image docker.io/rocm/vllm:rocm7.0.0_vllm_0.10.2_20251006 \
    --test-plan test \
    --gpu-devices 0
```

There are several interesting argument: `model-config` and `test-plan`. These are the purpose of this benchmark script. "What is best setting?" and "I just want to roll-out the tests". Check each section for more details.

### Environment File
Each model has optimal configuration and settings. To support this divergency, this benchmark script provide individual model config yaml file.

Following snippet shows the basic format of the model config file.
```yaml
# Default model configuration

env:
  # Example environment variables
  # HUGGING_FACE_HUB_TOKEN: "your_token_here"
  VLLM_ROCM_USE_AITER: 1

vllm_server_args:
  # Arguments for the vLLM server
  quantization: fp8
  kv-cache-dtype: auto
  
  # Add other vLLM server arguments here
  # example:
  # max_model_len: 1024
  # gpu-memory-utilization: 0.95
  # max-num-batched-token: 8192
  # swap-space: 16
  # block-size: 64
  # no-enable-prefix-caching: true
  # async-scheduling: true

compilation_config:
  cudagraph_mode: "FULL_AND_PIECEWISE"
```

For more available environment variables, please checke the following documents.
- [vLLM environment variables](https://docs.vllm.ai/en/stable/configuration/env_vars.html)
- [AMD vLLM V1 performance optimization](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/vllm-optimization.html)
- [AITER switches](https://rocm.docs.amd.com/en/develop/how-to/rocm-for-ai/inference-optimization/vllm-optimization.html#aiter-ai-tensor-engine-for-rocm-switches)

vLLM server arguments can be changed following the purpose of test. If you want optimize for TTFT, you need to test smaller `max-num-batched-token < 8192`. Or it is recommended to test 32-64k for low ITL or 64k+ for max throughput. But this can be vary depending on the model size and context length. Fill free to change and test your own.

I tries to put most of vLLM controls with this file while keeping the script argument is simple. Because this file can work as arguments set. The benchmark result will record this file and you can seperate the result among the all the mixed test results accordingly.

### Test Plan File
This benchmark script follow test plan in `configs/benchmark_plans/`. This plan file is custom format file that user can put comments with `#` prefix.

The benchmark script loads this file via `--test-plan` by identifying file name. Then, it parses along this order: request_rate, client_count, num_iteration, input_length, and output_length.

By specifying the benchmark plan, you can obtain multiple test results easily.

Following snippet show an example of the plan file.
```yaml
test_scenarios:
  - name: "1k input-1k output"
    description: "Balanced test with 1k input and 1k output tokens"
    request_rate: 0  # inf
    concurrency: [1, 2, 4, 8, 16, 32, 48, 64, 80, 96, 112, 128, 192, 256]
    input_length: 1024
    output_length: 1024
    num_iteration: 8
    batch_size: 256
```

`batch_size` means vllm engine's batch size and it is `max-num-seqs` in vllm.

There are several test cases in `configs/benchmark_plans`.
 - test
 - sample
 - decode_heavy
 - prefill_heavy
 - hybrid
 - throughput_control

## Benchmark Result

### Result Logs
All the benchmark results are stored in `logs/` directory following the model name and docker images tags. For instance, single benchmark's results are stored in `/logs/<model name>/<docker-tag>` directory.


## Other Features
### Manual test
To ease various testing, this script provides `--dry-run` mode. With this option, benchmark script prints out the command which will be used in the benchmark. You can copy the output and start own test.

Please mind that having full support vLLM command is not this script's objective.


### Profile (TBU)
For ease of analysis performance, this project provides `--profile` argument. Then it exports vllm server profile traces into `./profile` directory.

This repo also has a copy of vllm profile tool - `layerwise profiling`.

```
python examples/offline_inference/profiling.py \\
    --model neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8 --batch-size 4 \\
    --prompt-len 512 --max-num-batched-tokens 8196 --json Llama31-8b-FP8 \\
    --enforce-eager run_num_steps -n 2
```

then you can use various tools to analyze the json output terminal ascii tables:
```
python tools/profiler/print_layerwise_table.py \\
    --json-trace Llama31-8b-FP8.json --phase prefill --table summary
```
or create matplotlib stacked bar charts:
```
python tools/profiler/visualize_layerwise_profile.py \\
    --json-trace Llama31-8b-FP8.json \\
    --output-directory profile_breakdown --plot-metric pct_cuda_time
```

# TODO
1. Having test inferenceMax options:
    - https://github.com/InferenceMAX/InferenceMAX/blob/main/benchmarks/70b_fp4_mi355x_docker.sh
    - https://github.com/InferenceMAX/InferenceMAX/blob/main/benchmarks/70b_fp8_mi355x_docker.sh
1. Benchmark with other parallelism
1. Writing graph drawing code
1. Benchmark with PD disaggregation
