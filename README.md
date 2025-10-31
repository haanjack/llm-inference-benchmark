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

There are several interesting argument: `env-file` and `test-plan`. These are the purpose of this benchmark script. "What is best setting?" and "I just want to roll-out the tests". Check each section for more details.

### Environment File
The vLLM operation can be controled with environment variables. This benchmark script provides to custom the environment variables when it is executed.

For more available environment variables, please checke the following documents.
- [vLLM environment variables](https://docs.vllm.ai/en/stable/configuration/env_vars.html)
- [AITER switches](https://rocm.docs.amd.com/en/develop/how-to/rocm-for-ai/inference-optimization/vllm-optimization.html#aiter-ai-tensor-engine-for-rocm-switches)


This benchmark script provides some predefined sets for model architecture type and precision. Everyting is experimental, so fill free to test and change its settings.

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
```

There are several test cases are pre-written in:
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


### Profile
TBU

# TODO
1. Having test inferenceMax options:
    - https://github.com/InferenceMAX/InferenceMAX/blob/main/benchmarks/70b_fp4_mi355x_docker.sh
    - https://github.com/InferenceMAX/InferenceMAX/blob/main/benchmarks/70b_fp8_mi355x_docker.sh
1. Benchmark with other parallelism
1. 
1. Writing graph drawing code
1. Benchmark with PD disaggregation
