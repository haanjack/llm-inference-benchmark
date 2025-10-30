# Benchmark guide

This is benchmark script for extensive inference tests for various setups. This benchmark script help users to test with the following variants.
 - various environment variable sets
 - input/output sequence length
 - num concurrency
 - iteration based num prompts control
 - request rate control

## Basic usage of test
```bash
python run_vllm_benchmark.py \
    --env-file configs/envs/mi300x/vllm/baseline \
    --model-path models/Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 \
    --vllm-image docker.io/rocm/vllm:rocm7.0.0_vllm_0.10.2_20251006 \
    --test-plan test \
    --gpu-devices 0
```

## Test Plan File
This benchmark script follow test plan in `configs/plans/`. This plan file is custom format file that user can put comments with `#` prefix.

The benchmark script loads this file via `--test-plan` by identifying file name. Then, it parses along this order: request_rate, client_count, num_iteration, input_length, and output_length.

Following snippet show an example of this.
```
# default operation test plan
# request_rate - use 0 for inf
#
# test parameters format:
# request_rate client_count num_iteration input_length output_length

0 4 4 256 256
```

There are several test cases are pre-written in:
 - test
 - decode_heavy
 - prefill_heavy
 - hybrid
 - throughput_control



# TODO
1. Writing graph drawing code
1. Having test inferenceMax options:
    - https://github.com/InferenceMAX/InferenceMAX/blob/main/benchmarks/70b_fp4_mi355x_docker.sh
    - https://github.com/InferenceMAX/InferenceMAX/blob/main/benchmarks/70b_fp8_mi355x_docker.sh
1. Benchmark with DP enablement
1. Benchmark with PD disaggregation
