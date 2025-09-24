# Benchmark guide

This is simple scripts that helps extensive inference tests for various setups.

bash benchmark.sh <env> <model> <docker> <test-opt> <gpu-ids>
```
bash benchmark.sh attention_v1_aiter_on Qwen/Qwen3-32B-FP8 docker.io/rocm/vllm-dev:nightly_main_20250903 test 0
```

For tensor parallel,
```
# TP2
bash benchmark.sh attention_v1_aiter_on Qwen/Qwen3-32B-FP8 docker.io/rocm/vllm-dev:nightly_main_20250903 test 0,1
```

# TODO
1. Benchmark with DP enablement
2. Benchmark with PD disaggregation
