Benchmark guide

bash benchmark.sh <env> <model> <docker> <test-opt> <gpu-ids>
```
bash benchmark.sh attention_v1_aiter_on Qwen/Qwen3-32B-FP8 docker.io/rocm/vllm-dev:nightly_main_20250903 test 0
```

```
PROFILE=1 bash benchmark.sh ...
```
