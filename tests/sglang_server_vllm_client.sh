#!/bin/bash

python3 main.py \
    --model-config configs/models/default-sglang.yaml \
    --model-path-or-id amd/Llama-3.1-8B-Instruct-FP8-KV \
    --backend sglang \
    --image docker.io/rocm/sgl-dev:v0.5.5.post2-rocm700-mi35x-20251114 \
    --benchmark-client vllm \
    --test-plan sample \
    --gpu-devices 0

