#!/bin/bash

python3 main.py \
    --model-config configs/models/llama.yaml \
    --model-path-or-id amd/Llama-3.1-8B-Instruct-FP8-KV \
    --backend vllm \
    --vllm-image docker.io/rocm/vllm:latest \
    --benchmark-client genai-perf \
    --test-plan sample \
    --gpu-devices 0

