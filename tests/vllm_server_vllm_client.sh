#!/bin/bash

python3 main.py \
    --model-config configs/models/llama-vllm.yaml \
    --model-path-or-id amd/Llama-3.1-8B-Instruct-FP8-KV \
    --backend vllm \
    --image docker.io/rocm/vllm:latest \
    --benchmark-client vllm \
    --test-plan sample \
    --gpu-devices 0

