#!/bin/bash

python3 main.py \
    --model-config configs/models/gpt-oss.yaml \
    --model-path-or-id openai/gpt-oss-120b \
    --backend vllm \
    --vllm-image docker.io/rocm/vllm:rocm7.0.0_vllm_0.11.1_20251103 \
    --test-plan sample \
    --gpu-devices 0

