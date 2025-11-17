#!/bin/bash

python3 main.py \
    --model-config configs/models/llama.yaml \
    --model-path-or-id amd/Llama-3.1-8B-Instruct-FP8-KV \
    --backend sglang \
    --sglang-image sglang/sglang-llm:latest \
    --test-plan sample \
    --gpu-devices 0

