#!/bin/bash

python3 main.py \
    --model-config configs/models/gpt-oss.yaml \
    --model-path-or-id openai/gpt-oss-120b \
    --backend sglang \
    --sglang-image sglang/sglang-llm:latest \
    --test-plan sample \
    --gpu-devices 0

