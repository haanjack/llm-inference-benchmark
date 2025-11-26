#!/bin/bash

server_backend=${1:-"vllm"}
benchmark_client=${2:-"vllm"}
dry_run=${3:-"false"}

# check inputs are available options
server_backends=("sglang" "vllm")
benchmark_clients=("vllm" "genai-perf")

if [[ ! " ${server_backends[@]} " =~ " ${server_backend} " ]]; then
    echo "Invalid server backend: ${server_backend}"
    echo "Available options are: ${server_backends[@]}"
    exit 1
fi
if [[ ! " ${benchmark_clients[@]} " =~ " ${benchmark_client} " ]]; then
    echo "Invalid benchmark client: ${benchmark_client}"
    echo "Available options are: ${benchmark_clients[@]}"
    exit 1
fi

if [ "${dry_run}" == "true" ]; then
    dry_run="true"
else
    dry_run=""
fi

# set variables based on inputs
if [ "${server_backend}" == "sglang" ]; then
    image="docker.io/rocm/sgl-dev:v0.5.5.post3-rocm700-mi30x-20251123"
    model_config="configs/models/default-sglang.yaml"
    backend="sglang"
else
    image="docker.io/rocm/vllm:rocm7.0.0_vllm_0.11.1_20251103"
    model_config="configs/models/llama-vllm.yaml"
    backend="vllm"
fi

if [ "${benchmark_client}" == "genai-perf" ]; then
    benchmark_client="genai-perf"
else
    benchmark_client="vllm"
fi

python3 main.py \
    --model-config $model_config \
    --model-path-or-id amd/Llama-3.1-8B-Instruct-FP8-KV \
    --backend $backend \
    --image $image \
    --benchmark-client $benchmark_client \
    --test-plan test \
    --gpu-devices 0 \
    ${dry_run:+--dry-run}

