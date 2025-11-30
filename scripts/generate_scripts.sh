#!/bin/bash

# Args
model_config=${1:-"configs/models/llama-vllm.yaml"}
model_path_or_id=${2:-"amd/Llama-3.1-8B-Instruct-FP8-KV"}
server_backend=${3:-"vllm"}
benchmark_client=${4:-"vllm"}
test_plan=${5:-"sample"}
gpu_devices=${6:-"0"}

image="docker.io/rocm/vllm:rocm7.0.0_vllm_0.11.1_20251103"

# check inputs are available options
run_modes=("" "profile" "dry_run" "generate_script")
server_backends=("sglang" "vllm")
benchmark_clients=("vllm" "sglang" "genai-perf")

# Validate argument values
if [[ ! " ${run_modes[@]} " =~ " ${run_mode} " ]]; then
    echo "Invalid run mode: '${run_mode}'"
    echo "Available options are: ${run_modes[@]}"
    echo "Usage: $0 <model_config> <server_backend> <benchmark_client> [test_plan]"
    exit 1
fi

if [[ ! " ${server_backends[@]} " =~ " ${server_backend} " ]]; then
    echo "Invalid server backend: '${server_backend}'"
    echo "Available options are: ${server_backends[@]}"
    echo "Usage: $0 <model_config> <server_backend> <benchmark_client> [test_plan]"
    exit 1
fi
if [[ ! " ${benchmark_clients[@]} " =~ " ${benchmark_client} " ]]; then
    echo "Invalid benchmark client: '${benchmark_client}'"
    echo "Available options are: ${benchmark_clients[@]}"
    echo "Usage: $0 <model_config> <server_backend> <benchmark_client> [test_plan]"
    exit 1
fi

python3 main.py \
    --model-config $model_config \
    --model-path-or-id ${model_path_or_id} \
    --backend $server_backend \
    --image $image \
    --benchmark-client $benchmark_client \
    --test-plan ${test_plan} \
    --gpu-devices ${gpu_devices} \
    --generate-script

echo "Script generation complete!"
