#!/bin/bash

# Args
run_mode=${1:-""}
model_config=${2:-"configs/models/llama-vllm.yaml"}
model_path_or_id=${3:-"amd/Llama-3.1-8B-Instruct-FP8-KV"}
server_backend=${4:-"vllm"}
docker_image=${5:-"docker.io/rocm/vllm:rocm7.0.0_vllm_0.11.1_20251103"}
benchmark_client=${6:-"vllm"}
test_plan=${7:-"test"}
gpu_devices=${8:-"0"}
sub_task=${9:-""}

# check inputs are available options
run_modes=("" "profile" "dry_run" "generate_script")
server_backends=("sglang" "vllm")
benchmark_clients=("vllm" "sglang" "genai-perf")

# Validate argument values
if [[ ! " ${run_modes[@]} " =~ " ${run_mode} " ]]; then
    echo "Invalid run mode: '${run_mode}'"
    echo "Available options are: ${run_modes[@]}"
    echo "Usage: $0 <run_mode> <server_backend> <benchmark_client> [test_plan]"
    exit 1
fi

if [[ ! " ${server_backends[@]} " =~ " ${server_backend} " ]]; then
    echo "Invalid server backend: '${server_backend}'"
    echo "Available options are: ${server_backends[@]}"
    echo "Usage: $0 <run_mode> <server_backend> <benchmark_client> [test_plan]"
    exit 1
fi
if [[ ! " ${benchmark_clients[@]} " =~ " ${benchmark_client} " ]]; then
    echo "Invalid benchmark client: '${benchmark_client}'"
    echo "Available options are: ${benchmark_clients[@]}"
    echo "Usage: $0 <run_mode> <server_backend> <benchmark_client> [test_plan]"
    exit 1
fi

# check if model_config is exists
if [ ! -f "$model_config" ]; then
    echo "Model config file not found: '${model_config}'"
    exit 1
fi

extra_opt=""
if [ "${run_mode}" == "profile" ]; then
    extra_opt=""
elif [ "${run_mode}" == "dry_run" ]; then
    extra_opt="--dry-run"
elif [ "${run_mode}" == "generate_script" ]; then
    extra_opt="--generate-script"
fi

if [ -n "${sub_task}" ]; then
    extra_opt+=" --sub-task ${sub_task}"
fi

python3 main.py \
    --model-config $model_config \
    --model-path-or-id $model_path_or_id \
    --backend $server_backend \
    --image $docker_image \
    --benchmark-client $benchmark_client \
    --test-plan ${test_plan} \
    --gpu-devices ${gpu_devices} \
    ${extra_opt}
