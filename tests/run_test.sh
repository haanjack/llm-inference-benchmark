#!/bin/bash

# Args
run_mode=${1:-""}
server_backend=${2:-"vllm"}
benchmark_client=${3:-"vllm"}
test_plan=${4:-"test"}
gpu_devices=${5:-"0"}

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

run_mode_opt=""
if [ "${run_mode}" == "profile" ]; then
    run_mode_opt=""
elif [ "${run_mode}" == "dry_run" ]; then
    run_mode_opt="--dry-run"
elif [ "${run_mode}" == "generate_script" ]; then
    run_mode_opt="--generate-script"
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

python3 main.py \
    --model-config $model_config \
    --model-path-or-id amd/Llama-3.1-8B-Instruct-FP8-KV \
    --backend $backend \
    --image $image \
    --benchmark-client $benchmark_client \
    --test-plan ${test_plan} \
    --gpu-devices ${gpu_devices} \
    ${run_mode_opt}

if [ "${run_mode}" == "generate_script" ]; then
    # Find all generated scripts matching the pattern and prettify them (overwrite in place)
    echo "Prettifying generated scripts..."
    for script in scripts/generated/run-*-${test_plan}*.sh; do
        if [ -f "$script" ]; then
            echo "Prettifying (in-place): $script"
            python3 scripts/generated_script_prettifier.py -i "$script" -o "$script"
        fi
    done
    echo "All scripts prettified."
fi
