#!/bin/bash

run_mode="generate_script" # Options: "" | "profile" | "dry_run" | "generate_script"
test_plan="sample"
gpu_devices="0"
sub_task="1k1k"

server_backends=("vllm" "sglang")
benchmark_clients=("vllm" "sglang" "genai-perf")

image_vllm="docker.io/rocm/vllm:rocm7.0.0_vllm_0.11.1_20251103"
image_sgl="docker.io/rocm/sgl-dev:v0.5.5.post3-rocm700-mi30x-20251123"

declare -A model_and_configs
model_and_configs["llama"]="configs/models/llama- amd/Llama-3.1-8B-Instruct-FP8-KV"
model_and_configs["deepseek"]="configs/models/deepseek- deepseek-ai/DeepSeek-R1"
model_and_configs["qwen"]="configs/models/qwen- Qwen/Qwen3-32B"
model_and_configs["qwen-moe"]="configs/models/qwen-moe- Qwen/Qwen3-235B-A22B"
model_and_configs["gpt-oss"]="configs/models/gpt-oss- openai/gpt-oss-120b"

# rm -f scripts/generated/*.sh

for key in "${!model_and_configs[@]}"; do
    for server_backend in "${server_backends[@]}"; do
        for benchmark_client in "${benchmark_clients[@]}"; do
            IFS=' ' read -r -a values <<< "${model_and_configs[$key]}"
            model_config="${values[0]}"
            model_path_or_id="${values[1]}"

            if [[ "$server_backend" == "vllm" ]]; then
                image=${image_vllm}
                model_config="${model_config}vllm.yaml"
            else
                image=${image_sgl}
                model_config="${model_config}sglang.yaml"
            fi

            if [ ! -f "$model_config" ]; then
                echo "Model config file not found: '${model_config}', skipping..."
                continue
            fi

            echo bash tests/run_test.sh ${run_mode} ${model_config} ${model_path_or_id} ${server_backend} ${image} ${benchmark_client} ${test_plan} ${gpu_devices} ${sub_task}

        done
    done
done