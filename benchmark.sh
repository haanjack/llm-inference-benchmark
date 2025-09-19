#!/bin/bash

env_file=${1:-"baseline"}
model_dir=${2:-"dummy"}
docker_image=${3:-"docker.io/rocm/vllm:latest"}
mode=${4:-"test"}
gpu_ids=${5:-"0"}

tp_size=$TP_SIZE

if [[ ! -f "envs/${env_file}" ]]; then
    echo "Could not find env file. Please check env file name"
    ls "envs"
    exit 1
fi

echo "model_dir: ${model_dir}"
if [[ ! -d "${HOME}/models/$model_dir" ]]; then
    echo "Could not find model dir"
    exit 1
fi

if [[ $tp_size == "0" ]]; then
    echo "Please specify Tensor Parallel Size"
    exit 1
fi

if [[ "${mode}" != "all" ]]; then
    TP_SIZE=$tp_size bash run_vllm_benchmark.sh ${env_file} ${model_dir} ${docker_image} ${mode} ${gpu_ids}
else
    TP_SIZE=$tp_size bash run_vllm_benchmark.sh ${env_file} ${model_dir} ${docker_image} prefill ${gpu_ids}
    TP_SIZE=$tp_size bash run_vllm_benchmark.sh ${env_file} ${model_dir} ${docker_image} decode ${gpu_ids}
    TP_SIZE=$tp_size bash run_vllm_benchmark.sh ${env_file} ${model_dir} ${docker_image} middle ${gpu_ids}
fi

