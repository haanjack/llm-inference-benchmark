#!/bin/bash

env_file=${1:-"baseline"}
model_dir=${2:-"dummy"}
docker_image=${3:-"docker.io/rocm/vllm:latest"}
mode=${4:-"test"}
gpu_ids=${5:-"0"}

tp_size=$TP_SIZE

if [[ ! -f "${env_file}" ]]; then
    echo "Could not find env file. Please check env file name"
    ls "envs"
    exit 1
fi

echo "model_dir: ${model_dir}"
if [[ ! -d "${HOME}/$model_dir" ]]; then
    echo "Could not find model dir"
    exit 1
fi

if [[ $tp_size == "0" ]]; then
    echo "Please specify Tensor Parallel Size"
    exit 1
fi

if [[ "${mode}" == "dryrun" ]]; then
    python3 run_vllm_benchmark.py --env-file ${env_file} --model-path ${model_dir} --vllm-image ${docker_image} --bench-scope ${mode} --gpu-devices ${gpu_ids} --dry-run
    exit
fi

if [[ "${mode}" == "test" ]]; then
    # TP_SIZE=$tp_size bash run_vllm_benchmark.sh ${env_file} ${model_dir} ${docker_image} ${mode} ${gpu_ids}
    python3 run_vllm_benchmark.py --env-file ${env_file} --model-path ${model_dir} --vllm-image ${docker_image} --bench-scope ${mode} --gpu-devices ${gpu_ids}
elif [[ "${mode}" == "custom" ]]; then
    python3 run_vllm_benchmark.py --env-file ${env_file} --model-path ${model_dir} --vllm-image ${docker_image} --bench-scope ${mode} --gpu-devices ${gpu_ids} --custom-scope-file=configs/customs/custom_combination.txt
else
    # TP_SIZE=$tp_size bash run_vllm_benchmark.sh ${env_file} ${model_dir} ${docker_image} prefill ${gpu_ids}
    # TP_SIZE=$tp_size bash run_vllm_benchmark.sh ${env_file} ${model_dir} ${docker_image} decode ${gpu_ids}
    # TP_SIZE=$tp_size bash run_vllm_benchmark.sh ${env_file} ${model_dir} ${docker_image} middle ${gpu_ids}
    python3 run_vllm_benchmark.py --env-file ${env_file} --model-path ${model_dir} --vllm-image ${docker_image} --bench-scope prefill --gpu-devices ${gpu_ids}
    python3 run_vllm_benchmark.py --env-file ${env_file} --model-path ${model_dir} --vllm-image ${docker_image} --bench-scope decode --gpu-devices ${gpu_ids}
    python3 run_vllm_benchmark.py --env-file ${env_file} --model-path ${model_dir} --vllm-image ${docker_image} --bench-scope middle --gpu-devices ${gpu_ids}
fi

