#!/bin/bash

run_mode="profile" # Options: "" | "profile" | "dry_run" | "generate_script"
test_plan="test"
gpu_devices="0"

server_backends=("vllm" "sglang")
benchmark_clients=("vllm" "sglang" "genai-perf")

for server_backend in "${server_backends[@]}"; do
    for benchmark_client in "${benchmark_clients[@]}"; do
        bash tests/run_test.sh ${run_mode} ${server_backend}   ${benchmark_client} ${test_plan} ${gpu_devices}
    done
done