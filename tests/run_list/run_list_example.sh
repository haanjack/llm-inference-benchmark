# run_list format:
# backend docker_image model_path_or_id model_config test_plan benchmark_client gpu_devices sub_task(optional) async(optional)

################################################
# llama 70B
################################################
vllm docker.io/rocm/vllm:rocm7.0.0_vllm_0.11.2_20251210 amd/Llama-3.1-70B-Instruct-FP8-KV configs/models/llama-vllm.yaml sample vllm 0
vllm docker.io/rocm/vllm:rocm7.0.0_vllm_0.11.2_20251210 amd/Llama-3.1-70B-Instruct-FP8-KV configs/models/llama-vllm.yaml sample vllm 0,1
vllm docker.io/rocm/vllm:rocm7.0.0_vllm_0.11.2_20251210 amd/Llama-3.1-70B-Instruct-FP8-KV configs/models/llama-vllm.yaml sample vllm 0,1,2,3
vllm docker.io/rocm/vllm:rocm7.0.0_vllm_0.11.2_20251210 amd/Llama-3.1-70B-Instruct-FP8-KV configs/models/llama-vllm.yaml sample vllm 0,1,2,3,4,5,6,7

################################################
# llama 70B Async
################################################
vllm docker.io/rocm/vllm:rocm7.0.0_vllm_0.11.2_20251210 amd/Llama-3.1-70B-Instruct-FP8-KV configs/models/llama-vllm.yaml sample vllm 0 all async
vllm docker.io/rocm/vllm:rocm7.0.0_vllm_0.11.2_20251210 amd/Llama-3.1-70B-Instruct-FP8-KV configs/models/llama-vllm.yaml sample vllm 1,2 all async
vllm docker.io/rocm/vllm:rocm7.0.0_vllm_0.11.2_20251210 amd/Llama-3.1-70B-Instruct-FP8-KV configs/models/llama-vllm.yaml sample vllm 3,4,5,6 all async
vllm docker.io/rocm/vllm:rocm7.0.0_vllm_0.11.2_20251210 amd/Llama-3.1-70B-Instruct-FP8-KV configs/models/llama-vllm.yaml sample vllm 0,1,2,3,4,5,6,7