# run_list format:
# backend model_path_or_id model_config test_plan benchmark_client gpu_devices sub_task(optional)
vllm docker.io/rocm/vllm:rocm7.0.0_vllm_0.11.2_20251210 amd/Llama-3.1-8B-Instruct-FP8-KV configs/models/llama-vllm.yaml sample vllm 0

