# run_list format:
# backend model_path_or_id model_config test_plan benchmark_client gpu_devices sub_task(optional)
vllm amd/Llama-3.1-8B-Instruct-FP8-KV configs/models/llama-vllm.yaml sample vllm 0

