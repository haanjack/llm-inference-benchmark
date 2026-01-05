# run_list format:
# backend docker_image model_path_or_id model_config test_plan benchmark_client gpu_devices sub_task(optional) async(optional)

################################################
# llama 70B
################################################
vllm docker.io/rocm/vllm:rocm7.0.0_vllm_0.11.2_20251210                         amd/Llama-3.1-70B-Instruct-FP8-KV configs/models/llama-vllm.yaml sample vllm 0
vllm docker.io/rocm/vllm:rocm7.0.0_vllm_0.11.2_20251210                         amd/Llama-3.1-70B-Instruct-FP8-KV configs/models/llama-vllm.yaml sample vllm 1,2
vllm docker.io/rocm/vllm:rocm7.0.0_vllm_0.11.2_20251210                         amd/Llama-3.1-70B-Instruct-FP8-KV configs/models/llama-vllm.yaml sample vllm 3,4,5,6
vllm docker.io/rocm/vllm:rocm7.0.0_vllm_0.11.2_20251210                         amd/Llama-3.1-70B-Instruct-FP8-KV configs/models/llama-vllm.yaml sample vllm 0,1,2,3,4,5,6,7

## silo ai
vllm docker.io/amdsiloai/vllm:rocm7.2_preview_ubuntu_22.04_vllm_0.10.1_instinct_20251120  amd/Llama-3.1-70B-Instruct-FP8-KV configs/models/llama-vllm-silo.yaml sample vllm 0
vllm docker.io/amdsiloai/vllm:rocm7.2_preview_ubuntu_22.04_vllm_0.10.1_instinct_20251120  amd/Llama-3.1-70B-Instruct-FP8-KV configs/models/llama-vllm-silo.yaml sample vllm 1,2
vllm docker.io/amdsiloai/vllm:rocm7.2_preview_ubuntu_22.04_vllm_0.10.1_instinct_20251120  amd/Llama-3.1-70B-Instruct-FP8-KV configs/models/llama-vllm-silo.yaml sample vllm 0,1,2,3
vllm docker.io/amdsiloai/vllm:rocm7.2_preview_ubuntu_22.04_vllm_0.10.1_instinct_20251120  amd/Llama-3.1-70B-Instruct-FP8-KV configs/models/llama-vllm-silo.yaml sample vllm 0,1,2,3,4,5,6,7


################################################
# llama 405B
################################################
vllm docker.io/rocm/vllm:rocm7.0.0_vllm_0.11.2_20251210                         amd/Llama-3.1-405B-Instruct-FP8-KV configs/models/llama-vllm.yaml sample vllm 0
vllm docker.io/rocm/vllm:rocm7.0.0_vllm_0.11.2_20251210                         amd/Llama-3.1-405B-Instruct-FP8-KV configs/models/llama-vllm.yaml sample vllm 1,2
vllm docker.io/rocm/vllm:rocm7.0.0_vllm_0.11.2_20251210                         amd/Llama-3.1-405B-Instruct-FP8-KV configs/models/llama-vllm.yaml sample vllm 3,4,5,6
vllm docker.io/rocm/vllm:rocm7.0.0_vllm_0.11.2_20251210                         amd/Llama-3.1-405B-Instruct-FP8-KV configs/models/llama-vllm.yaml sample vllm 0,1,2,3,4,5,6,7

## silo ai
vllm docker.io/amdsiloai/vllm:rocm7.2_preview_ubuntu_22.04_vllm_0.10.1_instinct_20251120  amd/Llama-3.1-405B-Instruct-FP8-KV configs/models/llama-vllm-silo.yaml sample vllm 0
vllm docker.io/amdsiloai/vllm:rocm7.2_preview_ubuntu_22.04_vllm_0.10.1_instinct_20251120  amd/Llama-3.1-405B-Instruct-FP8-KV configs/models/llama-vllm-silo.yaml sample vllm 1,2
vllm docker.io/amdsiloai/vllm:rocm7.2_preview_ubuntu_22.04_vllm_0.10.1_instinct_20251120  amd/Llama-3.1-405B-Instruct-FP8-KV configs/models/llama-vllm-silo.yaml sample vllm 0,1,2,3
vllm docker.io/amdsiloai/vllm:rocm7.2_preview_ubuntu_22.04_vllm_0.10.1_instinct_20251120  amd/Llama-3.1-405B-Instruct-FP8-KV configs/models/llama-vllm-silo.yaml sample vllm 0,1,2,3,4,5,6,7

################################################
# CodeLlama-34b-hf
################################################
vllm docker.io/rocm/vllm:rocm7.0.0_vllm_0.11.2_20251210                         codellama/CodeLlama-34b-hf configs/models/llama-vllm.yaml sample vllm 0
vllm docker.io/rocm/vllm:rocm7.0.0_vllm_0.11.2_20251210                         codellama/CodeLlama-34b-hf configs/models/llama-vllm.yaml sample vllm 1,2
vllm docker.io/rocm/vllm:rocm7.0.0_vllm_0.11.2_20251210                         codellama/CodeLlama-34b-hf configs/models/llama-vllm.yaml sample vllm 3,4,5,6
vllm docker.io/rocm/vllm:rocm7.0.0_vllm_0.11.2_20251210                         codellama/CodeLlama-34b-hf configs/models/llama-vllm.yaml sample vllm 0,1,2,3,4,5,6,7

## silo ai
vllm docker.io/amdsiloai/vllm:rocm7.2_preview_ubuntu_22.04_vllm_0.10.1_instinct_20251120  codellama/CodeLlama-34b-hf configs/models/llama-vllm-silo.yaml sample vllm 0
vllm docker.io/amdsiloai/vllm:rocm7.2_preview_ubuntu_22.04_vllm_0.10.1_instinct_20251120  codellama/CodeLlama-34b-hf configs/models/llama-vllm-silo.yaml sample vllm 1,2
vllm docker.io/amdsiloai/vllm:rocm7.2_preview_ubuntu_22.04_vllm_0.10.1_instinct_20251120  codellama/CodeLlama-34b-hf configs/models/llama-vllm-silo.yaml sample vllm 3,4,5,6
vllm docker.io/amdsiloai/vllm:rocm7.2_preview_ubuntu_22.04_vllm_0.10.1_instinct_20251120  codellama/CodeLlama-34b-hf configs/models/llama-vllm-silo.yaml sample vllm 0,1,2,3,4,5,6,7


################################################
# gpt-oss-120b 70B
################################################
vllm docker.io/rocm/vllm:rocm7.0.0_vllm_0.11.1_20251103             openai/gpt-oss-120b configs/models/gpt-oss-vllm-fixed.yaml sample vllm 0
vllm docker.io/rocm/vllm:rocm7.0.0_vllm_0.11.1_20251103             openai/gpt-oss-120b configs/models/gpt-oss-vllm-fixed.yaml sample vllm 1,2
vllm docker.io/rocm/vllm:rocm7.0.0_vllm_0.11.1_20251103             openai/gpt-oss-120b configs/models/gpt-oss-vllm-fixed.yaml sample vllm 3,4,5,6
vllm docker.io/rocm/vllm:rocm7.0.0_vllm_0.11.1_20251103             openai/gpt-oss-120b configs/models/gpt-oss-vllm-fixed.yaml sample vllm 0,1,2,3,4,5,6,7

vllm docker.io/rocm/vllm:rocm7.0.0_vllm_0.11.2_20251210             openai/gpt-oss-120b configs/models/gpt-oss-vllm-0.11.2.yaml sample vllm 0
vllm docker.io/rocm/vllm:rocm7.0.0_vllm_0.11.2_20251210             openai/gpt-oss-120b configs/models/gpt-oss-vllm-0.11.2.yaml sample vllm 1,2
vllm docker.io/rocm/vllm:rocm7.0.0_vllm_0.11.2_20251210             openai/gpt-oss-120b configs/models/gpt-oss-vllm-0.11.2.yaml sample vllm 3,4,5,6
vllm docker.io/rocm/vllm:rocm7.0.0_vllm_0.11.2_20251210             openai/gpt-oss-120b configs/models/gpt-oss-vllm-0.11.2.yaml sample vllm 0,1,2,3,4,5,6,7

## silo ai
vllm docker.io/docker.io/amdsiloai/vllm:rocm7.2_preview_ubuntu_22.04_vllm_0.10.1_instinct_20251120 openai/gpt-oss-120b configs/models/gpt-oss-vllm-silo.yaml sample vllm     0
vllm docker.io/docker.io/amdsiloai/vllm:rocm7.2_preview_ubuntu_22.04_vllm_0.10.1_instinct_20251120 openai/gpt-oss-120b configs/models/gpt-oss-vllm-silo.yaml sample vllm     1,2
vllm docker.io/docker.io/amdsiloai/vllm:rocm7.2_preview_ubuntu_22.04_vllm_0.10.1_instinct_20251120 openai/gpt-oss-120b configs/models/gpt-oss-vllm-silo.yaml sample vllm     3,4,5,6
vllm docker.io/docker.io/amdsiloai/vllm:rocm7.2_preview_ubuntu_22.04_vllm_0.10.1_instinct_20251120 openai/gpt-oss-120b configs/models/gpt-oss-vllm-silo.yaml sample vllm     0,1,2,3,4,5,6,7

vllm docker.io/rocm/vllm:rocm7.0.0_vllm_0.11.2_20251210             openai/gpt-oss-120b configs/models/gpt-oss-vllm-2.yaml sample vllm     0
vllm docker.io/rocm/vllm:rocm7.0.0_vllm_0.11.2_20251210             openai/gpt-oss-120b configs/models/gpt-oss-vllm-2.yaml sample vllm     1,2
vllm docker.io/rocm/vllm:rocm7.0.0_vllm_0.11.2_20251210             openai/gpt-oss-120b configs/models/gpt-oss-vllm-2.yaml sample vllm     3,4,5,6
vllm docker.io/rocm/vllm:rocm7.0.0_vllm_0.11.2_20251210             openai/gpt-oss-120b configs/models/gpt-oss-vllm-2.yaml sample vllm     0,1,2,3,4,5,6,7

vllm docker.io/rocm/7.0:rocm7.0_ubuntu_22.04_vllm_0.10.1_instinct_20250927_rc1 openai/gpt-oss-120b configs/models/gpt-oss-vllm-silo.yaml sample vllm     0
vllm docker.io/rocm/7.0:rocm7.0_ubuntu_22.04_vllm_0.10.1_instinct_20250927_rc1 openai/gpt-oss-120b configs/models/gpt-oss-vllm-silo.yaml sample vllm     1,2
vllm docker.io/rocm/7.0:rocm7.0_ubuntu_22.04_vllm_0.10.1_instinct_20250927_rc1 openai/gpt-oss-120b configs/models/gpt-oss-vllm-silo.yaml sample vllm     3,4,5,6
vllm docker.io/rocm/7.0:rocm7.0_ubuntu_22.04_vllm_0.10.1_instinct_20250927_rc1 openai/gpt-oss-120b configs/models/gpt-oss-vllm-silo.yaml sample vllm     0,1,2,3,4,5,6,7

vllm docker.io/rocm/vllm-dev:nightly_main_20251212                  openai/gpt-oss-120b configs/models/gpt-oss-vllm-2.yaml sample vllm     0
vllm docker.io/rocm/vllm-dev:nightly_main_20251212                  openai/gpt-oss-120b configs/models/gpt-oss-vllm-2.yaml sample vllm     1,2
vllm docker.io/rocm/vllm-dev:nightly_main_20251212                  openai/gpt-oss-120b configs/models/gpt-oss-vllm-2.yaml sample vllm     3,4,5,6
vllm docker.io/rocm/vllm-dev:nightly_main_20251212                  openai/gpt-oss-120b configs/models/gpt-oss-vllm-2.yaml sample vllm     0,1,2,3,4,5,6,7


################################################
# deepseek-R1
################################################
vllm   rocm/7.x-preview:rocm7.2_preview_ubuntu_22.04_vlm_0.10.1_instinct_20251029   deepseek-ai/DeepSeek-R1-0528        configs/models/deepseek-vllm1.yaml          sample vllm 0,1,2,3,4,5,6,7 # silo
vllm   docker.io/rocm/vllm:rocm7.0.0_vllm_0.11.2_20251210                           deepseek-ai/DeepSeek-R1-0528        configs/models/deepseek-vllm1.yaml          sample vllm 0,1,2,3,4,5,6,7 # new
vllm   docker.io/rocm/vllm:rocm7.0.0_vllm_0.11.1_20251103                           deepseek-ai/DeepSeek-R1-0528        configs/models/deepseek-vllm1.yaml          sample vllm 0,1,2,3,4,5,6,7 # old

sglang docker.io/rocm/sgl-dev:v0.5.6.post1-rocm700-mi35x-20251211   deepseek-ai/DeepSeek-R1-0528        configs/models/deepseek-sglang1.yaml        sample vllm 0,1,2,3,4,5,6,7
sglang docker.io/lmsysorg/sglang:v0.5.6.post2-rocm700-mi35x         deepseek-ai/DeepSeek-R1-0528        configs/models/deepseek-sglang2.yaml        sample vllm 0,1,2,3,4,5,6,7

sglang docker.io/rocm/sgl-dev:v0.5.6.post1-rocm700-mi35x-20251211   amd/DeepSeek-R1-0528-MXFP4-Preview  configs/models/deepseek-sglang1-fp4.yaml    sample vllm 0,1,2,3
sglang docker.io/lmsysorg/sglang:v0.5.6.post2-rocm700-mi35x         amd/DeepSeek-R1-0528-MXFP4-Preview  configs/models/deepseek-sglang2-fp4.yaml    sample vllm 4,5,6,7
sglang docker.io/rocm/sgl-dev:v0.5.6.post1-rocm700-mi35x-20251211   amd/DeepSeek-R1-0528-MXFP4-Preview  configs/models/deepseek-sglang1-fp4.yaml    sample vllm 0,1,2,3,4,5,6,7

sglang docker.io/rocm/sgl-dev:v0.5.6.post1-rocm700-mi35x-20251211   deepseek-ai/DeepSeek-R1-0528        configs/models/deepseek-sglang1.yaml        sample vllm 0,1,2,3
sglang docker.io/lmsysorg/sglang:v0.5.6.post2-rocm700-mi35x         deepseek-ai/DeepSeek-R1-0528        configs/models/deepseek-sglang2.yaml        sample vllm 4,5,6,7
sglang docker.io/lmsysorg/sglang:v0.5.6.post2-rocm700-mi35x         amd/DeepSeek-R1-0528-MXFP4-Preview  configs/models/deepseek-sglang2-fp4.yaml    sample vllm 0,1,2,3,4,5,6,7


################################################
# KIMI K2
################################################

vllm   docker.io/rocm/vllm:rocm7.0.0_vllm_0.11.2_20251210           moonshotai/Kimi-K2-Instruct             configs/models/kimi-k2-vllm-0.yaml            sample vllm 0,1,2,3
vllm   docker.io/rocm/vllm:rocm7.0.0_vllm_0.11.2_20251210           moonshotai/Kimi-K2-Instruct             configs/models/kimi-k2-vllm-1.yaml            sample vllm 0,1,2,3
vllm   docker.io/rocm/vllm:rocm7.0.0_vllm_0.11.2_20251210           moonshotai/Kimi-K2-Instruct             configs/models/kimi-k2-vllm-2.yaml            sample vllm 0,1,2,3
vllm   docker.io/rocm/vllm:rocm7.0.0_vllm_0.11.2_20251210           moonshotai/Kimi-K2-Instruct             configs/models/kimi-k2-vllm-2.yaml            sample vllm 0,1,2,3,4,5,6,7

sglang docker.io/rocm/sgl-dev:v0.5.6.post1-rocm700-mi35x-20251211   moonshotai/Kimi-K2-Instruct             configs/models/kimi-k2-sglang1.yaml           sample vllm 0,1,2,3

vllm   docker.io/rocm/vllm:rocm7.0.0_vllm_0.11.2_20251210           moonshotai/Kimi-K2-Thinking             configs/models/kimi-k2-vllm-0.yaml            sample vllm 0,1,2,3
vllm   docker.io/rocm/vllm:rocm7.0.0_vllm_0.11.2_20251210           moonshotai/Kimi-K2-Thinking             configs/models/kimi-k2-vllm-1.yaml            sample vllm 0,1,2,3
vllm   docker.io/rocm/vllm:rocm7.0.0_vllm_0.11.2_20251210           moonshotai/Kimi-K2-Thinking             configs/models/kimi-k2-vllm-2.yaml            sample vllm 0,1,2,3

vllm   docker.io/rocm/vllm:rocm7.0.0_vllm_0.11.2_20251210           RedhatAI/Kimi-K2-Thinking-FP8-Block     configs/models/kimi-k2-vllm-2.yaml            sample vllm 0,1,2,3,4,5,6,7 # not covered
