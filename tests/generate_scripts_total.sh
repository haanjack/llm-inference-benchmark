#!/bin/bash


# LLaMA
bash tests/generate_scripts.sh "configs/models/llama-vllm.yaml"     "amd/Llama-3.1-8B-Instruct-FP8-KV" "vllm" "genai-perf" "sample" "0"

# DeepSeek/Kimi
bash tests/generate_scripts.sh "configs/models/deepseek-vllm.yaml"  "deepseek-ai/DeepSeek-R1"          "vllm" "genai-perf" "sample" "0"

# Qwen
bash tests/generate_scripts.sh "configs/models/qwen-vllm.yaml"      "Qwen/Qwen3-32B"                   "vllm" "genai-perf" "sample" "0"

# Qwen-MoE
bash tests/generate_scripts.sh "configs/models/qwen-moe-vllm.yaml"  "Qwen/Qwen3-235B-A22B"             "vllm" "genai-perf" "sample" "0"

# GPT-OSS
bash tests/generate_scripts.sh "configs/models/gpt-oss-vllm.yaml"   "openai/gpt-oss-120b"              "vllm" "genai-perf" "sample" "0"
