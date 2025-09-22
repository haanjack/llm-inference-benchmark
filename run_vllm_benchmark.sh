#!/bin/bash

# Load environment variables from the .env file
ENV_FILE=${1:-"baseline"}
ENV_FILE_PATH=$(pwd)/"envs"/${ENV_FILE} 
source $ENV_FILE_PATH

# Define the model to be used for both server and tokenizer
MODEL_NAME_OPT=${2:-"Meta-Llama-3-8B-Instruct-FP8"}
if [[ ${MODEL_NAME} == "" ]]; then
    MODEL_NAME=${MODEL_NAME_OPT}
fi
MODEL_NAME_PATH="/workspace/models/${MODEL_NAME}"

# Define the vLLM container image
VLLM_IMAGE_OPT=${3:-"docker.io/rocm/vllm:latest"}
if [[ ${VLLM_IMAGE} == "" ]]; then
    VLLM_IMAGE=${VLLM_IMAGE_OPT}
fi

# Define benchmark scape
BENCH_SCAPE_OPT=${4:-"test"}
if [[ ${BENCH_SCAPE_OPT} == "test" || ${BENCH_SCAPE_OPT} == "sweep" ]]; then
    CLIENT_COUNTS=(4)
    INPUT_LENGTHS=(2048)
    OUTPUT_LENGTH=(2048)
elif [[ ${BENCH_SCAPE_OPT} == "prefill" ]]; then
    CLIENT_COUNTS=(1 2 4 8 16 32 64 128)
    INPUT_LENGTHS=(256 512 1024 2048 4096 8192)
    OUTPUT_LENGTHS=(128)
elif [[ ${BENCH_SCAPE_OPT} == "decode" ]]; then
    CLIENT_COUNTS=(1 2 4 8 16 32 64 128)
    INPUT_LENGTHS=(128)
    OUTPUT_LENGTHS=(128 1024 2048 4096)
elif [[ ${BENCH_SCAPE_OPT} == "middle" ]]; then
    CLIENT_COUNTS=(1 2 4 8 16 32 64 128)
    INPUT_LENGTHS=(1024 2048 4096 8192)
    OUTPUT_LENGTHS=(1024 2048 4096)
fi

# GPU index
CUSTOM_VISIBLE_DEVICES_OPT=${5:-"${SLURM_JOB_GPUS}"}

# --quantization=${QUANTIZATION}
VLLM_ARGS="
--kv-cache-dtype=${KV_CACHE_DTYPE} 
--gpu_memory_utilization=${GPU_MEMORY_UTILIZATION:-"0.9"} 
--max-seq-len-to-capture ${MAX_SEQ_LEN_TO_CAPTURE:-"8192"} 
--max-num-batched-token ${MAX_NUM_BATCHED_TOKENS:-"8192"} 
--swap-space 64
--no-enable-prefix-caching 
--async-scheduling "
if [[ "$VLLM_USE_V1" == "0" ]]; then
    # Use CK Flash Attention
    VLLM_USE_TRITON_FLASH_ATTN=0
else
    if [[ $VLLM_ROCM_USE_AITER == 1 ]]; then
        # AITER MHA is on by default but does not support full graph capture yet
        if [[ $VLLM_ROCM_USE_AITER_MHA != 0 ]]; then
            VLLM_ARGS=$VLLM_ARGS # '--compilation-config {"full_cuda_graph":false}'
        fi

        if [[ ${VLLM_CUDAGRAPH_MODE} != "" ]]; then
            if [[ ${VLLM_CUDAGRAPH_MODE} == "NONE" ]]; then
                VLLM_ARGS="${VLLM_ARGS} --compilation-config {\"cudagraph_mode\": \"null\"} "
            elif [[ ${VLLM_CUDAGRAPH_MODE} == "PIECEWISE" ]]; then #default
                VLLM_ARGS="${VLLM_ARGS} --compilation-config {\"cudagraph_mode\": \"PIECEWISE\"} "
            elif [[ ${VLLM_CUDAGRAPH_MODE} == "FULL" ]]; then
 		VLLM_ARGS+=(--compilation-config='{"cudagraph_mode": "FULL"}')
#		VLLM_ARGS+=" --compilation-config='{\"cudagraph_mode\": \"FULL\"}'"
            elif [[ ${VLLM_CUDAGRAPH_MODE} == "FULL_DECODE_ONLY" ]]; then
                VLLM_ARGS="${VLLM_ARGS} --compilation-config {\"cudagraph_mode\": \"FULL_DECODE_ONLY\"} "
            elif [[ ${VLLM_CUDAGRAPH_MODE} == "FULL_AND_PIECEWISE" ]]; then
                VLLM_ARGS="${VLLM_ARGS} --compilation-config {\"cudagraph_mode\": \"FULL_AND_PIECEWISE\"} "
            elif [[ ${VLLM_CUDAGRAPH_MODE} == "MOE" ]]; then
                VLLM_ARGS="${VLLM_ARGS} --compilation-config \'{\"compile_sizes\":[1, 2, 4, 8, 16, 24, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192], \"cudagraph_capture_sizes\":[8192,4096,2048,1024,1008,992,976,960,944,928,912,896,880,864,848,832,816,800,784,768,752,736,720,704,688,672,656,640,624,608,592,576,560,544,528,512,496,480,464,448,432,416,400,384,368,352,336,320,304,288,272,256,248,240,232,224,216,208,200,192,184,176,168,160,152,144,136,128,120,112,104,96,88,80,72,64,56,48,40,32,24,16,8,4,2,1], \"full_cuda_graph\": true}\' "
            else
                echo "Unknown VLLM_CUDAGRAPH_MODE: ${VLLM_CUDAGRAPH_MODE}"
                exit 1
            fi
	    fi
    fi
fi

# GPU information handling
IFS=',' read -r -a gpu_array <<< "$CUSTOM_VISIBLE_DEVICES_OPT"
FIRST_GPU_ID=${gpu_array[0]}
NUM_GPUS=${#gpu_array[@]}
echo "> $NUM_GPUS GPU(s) is/are allocated"

# Define the container name
PROCESS_NAME="${ENV_FILE}"
SLURM_JOB_ID=${SLURM_JOB_ID:-""}
CONTAINER_NAME="${PROCESS_NAME}_${SLURM_JOB_ID}_${FIRST_GPU_ID}" # Unique name to avoid conflicts
if [[ ${SLURM_JOB_ID} == "" ]]; then
    CONTAINER_NAME="${PROCESS_NAME}_${FIRST_GPU_ID}" # Unique name to avoid conflicts
fi

if [ -z "$NUM_GPUS" ]; then
  echo "No GPU is allocated"
  exit 1
fi

# parallelism
if [[ $TP_SIZE != "" ]]; then
    TENSOR_PARALLEL_SIZE=$TP_SIZE
fi

# -- start vllm server ---
VLLM_PORT=$((23400 + ${FIRST_GPU_ID}))

# Setting log directory
IMAGE_TAG=${VLLM_IMAGE##*:}

LOG_DIR="logs/${MODEL_NAME}/${IMAGE_TAG}"
if [[ "$BENCH_SCAPE_OPT" == "sweep" ]]; then
    LOG_DIR="logs/${MODEL_NAME}/${IMAGE_TAG}"
fi
mkdir -p ${LOG_DIR}
CONSOLIDATED_LOG="$LOG_DIR/benchmark_result.log"
SERVER_LOG_FILE="$LOG_DIR/server_log.txt"

if [[ ${PROFILE} ]]; then
    VLLM_TORCH_PROFILER_DIR_OPT="-e VLLM_TORCH_PROFILER_DIR=$LOG_DIR"
    PROFILE_OPT="--profile"
fi

# Header for terminal output
printf "%-30s\t%-8s\t%-8s\t%-8s\t%-8s\t%-12s\t%-12s\t%-12s\t%-12s\t%-12s\t%-12s\t%-12s\t%-12s\t%-12s\t%-12s\t%-12s\t%-12s\t%-12s\n" \
"ENV" "TP Size" "Client Count" "Input Length" "Output Length" "Mean TTFT (ms)" "Median TTFT" "P99 TTFT" "Mean TPOT" "Median TPOT" "P99 TPOT" \
"Mean ITL" "Median ITL" "P99 ITL" "Mean E2EL" "Request THPT (req/s)" "Output token THPT" "Total Token THPT"
echo "------------------------------------------------------------------------------------------------------------------------------------------------"

RESULT_FILE="$LOG_DIR/result_list"
if [[ -f "$RESULT_FILE" ]]; then
    echo "Result file already exists. Appending to it."
else
    echo "Creating new result file."
    echo "env,TP Size,Client Count,Input Length,Output Length,Mean TTFT (ms),Median TTFT (ms),P99 TTFT (ms),Mean TPOT (ms),Median TPOT (ms),P99 TPOT (ms),Mean ITL (ms),Median ITL (ms),P99 ITL (ms),Mean E2EL (ms),Request Throughput (req/s),Output token throughput (tok/s),Total Token throughput (tok/s)" > "$RESULT_FILE"
    echo "" > "$CONSOLIDATED_LOG"
fi

if [ $(docker ps -a -q -f name=$CONTAINER_NAME) ]; then
    echo "Removing container ${CONTAINER_NAME} ..."
    docker rm -f $CONTAINER_NAME
fi


# Benchmark space: this should be specified with env file
# CLIENT_COUNTS=(1 2 4 8 16 24 32 40 48 64 96 128 160 192 224 256)
# INPUT_LENGTHS=(256 512 1024 2048 4096 8192)
# OUTPUT_LENGTHS=(256 512 1024 2048 4096 8192)
if [[ $CLIENT_COUNT != "" ]]; then
    CLIENT_COUNTS=($CLIENT_COUNT)
fi
if [[ $INPUT_LENGTH != "" ]]; then
    INPUT_LENGTHS=($INPUT_LENGTH)
fi
if [[ $OUTPUT_LENGTH != "" ]]; then
    OUTPUT_LENGTHS=($OUTPUT_LENGTH)
fi

# Prepare output    
MAX_MODEL_LEN=$((${INPUT_LENGTHS[-1]} + ${OUTPUT_LENGTHS[-1]} + 256))

set -x
docker run -d --name "$CONTAINER_NAME" \
    -v "$HF_HOME:/root/.cache/huggingface" \
    --device /dev/kfd --device /dev/dri --device /dev/mem \
    --group-add video \
    --ipc=host --network=host \
    --cap-add=CAP_SYS_ADMIN \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --env-file ${ENV_FILE_PATH} \
    -e VLLM_USE_TRITON_FLASH_ATTN=${VLLM_USE_TRITON_FLASH_ATTN} \
    -e CUDA_VISIBLE_DEVICES=${CUSTOM_VISIBLE_DEVICES_OPT} \
    ${VLLM_TORCH_PROFILER_DIR_OPT} \
    --volume ${HOME}:/workspace/ \
    ${VLLM_IMAGE} \
    vllm serve \
    ${MODEL_NAME_PATH} \
    --disable-log-requests \
    --trust-remote-code \
    --max-model-len=${MAX_MODEL_LEN} \
    --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} \
    --distributed-executor-backend mp \
    --port=${VLLM_PORT} \
    ${VLLM_ARGS} \
    --host 0.0.0.0 > /dev/null 1>&2 &
set +x

# --- Server Health Check ---

# echo "Waiting for the server to become available..."
sleep 5
docker logs -f ${CONTAINER_NAME} > /dev/null 2>> ${SERVER_LOG_FILE} &
LOG_COLLECTOR_PID=$! # saving logging background process to terminiate this

TIMEOUT=6000 # 60-minute timeout
START_TIME=$SECONDS

while ! curl --silent --fail http://localhost:${VLLM_PORT}/v1/models > /dev/null; do
    if (( SECONDS - START_TIME > TIMEOUT )); then
# set -x
        docker logs "$CONTAINER_NAME" 1>&2
        echo "Error: Server failed to start within $TIMEOUT seconds."
        docker rm -f "$CONTAINER_NAME"
        exit 1
# set +x
    fi
    sleep 5
done

# collect server initialization logs
docker logs "$CONTAINER_NAME" >> "$CONSOLIDATED_LOG"
echo "Server is up and running."

# warmup
# set -x
NUM_PROMPTS=16
docker exec "$CONTAINER_NAME" \
    python /app/vllm/benchmarks/benchmark_serving.py \
        --backend vllm \
        --host localhost \
        --port ${VLLM_PORT} \
        --model $MODEL_NAME_PATH \
        --dataset-name random \
        --ignore-eos \
        --trust-remote-code \
        --num-prompts $NUM_PROMPTS \
        --max-concurrency 4 \
        --random-input-len 256 \
        --random-output-len 256 \
        --tokenizer $MODEL_NAME_PATH > /dev/null
# set +x


# benchmark loop
for OUTPUT_LENGTH in "${OUTPUT_LENGTHS[@]}"; do
for INPUT_LENGTH in "${INPUT_LENGTHS[@]}"; do
for CLIENT_COUNT in "${CLIENT_COUNTS[@]}"; do
    # --- Server Startup ---

    # check if benchmark is done
    unique_key="$ENV_FILE,$TENSOR_PARALLEL_SIZE,$CLIENT_COUNT,$INPUT_LENGTH,$OUTPUT_LENGTH"
    readline=$(grep -F "$unique_key" "$RESULT_FILE")

    if [ -n "$readline" ]; then
        IFS=',' read -r s_env f_tp f_bs f_in f_out f_ttft_mean f_ttft_median f_ttft_p99 f_tpot_mean f_tpot_median f_tpot_p99 f_itl_mean f_itl_median f_itl_p99 f_e2el_mean f_rthpt f_othpt f_tthpt <<< "$readline"

        printf "%-30s\t%-8s\t%-8s\t%-8s\t%-8s\t%-12s\t%-12s\t%-12s\t%-12s\t%-12s\t%-12s\t%-12s\t%-12s\t%-12s\t%-12s\t%-12s\t%-12s\t%-12s\n" \
        "$s_env" "$f_tp" "$f_bs" "$f_in" "$f_out" "$f_ttft_mean" "$f_ttft_median" "$f_ttft_p99" "$f_tpot_mean" "$f_tpot_median" "$f_tpot_p99" "$f_itl_mean" "$f_itl_median" "$f_itl_p99" "$f_e2el_mean" "$f_rthpt" "$f_othpt" "$f_tthpt"
        continue
    fi
    
    # Start the vLLM server in the background, redirecting output to /dev/null
    # echo "Running the benchmark client..."
    
    # Run the benchmark and save the results
    LOG_FILE="$LOG_DIR/vllm_tp${TENSOR_PARALLEL_SIZE}_i${INPUT_LENGTH}_o${OUTPUT_LENGTH}_c${CLIENT_COUNT}.log"
    # set -x
    NUM_ITERATION=8
    NUM_PROMPTS=$((${CLIENT_COUNT} * ${NUM_ITERATION}))
    docker exec "$CONTAINER_NAME" \
	vllm bench serve \
	    --model ${MODEL_NAME_PATH} \
        --backend vllm \
        --host localhost \
        --port ${VLLM_PORT} \
        --model $MODEL_NAME_PATH \
        --dataset-name random \
        --ignore-eos \
        --trust-remote-code \
        --num-prompts $NUM_PROMPTS \
        --max-concurrency $CLIENT_COUNT \
        --random-input-len $INPUT_LENGTH \
        --random-output-len $OUTPUT_LENGTH \
        --tokenizer $MODEL_NAME_PATH \
        --disable-tqdm \
        --percentile-metrics ttft,tpot,itl,e2el > "$LOG_FILE" 2>&1
        ${PROFILE_OPT} \
    # set +x

    # --- Reporting ---

    # Append to consolidated log
    {
        echo -e "\n========== CLIENT COUNT: $CLIENT_COUNT, INPUT LENGTH: $INPUT_LENGTH, OUTPUT_LENGTH: $OUTPUT_LENGTH ==========\n"
        cat "$LOG_FILE"
        echo -e "\n==============================================\n"
    } >> "$CONSOLIDATED_LOG"

    echo $LOG_FILE

    # Extract metrics
    PREFILL_LATENCY_MEAN=$(awk '/Mean TTFT \(ms\)/ { printf "%.2f", $NF }' "$LOG_FILE")
    PREFILL_LATENCY_MEDIAN=$(awk '/Median TTFT \(ms\)/ { printf "%.2f", $NF }' "$LOG_FILE")
    PREFILL_LATENCY_P99=$(awk '/P99 TTFT \(ms\)/ { printf "%.2f", $NF }' "$LOG_FILE")
    DECODE_TPUT_MEAN=$(awk '/Mean TPOT \(ms\)/ { printf "%.2f", $NF }' "$LOG_FILE")
    DECODE_TPUT_MEDIAN=$(awk '/Median TPOT \(ms\)/ { printf "%.2f", $NF }' "$LOG_FILE")
    DECODE_TPUT_P99=$(awk '/P99 TPOT \(ms\)/ { printf "%.2f", $NF }' "$LOG_FILE")
    ITL_MEAN=$(awk '/Mean ITL \(ms\)/ { printf "%.2f", $NF }' "$LOG_FILE")
    ITL_MEDIAN=$(awk '/Median ITL \(ms\)/ { printf "%.2f", $NF }' "$LOG_FILE")
    ITL_P99=$(awk '/P99 ITL \(ms\)/ { printf "%.2f", $NF }' "$LOG_FILE")
    E2E_LATENCY=$(awk '/Mean E2EL \(ms\)/ { printf "%.2f", $NF }' "$LOG_FILE")

    REQUEST_TPUT=$(awk '/Request throughput \(req\/s\)/ { print $NF }' "$LOG_FILE")
    OUTPUT_TOKEN_THROUGHPUT=$(awk '/Output token throughput \(tok\/s\)/ { print $NF }' "$LOG_FILE")
    TOTAL_TOKEN_THROUGHPUT=$(awk '/Total Token throughput \(tok\/s\)/ { print $NF }' "$LOG_FILE")

    # Fallbacks for empty values
    PREFILL_LATENCY_MEAN=${PREFILL_LATENCY_MEAN:-0.0000}
    DECODE_TPUT_MEAN=${DECODE_TPUT_MEAN:-0.00}
    E2E_LATENCY=${E2E_LATENCY:-0.0000}
    REQUEST_TPUT=${REQUEST_TPUT:-0.00}

    # Print formatted row to terminal
    printf "%-30s\t%-8s\t%-8s\t%-8s\t%-8s\t%-12s\t%-12s\t%-12s\t%-12s\t%-12s\t%-12s\t%-12s\t%-12s\t%-12s\t%-12s\t%-12s\t%-12s\t%-12s\n" \
    "$ENV_FILE" "$TENSOR_PARALLEL_SIZE" "$CLIENT_COUNT" "$INPUT_LENGTH" "$OUTPUT_LENGTH" "$PREFILL_LATENCY_MEAN" "$PREFILL_LATENCY_MEDIAN" "$PREFILL_LATENCY_P99" "$DECODE_TPUT_MEAN" "$DECODE_TPUT_MEDIAN" "$DECODE_TPUT_P99" "$ITL_MEAN" "$ITL_MEDIAN" "$ITL_P99" "$E2E_LATENCY" "$REQUEST_TPUT" "$OTUPUT_TOKEN_THROUGHPUT" "$TOTAL_TOKEN_THROUGHPUT"

    if [[ $PREFILL_LATENCY_MEAN == "0.0000" && $DECODE_TPUT_MEAN == "0.00" && $E2E_LATENCY == "0.0000" && $REQUEST_TPUT == "0.00" ]]; then
        echo "Benchmark failed or returned zero metrics. Skipping result saving."
        continue
    fi

    # Save to CSV
    echo "$ENV_FILE,$TENSOR_PARALLEL_SIZE,$CLIENT_COUNT,$INPUT_LENGTH,$OUTPUT_LENGTH,$PREFILL_LATENCY_MEAN,$PREFILL_LATENCY_MEDIAN,$PREFILL_LATENCY_P99,$DECODE_TPUT_MEAN,$DECODE_TPUT_MEDIAN,$DECODE_TPUT_P99,$ITL_MEAN,$ITL_MEDIAN,$ITL_P99,$E2E_LATENCY,$REQUEST_TPUT,$OUTPUT_TOKEN_THROUGHPUT,$TOTAL_TOKEN_THROUGHPUT," >> "$RESULT_FILE"

    # --- Cleanup ---

    # echo "Stopping the vLLM server..."
    # kill ${LOG_COLLECTOR_PID}
    # docker rm -f "$CONTAINER_NAME" > /dev/null

done # CLIENT_COUNT
done # INPUT_LENGTH
done # OUTPUT_LENGTH

if [ $(docker ps -a -q -f name=$CONTAINER_NAME) ]; then
    echo "Removing container ${CONTAINER_NAME} ..."
    docker rm -f $CONTAINER_NAME
fi

echo -e "\nBenchmarking complete."
echo "📄 Results:          $RESULT_FILE"
echo "📜 Consolidated log: $CONSOLIDATED_LOG"


echo "Benchmark complete."

