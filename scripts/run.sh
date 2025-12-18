#!/bin/bash
# this script is used to run model benchmarks in benchmark mode with distributed hosts

run_mode=${1:-"benchmark"} # Options: "" | "benchmark" | "dry_run" | "generate_script"
run_list_file=${2:-"scripts/run_list_example.sh"}
test_plan_override=${3:-""} # optional test plan to override run_list test_plan

# list of hostnames to distribute benchmarks across
# e.g. host_list=("host1" "host2" "host3")
host_list=()

# docker images for different backends
# image_vllm="docker.io/rocm/vllm:rocm7.0.0_vllm_0.11.2_20251210"
# image_sgl="docker.io/rocm/sgl-dev:v0.5.6.post1-rocm700-mi35x-20251211"

# set to true to remove model checkpoints after use
remove_checkpoint=false


###### don't modify below this line ######

# load run_list from file
run_list=()
while IFS= read -r line || [ -n "$line" ]; do
    # Skip empty lines and comments
    if [[ -z "$line" || "$line" =~ ^# ]]; then
        continue
    fi
    run_list+=("$line")
done < "$run_list_file"

if [ "$run_mode" == "generate_script" ]; then
    if [ ! -d scripts/generated ]; then
        mkdir -p scripts/generated
    fi
    rm -f scripts/generated/*.sh
fi

current_host=$(hostname)
# current_host="mi355-gpu-2" # for testing purpose

# distribute benchmark_map keys across hosts
# each host get different model_path_or_id keys
# when multiple hosts run the same script, they will run different models
# this is to avoid multiple hosts downloading the same model at the same time
# each host will run approximately equal number of models
if [ "$run_mode" == "benchmark" ]; then
    # Coalesce benchmark list based on model_path_or_id
    declare -A benchmark_map
    model_key_list=()
    for entry in "${run_list[@]}"; do
        IFS=' ' read -r -a values <<< "$entry"
        model_path_or_id="${values[2]}"
        benchmark_map["$model_path_or_id"]+="$entry |"
    done
    for key in "${!benchmark_map[@]}"; do
        model_key_list+=("$key")
    done

    # Select keys for the current host
    if [ ${#host_list[@]} -ne 0 ]; then
        selected_keys=()
        model_key_index=0
        for model_key in "${model_key_list[@]}"; do
            if [ $((model_key_index % ${#host_list[@]})) -eq $(echo ${host_list[@]} | tr ' ' '\n' | grep -n "^$current_host$" | cut -d: -f1-1 | awk '{print $1-1}') ]; then
                selected_keys+=("$model_key")
            fi
            model_key_index=$((model_key_index + 1))
        done
        echo "Selected keys for host $current_host: ${selected_keys[*]}"
    else
        selected_keys=("${model_key_list[@]}")
    fi

    run_list=()
    for key in "${selected_keys[@]}"; do
        entries_string="${benchmark_map[$key]}"
        # Remove trailing pipe
        entries_string="${entries_string%|}"

        # Split by pipe into an array and trim whitespace
        IFS='|' read -r -a entries <<< "$entries_string"
        for entry in "${entries[@]}"; do
            # trim leading/trailing spaces
            entry="${entry#"${entry%%[![:space:]]*}"}"
            entry="${entry%"${entry##*[![:space:]]}"}"
            if [ -n "$entry" ]; then
                run_list+=("$entry")
            fi
        done
    done
fi

# Count how many times each model appears in run_list
declare -A model_usage_count
for entry in "${run_list[@]}"; do
    IFS=' ' read -r -a values <<< "$entry"
    model_path_or_id="${values[2]}"
    ((model_usage_count["$model_path_or_id"]++))
done

# track async runs
declare -a bg_pids bg_models bg_entries

# run benchmarks
for entry in "${run_list[@]}"; do
    IFS=' ' read -r -a values <<< "$entry"
    server_backend="${values[0]}"
    docker_image="${values[1]}"
    model_path_or_id="${values[2]}"
    model_config="${values[3]}"
    test_plan="${values[4]}"
    benchmark_client="${values[5]}"
    gpu_devices="${values[6]}"
    sub_task="${values[7]:-"all"}"
    run_async="${values[8]:-""}"

    if [ ! -f "$model_config" ]; then
        echo "Model config file not found: '${model_config}', skipping..."
        continue
    fi

    if [ -n "$test_plan_override" ]; then
        test_plan="$test_plan_override"
    fi

    # Async mode: launch and move on
    if [[ "$run_async" == "async" ]]; then
        bash scripts/run_test.sh ${run_mode} ${model_config} ${model_path_or_id} ${server_backend} ${docker_image} ${benchmark_client} ${gpu_devices} ${test_plan} ${sub_task} &
        pid=$!
        echo "Launched async PID ${pid} for model '${model_path_or_id}' on GPUs '${gpu_devices}'"
        bg_pids+=("$pid")
        bg_models+=("$model_path_or_id")
        bg_entries+=("$entry")
        continue
    fi

    # Wait for pending async jobs before running sync job
    if [ ${#bg_pids[@]} -gt 0 ]; then
        echo "Waiting for ${#bg_pids[@]} async jobs before running sync job..."
        for i in "${!bg_pids[@]}"; do
            wait "${bg_pids[$i]}"
        done
        bg_pids=()
        bg_models=()
        bg_entries=()
    fi

    # echo "Running benchmark for model: '${model_path_or_id}' with config: '${model_config}' on backend: '${server_backend}' using docker image: '${docker_image}'"
    bash scripts/run_test.sh ${run_mode} ${model_config} ${model_path_or_id} ${server_backend} ${docker_image} ${benchmark_client} ${gpu_devices} ${test_plan} ${sub_task}
    test_status=$?

    # Remove checkpoint to save space
    if [ $test_status -ne 0 ]; then
        echo "Benchmark failed for model: '${model_path_or_id}'"
        continue
    else
        # Check if this model is still needed for remaining tests
        model_still_needed=false
        current_index=$((BASH_LINENO[0]))
        for remaining_entry in "${run_list[@]}"; do
            IFS=' ' read -r -a remaining_values <<< "$remaining_entry"
            remaining_model="${remaining_values[2]}"

            # Skip if we haven't processed this entry yet
            if [ "$remaining_entry" == "$entry" ]; then
                continue
            fi

            # Check if any remaining entry uses the same model
            if [ "$remaining_model" == "$model_path_or_id" ]; then
                model_still_needed=true
                break
            fi
        done

        # Only remove checkpoint if no more tests need this model
        if [ "$model_still_needed" == false ]; then
            if [ "$remove_checkpoint" = true ]; then
                echo "No more tests need model '${model_path_or_id}', removing checkpoint..."
                rm -rf ~/models/$model_path_or_id
            fi
        else
            echo "Model '${model_path_or_id}' still needed for remaining tests, keeping checkpoint..."
        fi
    fi
done

# Wait for any remaining background jobs
if [ ${#bg_pids[@]} -gt 0 ]; then
    echo "Waiting for ${#bg_pids[@]} final async jobs..."
    for i in "${!bg_pids[@]}"; do
        pid="${bg_pids[$i]}"
        model="${bg_models[$i]}"
        if wait "$pid"; then
            echo "Async job PID ${pid} for model '${model}' completed successfully."
        else
            echo "Async job PID ${pid} for model '${model}' failed."
        fi
    done
fi