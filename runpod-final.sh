#!/bin/bash

### FUNCTIONS ###

check_variables() {
    : "${HF_TOKEN:=$(read -r -p 'Please set HF_TOKEN: ' tmp && echo $tmp)}"
    : "${MODEL:=$(read -r -p 'Please set MODEL: ' tmp && echo $tmp)}"
    : "${BENCHMARK:=$(read -r -p 'Please set BENCHMARK: ' tmp && echo $tmp)}"
    : "${GITHUB_API_TOKEN:=$(read -r -p 'Please set GITHUB_API_TOKEN: ' tmp && echo $tmp)}"

    # Validate GitHub token
    if ! curl -H "Authorization: token $GITHUB_API_TOKEN" -s https://api.github.com/user | grep -q login; then
        echo "Invalid GITHUB_API_TOKEN. Please check and try again."
        exit 1
    fi
}

setup_cuda_devices() {
    local gpu_count=$(nvidia-smi -L | wc -l)
    if [ "$gpu_count" -eq 0 ]; then
        echo "No NVIDIA GPUs detected. Exiting."
        exit 1
    fi
    seq -s, 0 $((gpu_count - 1))
}

bootstrap() {
    export MAKEFLAGS="-j$(nproc)"
    export CMAKE_BUILD_PARALLEL_LEVEL=$(nproc)

    echo "Updating and installing necessary packages..."
    DEBIAN_FRONTEND=noninteractive apt-get update -y && apt-get install -y screen git git-lfs python3 python3-pip jq

    echo "Cloning and setting up lm-evaluation-harness repository..."
    [[ ! -d "lm-evaluation-harness" ]] && git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness || echo "Directory already exists, skipping git clone."
    cd lm-evaluation-harness

    cat <<EOF >requirements.txt
-e .[openai,vllm,math,sentencepiece,zeno,hf_transfer] ;
pytablewriter
einops
protobuf
EOF

    pip install --upgrade --no-cache-dir pip setuptools wheel
    pip install -r requirements.txt
}

eval_function() {
    local common_args="--model hf --model_args pretrained=$MODEL,dtype=auto --device cuda:$cuda_devices --batch_size auto --output_path ./${BENCHMARK}.json --trust_remote_code"

    case "$BENCHMARK" in
    "tiny")
        pip install git+https://github.com/felipemaiapolo/tinyBenchmarks --no-cache-dir --prefer-binary
        TASKS="tinyArc,tinyHellaswag,tinyMMLU,tinyTruthfulQA,tinyTruthfulQA_mc1,tinyWinogrande"
        lm_eval $common_args --tasks "$TASKS"
        ;;
    "tinychat")
        pip install git+https://github.com/felipemaiapolo/tinyBenchmarks --no-cache-dir --prefer-binary
        TASKS="tinyArc,tinyHellaswag,tinyMMLU,tinyTruthfulQA,tinyTruthfulQA_mc1,tinyWinogrande"
        lm_eval $common_args --tasks "$TASKS" --apply_chat_template --fewshot_as_multiturn
        ;;
    "nous" | "nous:*")
        declare -A benchmarks=(
            ["agieval"]="agieval_aqua_rat,agieval_logiqa_en,agieval_lsat_ar,agieval_lsat_lr,agieval_lsat_rc,agieval_sat_en,agieval_sat_en_without_passage,agieval_sat_math"
            ["gpt4all"]="hellaswag,openbookqa,winogrande,arc_easy,arc_challenge,boolq,piqa"
            ["truthfulqa"]="truthfulqa_mc"
            ["bigbench"]="bigbench_causal_judgement,bigbench_date_understanding,bigbench_disambiguation_qa,bigbench_geometric_shapes,bigbench_logical_deduction_five_objects,bigbench_logical_deduction_seven_objects,bigbench_logical_deduction_three_objects,bigbench_movie_recommendation,bigbench_navigate,bigbench_reasoning_about_colored_objects,bigbench_ruin_names,bigbench_salient_translation_error_detection,bigbench_snarks,bigbench_sports_understanding,bigbench_temporal_sequences,bigbench_tracking_shuffled_objects_five_objects,bigbench_tracking_shuffled_objects_seven_objects,bigbench_tracking_shuffled_objects_three_objects"
        )
        for bm in "${!benchmarks[@]}"; do
            lm_eval $common_args --tasks "${benchmarks[$bm]}" --output_path ./${bm}.json
        done
        ;;
    *)
        echo "Invalid benchmark specified"
        exit 1
        ;;
    esac
}

upload_results() {
    local directory=$1
    local elapsed_time=$2

    python3 ../main.py "$directory" "$elapsed_time"
}

### MAIN EXECUTION ###

set -e
trap 'echo "An error occurred. Exiting..."' ERR

check_variables
cuda_devices=$(setup_cuda_devices)
bootstrap

start_time=$(date +%s)
eval_function
end_time=$(date +%s)

elapsed_time=$((end_time - start_time))
upload_results "$(pwd)" "$elapsed_time"
