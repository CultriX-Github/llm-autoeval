#!/bin/bash

### FUNCTIONS ###

check_variables() {
    : "${HF_TOKEN:=$(read -r -p 'Please set HF_TOKEN: ' tmp && echo $tmp)}"
    : "${MODEL:=$(read -r -p 'Please set MODEL: ' tmp && echo $tmp)}"
    : "${BENCHMARK:=$(read -r -p 'Please set BENCHMARK: ' tmp && echo $tmp)}"
    : "${GITHUB_API_TOKEN:=$(read -r -p 'Please set GITHUB_API_TOKEN: ' tmp && echo $tmp)}"
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

generate_summary_and_upload() {
    local directory=$1
    local elapsed_time=$2

    python3 - <<EOF
import os
import json
import time
from llm_autoeval.table import make_table, make_final_table
from llm_autoeval.upload import upload_to_github_gist

directory = "$directory"
elapsed_time = float("$elapsed_time")
model = os.getenv("MODEL")
benchmark = os.getenv("BENCHMARK")
github_api_token = os.getenv("GITHUB_API_TOKEN")

tasks = []
if benchmark == "openllm":
    tasks = ["ARC", "HellaSwag", "MMLU", "TruthfulQA", "Winogrande", "GSM8K"]
elif benchmark == "nous":
    tasks = ["AGIEval", "GPT4All", "TruthfulQA", "Bigbench"]
elif benchmark == "tiny":
    tasks = ["tinyArc", "tinyHellaswag", "tinyMMLU", "tinyTruthfulQA", "tinyTruthfulQA_mc1", "tinyWinogrande"]
elif benchmark == "tinychat":
    tasks = ["tinyArc", "tinyHellaswag", "tinyMMLU", "tinyTruthfulQA", "tinyTruthfulQA_mc1", "tinyWinogrande"]
else:
    raise ValueError(f"Invalid BENCHMARK value: {benchmark}")

tables = []
averages = []

for task in tasks:
    file_path = os.path.join(directory, f"{task.lower()}.json")
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            data = json.load(f)
        table, average = make_table(data, task)
    else:
        table = f"### {task}\nError: File does not exist\n\n"
        average = None

    tables.append(table)
    averages.append(average)

summary = ""
for i, task in enumerate(tasks):
    summary += tables[i]
    if averages[i] is not None:
        summary += f"Average: {averages[i]}%\n\n"
    else:
        summary += "Average: Not available due to error\n\n"

# Calculate final average
if all(isinstance(avg, float) for avg in averages):
    final_average = round(sum(averages) / len(averages), 2)
    summary += f"Average score: {final_average}%\n"
else:
    summary += "Average score: Not available due to errors\n"

# Add elapsed time
elapsed = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
summary += f"\nElapsed time: {elapsed}"

# Final table
final_table = make_final_table({k: v for k, v in zip(tasks, averages)}, model)
summary = final_table + "\n\n" + summary

# Upload to GitHub Gist
upload_to_github_gist(summary, f"{model.split('/')[-1]}-{benchmark.capitalize()}.md", github_api_token)
EOF
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
generate_summary_and_upload "$(pwd)" "$elapsed_time"
