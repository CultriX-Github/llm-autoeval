#!/bin/bash

### FUNCTIONS ###
check_variables() {
        # Verify that critical environment variables are set
        : "${HF_TOKEN:=$(read -r -p 'Please set HF_TOKEN: ' tmp && echo $tmp)}"
        : "${MODEL:=$(read -r -p 'Please set MODEL: ' tmp && echo $tmp)}"
        : "${BENCHMARK:=$(read -r -p 'Please set BENCHMARK: ' tmp && echo $tmp)}"
        : "${GITHUB_API_TOKEN:=$(read -r -p 'Please set GITHUB_API_TOKEN: ' tmp && echo $tmp)}"
        : "${CHAT_MODEL:=$(read -r -p 'Please set CHAT_MODEL: ' tmp && echo $tmp)}"
}

setup_cuda_devices() {
        local gpu_count=$(nvidia-smi -L | wc -l)
        if [ $gpu_count -eq 0 ]; then
                echo "No NVIDIA GPUs detected. Exiting."
                exit 1
        fi
        cuda_devices=$(seq -s, 0 $((gpu_count - 1)))
        echo $cuda_devices
}

bootstrap() {
        export MAKEFLAGS="-j$(nproc)"
        export CMAKE_BUILD_PARALLEL_LEVEL=$(nproc)

        echo "Updating and installing necessary packages..."
        DEBIAN_FRONTEND=noninteractive apt-get update -y && apt-get install -y screen git git-lfs

        echo "Cloning and setting up lm-evaluation-harness repository..."
        [[ ! -d "lm-evaluation-harness" ]] && git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness || echo "Directory already exists, skipping git clone."
        cd lm-evaluation-harness

        cat <<EOF >requirements.txt
-e .[openai,vllm,math,sentencepiece,zeno,hf_transfer]
pytablewriter einops protobuf tinyBenchmarks
EOF

        pip install --upgrade --no-cache-dir --prefer-binary pip setuptools wheel
        pip install -r requirements.txt --progress-bar on --use-feature=fast-deps --prefer-binary || {
                echo "Failed to install Python libraries"
                exit 1
        }
}

huggingface_login() {
        echo "Logging into Hugging Face CLI..."
        huggingface-cli login --token "$HF_TOKEN"
}

upload_results() {
        local end=$(date +%s)
        echo "Elapsed Time: $((end - start)) seconds"
        python ../main.py . $((end - start))
}

eval_function() {
        export BENCHMARKRUN=$BENCHMARK:$CHAT_MODEL
        export TASKS="tinyArc,tinyHellaswag,tinyMMLU,tinyTruthfulQA,tinyTruthfulQA_mc1,tinyWinogrande"
        local common_args="--model hf --model_args pretrained=$MODEL,dtype=auto --tasks $TASKS --device cuda:$cuda_devices --batch_size auto --output_path ./${BENCHMARK}.json --trust_remote_code"

        case "$BENCHMARKRUN" in
        "tiny:no")
                lm_eval $common_args
                ;;
        "tiny:yes")
                lm_eval $common_args --apply_chat_template --fewshot_as_multiturn
                ;;
        "tiny:both")
                lm_eval $common_args
                lm_eval $common_args --apply_chat_template --fewshot_as_multiturn
                ;;
        *)
                echo "Invalid benchmark specified"
                exit 1
                ;;
        esac
        upload_results
}

### EXECUTION ###
set -e
trap 'echo "An error occurred. Exiting..."' ERR
start=$(date +%s)

check_variables
cuda_devices=$(setup_cuda_devices)
bootstrap
huggingface_login

echo "Starting screen session..."
screen -dmL -S eval bash -c "$(declare -f eval_function); eval_function; exec bash"
echo "[$(date +%F_%T)]: Starting evaluation in screen session: eval"
wait
