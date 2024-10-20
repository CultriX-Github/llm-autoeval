function install_dependencies() {
	echo "Updating package lists..."
	DEBIAN_FRONTEND=noninteractive apt-get update -y &&
		DEBIAN_FRONTEND=noninteractive apt-get install -y vim git-lfs
	echo "Installing Python libraries..."
 	pip install -U requests accelerate sentencepiece pytablewriter einops protobuf accelerate flash-attention --progress-bar on --use-feature=fast-deps --use-feature=fast-mirror -j $(nproc) || {
		echo "Failed to install Python libraries"
		exit 1
	}
}

function setup_cuda_devices() {
	local gpu_count=$(nvidia-smi -L | wc -l)
	if [ $gpu_count -eq 0 ]; then
		echo "No NVIDIA GPUs detected. Exiting."
		exit 1
	fi
	local cuda_devices=""
	for ((i = 0; i < gpu_count; i++)); do
		[ $i -gt 0 ] && cuda_devices+=","
		cuda_devices+="$i"
	done
	echo $cuda_devices
}

function upload_results() {
	end=$(date +%s)
	echo "Elapsed Time: $((end - start)) seconds"
	python ../main.py . $((end - start))
}

function run_benchmark() {
	local benchmark=$1
	local model=$2
	local trust_remote_code=$3
	local cuda_devices=$4
	export BENCHMARK=$benchmark
	export MODEL=$model
	export TRUST_REMOTE_CODE=True

	if [ "$benchmark" == "nous" ]; then
		git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
		cd lm-evaluation-harness || exit 1
		pip install --upgrade pip --progress-bar on --use-feature=fast-deps --use-feature=fast-mirror -j $(nproc) || exit 1
                pip install -e ".[openai,vllm]" --progress-bar on --use-feature=fast-deps --use-feature=fast-mirror -j $(nproc) || exit 1

		# Several benchmarks are run with different tasks, each writing results to a JSON file.
		benchmark="agieval"
		TRUST_REMOTE_CODE=True lm-eval \
			--model hf \
			--model_args pretrained=$MODEL,trust_remote_code=True \
			--tasks agieval_aqua_rat,agieval_logiqa_en,agieval_lsat_ar,agieval_lsat_lr,agieval_lsat_rc,agieval_sat_en,agieval_sat_en_without_passage,agieval_sat_math \
			--device cuda:$cuda_devices \
			--batch_size auto \
			--output_path ./${benchmark}.json \
                        --trust_remote_code

		benchmark="gpt4all"
		TRUST_REMOTE_CODE=True lm-eval \
			--model hf \
			--model_args pretrained=$MODEL,trust_remote_code=True \
			--tasks hellaswag,openbookqa,winogrande,arc_easy,arc_challenge,boolq,piqa \
			--device cuda:$cuda_devices \
			--batch_size auto \
			--output_path ./${benchmark}.json \
                        --trust_remote_code

		benchmark="truthfulqa"
		TRUST_REMOTE_CODE=True lm-eval \
			--model hf \
			--model_args pretrained=$MODEL,trust_remote_code=True \
			--tasks truthfulqa_mc \
			--device cuda:$cuda_devices \
			--batch_size auto \
			--output_path ./${benchmark}.json \
                        --trust_remote_code

		benchmark="bigbench"
		TRUST_REMOTE_CODE=True lm-eval \
			--model hf \
			--model_args pretrained=$MODEL,trust_remote_code=True \
			--tasks bigbench_causal_judgement,bigbench_date_understanding,bigbench_disambiguation_qa,bigbench_geometric_shapes,bigbench_logical_deduction_five_objects,bigbench_logical_deduction_seven_objects,bigbench_logical_deduction_three_objects,bigbench_movie_recommendation,bigbench_navigate,bigbench_reasoning_about_colored_objects,bigbench_ruin_names,bigbench_salient_translation_error_detection,bigbench_snarks,bigbench_sports_understanding,bigbench_temporal_sequences,bigbench_tracking_shuffled_objects_five_objects,bigbench_tracking_shuffled_objects_seven_objects,bigbench_tracking_shuffled_objects_three_objects \
			--device cuda:$cuda_devices \
			--batch_size auto \
			--output_path ./${benchmark}.json \
                        --trust_remote_code

		upload_results . $end

	elif [ "$benchmark" == "openllm" ]; then
		git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
		cd lm-evaluation-harness || exit 1
		pip install --upgrade pip --progress-bar on --use-feature=fast-deps --use-feature=fast-mirror -j $(nproc) || exit 1
                pip install -e ".[openai,vllm]" --progress-bar on --use-feature=fast-deps --use-feature=fast-mirror -j $(nproc) || exit 1
		pip install -U requests accelerate sentencepiece pytablewriter einops protobuf accelerate || exit 1

		# Several benchmarks are run with different tasks, each writing results to a JSON file.
		benchmark="arc"
		lm_eval --model vllm \
			--model_args pretrained=${MODEL},dtype=auto,gpu_memory_utilization=0.8,trust_remote_code=True \
			--tasks arc_challenge \
			--num_fewshot 25 \
			--batch_size auto \
			--output_path ./${benchmark}.json \
                        --trust_remote_code

		benchmark="hellaswag"
		lm_eval --model vllm \
			--model_args pretrained=${MODEL},dtype=auto,gpu_memory_utilization=0.8,trust_remote_code=True \
			--tasks hellaswag \
			--num_fewshot 10 \
			--batch_size auto \
			--output_path ./${benchmark}.json \
                        --trust_remote_code

		benchmark="truthfulqa"
		lm_eval --model vllm \
			--model_args pretrained=${MODEL},dtype=auto,gpu_memory_utilization=0.8,trust_remote_code=True \
			--tasks truthfulqa \
			--num_fewshot 0 \
			--batch_size auto \
			--output_path ./${benchmark}.json \
                        --trust_remote_code

		benchmark="winogrande"
		lm_eval --model vllm \
			--model_args pretrained=${MODEL},dtype=auto,gpu_memory_utilization=0.8,trust_remote_code=True \
			--tasks winogrande \
			--num_fewshot 5 \
			--batch_size auto \
			--output_path ./${benchmark}.json \
                        --trust_remote_code

		benchmark="gsm8k"
		lm_eval --model vllm \
			--model_args pretrained=${MODEL},dtype=auto,gpu_memory_utilization=0.8,trust_remote_code=True \
			--tasks gsm8k \
			--num_fewshot 5 \
			--batch_size auto \
			--output_path ./${benchmark}.json \
                        --trust_remote_code

		upload_results . $end

	else
		echo "Invalid benchmark specified"
		return
	fi
}

# Main script starts here
echo "Benchmarking Model Script"
read -p "The model you want to benchmark: " MODEL
export MODEL=$MODEL
read -p "Your api token for uploading the results as a gist: " GITHUB_API_TOKEN
export GITHUB_API_TOKEN=$GITHUB_API_TOKEN
export TRUST_REMOTE_CODE=True
read -p "Do not delete Pod when done? (True = Keep |False = Delete): " DEBUG
while [[ $DEBUG != "True" && $DEBUG != "False" ]]; do
	echo "Invalid value. Please enter 'True' or 'False'."
	read -p "Do not delete Pod when done? (True = Keep|False = Delete): " DEBUG
done

read -p "Enter which benchmark you want to run (nous/openllm): " BENCHMARK
while [[ $BENCHMARK != "nous" && $BENCHMARK != "openllm" ]]; do
	echo "Invalid benchmark ($BENCHMARK). Please enter 'nous' or 'openllm'."
	read -p "Enter benchmark: " BENCHMARK
done

start=$(date +%s)
cuda_devices=$(setup_cuda_devices)

install_dependencies
run_benchmark $BENCHMARK $MODEL $TRUST_REMOTE_CODE $cuda_devices

if [ "$DEBUG" == "False" ]; then
	runpodctl remove pod $RUNPOD_POD_ID
fi

echo "Benchmark completed successfully."
