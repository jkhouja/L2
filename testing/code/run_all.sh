#!/bin/bash
OPEN=("Phi4" 
	"Llama_3.3_70B"
	"Ours_1B"
	"Llama_3.2_3B_Inst"
	"Llama_3.2_1B_Inst")

MODEL_NAMES=("GPT_4o"
	"Command_R+"
	"Sonnet"
    	"Gemini_1.5_Pro"
    	"o1")


# Loop through each model name and execute the Python script
for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    echo "--- Running benchmark for model: $MODEL_NAME"
    if [ -z "$2" ]; then
        echo "No argument supplied. Running on all data on GPU: $1"
        CUDA_VISIBLE_DEVICES=$1 python benchmark_model.py "$MODEL_NAME"
    else
        echo "Running on $1 examples using GPU: $2"
        CUDA_VISIBLE_DEVICES=$2 python benchmark_model.py "$MODEL_NAME"  -questions_limit $1
    fi

    # Check if the last command failed
    if [ $? -ne 0 ]; then
        echo "--- Running Failed for model: $MODEL_NAME"
    fi
done
