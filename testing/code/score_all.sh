#!/bin/bash
MODEL_NAMES=("Phi4"
    "Llama_3.3_70B"
	"GPT_4o"
	"Ours_1B"
	"Ours_3B"
	"Llama_3.2_3B_Inst"
	"Llama_3.2_1B_Inst"
    "Aya_23_35B"
	"Sonnet"
    "Gemini_1.5_Pro"
    "o1-preview")

# Loop through each model name and execute the Python script
for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    echo "--- Scoring model: $MODEL_NAME"
    python scoring.py "$MODEL_NAME" --cot True

    # Check if the last command failed
    if [ $? -ne 0 ]; then
        echo "--- Scoring Failed for model: $MODEL_NAME"
    fi
done
