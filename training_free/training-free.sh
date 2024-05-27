#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 model_name"
    exit 1
fi

MODEL_NAME=$1

python curiosity-driven.py $MODEL_NAME
if [ $? -ne 0 ]; then
    echo "Curiosity-driven.py failed, stopping execution."
    exit 1
fi

cd ../evaluation || exit
python llm_evaluation.py $MODEL_NAME
if [ $? -ne 0 ]; then
    echo "Execution of llm_evaluation.py failed."
    exit 1
fi

echo "Training-free method is executed successfully."
