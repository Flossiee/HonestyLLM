#!/bin/bash

usage() {
    echo "Usage: $0 model_type model_name"
    echo "model_type: 'online' or 'local'"
    exit 1
}

if [ "$#" -ne 2 ]; then
    usage
fi

MODEL_TYPE=$1
MODEL_NAME=$2

python curiosity-driven.py $MODEL_TYPE $MODEL_NAME
if [ $? -ne 0 ]; then
    echo "curiosity-driven.py failed, stopping execution."
    exit 1
fi

cd ../evaluation || exit
python llm_evaluation.py $MODEL_NAME
if [ $? -ne 0 ]; then
    echo "Execution of llm_evaluation.py failed."
    exit 1
fi

echo "Method is executed successfully."
