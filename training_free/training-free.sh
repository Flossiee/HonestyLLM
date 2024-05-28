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

case $MODEL_TYPE in
    online)
        # Execute existing scripts with the online model
        python curiosity-driven.py $MODEL_NAME
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
        ;;

    local)
        LOCAL_MODEL_PATH=""
        echo "Enter the local model path (press Enter to use default):"
        read LOCAL_MODEL_PATH
        LOCAL_MODEL_PATH="${LOCAL_MODEL_PATH:-None}"

        local_generation() {
            echo "Executing local generation with model name: $MODEL_NAME and path: $LOCAL_MODEL_PATH"
            # Add your local generation code here
        }

        local_generation
        ;;

    *)
        echo "Invalid model type: $MODEL_TYPE"
        usage
        ;;
esac

echo "Method is executed successfully."
