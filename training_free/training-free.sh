#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 model_name"
    exit 1
fi

MODEL_NAME=$1

# 执行Python脚本，传递模型名称
python curiosity-driven.py $MODEL_NAME
