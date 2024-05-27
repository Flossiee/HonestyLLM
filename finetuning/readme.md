# README

## Overview

This repository contains scripts and configurations for fine-tuning, merging, and running inference with Llama models using `LLaMA-Factory`.

## Requirements

- `LLaMA-Factory` installed


- Install LLaMA-Factory

```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e .[torch,metrics]
```

## Run Fine-tuning


## Fine-Tuning

To fine-tune the model, use the following command:

```bash
llamafactory-cli train train_config.yaml
```

Replace `train_config.yaml` with one setting in `finetuning/*.yaml`

## Merging Stage 1 Model

After fine-tuning, merge the stage 1 model using:

```bash
llamafactory-cli export merge_lora_dpo.yaml
```

Make sure `merge_lora_dpo.yaml` is configured with the appropriate merging parameters.

## Running Model Inference

To run model inference, execute:

```bash
llamafactory-cli api model_inference.yaml
```

Ensure `model_inference.yaml` contains the correct inference settings.
