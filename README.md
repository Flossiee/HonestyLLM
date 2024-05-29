# HonestyLLM
This repository contains scripts and configurations for our paper "The Best of Both Worlds: Toward an Honest and
Helpful Large Language Mode".

## Table of Contents
- 🕰️ [Introduction](#introduction)
- 📑 [HoneSet](#Honeset)
- 💡 [Training-free enhancement](#training-free-enhancement)
- ✨ [Improvement through fine-tuning](#improvement-through-fine-tuning)
- 🔗 [Citation](#citation)

## Introduction
This repository focuses on enhancing the honesty and helpfulness of Large Language Models (LLMs) in real-world applications. Our work introduces novel methodologies and datasets to evaluate and improve the reliability of LLMs.
<div align="center"><img src="image/intro.png" width="75%"></div>

### Components
- HoneSet Dataset: A novel dataset containing 930 queries across six categories, crafted to evaluate the honesty of LLMs.
- Two Enhancement Approaches:

  - Training-Free Enhancement: Leverages curiosity-driven prompting to help LLMs express uncertainty and refine their responses.
  - Fine-Tuning-Based Improvement: Utilizes a curriculum learning inspired two-stage process to teach LLMs to differentiate between honest and dishonest responses, followed by a phase to boost their helpfulness.
<div align="center"><img src="image/architecture.png"></div>

## HoneSet
- Honeset is located in `dataset/HoneSet.json` which contains 930 data items across 6 categories as follows:

| Category                                         |
|--------------------------------------------------|
| Latest Information with External Services        |
| User Input Not Enough Or With Wrong Information  |
| Self Identity Cognition                          |
| Modality Mismatch                                |
| Professional Capability in Specific Domain       |
| Interactivity Sensory Processing                 |


## Training-free Enhancement
### Configuration Steps
- **Edit Configuration:**
   - Navigate to the `training_free/config.yaml` file.
   - Replace your API key and any other necessary configurations within this file.
- **Script Location:**
   - Ensure that you are in the directory containing the `training_free.sh` script.
- **Set Model Parameters:**
  - `model_type` can be `online` or `local`
  - `model_name` can be as follows:

    | Model_name input | Model        |
    |------------------|--------------|
    | gpt-4            | GPT-4        |
    | chatgpt          | ChatGPT      |
    | claude           | Claude3-Opus |
    | llama3-70b       | Llama3-70b   |
    | llama3-8b        | Llama3-8b    |
    | mixtral-8x7b     | Mixtral-8x7b |
    | llama2-7b        | Llama2-7b    |
    | llama2-13b       | Llama2-13b   |
    | llama2-70b       | Llama2-70b   |
    | mistral-7b       | Mistral-7b   |

```bash
./run_model.sh [model_type] [model_name]
```

## Improvement through fine-tuning
### Overview

This repository contains scripts and configurations for fine-tuning, merging, and running inference with Llama models using `LLaMA-Factory`.
### Requirements

- `LLaMA-Factory` installed
- Install LLaMA-Factory

```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e .[torch,metrics]
```

### Run Fine-tuning
### Fine-Tuning

To fine-tune the model, use the following command:

```bash
llamafactory-cli train train_config.yaml
```

Replace `train_config.yaml` with one setting in `finetuning/*.yaml`

### Merging Stage 1 Model

After fine-tuning, merge the stage 1 model using:

```bash
llamafactory-cli export merge_lora_dpo.yaml
```

Make sure `merge_lora_dpo.yaml` is configured with the appropriate merging parameters.

### Running Model Inference

To run model inference, execute:

```bash
llamafactory-cli api model_inference.yaml
```

Ensure `model_inference.yaml` contains the correct inference settings.

## Updates
## Citation