# HonestyLLM
This repository contains scripts and configurations for our paper "The Best of Both Worlds: Toward an Honest and
Helpful Large Language Mode". [![Paper](https://img.shields.io/badge/Paper-%F0%9F%8E%93-lightblue?style=flat-square)](https://arxiv.org/abs/2406.00380)

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
### Requirements
- Python 3.x 
- Libraries: openai, replicate, requests, tenacity, concurrent.futures, anthropic, torch, yaml, argparse, dotenv
- API keys and model mappings for Azure, replicate, deepinfra and other services.
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

### Command Line Arguments
- Online Mode
When running the script in `online` mode, use the following parameters:
```bash
./training_free.sh online [model_name]
```
- local Mode
When running the script in `local` mode, you can specify additional parameters:
- `--temperature` (default = 0): Controls the randomness of the response generation. Higher values produce more varied outputs.
- `--repetition_penalty` (default = 1.0): Penalizes repetition to encourage more diverse responses.
- `--num_gpus` (default = 1): Specifies the number of GPUs to use.
- `--max_length` (default = 2048): Limits the number of tokens in the response.
- `--debug` (default = false): Enables debug mode for more verbose output.
- `--model_path` (default = ''): The path to the model files, necessary in local mode.
- `--filename` (default = ''): Specifies the output filename.
- `--test_type` (default = 'plugin'): Sets the type of testing or processing.
- `--online` (default = 'False'): Indicates whether to run the model in online mode.
```bash
./training_free.sh local [model_name] --temperature [value] --repetition_penalty [value] --num_gpus [value] --max_length [value] --debug --model_path [path] --filename [filename] --test_type [type] 
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

## Citation

```
@misc{gao2024best,
      title={The Best of Both Worlds: Toward an Honest and Helpful Large Language Model}, 
      author={Chujie Gao and Qihui Zhang and Dongping Chen and Yue Huang and Siyuan Wu and Zhengyan Fu and Yao Wan and Xiangliang Zhang and Lichao Sun},
      year={2024},
      eprint={2406.00380},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
