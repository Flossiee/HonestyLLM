from openai import AzureOpenAI
import json
import time
import os
import replicate
import requests
from tenacity import retry, wait_random_exponential, stop_after_attempt
import concurrent.futures
import anthropic
import sys
import prompt_loader
from functools import partial
import yaml
import argparse
import torch
import requests
from fastchat.model import load_model, get_conversation_template, add_model_args
from dotenv import load_dotenv


def load_config():
    with open('config.yaml', 'r') as file:
        return yaml.safe_load(file)


config = load_config()




def prompt2conversation(prompt, model_path):
    msg = prompt
    conv = get_conversation_template(model_path)
    conv.set_system_message('')
    conv.append_message(conv.roles[0], msg)
    conv.append_message(conv.roles[1], None)
    conversation = conv.get_prompt()
    return conversation


def generate_output(model, tokenizer, prompt, device, max_new_tokens, temperature, model_path):
    inputs = tokenizer([prompt], return_tensors="pt")
    prompt = prompt2conversation(prompt, model_path=model_path)
    inputs = tokenizer([prompt])
    inputs = {k: torch.tensor(v).to(device) for k, v in inputs.items()}
    if 'token_type_ids' in inputs:
        del inputs['token_type_ids']
    print(type(temperature))
    output_ids = model.generate(
        **inputs,
        do_sample=True if temperature > 1e-5 else False,
        temperature=temperature,
        max_new_tokens=max_new_tokens
    )
    if model.config.is_encoder_decoder:
        output_ids = output_ids[0]
    else:
        output_ids = output_ids[0][len(inputs["input_ids"][0]):]
    generated_text = tokenizer.decode(
        output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
    )
    print(generated_text)

    return generated_text


@retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(6))
def get_azure(string, model):
    client = AzureOpenAI(
        api_key=config['api_config']['azure']['api_key'],
        api_version=config['api_config']['azure']['api_version'],
        azure_endpoint=config['api_config']['azure']['azure_endpoint']
    )
    try:
        chat_completion = client.chat.completions.create(
            model=config['api_config']['azure']['model_mapping'][model],
            messages=[
                {"role": "user", "content": string}
            ],
            temperature=0,
            top_p=1,
        )
        print(chat_completion.choices[0].message.content)
        return chat_completion.choices[0].message.content
    except Exception as e:
        print(f"Failed due to {e}")
        return None


def get_replicate(string, model):
    attempts = 0
    while attempts < 3:
        try:
            input = {
                "prompt": string,
                "temperature": 0,
                "top_p": 1,
                "max_tokens": 2500
            }
            os.environ["REPLICATE_API_TOKEN"] = config['api_config']['replicate']['api_key']
            res = replicate.run(config['api_config']['replicate']['model_mapping'][model], input=input)
            res = "".join(res)
            print(res)
            return res
        except Exception as e:
            print(f"Attempt {attempts + 1} failed: {e}")
            attempts += 1
            time.sleep(10)
    print("Failed after 3 attempts.")
    return None


def get_deepinfra(string, model):
    api_key = config['api_config']['deepinfra']['api_key']
    base_url = config['api_config']['deepinfra']['base_url']
    headers = {
        'Authorization': f'Bearer {api_key}'
    }
    data = {
        'model': config['api_config']['deepinfra']['model_mapping'][model],
        'messages': [{"role": "user", "content": string}],
        'max_tokens': 2500,
        'temperature': 0,
        'top_p': 1,
    }

    response = requests.post(f"{base_url}/chat/completions", json=data, headers=headers)
    time.sleep(1)

    if response.status_code == 200:
        response_json = response.json()
        content = response_json['choices'][0]['message']['content']
        print(content)
        return content
    else:
        print(response.json().get('error', 'No error information available.'))
        return None


def get_claude(text):
    try:
        client = anthropic.Anthropic(
            api_key=config['api_config']['anthropic']['api_key'],
        )
        message = client.messages.create(
            model=config['api_config']['anthropic']['api_version'],
            max_tokens=2500,
            temperature=0,
            top_p=1,
            messages=[
                {"role": "user", "content": text}
            ]
        )
        print(message.content[0].text)
        return message.content[0].text
    except Exception as e:
        print(f"Failed to get response: {e}")
        return None


def process_merge_query(item):
    string = prompt_loader.get_merge_prompt().replace('[question]', item['instruction']).replace('answer', item['raw_ans']).replace('[confusion]', item['res'])
    return string


def process_azure_item(item, model):
    item['raw_ans'] = get_azure(item.get("instruction"), model)
    item['res'] = get_azure(prompt_loader.get_curiosity_driven_prompt().replace('[question]', item['instruction']), model)
    item['merge_ans'] = get_azure(process_merge_query(item), model)
    return item


def process_replicate_item(item, model):
    item['raw_ans'] = get_replicate(item.get("instruction"), model)
    item['res'] = get_replicate(prompt_loader.get_curiosity_driven_prompt().replace('[question]', item['instruction']), model)
    item['merge_ans'] = get_replicate(process_merge_query(item), model)
    return item


def process_deepinfra_item(item, model):
    item['raw_ans'] = get_deepinfra(item.get("instruction"), model)
    item['res'] = get_deepinfra(prompt_loader.get_curiosity_driven_prompt().replace('[question]', item['instruction']), model)
    item['merge_ans'] = get_deepinfra(process_merge_query(item), model)
    return item


def process_claude_item(item):
    item['raw_ans'] = get_claude(item.get("instruction"))
    item['res'] = get_claude(prompt_loader.get_curiosity_driven_prompt().replace('[question]', item['instruction']))
    item['merge_ans'] = get_claude(process_merge_query(item))
    return item


def process_online(model_name):
    print("Processing online with model name:", model_name)
    json_path = "../dataset/HoneSet.json"

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if model_name in ['gpt-4', 'chatgpt']:
        process_function = process_azure_item
    elif model_name in ['llama2-7b', 'llama2-13b', 'llama2-70b', 'mistrual-7b', 'mixtral-8x7b']:
        process_function = process_deepinfra_item
    elif model_name in ['llama3-70b', 'llama3-8b']:
        process_function = process_replicate_item
    elif model_name == 'claude':
        process_function = process_claude_item
    else:
        print("Error: Model not supported")
        sys.exit(2)

    func = partial(process_function, model=model_name)

    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        save_data = list(executor.map(func, data))

    with open(f'../dataset/{model_name}_HoneSet.json', 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=4)

    print("Online processing completed for model:", model_name)




def process_local(args):
    model_path = args.model_path
    print("Processing online with model name:", model_path)


    model, tokenizer = load_model(
        args.model_path,
        device=args.device,
        num_gpus=args.num_gpus,
        max_gpu_memory=args.max_gpu_memory,
        load_8bit=args.load_8bit,
        cpu_offloading=args.cpu_offloading,
        revision=args.revision,
        debug=args.debug,
    )

    json_path = "../dataset/HoneSet.json"

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    save_data = []
    for item in data:
        item['raw_ans'] = generate_output(model, tokenizer, item.get("instruction"), args.device, args.max_length, args.temperature, model_path)
        item['res'] = generate_output(model, tokenizer, prompt_loader.get_curiosity_driven_prompt().replace('[question]', item['instruction']), args.device, args.max_length, args.temperature, model_path)
        item['merge_ans'] = generate_output(model, tokenizer, process_merge_query(item), args.device, args.max_length, args.temperature, model_path)

        save_data.append(item)

    with open(f'../dataset/{model_path}_HoneSet.json', 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=4)

    print("Local processing completed for model:", model_path)


def main():
    parser = argparse.ArgumentParser()
    add_model_args(parser)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--model_path", type=str, default='')
    parser.add_argument("--filename", type=str, default='')
    parser.add_argument("--test_type", type=str, default='plugin')
    parser.add_argument("--online", type=str, default='False')
    args = parser.parse_args()

    if args.online:
        process_online(args.model_path)
    else:
        load_dotenv()
        process_local(args)


if __name__ == "__main__":
    main()
