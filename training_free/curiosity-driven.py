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


def load_config():
    with open('config.yaml', 'r') as file:
        return yaml.safe_load(file)


config = load_config()


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


def process_local(model_name):
    print("local")


def main():
    if len(sys.argv) != 3:
        print("Usage: python curiosity-driven.py model_type model_name")
        sys.exit(1)

    model_type = sys.argv[1]
    model_name = sys.argv[2]

    if model_type == 'online':
        process_online(model_name)
    elif model_type == 'local':
        process_local(model_name)
    else:
        print("Invalid model type")
        sys.exit(2)


if __name__ == "__main__":
    main()
