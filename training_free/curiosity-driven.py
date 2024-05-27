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

@retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(6))
def get_azure(string, model):
    # model_mapping = {'gpt-4': 'YOUR_GPT4_VERSION', 'chatgpt': 'YOUR_CHATGPT_VERSION'}
    # try:
    #     client = AzureOpenAI(
    #         api_key='AZURE_KEY',
    #         api_version="AZURE_API_VERSION",
    #         azure_endpoint="AZURE_CHECKPOINT"
    #     )
    model_mapping = {'gpt-4': 'yuehuang-gpt-4', 'chatgpt': 'TrustLLM_chatgpt_1106'}  # 这里chatgpt没有换
    try:
        client = AzureOpenAI(
            # 15w
            api_key='428527aa7f804ebd866f9bb76bbe3ffe',
            # 8w
            # api_key='1f462c580d06407eb49954553ab22ff7',
            api_version="2023-12-01-preview",  # 15w
            # api_version="2023-08-01-preview",  # 8w
            azure_endpoint="https://yuehuang-15w.openai.azure.com/"
            # azure_endpoint="https://trustllm-gpt-4.openai.azure.com/"

        )
        chat_completion = client.chat.completions.create(
            model=model_mapping[model],
            messages=[
                {"role": "user", "content": string}
            ],
            temperature=0,
            top_p=1,
        )
        print(chat_completion.choices[0].message.content)
        return chat_completion.choices[0].message.content
    except:
        return None


def get_replicate(string, model):
    model_mapping = {
        'llama3-70b': 'meta/meta-llama-3-70b-instruct',
        'llama3-8b': 'meta/meta-llama-3-8b-instruct'
    }
    attempts = 0
    while attempts < 3:
        try:
            input = {
                "prompt": string,
                "temperature": 0,
                "top_p": 1,
                "max_tokens": 2500
            }
            os.environ["REPLICATE_API_TOKEN"] = "YOUR_REPLICATE_API_KEY"
            res = replicate.run(model_mapping[model], input=input)
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
    model_mapping = {
        'llama2-7b': 'meta-0519_llama/Llama-2-7b-chat-hf',
        'llama2-13b': 'meta-0519_llama/Llama-2-13b-chat-hf',
        'llama2-70b': 'meta-0519_llama/Llama-2-70b-chat-hf',
        'mistrual-7b': 'mistralai/Mistral-7B-Instruct-v0.1',
        'mixtral-8x7b': 'mistralai/Mixtral-8x7B-Instruct-v0.1'
    }
    # api_key = 'YOUR_DEEPINFRA_API_KEY'
    # base_url = 'YOUR_BASE_URL'
    api_key = 'kI3jALDHZxJPSEdRJrN7v3RUFwaVJSUg'
    base_url = 'https://api.deepinfra.com/v1/openai'
    headers = {
        'Authorization': f'Bearer {api_key}'
    }
    data = {
        'model': model_mapping[model],
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
        return "API call failed, status code: {}".format(response.status_code)


def get_claude(text):
    try:
        client = anthropic.Anthropic(
            api_key="YOUR_CLAUDE_API_KEY",
        )

        message = client.messages.create(
            model="YOUR_MODEL_VERSION",
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


def process_claude_item(item, model):
    item['raw_ans'] = get_claude(item.get("instruction"))
    item['res'] = get_claude(prompt_loader.get_curiosity_driven_prompt().replace('[question]', item['instruction']))
    item['merge_ans'] = get_claude(process_merge_query(item))
    return item


def main():
    if len(sys.argv) != 2:
        print("Usage: python training-free.py model_name")
        sys.exit(1)

    model_name = sys.argv[1]
    json_path = "../dataset/HoneSet.json"

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # map model and its process method
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

    # parallelly process data items
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        save_data = list(executor.map(func, data))

    # write back into a new json file
    with open(f'../dataset/{model_name}_HoneSet.json', 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=4)


if __name__ == "__main__":
    main()
