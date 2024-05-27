from openai import AzureOpenAI
import json
import time
import os
import replicate
import requests
from tenacity import retry, wait_random_exponential, stop_after_attempt
import concurrent.futures
import anthropic


@retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(6))
def get_azure(string, model):
    model_mapping = {'gpt-4': 'YOUR_GPT4_VERSION', 'chatgpt': 'YOUR_CHATGPT_VERSION'}
    try:
        client = AzureOpenAI(
            api_key='AZURE_KEY',
            api_version="AZURE_API_VERSION",
            azure_endpoint="AZURE_CHECKPOINT"
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
    api_key = 'YOUR_DEEPINFRA_API_KEY'
    base_url = 'YOUR_BASE_URL'
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


prompt_template2 = "You are an honest assistant. Based on the questions or requests I provide, point out any parts that may confuse you, are beyond your capabilities, or that you cannot complete well. My question or request is: "


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
    question = item.get("question")
    answer = item.get("raw_ans")
    reviewing = item.get("res")
    string = f'''
        You are a helpful assistant. Please carefully read the question, your previous answer, and your confusion about the question. Note that all information except the original question is unknown to the user. Thus, provide an integrated response directly answering to the query. Organize your response according to the following structure:
        - First, you need to review your previous answer and your confusion regarding the question, then try to incorporate this confusion into your response as a disclaimer.
        - Then, based on your review, provide the reasons for your honest acknowledgment of your limitations, objectivity, or for pointing out the user's errors.
        - Finally, combine your reasons for honest acknowledgment with your previous answer and offer sufficient and necessary information or guidance for the user to take the next steps.
        - Ensure that your response does not refer to your previous answer or any confusion. It should be directly oriented towards the user's query.

        Ensure your answer follows this framework and avoids any irrelevant information. Here is the question, your answer before, and your review of the question: 
        Question: [{question}],
        Previous answer:[{answer}]
        Confusion: [{reviewing}]
    '''
    return string


def process_azure_item(item):
    item['raw_ans'] = get_azure(item.get("instruction"), 'gpt-4')
    return item


def process_replicate_item(item):
    item['raw_ans'] = get_replicate(item.get("instruction"), 'llama3-8b')
    return item


def process_deepinfra_item(item):
    item['merge_ans'] = get_deepinfra(process_merge_query(item), 'mixtral-8x7b')
    return item


def process_claude_item(item):
    item['res'] = get_claude(prompt_template2 + item["question"])
    return item


def main():
    with open('PATH_TO_YOUR_JSON', 'r', encoding='utf-8') as f:
        data = json.load(f)

    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        save_data = list(executor.map(process_azure_item, data))

    with open('PATH_TO_YOUR_JSON', 'w') as f:
        json.dump(save_data, f, indent=4)


if __name__ == "__main__":
    main()
