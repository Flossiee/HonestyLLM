from openai import AzureOpenAI
import json
from tenacity import retry, wait_random_exponential, stop_after_attempt
import concurrent.futures


@retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(6))
def get_res(string, model):
    model_mapping = {'gpt-4': 'YOUR_GPT4_VERSION', 'chatgpt': 'YOUR_ChatGPT_VERSION'}
    client = AzureOpenAI(
            api_key='YOUR_AZURE_KEY',
            api_version="YOUR_API_VERSION",
            azure_endpoint="YOUR_AZURE_CHECKPOINT"
        )
    try:
        chat_completion = client.chat.completions.create(
            model=model_mapping[model],
            messages=[
                {"role": "user", "content": string}
            ],
            temperature=0
        )
        print(chat_completion.choices[0].message.content)
        return chat_completion.choices[0].message.content
    except:
        return None

def process_item(el):
    if el.get("eval_res_dpo") is None:
        # prompt = prompt_template['prompt'].replace('[question]', el['question']).replace('[measure]', prompt_template['action'][el['category']]).replace('[answer]', el['raw_ans']).replace('[category]', el['category'])
        dpo_res = el.get('dpo_res')
        prompt = prompt_template['prompt'].replace('[question]', el['instruction']).replace('[measure]',
                                                                                         prompt_template['action'][
                                                                                             el['category']]).replace(
            '[answer]', dpo_res).replace('[category]', el['category'])
        el['eval_res_dpo'] = get_res(prompt, 'gpt-4')
    return el


with open('YOUR_JSON_FILE_PATH', 'r') as f:
    prompt_template = json.load(f)

with open('../curiosity-driven/evaluation/finetune_data/course_learning/dataset/DPO_multistages/0520_threshold_mistral/7_all/extra1.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
    save_data = list(executor.map(process_item, data))

with open('YOUR_JSON_FILE_PATH', 'w') as f:
    json.dump(save_data, f, indent=4)
 
