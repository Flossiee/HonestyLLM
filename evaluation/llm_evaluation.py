from openai import AzureOpenAI
import json
from tenacity import retry, wait_random_exponential, stop_after_attempt
import concurrent.futures
import sys

@retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(6))
def call_azure(string, model):
    model_mapping = {'gpt-4': 'YOUR_gpt4_VERSION'}
    client = AzureOpenAI(
        api_key='AZURE_API_KEY',
        api_version="AZURE_API_VERSION",
        azure_endpoint="AZURE_ENDPOINT"
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


def eval_honest_rate(el, ans_type):
    with open('prompt_template/eval_template.json', 'r') as f:
        prompt_template = json.load(f)

    eval_honesty_template = next(item for item in prompt_template if item['key'] == "eval_honesty")
    action = eval_honesty_template['action']

    if el.get(f"eval_{ans_type}") is None:
        res = el.get(f"{ans_type}")
        prompt = eval_honesty_template['prompt']
        prompt_processed = prompt.replace('[question]', el['instruction']).replace('[measure]', action[el['category']]).replace(
            '[answer]', res).replace('[category]', el['category'])
        el[f'eval_{ans_type}'] = call_azure(prompt_processed, 'gpt-4')
    return el


def scoring_prompt(item, ans_type):
    question = item.get("instruction")
    answer = item.get(ans_type)
    category = item.get("category")

    # Load category mapping
    with open('prompt_template/cate_define.json', 'r') as f:
        cate_define = json.load(f)
    category_definition = cate_define["definition"].get(category, "No definition found for this category")

    # Load prompt_template and find the eval_scoring template
    with open('prompt_template/eval_template.json', 'r') as f:
        prompt_template = json.load(f)
    eval_scoring_template = next(item for item in prompt_template if item['key'] == "eval_scoring")

    prompt = eval_scoring_template['prompt']
    prompt_processed = prompt.replace('[category]', category)\
                             .replace('[category_definition]', category_definition)\
                             .replace('[question]', question)\
                             .replace('[answer]', answer)
    return prompt_processed


def score_extraction(item, ans_type):
    with open('prompt_template/eval_template.json', 'r') as f:
        prompt_template = json.load(f)

    # Find the correct template based on the 'key'
    scoring_extraction_template = next((template for template in prompt_template if template['key'] == "scoring_extraction"), None)
    scoring_sentence = item.get(ans_type)

    if scoring_extraction_template:
        string = scoring_extraction_template['prompt'].replace("[scoring_sentence]", scoring_sentence)
    else:
        string = "Template for scoring extraction not found."

    return string


def eval_scoring(item, ans_type):
    # scoring process
    if item.get(f"scoring_{ans_type}") is None and f"{ans_type}" in item:
        item[f"scoring_{ans_type}"] = call_azure(scoring_prompt(item, f"{ans_type}"), 'gpt-4')

    # extract overall score
    if f"scoring_{ans_type}" in item and item.get(f"scoring_{ans_type}_overall") is None:
        item[f"scoring_{ans_type}_overall"] = call_azure(score_extraction(item, f"scoring_{ans_type}"), 'gpt-4')
    return item


def process_evaluation(item):
    item = eval_honest_rate(item, "raw_ans")
    item = eval_scoring(item, "raw_ans")
    item = eval_honest_rate(item, "merge_ans")
    item = eval_scoring(item, "merge_ans")
    return item


def main():
    if len(sys.argv) != 2:
        print("Usage: python training-free.py model_name")
        sys.exit(1)
    model_name = sys.argv[1]

    with open(f'../dataset/{model_name}_HoneSet.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        save_data = list(executor.map(process_evaluation, data))

    with open(f'../dataset/{model_name}_HoneSet_eval.json', 'w') as f:
        json.dump(save_data, f, indent=4)


if __name__ == "__main__":
    main()