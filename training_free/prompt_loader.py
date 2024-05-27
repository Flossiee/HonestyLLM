import json


def load_prompts(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
        prompt = {item["key"]: item["prompt"] for item in data}
    return prompt


def get_curiosity_driven_prompt():
    prompts = load_prompts('../evaluation/prompt_template/training-free_prompt.json')
    curiosity_prompt = prompts['curiosity_driven_prompt']
    print(curiosity_prompt)
    return curiosity_prompt


def get_merge_prompt():
    prompts = load_prompts('../evaluation/prompt_template/training-free_prompt.json')
    merge_prompt = prompts['merge_prompt']
    print(merge_prompt)
    return merge_prompt
