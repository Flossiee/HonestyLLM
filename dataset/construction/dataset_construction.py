import openai
from tenacity import retry, wait_random_exponential, stop_after_attempt
import csv

openai.api_type = "YOUR_API_TYPE"
openai.api_base = "YOUR_API_BASE_URL"
openai.api_version = "YOUR_API_VERSION"
openai.api_key = 'YOUR_API_KEY'


@retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(6))
def get_res(string):
    chat_completion = openai.ChatCompletion.create(
        engine="YOUR_ENGINE_ID",
        messages=[
            {"role": "user",
             "content": string}
        ],
        temperature=0,
    )
    print(chat_completion.choices[0].message.content)
    return chat_completion.choices[0].message.content


def process_item(prompt):
    try:
        return get_res(prompt)
    except Exception as e:
        print(f"Error processing item")
        print(f"Exception: {e}")
        return None


def main():
    prompt = "PROMPT_FOR_EACH_CATEGORY"
    with open('PATH/TO/YOUR/FILE', 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        for _ in range(1):
            response = process_item(prompt)
            if response:
                writer.writerow([response])

    print("done!(csv saved)")


if __name__ == "__main__":
    main()
