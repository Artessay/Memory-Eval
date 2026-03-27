import time
from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1",
    timeout=3600
)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "How are you feeling today?"
            }
        ]
    }
]

start = time.time()
response = client.chat.completions.create(
    # model="qwen3.5-27b",
    model="qwen3.5-122b-a10b",
    messages=messages,
    max_tokens=2048,
    extra_body={
        "chat_template_kwargs": {"enable_thinking": False},
    },
)
print(f"Response costs: {time.time() - start:.2f}s")
print(f"Generated text: {response.choices[0].message.content}")