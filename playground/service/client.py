import os
import time
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    # base_url="http://localhost:8000/v1",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
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
    # model="qwen3.5-122b-a10b",
    model="qwen3.5-flash",
    messages=messages,
    max_tokens=2048,
    extra_body={
        "enable_thinking": False,
        "chat_template_kwargs": {"enable_thinking": False},
    },
)
print(f"Response costs: {time.time() - start:.2f}s")
print(f"Generated text: {response.choices[0].message.content}")