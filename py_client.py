from openai import OpenAI

# Initializing client pointing to your vLLM server
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key = "your-secret-key-here" # If you set --api-key
)

# Using the chat completions API
response = client.chat.completions.create(
    model="qwen/qwen3-4B",
    messages=[
        {
         "role": "system", 
         "content": "You are a helpful assitant."
        },
        {
         "role": "user",
         "content": "Expalin LLMs transformer architecture in simple terms.",
        }
    ],
    max_tokens=1024,
    temperature=0.7,
    top_p=0.9,
)

print(response.choices[0].message.content)