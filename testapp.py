import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
NVIDIA_API_KEY=os.getenv("NVIDIA_API_KEY")
client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key =  NVIDIA_API_KEY
)

completion = client.chat.completions.create(
  model="meta/llama-3.3-70b-instruct",
  messages=[{"role":"user","content":"Provide me an article on Machine Learning"}],
  temperature=0.2,
  top_p=0.7,
  max_tokens=1024,
  stream=False
)

print(completion.choices[0].message)
