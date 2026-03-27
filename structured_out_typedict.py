from langchain_ollama import ChatOllama
from dotenv import load_dotenv
from typing import TypedDict
import os
load_dotenv()

model = ChatOllama(
    model = "qwen2.5:14b-instruct",
    base_url = os.getenv("OLLAMA_BASE_URL")
)

class Reviews(TypedDict):
    summary : str
    sentiment:str

structured_model = model.with_structured_output(Reviews)
result = structured_model.invoke(""" 
the hardware is the greate , but the software feels laggy. there are too many pre-installed apps that i can't remove. overall a best experiance""")

print(result)

print("----------------------")

print(result['summary'])

print("----------------------")

print(f' Sentiment : {result['sentiment']}')