from langchain_ollama import ChatOllama
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatOllama(
    model="llama3.1:8b",
    base_url=os.getenv("OLLAMA_BASE_URL"),
    temperature=0     # tmperature is set to 0 for deterministic output or higher values for more creative output
)

response = llm.invoke("write a five line poem on cricket")

print(response.content) 