from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

embedding = OllamaEmbeddings(
    model = "nomic-embed-text:latest",
    base_url=os.getenv("OLLAMA_BASE_URL")
)
result = embedding.embed_query("What is the capital of india?")
print(str(result))