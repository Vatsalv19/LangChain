from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

embedding = OllamaEmbeddings(
    model = "nomic-embed-text:latest",
    base_url=os.getenv("OLLAMA_BASE_URL")
)
docuements = [
"Delhi is the capital of india",
"Kolkata is the capital of west bengal",
"Chennai is the capital of tamil nadu"

]
result = embedding.embed_documents(docuements)
print(str(result))