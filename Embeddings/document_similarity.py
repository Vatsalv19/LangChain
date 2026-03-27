from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import os
import numpy as np

load_dotenv()

embedding = OllamaEmbeddings(
    model = "mxbai-embed-large:latest",
    base_url=os.getenv("OLLAMA_BASE_URL")
)
document=[
    "Virat Kohli is an Indian cricketer and former captain of the Indian national team.",
    "Sachin Tendulkar is a former Indian cricketer and one of the greatest batsmen in the history of cricket.",
    "M.S. Dhoni is a former Indian cricketer and captain of the Indian national team.",
    "Rohit Sharma is an Indian cricketer and the current captain of the Indian national team.",
    "Jasprit Bumrah is an Indian cricketer and one of the best fast bowlers in the world."
]
query ="tell me about jasprit bumrah"
document_embeddings = embedding.embed_documents(document)
query_embedding = embedding.embed_query(query)

scores = cosine_similarity([query_embedding],document_embeddings)[0]
index , score = (sorted(list(enumerate(scores)), key=lambda x: x[1])[-1])

print(document[index])
print("Similarity Score: ",score)
