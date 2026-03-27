from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
import os

load_dotenv()

model = ChatOllama(
    model="qwen2.5:14b-instruct",
    base_url=os.getenv("OLLAMA_BASE_URL")
)
messages = [
    SystemMessage(content = "You are a helpful assistant ."),
    HumanMessage(content="Tell me about Langchain")
]
result = model.invoke(messages)

messages.append(AIMessage(content = result.content))

print(messages)