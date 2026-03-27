from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
import os

load_dotenv()

model = ChatOllama(
    model="qwen2.5:14b-instruct",
    base_url=os.getenv("OLLAMA_BASE_URL")
)

chat_history = [
    SystemMessage(content = "You are a helpful assistant .")

]

while True:
    user_input = input("You: ")
    chat_history.append(HumanMessage(content=user_input))
    if user_input.lower() == 'exit':
        break
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print("Ollama:", result.content)

print(chat_history)