from langchain_ollama import ChatOllama
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate  

import os

load_dotenv()

model = ChatOllama(
    model="qwen2.5:14b-instruct",
    base_url=os.getenv("OLLAMA_BASE_URL")
)

template1 = PromptTemplate(
    template = 'write a detail report on {topic}',
    input_variables = ['topic']
)
template2 = PromptTemplate(
    template = 'write a 1 line summary on following text : {text}',
    input_variables = ['text']
)

prompt1 = template1.invoke({'topic':'Black hole'})

result = model.invoke(prompt1)

prompt2 = template2.invoke({'text': result.content})

result2 = model.invoke(prompt2)

print(result2.content)