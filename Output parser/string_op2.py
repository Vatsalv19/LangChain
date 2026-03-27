from langchain_ollama import ChatOllama
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate  
from langchain_core.output_parsers import StrOutputParser

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

parser = StrOutputParser()   # create an output parser to parse the output of the model into a string

chain = template1 | model | parser | template2 | model | parser     # chain the prompts and model together

result = chain.invoke({'topic': 'Black Hole'})

print(result)
