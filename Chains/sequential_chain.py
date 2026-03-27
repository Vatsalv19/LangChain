from langchain_ollama import ChatOllama
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate 
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()

prompt = PromptTemplate(
    template = 'generate 5 interesting facts about {topic}',
    input_variables = ['topic']
)

model = ChatOllama(
    model="qwen2.5:14b-instruct",
    base_url=os.getenv("OLLAMA_BASE_URL")
)

prompt = PromptTemplate(
    template = 'generate a detailed report on {topic}',
    input_variables = ['topic']
)

prompt2 = PromptTemplate(
    template = 'write a 5 point summary on following text : {text}',
    input_variables = ['text']
)

parser = StrOutputParser()

chain = prompt | model | parser | prompt2 | model | parser    # chain the prompts and model together

result = chain.invoke(({'topic' : 'milky way galaxy'}))
print(result)