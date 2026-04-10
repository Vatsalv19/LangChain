from langchain_community.document_loaders import TextLoader
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

model = ChatOllama(
    model = "qwen2.5:14b-instruct",
    base_url = os.getenv("OLLAMA_BASE_URL")
)

prompt = PromptTemplate(

template='Write a summary for following poem - \n {poem}',
input_variables=['poem']
)

parser = StrOutputParser()


loader = TextLoader('poem.txt' , encoding='utf-8')

docs = loader.load()

print(type(docs))

print(docs[0].page_content)

print(docs[0].metadata)

chain = prompt | model | parser

result = chain.invoke({'poem' : docs[0].page_content})
print('\n')
print(result)