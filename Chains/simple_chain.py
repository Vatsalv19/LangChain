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

parser = StrOutputParser()

chain = prompt | model | parser    # | is called the chaining operator which is used to chain the prompt and model together

result = chain.invoke(({'topic' : 'milky way galaxy'}))
print(result)

chain.get_graph().print_ascii()