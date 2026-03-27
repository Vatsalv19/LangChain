from langchain_ollama import ChatOllama
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate  
from langchain_core.output_parsers import JsonOutputParser

import os

load_dotenv()

model = ChatOllama(
    model="qwen2.5:14b-instruct",
    base_url=os.getenv("OLLAMA_BASE_URL")
)

parser = JsonOutputParser()   # create an output parser to parse the output of the model into a json object

template1 = PromptTemplate(
    template = 'Give the name , age , and city  of fictional person {format_instruction}',

    input_variables = [],
    partial_variables= {'format_instruction': parser.get_format_instructions()}
)

chain = template1 | model | parser     # chain the prompt and model together

result = chain.invoke({})
print(result)

# prompt = template1.format()
# result = model.invoke(prompt)
# print(result)


# final_result = parser.parse(result.content)   # parse the output of the model into a json object

# print(final_result)