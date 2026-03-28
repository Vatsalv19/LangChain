from langchain_ollama import ChatOllama
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate 
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch , RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field
from typing import Literal
import os


load_dotenv()

model = ChatOllama(
    model = "qwen2.5:14b-instruct",
    base_url=os.getenv("OLLAMA_BASE_URL")
)

parser = StrOutputParser()

class Feedback(BaseModel):
     sentiment : Literal['positive','negative'] = Field(description='Give the sentiment of the feedback')


parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template = 'classify the following feedback text into positive or negative \n {feedback} \n {format_instruction} ',
    input_variables=['feedback'],
    partial_variables={'format_instruction': parser2.get_format_instructions()}
)


classifier_chain = prompt1 | model | parser2

prompt2 = PromptTemplate(
     template = 'Write an appropiate response to this Positive feedback \n {feedback} ',
     input_variables=['feedback']

)
prompt3 = PromptTemplate(
     template = 'Write an appropiate response to this Negative feedback \n {feedback} ',
     input_variables=['feedback']

)


branch_chain = RunnableBranch(
     (lambda x: x.sentiment == 'positive' , prompt2 | model | parser),
     (lambda x: x.sentiment == 'negative' , prompt3 | model | parser),
  
      RunnableLambda(lambda x : "Could not find sentiment")

)

chain = classifier_chain | branch_chain

result = chain.invoke({'feedback' : 'This is best phone ever'})
print(result)

