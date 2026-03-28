from langchain_ollama import ChatOllama
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate 
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from ollama_ import model


import os

load_dotenv()

prompt = PromptTemplate(
    template = 'generate 5 interesting facts about {topic}',
    input_variables = ['topic']
)

model1 = ChatOllama(
    model="qwen2.5:14b-instruct",
    base_url=os.getenv("OLLAMA_BASE_URL")
)
model2 = ChatOllama(
    model="qwen2.5:3b-instruct",
    base_url=os.getenv("OLLAMA_BASE_URL")
)

prompt1 = PromptTemplate(
    template = 'generate a short and simple notes from the following text : {text}',
    input_variables = ['text']
 
)

prompt2 = PromptTemplate(
    template = 'generate 5 question and answer(quiz)  from the following text : {text}',
    input_variables = ['text']
)

prompt3 = PromptTemplate(
    template = 'merge the provided notes and quiz into a sngle document {notes} {quiz}',
    input_variables = ['notes', 'quiz']
)
parser = StrOutputParser()

parallel_chains = RunnableParallel({             # parallel chain to generate notes and quiz at the same time
    'notes': prompt1 | model1 |parser,           # chain to generate notes
    'quiz': prompt2 | model2 | parser            # chain to generate quiz

})

merge_chain = prompt3 | model1 | parser

chain = parallel_chains | merge_chain

result = chain.invoke({'text': """
TensorFlow is an open-source machine learning framework developed by the Google Brain team. It provides a comprehensive ecosystem for building and deploying machine learning models, including tools for data preprocessing, model training, and deployment. TensorFlow supports a wide range of applications, from natural language processing to computer vision, and is widely used in both research and industry for developing AI solutions.

Key Features:

Flexible Architecture: TensorFlow supports multiple programming paradigms including eager execution and graph-based computation, making it suitable for both research and production environments.
Scalability: Built to scale from mobile devices to large-scale distributed systems, enabling deployment across various platforms.
Ecosystem: Includes specialized libraries like TensorFlow Lite for mobile, TensorFlow.js for web, TensorFlow Hub for pre-trained models, and TensorFlow Extended (TFX) for production ML pipelines.
Performance: Highly optimized for GPUs and TPUs, leveraging hardware acceleration for faster computations.
Core Components:

Tensors: Multi-dimensional arrays that form the fundamental data structure
Operations: Mathematical functions that manipulate tensors
Layers: High-level building blocks for constructing neural networks
Models: Full neural network architectures combining layers and operations
"""})
print(result)


chain.get_graph().print_ascii()