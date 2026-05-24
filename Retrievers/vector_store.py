from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv
import os
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

load_dotenv()



documents =[
    Document(page_content = " LangChain is a framework for developing applications powered by language models. It can be used for chatbots, Generative Question-Answering (GQA), summarization, and much more."),
    Document(page_content = "The core idea of the library is that we can “chain” together different components to create more advanced use cases around LLMs. Chains may consist of multiple components from several modules, or may just consist of a single component. The chains themselves are also modular, so you can import them and use them (or remix them) in your own code."),
    Document(page_content = "The library also provides end-to-end chains for common applications, like summarization or question-answering, that you can use right away for your projects. These chains are built using the components described above, and are designed to be easily customizable and extensible."),    
    Document(page_content = "LangChain is a framework for developing applications powered by language models. It can be used for chatbots, Generative Question-Answering (GQA), summarization, and much more."),
    Document(page_content = "The core idea of the library is that we can “chain” together different components to create more advanced use cases around LLMs. Chains may consist of multiple components from several modules, or may just consist of a single component. The chains themselves are also modular, so you can import them and use them (or remix them) in your own code."),
]

model = OllamaEmbeddings(
    model="qwen3-embedding:0.6b",
    base_url=os.getenv("OLLAMA_BASE_URL")
)
vectorestore = Chroma.from_documents(
   documents=documents,
   embedding=model,
   collection_name="langchain_docs"
      )
retriever = vectorestore.as_retriever(search_kwargs={"k":2})

query = "What is LangChain ?"
result = retriever.invoke(query)

for i ,doc in enumerate(result):
    print(f"Result {i+1} : {doc.page_content}")