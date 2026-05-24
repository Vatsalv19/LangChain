from youtube_transcript_api import ( 
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound
)
import browser_cookie3
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

import os

load_dotenv()
cookies = browser_cookie3.chrome()
# LLM
llm = ChatOllama(
    model="llama3.2:3b",
    base_url=os.getenv("OLLAMA_BASE_URL")
)

# Embedding model
embedding = OllamaEmbeddings(
    model="qwen3-embedding:0.6b",
    base_url=os.getenv("OLLAMA_BASE_URL")
)
video_id = "HyNa3XXe91c"

try:
    # Fetch transcript
    transcript_list = YouTubeTranscriptApi.get_transcript(
        video_id,
        languages=['en'],
        cookies=cookies
    )

    # Convert transcript to text
    transcript = " ".join(
        chunk['text'] for chunk in transcript_list
    )

    print("\nTranscript fetched successfully\n")

except TranscriptsDisabled:
    print("Transcripts are disabled for this video.")
    exit()

except NoTranscriptFound:
    print("No transcript found.")
    exit()

except Exception as e:
    print("Error:", e)
    exit()

# Split transcript
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.create_documents([transcript])

# Create vector store
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embedding,
    collection_name="youtube_transcript"
)

# Retriever
retriever = vector_store.as_retriever(
    search_type='similarity',
    search_kwargs={"k": 4}
)

# User question
question = "What is the video about ?"

# Retrieve relevant docs
retriever_docs = retriever.invoke(question)

# Combine context
context = "\n".join(
    [doc.page_content for doc in retriever_docs]
)

# Prompt
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful assistant.

Answer only from the provided transcript context.

If the context is insufficient to answer the question,
say "I don't know".

Context:
{context}

Question:
{question}
"""
)

# Final prompt
final_prompt = prompt.format(
    context=context,
    question=question
)

# Generate answer
answer = llm.invoke(final_prompt)

print("\nAnswer:\n")
print(answer.content)