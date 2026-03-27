from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv


load_dotenv()   # Load environment variables from .env file

llm = GoogleGenerativeAI(          # Initialize the language model with specified parameters
    model="gemini-2.5-flash", 
    temperature=0.7
)
response = llm.invoke("What is the capital of France?")
print(response)