from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()   # Load environment variables from .env file

model = GoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature = 1.5,    # Set the temperature for response generation (controls creativity)
    # max_tokens = 100     # Set the maximum number of tokens in the response
)
result = model.invoke("write 5 line poem on cricket")

print(result)