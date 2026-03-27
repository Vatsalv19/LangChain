from langchain_ollama import ChatOllama
from dotenv import load_dotenv
import os
import streamlit as st
from langchain_core.prompts import PromptTemplate

load_dotenv()

model = ChatOllama(
    model = "qwen2.5:14b-instruct",
    base_url = os.getenv("OLLAMA_BASE_URL")
)

st.header("Ollama research assistant")
paper_input = st.selectbox("Select Research paper name",["Select..", "Attention is all you need", "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", "GPT-3: Language Models are Few-Shot Learners","Difussion Models","Transformers in Computer Vision"])

style_input = st.selectbox("Select the style of answer",["Select..", "Detailed answer", "Short answer", "Bullet points", "Summary"])

lenght_input =st.selectbox("Select the lenght of answer",["Select..", "Short answer", "Medium answer", "Long answer"])


#template
template = PromptTemplate(
    template = """You are a research assistant. You will be given the name of a research paper, the style of answer and the lenght of answer. Based on these inputs, you will provide an answer to the user's question.

Paper: {paper_input}
Style: {style_input}
Length: {length_input}
""",
input_variables = ["paper_input", "style_input", "length_input"],
validate_template = True
)

prompt = template.invoke({'paper_input': paper_input, 'style_input': style_input, 'length_input': lenght_input})



# user_input = st.text_input("Ask me anything")

if st.button("Submit"):
  
    result = model.invoke(prompt)
    st.write(result.content)
