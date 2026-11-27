
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite"
)

prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in simple words with an example."
)

# Connect prompt and LLM using LCEL
chain = prompt | llm

response = chain.invoke({
    "topic": "Artificial Intelligence"
})

print(response.text)
