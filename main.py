
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite"
)

response = llm.invoke(
    "Explain LangChain in simple words with an example."
)

print(response.text)
