
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite"
)

prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in 3 simple points."
)

parser = StrOutputParser()

chain = prompt | llm | parser

response = chain.invoke({
    "topic": "Machine Learning"
})

print(response)
print(type(str(response)))