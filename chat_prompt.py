
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# Step 1: Initialize Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite"
)

# Step 2: Create a chat prompt
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful Python programming teacher."),
    ("human", "Explain {topic} with a simple example.")
])

# Step 3: Insert dynamic input
messages = chat_prompt.format_messages(
    topic="OOP concepts"
)

# Step 4: Send messages to Gemini
response = llm.invoke(messages)

# Step 5: Print response
print(response.text)
