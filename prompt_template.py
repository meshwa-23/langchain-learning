
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

load_dotenv()

# 1. Initialize Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite"
)

# 2. Create a prompt template
prompt = PromptTemplate.from_template(
    "Explain {topic} in simple words with one example."
)

# 3. Provide dynamic input
formatted_prompt = prompt.format(
    topic="Machine Learning"
)

print("Formatted Prompt:")
print(formatted_prompt)

# 4. Send prompt to Gemini
response = llm.invoke(formatted_prompt)

# 5. Display response
print("\nAI Response:")
print(response.text)
