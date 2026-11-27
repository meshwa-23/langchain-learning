
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite"
)

parser = JsonOutputParser()

prompt = PromptTemplate(
    template="""
    Give information about {technology}.
    Return the answer with these fields:
        name, type, use_cases.

        {format_instructions}
        """,
        input_variables=["technology"],
        partial_variables={
            "format_instructions": parser.get_format_instructions()
        }
    )

chain = prompt | llm | parser

response = chain.invoke({
    "technology": "Python"
})

print(response)
print(type(response))
