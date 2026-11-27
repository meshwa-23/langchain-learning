
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

load_dotenv()

# Initialize embedding model
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

# Sample documents
documents = [
    Document(page_content="Python is a popular programming language."),
    Document(page_content="Machine learning helps computers learn from data."),
    Document(page_content="LangChain is used to build applications with LLMs."),
    Document(page_content="Dogs and cats are common household pets.")
]

# Create vector database
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    collection_name="my_first_collection",
    persist_directory="./chroma_db"
)

print("Documents stored successfully!")

# Search for similar documents
query = "Tell me about artificial intelligence"

results = vectorstore.similarity_search(query, k=2)

print("\nSearch Results:")

for result in results:
    print(result.page_content)
