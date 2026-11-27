
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
    Document(page_content="Python is used for AI and machine learning."),
    Document(page_content="Java is widely used for backend development."),
    Document(page_content="LangChain helps developers build LLM applications."),
    Document(page_content="ChromaDB stores vector embeddings.")
]

# Create vector database
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    collection_name="retriever_demo"
)

# Create retriever
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 2}
)

# Search
query = "Which technology is used for artificial intelligence?"

results = retriever.invoke(query)

print("\nRetrieved Documents:")

for doc in results:
    print(doc.page_content)
