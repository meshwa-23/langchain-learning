
from dotenv import load_dotenv

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# 1. Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite"
)

# 2. Initialize embedding model
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

# 3. Load PDF
loader = PyPDFLoader("sample.pdf")
documents = loader.load()

print("PDF loaded successfully!")

# 4. Split PDF into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(documents)

print("Total chunks:", len(chunks))

# 5. Store chunks in ChromaDB
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="my_rag_collection",
    persist_directory="./rag_database"
)

print("Documents stored in vector database!")

# 6. Create retriever
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

# 7. Create prompt template
prompt = ChatPromptTemplate.from_template("""
Answer the question using only the following context.

If the answer is not present in the context,
say: I could not find this information in the document.

Context:
    {context}

    Question:
        {question}
        """)

# 8. Ask user a question
question = input("\nAsk a question about your PDF: ")

# 9. Retrieve relevant chunks
results = retriever.invoke(question)

context = "\n\n".join(
    doc.page_content for doc in results
)

# 10. Generate answer
chain = prompt | llm

response = chain.invoke({
    "context": context,
    "question": question
})

print("\nAI Answer:")
print(response.text)
