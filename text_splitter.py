
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# Load PDF
loader = PyPDFLoader("sample.pdf")
documents = loader.load()

# Create text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

# Split documents into chunks
chunks = text_splitter.split_documents(documents)

print("Total chunks:", len(chunks))

# Print first chunk
print("\nFirst Chunk:")
print(chunks[0].page_content)

# Print metadata
print("\nMetadata:")
print(chunks[0].metadata)
