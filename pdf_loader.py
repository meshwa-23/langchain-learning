
from langchain_community.document_loaders import PyPDFLoader

# Load PDF
loader = PyPDFLoader("sample.pdf")

# Extract pages
documents = loader.load()

# Print total pages
print("Total Pages:", len(documents))

# Print first page content
print("\nFirst Page Content:")
print(documents[0].page_content)

# Print metadata
print("\nMetadata:")
print(documents[0].metadata)
