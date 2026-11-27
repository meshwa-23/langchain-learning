
from langchain_community.document_loaders import TextLoader

# Step 1: Load the text file
loader = TextLoader("sample.txt", encoding="utf-8")

# Step 2: Load documents
documents = loader.load()

# Step 3: Print document content
print("Document Content:")
print(documents[0].page_content)

# Step 4: Print metadata
print("\nDocument Metadata:")
print(documents[0].metadata)

# Step 5: Check document type
print("\nDocument Type:")
print(type(documents[0]))
