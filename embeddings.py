
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from math import sqrt

load_dotenv()

# Initialize embedding model
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

# Sample sentences
texts = [
    "I love dogs and puppies.",
    "I enjoy playing with animals.",
    "I like programming in Python."
]

# Convert sentences into vectors
vectors = embeddings.embed_documents(texts)

# Display vector information
print("Total sentences:", len(vectors))
print("Vector dimensions:", len(vectors[0]))

print("\nFirst sentence vector (first 10 values):")
print(vectors[0][:10])

# Calculate cosine similarity
def cosine_similarity(v1, v2):
    dot_product = sum(a * b for a, b in zip(v1, v2))
    magnitude1 = sqrt(sum(a * a for a in v1))
    magnitude2 = sqrt(sum(b * b for b in v2))

    return dot_product / (magnitude1 * magnitude2)

print("\nSimilarity between sentences:")

print(
    "Sentence 1 and 2:",
    cosine_similarity(vectors[0], vectors[1])
)

print(
    "Sentence 1 and 3:",
    cosine_similarity(vectors[0], vectors[2])
)
