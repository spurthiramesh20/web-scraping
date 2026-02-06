
import os
from dotenv import load_dotenv
from google import genai
from qdrant_client import QdrantClient
from fastembed import TextEmbedding
import numpy as np

# Load environment variables
load_dotenv()

# Setup
api_key = os.getenv("OPENAI_API_KEY")
qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")

client_ai = genai.Client(api_key=api_key)
client_db = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
embed_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

COLLECTION_NAME = "tarento_web_data"

# Test query
query = "What services does Tarento provide?"
print(f"Query: {query}")

# Get embedding
query_embeddings = list(embed_model.embed([query]))
query_embedding = np.array(query_embeddings[0], dtype=np.float32).flatten()

# Search in Qdrant Cloud using query_points
try:
    search_results = client_db.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding.tolist(),
        limit=3
    ).points
    results = [(hit.score, hit.payload['text']) for hit in search_results]
except Exception as e:
    print(f"Search error: {e}")
    results = []

if results:
    results.sort(reverse=True)
    context = "\n".join([text for _, text in results[:3]])
    
    # Ask Gemini
    prompt = f"""Answer based on this context:
    
{context}

Question: {query}"""
    
    response = client_ai.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )
    
    print("\n--- Response ---")
    print(response.text)
else:
    print("No data found in Qdrant Cloud")
