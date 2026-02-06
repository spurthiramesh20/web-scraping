import os
import uuid  # To generate unique IDs for chunks
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding
from dotenv import load_dotenv

load_dotenv()

# 1. Initialize models manually to avoid the "Named Vector" conflict
embed_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))

def upload_to_qdrant_clean(documents):
    points = []
    
    print("🧠 Generating embeddings for Docling chunks...")
    for doc in documents:
        # Generate the vector for this specific chunk
        vector = list(embed_model.embed([doc.page_content]))[0].tolist()
        
        # Create a Qdrant Point
        points.append(
            models.PointStruct(
                id=str(uuid.uuid4()), # Generates a unique ID
                vector=vector,        # This is an "unnamed" vector, matching your DB
                payload={
                    "text": doc.page_content,
                    "source_url": doc.metadata.get("source", "https://www.tarento.com"),
                    "title": "Tarento Document"
                }
            )
        )

    print(f"📡 Upserting {len(points)} points...")
    client.upsert(
        collection_name="tarento_web_data",
        points=points
    )
    print("✅ Success! Docling data is now in your collection.")

# Use this in your if __name__ == "__main__" block