import json
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding

# 1. Load environment variables
load_dotenv()

# Configuration
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "tarento_web_data"
DATA_FILE = "scraped_data.jsonl"

def store_vectors():
    # Initialize Qdrant Client
    if not QDRANT_URL or not QDRANT_API_KEY:
        print("Error: QDRANT_URL or QDRANT_API_KEY not found in .env file.")
        return

    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    # Initialize Embedding Model (FastEmbed)
    # Using BAAI/bge-small-en-v1.5 (dimension: 384)
    embed_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    
    # 2. Recreate collection for a clean slate
    print(f"Creating collection: {COLLECTION_NAME}...")
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE)
    )

    # 3. Process the JSONL file
    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found.")
        return

    points = []
    print(f"Reading data from {DATA_FILE}...")
    
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                data = json.loads(line)
                text_content = data.get("text", "")
                source_url = data.get("url", "")

                if not text_content:
                    continue

                # Generate vector for this specific chunk
                # list() is used because embed() returns a generator
                vector = list(embed_model.embed([text_content]))[0].tolist()

                # Build the point with specific metadata (URL)
                points.append(models.PointStruct(
                    id=i,
                    vector=vector,
                    payload={
                        "text": text_content,
                        "source_url": source_url
                    }
                ))
                
                # Batch upload every 100 points to avoid memory issues
                if len(points) >= 100:
                    client.upsert(collection_name=COLLECTION_NAME, points=points)
                    print(f"Uploaded {i+1} points...")
                    points = []

            except json.JSONDecodeError as e:
                print(f"Skipping malformed line {i}: {e}")

    # Final upload for remaining points
    if points:
        client.upsert(collection_name=COLLECTION_NAME, points=points)

    print(f"✅ Success! All Tarento data is now vectorized and stored in Qdrant.")

if __name__ == "__main__":
    store_vectors()