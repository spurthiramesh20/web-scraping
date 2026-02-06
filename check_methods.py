from qdrant_client import QdrantClient
import os
from dotenv import load_dotenv

load_dotenv()
qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")

client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

# List collections
collections = client.get_collections()
print("Collections:", collections)

# Check available methods
print("\nAvailable methods:")
for method in dir(client):
    if 'search' in method.lower() or 'query' in method.lower():
        print(f"  - {method}")
