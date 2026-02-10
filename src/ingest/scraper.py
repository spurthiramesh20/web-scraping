import os
import uuid
import requests
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding
from dotenv import load_dotenv

# 1. Environment Setup
load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "tarento_knowledge"

if not QDRANT_URL:
    print("❌ ERROR: QDRANT_URL not found in .env file!")
    exit()

# 2. Initializing Clients
print("⚙️ Initializing clients...")
client_db = QdrantClient(
    url=QDRANT_URL, 
    api_key=QDRANT_API_KEY,
    prefer_grpc=False  # Optimized for your GCP Cloud Cluster
)
embed_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

BASE_URL = "https://www.tarento.com"
MAX_PAGES = 40 

def is_internal(url):
    return urlparse(url).netloc == urlparse(BASE_URL).netloc

def run_local_recursive_crawl():
    to_visit = [BASE_URL]
    visited = set()
    points = []

    print(f"🚀 Starting BFS discovery crawl from {BASE_URL}...")

    # The BFS Loop
    while to_visit and len(visited) < MAX_PAGES:
        current_url = to_visit.pop(0)
        if current_url in visited:
            continue
        
        visited.add(current_url)
        print(f"🔍 Scraping ({len(visited)}/{MAX_PAGES}): {current_url}")

        try:
            response = requests.get(current_url, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')

            # Discovery: Find internal links for the next "level" of BFS
            for a_tag in soup.find_all('a', href=True):
                full_link = urljoin(BASE_URL, a_tag['href']).split('#')[0].rstrip('/')
                if is_internal(full_link) and full_link not in visited:
                    to_visit.append(full_link)


            # Physically remove the elements that cause hallucinated links
            for noise in soup(['header', 'footer', 'nav', 'script', 'style', 'aside']):
                noise.decompose()

            main_body = soup.find('main') or soup.find('article') or soup.find('body')
            clean_text = main_body.get_text(separator=' ', strip=True)

            if len(clean_text) > 300:
                # Generate embedding
                vector = list(embed_model.embed([clean_text]))[0].tolist()
                points.append(models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={
                        "text": clean_text, 
                        "source_url": current_url,
                        "title": soup.title.string if soup.title else "Tarento Page"
                    }
                ))
        except Exception as e:
            print(f"⚠️ Error on {current_url}: {e}")

    # 3. Updating Qdrant Cloud
    print(f"📡 Found {len(points)} quality pages. Updating Qdrant...")

    try:
        client_db.delete_collection(collection_name=COLLECTION_NAME)
        print("🗑️ Old collection cleared.")
    except Exception:
        print("ℹ️ Starting with a fresh collection.")

    client_db.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=384, 
            distance=models.Distance.COSINE
        )
    )

    if points:
        client_db.upsert(collection_name=COLLECTION_NAME, points=points)
        print(f"✅ Success! {len(points)} clean pages are now in the cloud.")
    else:
        print("❌ No valid content found to upload.")

if __name__ == "__main__":
    run_local_recursive_crawl()