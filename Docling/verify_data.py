import os
import uuid
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType

load_dotenv()

# --- PART 1: THE UPLOADER (Define this first!) ---
embed_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))

def upload_to_qdrant_clean(documents):
    points = []
    for doc in documents:
        vector = list(embed_model.embed([doc.page_content]))[0].tolist()
        points.append(
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "text": doc.page_content,
                    "source_url": doc.metadata.get("source", "https://www.tarento.com"),
                    "title": "Tarento Auto-Crawl"
                }
            )
        )
    client.upsert(collection_name="tarento_web_data", points=points)

# --- PART 2: THE SPIDER ---
def crawl_tarento(seed_url="https://www.tarento.com", limit=40):
    visited = set()
    to_visit = [seed_url]
    domain = urlparse(seed_url).netloc
    count = 0

    while to_visit and count < limit:
        url = to_visit.pop(0)
        if url in visited: continue
        
        try:
            print(f"🕷️ [{count+1}/{limit}] Crawling: {url}")
            # Discover links
            res = requests.get(url, timeout=5)
            visited.add(url)
            soup = BeautifulSoup(res.text, 'html.parser')
            
            for a in soup.find_all('a', href=True):
                link = urljoin(seed_url, a['href']).split('#')[0].rstrip('/')
                if urlparse(link).netloc == domain and link not in visited:
                    if not any(x in link for x in ['.pdf', '.jpg', '.png', '/careers']):
                        to_visit.append(link)

            # Process with Docling
            loader = DoclingLoader(file_path=url, export_type=ExportType.DOC_CHUNKS)
            docs = loader.load()
            upload_to_qdrant_clean(docs)
            count += 1
            
        except Exception as e:
            print(f"⚠️ Skipping {url}: {e}")

if __name__ == "__main__":
    crawl_tarento()