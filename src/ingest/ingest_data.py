import os
import time
import uuid
from firecrawl import Firecrawl
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding
from dotenv import load_dotenv

load_dotenv()

app = Firecrawl(api_key=os.getenv("FIRECRAWL_API_KEY"))
client_db = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
embed_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
COLLECTION_NAME = "tarento_web_data"

def run_deep_ingestion(start_url="https://www.tarento.com"):
    # 1. Start Job
    crawl_job = app.start_crawl(
        url=start_url,
        params={
            "limit": 30,
            "scrapeOptions": {"formats": ["markdown"]},
        },
    )
    job_id = crawl_job["id"]
    
    # 2. Polling for results
    while True:
        status_data = app.get_crawl_status(job_id)
        if status_data.get("status") == "completed":
            crawl_result = status_data.get("data", [])
            break
        time.sleep(5)

    # 3. Processing with Paragraph-Aware Chunking
    points = []
    for page in crawl_result:
        markdown_text = page.get("markdown", "")
        metadata = page.get("metadata", {})
        actual_url = metadata.get("sourceURL") or metadata.get("url")
        title = metadata.get("title") or "Tarento Solution"

        # Split by double newline to preserve paragraph context
        paragraphs = markdown_text.split("\n\n")
        for chunk in paragraphs:
            if len(chunk) < 60 or "SAY HELLO" in chunk: # Filter noise
                continue
            
            vector = list(embed_model.embed([chunk]))[0].tolist()
            points.append(
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={
                        "text": chunk,
                        "source_url": actual_url,
                        "title": title,
                    },
                )
            )

    client_db.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"✅ Ingested {len(points)} high-quality chunks.")

if __name__ == "__main__":
    run_deep_ingestion()