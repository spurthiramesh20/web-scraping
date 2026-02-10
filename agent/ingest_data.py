import os
import time
from firecrawl import Firecrawl
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding
from dotenv import load_dotenv

load_dotenv()

# 1. Initialize Clients (Note: Using 'Firecrawl' class for v2)
app = Firecrawl(api_key=os.getenv("FIRECRAWL_API_KEY"))
client_db = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
embed_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
COLLECTION_NAME = "tarento_web_data"


def run_deep_ingestion(start_url="https://www.tarento.com"):
    print(f"Starting async crawl on {start_url}...")

    # 2. Start the job without waiting
    # We use limit=20 to stay within free tier limits/testing time
    crawl_job = app.start_crawl(
        url=start_url,
        params={
            "limit": 20,
            "scrapeOptions": {"formats": ["markdown"]},
        },
    )
    job_id = crawl_job["id"]
    print(f"Job initiated! ID: {job_id}")

    # 3. Polling Loop: Wait for completion with feedback
    crawl_result = None
    while True:
        status_data = app.get_crawl_status(job_id)
        status = status_data.get("status")
        completed = status_data.get("completed", 0)
        total = status_data.get("total", 0)

        print(f"Status: {status} | Progress: {completed}/{total} pages")

        if status == "completed":
            crawl_result = status_data.get("data", [])
            break
        if status == "failed":
            print(f"Crawl failed: {status_data.get('error')}")
            return

        time.sleep(5)  # Check every 5 seconds

    # 4. Processing Phase
    points = []
    point_id = 0

    for page in crawl_result:
        markdown_text = page.get("markdown", "")
        metadata = page.get("metadata", {})
        # This is where the deep citation link comes from!
        actual_url = metadata.get("sourceURL") or metadata.get("url") or start_url
        title = metadata.get("title") or "Tarento Page"

        print(f"Processing: {actual_url}")

        # Chunking for BGE (Max 512 tokens, ~1000 chars is safe)
        chunks = [markdown_text[i : i + 1000] for i in range(0, len(markdown_text), 1000)]

        for chunk in chunks:
            if not chunk.strip():
                continue
            # Generate the 384-dimensional vector
            vector = list(embed_model.embed([chunk]))[0].tolist()

            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "text": chunk,
                        "source_url": actual_url,
                        "title": title,
                    },
                )
            )
            point_id += 1

    # 5. Push to Qdrant Cloud
    if points:
        client_db.upsert(collection_name=COLLECTION_NAME, points=points)
        print(f"Successfully ingested {point_id} points with deep links!")
    else:
        print("No data found to ingest.")


if __name__ == "__main__":
    run_deep_ingestion()
