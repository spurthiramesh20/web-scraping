import os
from firecrawl import FirecrawlApp
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding
from dotenv import load_dotenv

load_dotenv()

# 1. Initialize Clients
app = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY"))
client_db = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
embed_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
COLLECTION_NAME = "tarento_web_data"

def run_deep_ingestion(start_url="https://www.tarento.com"):
    print(f"🕵️  Starting recursive crawl on {start_url}...")
    
    # Firecrawl will find deep links (services, careers, contact) automatically
    # 'crawl' waits until the job is done by default in the Python SDK
    crawl_result = app.crawl(
    start_url,
    limit=20,
    scrape_options={"formats": ["markdown"]},
    timeout=180,          # total crawl time limit in seconds
    request_timeout=20,   # per-request timeout in seconds
    poll_interval=2,
)


    points = []
    point_id = 0

    for page in (crawl_result.data or []):
        markdown_text = page.markdown or ""
        # This metadata contains the ACTUAL sub-page URL (the deep link)
        metadata = page.metadata_dict
        actual_url = metadata.get("source_url") or metadata.get("url") or start_url
        title = metadata.get("title") or "Tarento Page"

        print(f"📄 Processing: {actual_url}")

        # Chunk the markdown into manageable pieces for BGE-small (512 token limit)
        # We split by ~1000 characters to be safe
        chunks = [markdown_text[i:i+1000] for i in range(0, len(markdown_text), 1000)]

        for chunk in chunks:
            # Generate the 384-dimensional vector
            vector = list(embed_model.embed([chunk]))[0].tolist()

            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "text": chunk,
                        "source_url": actual_url,
                        "title": title
                    }
                )
            )
            point_id += 1

    # Push to Qdrant Cloud
    client_db.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"✅ Successfully ingested {point_id} points from deep-crawled pages!")

if __name__ == "__main__":
    run_deep_ingestion()
