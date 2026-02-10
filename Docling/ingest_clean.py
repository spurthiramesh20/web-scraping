import os
import uuid  
from urllib.parse import urljoin, urlparse

import requests
import streamlit as st
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from fastembed import TextEmbedding
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
from qdrant_client import QdrantClient, models

load_dotenv()

# Initialize models manually to avoid the "Named Vector" conflict
embed_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
COLLECTION_NAME = "tarento_web_data"


def upload_to_qdrant_clean(documents, source_url=None, title=None):
    points = []

    print("Generating embeddings for Docling chunks...")
    for doc in documents:
        # Generate the vector for this specific chunk
        vector = list(embed_model.embed([doc.page_content]))[0].tolist()

        # Create a Qdrant Point
        points.append(
            models.PointStruct(
                id=str(uuid.uuid4()),  # Generates a unique ID
                vector=vector,  # This is an "unnamed" vector, matching your DB
                payload={
                    "text": doc.page_content,
                    "source_url": source_url or doc.metadata.get("source", "https://www.tarento.com"),
                    "title": title or doc.metadata.get("title", "Tarento Page"),
                },
            )
        )

    print(f"Upserting {len(points)} points...")
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print("Success! Docling data is now in your collection.")
    return len(points)


def crawl_tarento(seed_url="https://www.tarento.com", limit=40, status_cb=None):
    visited = set()
    to_visit = [seed_url]
    domain = urlparse(seed_url).netloc
    count = 0
    total_points = 0

    while to_visit and count < limit:
        url = to_visit.pop(0)
        if url in visited:
            continue

        try:
            if status_cb:
                status_cb(f"Crawling {count + 1}/{limit}: {url}")
            res = requests.get(url, timeout=8)
            visited.add(url)
            soup = BeautifulSoup(res.text, "html.parser")

            for a in soup.find_all("a", href=True):
                link = urljoin(seed_url, a["href"]).split("#")[0].rstrip("/")
                if urlparse(link).netloc == domain and link not in visited:
                    if not any(x in link for x in [".pdf", ".jpg", ".png", "/careers"]):
                        to_visit.append(link)

            page_title = (soup.title.string.strip() if soup.title and soup.title.string else "Tarento Page")
            loader = DoclingLoader(file_path=url, export_type=ExportType.DOC_CHUNKS)
            docs = loader.load()
            total_points += upload_to_qdrant_clean(docs, source_url=url, title=page_title)
            count += 1
        except Exception as e:
            if status_cb:
                status_cb(f"Skipping {url}: {e}")

    return count, total_points


def main():
    st.set_page_config(page_title="Tarento AI Assistant", page_icon="T")
    st.title("Tarento AI Assistant")
    st.caption("Runs on Streamlit (localhost:8501 by default).")

    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_key = os.getenv("QDRANT_API_KEY")

    st.subheader("Ingest")
    seed_url = st.text_input("Seed URL", value="https://www.tarento.com")
    limit = st.number_input("Max pages", min_value=1, max_value=200, value=40, step=1)

    log_area = st.empty()

    def push_log(msg):
        log_area.markdown(msg)

    if st.button("Run Crawl + Upload"):
        if not qdrant_url or not qdrant_key:
            st.error("QDRANT_URL and QDRANT_API_KEY must be set in .env.")
        else:
            with st.status("Crawling and uploading...", expanded=True) as status:
                pages, points = crawl_tarento(
                    seed_url=seed_url, limit=int(limit), status_cb=push_log
                )
                status.update(label=f"Done. Pages: {pages}, Points: {points}", state="complete")


if __name__ == "__main__":
    main()
