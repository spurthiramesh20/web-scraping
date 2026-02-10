import os
import re
import sys
from pathlib import Path

import streamlit as st
from qdrant_client import QdrantClient
from fastembed import TextEmbedding
from dotenv import load_dotenv

# Ensure repo root is on sys.path for imports like gemini.py
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gemini import get_assistant_response

# Optional: reuse crawler logic from ingest_clean if you want to run crawl on startup
from Docling.ingest_clean import crawl_tarento

load_dotenv()
st.set_page_config(page_title="Tarento AI Assistant", page_icon="T", layout="wide")

COLLECTION_NAME = "tarento_web_data"

@st.cache_resource
def initialize_system():
    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
    )
    model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    return client, model

client_db, embed_model = initialize_system()


def search_knowledge_base(query):
    try:
        query_vec = list(embed_model.embed([query]))[0].tolist()
        results = client_db.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vec,
            limit=8,
        ).points
        return [
            {
                "text": hit.payload.get("text", ""),
                "url": hit.payload.get("source_url", ""),
                "title": hit.payload.get("title", "Tarento Page"),
                "score": hit.score,
            }
            for hit in results
            if hit.score > 0.30
        ]
    except Exception as e:
        st.error(f"Search Error: {e}")
        return []


def dedupe_citations(citations, max_items=5):
    seen = set()
    unique = []
    for c in citations:
        url = c.get("url")
        if not url or url in seen:
            continue
        seen.add(url)
        unique.append(c)
        if len(unique) >= max_items:
            break
    return unique


def get_response_and_link(prompt, context, instructions):
    full_response = get_assistant_response(prompt, context, instructions)
    url_match = re.search(r"SOURCE_LINK:\s*\[?(https?://[^\s\]]+)\]?", full_response)

    clean_answer = full_response
    source_url = None

    if url_match:
        source_url = url_match.group(1)
        clean_answer = re.sub(r"SOURCE_LINK:.*", "", full_response).strip()

    return clean_answer, source_url


def load_system_prompts():
    try:
        with open(REPO_ROOT / "prompts_v1.txt", "r") as f:
            v1 = f.read()
        with open(REPO_ROOT / "prompts_v2.txt", "r") as f:
            v2 = f.read()
        return f"{v1}\n\nSTRICT GUIDELINES:\n{v2}"
    except Exception:
        return "You are a Tarento Assistant. Help users find services and careers."


STRICT_INSTRUCTIONS = load_system_prompts()

# Trigger crawl + upload ONCE per session (not from UI)
if "docling_crawl_done" not in st.session_state:
    st.session_state.docling_crawl_done = False

if not st.session_state.docling_crawl_done:
    with st.status("Docling crawl + upload running...", expanded=True) as status:
        try:
            pages, points = crawl_tarento(seed_url="https://www.tarento.com", limit=40)
            status.update(label=f"Docling ingest done. Pages: {pages}, Points: {points}", state="complete")
            st.session_state.docling_crawl_done = True
        except Exception as e:
            status.update(label=f"Docling ingest failed: {e}", state="error")

# Chat UI
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("Tarento AI Assistant (Docling)")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("link"):
            st.markdown(f"Info: [{msg.get('title', 'Tarento')}]({msg['link']})")
        if msg.get("citations"):
            st.markdown("Sources:")
            for c in msg["citations"]:
                title = c.get("title", "Tarento Page")
                url = c.get("url", "")
                if url:
                    st.markdown(f"- [{title}]({url})")

if prompt := st.chat_input("Ask about Tarento..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").markdown(prompt)

    with st.chat_message("assistant"):
        answer = ""
        current_link = None
        current_title = "Tarento Official"

        with st.status("Consulting Tarento Records...", expanded=False) as status:
            citations = search_knowledge_base(prompt)
            citation_list = dedupe_citations(citations, max_items=5)

            if citations:
                context_text = "\n".join(
                    [f"Data: {c['text']} | Link: {c['url']}" for c in citations]
                )
                status.update(label="Information retrieved.", state="complete")
            else:
                context_text = "NO_CONTEXT_AVAILABLE"
                status.update(label="No specific internal documents found.", state="complete")

            answer, current_link = get_response_and_link(
                prompt, context_text, STRICT_INSTRUCTIONS
            )

        if current_link:
            for c in citations:
                if c["url"] == current_link:
                    current_title = c.get("title", "Tarento Official Page").split("|")[0].strip()
                    break

        st.markdown(answer)
        if current_link:
            st.markdown(f"Info: [{current_title}]({current_link})")
        if citation_list:
            st.markdown("Sources:")
            for c in citation_list:
                title = c.get("title", "Tarento Page")
                url = c.get("url", "")
                if url:
                    st.markdown(f"- [{title}]({url})")

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "link": current_link,
                "title": current_title,
                "citations": citation_list,
            }
        )
