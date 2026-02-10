import json
import os
import re
import streamlit as st
from qdrant_client import QdrantClient
from fastembed import TextEmbedding
from dotenv import load_dotenv
import sys
from pathlib import Path

# --- 1. SESSION INITIALIZATION (Must be first) ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Module path setup for internal imports
if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from agent.gemini import get_assistant_response
else:
    from .agent.gemini import get_assistant_response

# --- 2. CONFIGURATION & CLIENTS ---
load_dotenv()
st.set_page_config(page_title="Tarento AI", page_icon="⚙️", layout="centered")

@st.cache_resource
def initialize_system():
    # Using REST (HTTP) for better Cloud stability
    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        prefer_grpc=False 
    )
    model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    return client, model, "tarento_knowledge"

client_db, embed_model, COLLECTION_NAME = initialize_system()

# --- 3. HELPER FUNCTIONS ---

def search_knowledge_base(query):
    """Searches Qdrant and returns cleaned results with scores."""
    try:
        query_vec = list(embed_model.embed([query]))[0].tolist()
        results = client_db.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vec,
            limit=3,
        ).points
        
        return [
            {
                "text": hit.payload.get("text", ""),
                "url": hit.payload.get("source_url", ""),
                "title": hit.payload.get("title", "Tarento Solution"),
                "score": hit.score,
            }
            for hit in results
        ]
    except Exception as e:
        st.error(f"Search Error: {e}")
        return []

def get_response_and_link(prompt, context, instructions):
    """Queries Gemini and extracts the source link if present."""
    # Pass chat history if your get_assistant_response supports it
    full_response = get_assistant_response(prompt, context, st.session_state.messages)
    
    # Regex to find the link format used in the prompt instructions
    url_match = re.search(r"SOURCE_LINK:\s*(https?://[^\s\]\)\n]+)", full_response)
    source_url = url_match.group(1) if url_match else None
    
    # Remove the metadata tag from the final display
    clean_answer = re.sub(r"SOURCE_LINK:.*", "", full_response).strip()
    return clean_answer, source_url

def load_system_prompts():
    """Loads the agent's persona from docs_agent_prompt.py."""
    try:
        prompt_path = Path(__file__).resolve().parents[1] / "prompts" / "docs_agent_prompt.py"
        with open(prompt_path, "r", encoding="utf-8") as f:
            content = f.read()
        match = re.search(r"docs_agent_prompt\s*=\s*['\"]{3}(.*?)['\"]{3}", content, re.S)
        return match.group(1).strip() if match else "You are a Tarento Assistant."
    except Exception:
        return "You are a Tarento Assistant. Use context to answer concisely."

STRICT_INSTRUCTIONS = load_system_prompts()

# --- 4. UI LAYOUT ---
st.image("https://www.tarento.com/images/logo.png", width=180)
st.title("Tarento Knowledge Assistant")
st.markdown("*Your expert guide to Tarento's Digital Transformation solutions.*")

# Display historical messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("link"):
            st.markdown(f"🔗 **[More about {message.get('title', 'this topic')}]({message['link']})**")

# --- 5. CHAT LOGIC ---
if prompt := st.chat_input("How can Tarento help you today?"):
    # User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistant Response
    with st.chat_message("assistant"):
        with st.status("🔍 Consulting Knowledge Base...", expanded=False) as status:
            citations = search_knowledge_base(prompt)
            
            if citations and citations[0]['score'] > 0.45: # Filter noise
                context_text = "\n\n".join([f"Data: {c['text']} | Link: {c['url']}" for c in citations])
                status.update(label="Information retrieved.", state="complete")
            else:
                context_text = "NO_CONTEXT_AVAILABLE"
                status.update(label="No specific documents found.", state="complete")

        answer, primary_link = get_response_and_link(prompt, context_text, STRICT_INSTRUCTIONS)
        
        # Display clean text
        st.markdown(answer)

        # Handle Relevant Links (THE "HI" FIX)
        final_link = None
        final_title = ""
        
        if primary_link and citations:
            best_match = citations[0]
            # Only show link if the database actually has a strong match
            if best_match['score'] > 0.60:
                final_link = primary_link
                final_title = best_match['title']
                st.markdown(f"🔗 **[More about {final_title}]({final_link})**")

        # Save to Session State
        st.session_state.messages.append({
            "role": "assistant", 
            "content": answer, 
            "link": final_link, 
            "title": final_title
        })