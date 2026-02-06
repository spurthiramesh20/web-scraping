import json
import os
import re
import streamlit as st
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding
from dotenv import load_dotenv
from gemini import get_assistant_response 

# 1. PAGE CONFIGURATION & THEME
load_dotenv()
st.set_page_config(page_title="Tarento AI", page_icon="🚀", layout="wide")


# 2. INITIALIZE DATABASE & EMBEDDING MODEL
@st.cache_resource
def initialize_system():
    # Setup Qdrant Cloud Client
    client = QdrantClient(
        url=os.getenv("QDRANT_URL"), 
        api_key=os.getenv("QDRANT_API_KEY")
    )
    # Load Local Embedding Model
    model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    return client, model, "tarento_web_data"

client_db, embed_model, COLLECTION_NAME = initialize_system()

# 3. CORE LOGIC FUNCTIONS
def search_knowledge_base(query):
    try:
        query_vec = list(embed_model.embed([query]))[0].tolist()
        results = client_db.query_points(
            collection_name=COLLECTION_NAME, 
            query=query_vec, 
            limit=8  # Higher limit helps Gemini see multiple routing options
        ).points
        
        # Return unique results with a safety threshold
        return [{"text": hit.payload.get("text", ""), "url": hit.payload.get("source_url", "")} 
                for hit in results if hit.score > 0.65]
    except Exception as e:
        print(f"Search Error: {e}")
        return []

def get_response_and_link(prompt, context, instructions):
    # Call Gemini logic
    full_response = get_assistant_response(prompt, context, instructions)
    
    # Extract URL using Regex looking for "SOURCE_LINK: [URL]"
    url_match = re.search(r"SOURCE_LINK:\s*\[?(https?://[^\s\]]+)\]?", full_response)
    
    clean_answer = full_response
    source_url = None
    
    if url_match:
        source_url = url_match.group(1)
        # Clean the text so the raw SOURCE_LINK tag isn't visible to user
        clean_answer = re.sub(r"SOURCE_LINK:.*", "", full_response).strip()
        
    return clean_answer, source_url

def load_system_prompts():
    try:
        with open("prompts_v1.txt", "r") as f: v1 = f.read()
        with open("prompts_v2.txt", "r") as f: v2 = f.read()
        return f"{v1}\n\nSTRICT GUIDELINES:\n{v2}"
    except:
        return "You are a Tarento Assistant. Help users find services and careers."

STRICT_INSTRUCTIONS = load_system_prompts()

# 4. CHAT INTERFACE & HISTORY
st.title("🚀 Tarento AI Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history with persistent buttons
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("link"):
            # Re-generate the correct label based on the stored link
            l_val = msg["link"].lower()
            if "career" in l_val: label = "🎯 View Openings"
            elif "service" in l_val: label = "🛠️ Explore Services"
            elif "contact" in l_val: label = "📞 Contact Info"
            elif "about" in l_val: label = "🏢 About Tarento"
            else: label = "📖 View Details"
            
            st.divider()
            st.link_button(label, msg["link"], use_container_width=True)

# 5. NEW USER INPUT
if prompt := st.chat_input("Ask about Tarento services, careers, or locations..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").markdown(prompt)

    with st.chat_message("assistant"):
        # Handle simple greetings without calling the DB (Clean Greetings)
        if prompt.lower().strip() in ["hi", "hello", "hey", "greetings"]:
            answer = "Hello! I'm the Tarento AI Assistant. How can I help you explore our services, career opportunities, or global offices today?"
            link = None
        else:
            with st.status("Searching Tarento Knowledge...", expanded=False) as status:
                citations = search_knowledge_base(prompt)
                # Formulate context for Gemini
                context_text = "\n".join([f"Data: {c['text']} | Link: {c['url']}" for c in citations])
                
                answer, link = get_response_and_link(prompt, context_text, STRICT_INSTRUCTIONS)
                status.update(label="Ready!", state="complete")

        st.markdown(answer)
        
        # Display the button only if Gemini provided a link
        if link:
            l_val = link.lower()
            if "career" in l_val: label = "🎯 View Openings"
            elif "service" in l_val: label = "🛠️ Explore Services"
            elif "contact" in l_val: label = "📞 Contact Info"
            elif "about" in l_val: label = "🏢 About Tarento"
            else: label = "📖 View Details"
            
            st.divider()
            st.link_button(label, link, use_container_width=True)

        # SAVE ASSISTANT MESSAGE & LINK TO HISTORY
        st.session_state.messages.append({
            "role": "assistant", 
            "content": answer, 
            "link": link
        })