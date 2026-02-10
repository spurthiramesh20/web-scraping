import os
import streamlit as st
from qdrant_client import QdrantClient
from fastembed import TextEmbedding
from groq import Groq  
from dotenv import load_dotenv
from fastembed import TextEmbedding

load_dotenv()

# 1. Initialize Clients
# No need for LangChain wrappers here
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
qdrant_client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
embed_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

COLLECTION_NAME = "tarento_web_data"

st.title("⚡ Tarento Assistant")

# --- Retrieval & Chat Logic ---
if prompt := st.chat_input("Ask about Tarento..."):
    with st.chat_message("user"):
        st.markdown(prompt)

    # A. Search Qdrant
    q_vec = list(embed_model.embed([prompt]))[0].tolist()
    search_results = qdrant_client.search(
        collection_name=COLLECTION_NAME, 
        query_vector=q_vec, 
        limit=3
    )
    context = "\n".join([res.payload['text'] for res in search_results])

    # B. Generate with Groq SDK
    with st.chat_message("assistant"):
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": f"You are a Tarento expert. Use this context: {context}"
                },
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="llama-3.3-70b-versatile", 
        )
        
        response_text = chat_completion.choices[0].message.content
        st.markdown(response_text)