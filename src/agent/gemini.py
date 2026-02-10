import os
import re
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def load_prompt_from_python(path):
    try:
        if not os.path.exists(path):
            return "You are a Tarento assistant. Answer only using Tarento context."
            
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # This regex looks specifically for the content INSIDE the triple quotes
        # It handles 'docs_agent_prompt = ' with single or double triple-quotes
        match = re.search(r"docs_agent_prompt\s*=\s*['\"]{3}(.*?)['\"]{3}", content, re.S)
        
        if match:
            return match.group(1).strip()
        
        # Fallback: if regex fails, we don't want the whole file, just a default
        return "You are a Tarento Solutions Architect. Use the provided context."
    except Exception as e:
        print(f"Error loading prompt: {e}")
        return "You are a Tarento assistant."

# Initialize once
SYSTEM_PROMPT = load_prompt_from_python(os.path.join("prompts", "docs_agent_prompt.py"))

def get_assistant_response(prompt, context, history):
    try:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        
        # Add History
        if isinstance(history, list):
            for h in history[-3:]: # Increased to last 3 for better context
                if isinstance(h, dict):
                    user_val = h.get("user") or h.get("content")
                    ast_val = h.get("assistant") or h.get("role")
                    if user_val and ast_val:
                        messages.append({"role": "user", "content": str(user_val)})
                        messages.append({"role": "assistant", "content": str(ast_val)})

        # Prepare the grounded input
        # Note: We tell the AI clearly what is context vs what is the question
        grounded_input = (
            f"TECHNICAL CONTEXT:\n{context}\n\n"
            f"USER QUESTION: {prompt}"
        )
        messages.append({"role": "user", "content": grounded_input})

        # Groq API Call
        chat_completion = groq_client.chat.completions.create(
            messages=messages,
            model="llama-3.1-8b-instant",
            temperature=0, # Keep it strictly factual
        )
        
        return chat_completion.choices[0].message.content

    except Exception as e:
        return f"System Error: {str(e)}"
