import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def load_prompt(filename):
    """Utility to load prompt files."""
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return "You are a Tarento assistant. Answer only using Tarento context."

# Load your strict prompt files
SYSTEM_PROMPT = load_prompt("prompts_v1.txt")
GUARDRAIL_PROMPT = load_prompt("prompts_v2.txt")

def get_assistant_response(prompt, context, history):
    """
    Strictly filters out general questions and only answers Tarento-related queries.
    """
    try:
        # Step 1: Combine the system persona with the guardrail rules from v2
        full_system_instruction = f"{SYSTEM_PROMPT}\n\n{GUARDRAIL_PROMPT}"
        
        # Step 2: Create the grounded user message
        grounded_input = f"Context provided from Tarento database:\n{context}\n\nUser Question: {prompt}"

        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": full_system_instruction},
                {"role": "user", "content": grounded_input}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0, # Keep it deterministic
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"System Error: {str(e)}"
    
