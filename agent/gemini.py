import os
import re
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def load_prompt_from_python(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        match = re.search(r"docs_agent_prompt\\s*=\\s*'''(.*)'''", content, re.S)
        if match:
            return match.group(1).strip()
        match = re.search(r'docs_agent_prompt\\s*=\\s*\"\"\"(.*)\"\"\"', content, re.S)
        if match:
            return match.group(1).strip()
        return content.strip()
    except Exception:
        return "You are a Tarento assistant. Answer only using Tarento context."


SYSTEM_PROMPT = load_prompt_from_python(os.path.join("prompts", "docs_agent_prompt.py"))


def get_assistant_response(prompt, context, history):
    """
    Strictly filters out general questions and only answers Tarento-related queries.
    """
    try:
        grounded_input = f"Context provided from Tarento database:\n{context}\n\nUser Question: {prompt}"

        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": grounded_input},
            ],
            model="llama-3.3-70b-versatile",
            temperature=0,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"System Error: {str(e)}"
