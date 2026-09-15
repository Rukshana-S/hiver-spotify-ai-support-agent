import re
import requests
from backend.config import OPENROUTER_API_KEY

def clean_text(text: str) -> str:
    """Cleans general text."""
    if not isinstance(text, str):
        return ""
    return re.sub(r'\s+', ' ', text).strip()

def clean_support_text(text: str) -> str:
    """Cleans Spotify-specific support text."""
    text = clean_text(text)
    return text.replace("@SpotifyCares", "").strip()

def summarize_issue(text: str) -> str:
    """Summarizes the support issue."""
    # Dummy summarization logic, replace with notebook implementation
    return text[:150] + "..." if len(text) > 150 else text

def build_prompt(query: str, retrieved_cases: list) -> str:
    """Builds the RAG prompt for the LLM."""
    context = "\n".join([f"- {case['text']}" for case in retrieved_cases])
    return f"Context from past Spotify support cases:\n{context}\n\nCustomer Query: {query}\n\nProvide a helpful, grounded response."

def call_llm(prompt: str) -> str:
    """Calls OpenRouter LLM."""
    if not OPENROUTER_API_KEY:
        return "Error: OPENROUTER_API_KEY is not set."
        
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "openai/gpt-3.5-turbo", # Default model, adjust as needed
        "messages": [{"role": "user", "content": prompt}]
    }
    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        return f"Error generating reply: {str(e)}"

def post_process_reply(reply: str) -> str:
    """Post-processes the generated reply."""
    return reply.strip()

def generate_reply(query: str, retrieved_cases: list) -> str:
    """End-to-end generation combining prompt building and LLM call."""
    prompt = build_prompt(query, retrieved_cases)
    reply = call_llm(prompt)
    return post_process_reply(reply)
