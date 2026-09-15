import pandas as pd
import numpy as np
import faiss
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from backend.intent import detect_intent
from backend.retrieval import retrieve_cases
from backend.escalation import confidence_from_distance, decide_action
from backend.rag import generate_reply

app = FastAPI(title="Hiver Spotify AI Support Agent")

# Global variables to hold models and data loaded at startup
embedding_model = None
faiss_index = None
df = None

@app.on_event("startup")
def load_resources():
    """Loads models and datasets on server startup."""
    global embedding_model, faiss_index, df
    try:
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        # Expecting these files to be provided by the user from their notebook run
        faiss_index = faiss.read_index("models/spotify_faiss.index")
        df = pd.read_csv("processed/conversation_pairs_cleaned.csv")
        print("Models and data loaded successfully.")
    except Exception as e:
        print(f"Warning: Could not load models or data. Please ensure 'models/spotify_faiss.index' and 'processed/conversation_pairs_cleaned.csv' exist. Error: {e}")

class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    query: str
    intent: str
    confidence: float
    action: str
    reason: str
    reply: str
    retrieved_cases: list

@app.get("/")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "models_loaded": faiss_index is not None}

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    """Main support endpoint for processing user queries."""
    query = request.query
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
        
    intent = detect_intent(query)
    
    retrieved_cases = []
    confidence = 0.0
    
    # Only retrieve if models are successfully loaded
    if embedding_model and faiss_index is not None and df is not None:
        query_embedding = embedding_model.encode(query)
        retrieved_cases = retrieve_cases(query_embedding, faiss_index, df)
        if retrieved_cases:
            # Calculate confidence using the top retrieved case's distance
            top_distance = retrieved_cases[0]["distance"]
            confidence = confidence_from_distance(top_distance)
    else:
        # Fallback confidence if models are missing
        confidence = 50.0

    action = decide_action(intent, confidence)
    
    reply = ""
    reason = ""
    
    if action == "Escalate":
        reason = "Confidence is too low or a sensitive intent was detected (e.g., Billing)."
        reply = "I'm escalating this issue to a human support agent who can assist you further."
    else:
        reply = generate_reply(query, retrieved_cases)
        reason = "High confidence and safe intent."
        
    return ChatResponse(
        query=query,
        intent=intent,
        confidence=confidence,
        action=action,
        reason=reason,
        reply=reply,
        retrieved_cases=retrieved_cases
    )
