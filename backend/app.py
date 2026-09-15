import pandas as pd
import numpy as np
import faiss
import uuid
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, validator
from sentence_transformers import SentenceTransformer

from backend.intent import detect_intent
from backend.retrieval import retrieve_cases
from backend.escalation import confidence_from_distance, decide_action
from backend.rag import generate_reply

app = FastAPI(
    title="Hiver Spotify AI Support Agent",
    description="Enterprise-grade AI support API for autonomous Spotify customer service resolution.",
    version="1.0.0",
    contact={
        "name": "Hiver AI Engineering Team",
        "url": "https://hiverhq.com",
    },
)

# Exception Handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Returns a structured 400 error for validation failures."""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"status": "error", "message": "Customer message cannot be empty or invalid."},
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Returns a structured 500 error for internal server errors."""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"status": "error", "message": "Unable to generate a support response."},
    )

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
        faiss_index = faiss.read_index("models/spotify_faiss.index")
        df = pd.read_csv("processed/conversation_pairs_cleaned.csv")
        print("Models and data loaded successfully.")
    except Exception as e:
        print(f"Warning: Could not load models or data. Error: {e}")

# Pydantic Models
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=5, max_length=1500, description="The customer's support message.", example="I got charged twice for my Premium subscription this month!")
    
    @validator('message')
    def message_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError('Customer message cannot be empty.')
        return v.strip()

class Decision(BaseModel):
    action: str = Field(..., description="The action taken by the AI (e.g., Auto Reply, Escalate).", example="Escalate")
    reason: str = Field(..., description="The reasoning behind the decision.", example="Sensitive billing requests require secure account verification.")

class EvidenceItem(BaseModel):
    case_id: int = Field(..., description="The historical case ID.")
    customer_issue: str = Field(..., description="Summary of the historical customer issue.")
    resolution: str = Field(..., description="How the historical issue was resolved.")
    similarity: float = Field(..., description="Similarity percentage to the current query.", example=96.86)

class ChatResponse(BaseModel):
    request_id: str = Field(..., description="Unique UUID for this request.")
    timestamp: str = Field(..., description="ISO-8601 UTC timestamp.")
    status: str = Field("success", description="Status of the request.")
    query: str = Field(..., description="The sanitized customer query.")
    intent: str = Field(..., description="The classified intent of the query.")
    confidence: float = Field(..., description="Confidence score of the AI resolution (0-100).", example=96.86)
    decision: Decision
    reply: str = Field(..., description="The generated AI response or escalation message.")
    evidence: list[EvidenceItem] = Field(default_factory=list, description="Knowledge-base articles mapped from historical cases.")

@app.get("/", tags=["Health"])
def health_check():
    """Health check endpoint to verify the service is running."""
    return {"status": "healthy", "models_loaded": faiss_index is not None}

@app.post("/chat", response_model=ChatResponse, tags=["Support AI"], operation_id="process_customer_query", summary="Process a customer support query")
def chat_endpoint(request: ChatRequest):
    """
    Processes an incoming customer support message.
    
    This endpoint:
    - Analyzes the intent of the message.
    - Searches the FAISS index for similar historical Spotify cases.
    - Uses RAG to generate a professional support reply.
    - Calculates a confidence score.
    - Escalates to a human agent if the confidence is low or the issue is sensitive.
    """
    query = request.message
    intent = detect_intent(query)
    
    retrieved_cases = []
    confidence = 0.0
    
    if embedding_model and faiss_index is not None and df is not None:
        query_embedding = embedding_model.encode(query)
        retrieved_cases = retrieve_cases(query_embedding, faiss_index, df)
        if retrieved_cases:
            top_distance = retrieved_cases[0]["distance"]
            confidence = confidence_from_distance(top_distance)
    else:
        confidence = 50.0

    action = decide_action(intent, confidence)
    
    reply = ""
    reason = ""
    evidence_list = []
    
    # Process Evidence into Knowledge Base Format
    for case in retrieved_cases:
        raw_text = case.get("text", "")
        customer_issue = raw_text
        resolution = "No documented resolution available."
        
        # If the text was formatted by retrieval.py as "Customer: ... | Agent: ..."
        if " | Agent: " in raw_text:
            parts = raw_text.split(" | Agent: ")
            customer_issue = parts[0].replace("Customer: ", "").strip()
            resolution = parts[1].strip()
            
        similarity = confidence_from_distance(case.get("distance", 10.0))
            
        evidence_list.append(EvidenceItem(
            case_id=case.get("case_id"),
            customer_issue=customer_issue,
            resolution=resolution,
            similarity=similarity
        ))

    if action == "Escalate":
        reason = "Sensitive requests (like billing) or low confidence issues require secure human verification."
        reply = "I'm escalating this issue to a human support agent who can securely assist you further."
    else:
        reason = "High confidence historical match and safe intent."
        # Inject professional persona into the query without modifying core RAG logic
        professional_query = f"Act as a professional Hiver email support agent. Do not use Twitter slang, DMs, or agent initials. Here is the customer issue: {query}"
        reply = generate_reply(professional_query, retrieved_cases)
        
    return ChatResponse(
        request_id=str(uuid.uuid4()),
        timestamp=datetime.now(timezone.utc).isoformat(),
        status="success",
        query=query,
        intent=intent,
        confidence=confidence,
        decision=Decision(action=action, reason=reason),
        reply=reply,
        evidence=evidence_list
    )
