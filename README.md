# Hiver AI Support Agent - Spotify Edition

An AI-powered customer support agent built to resolve Spotify Customer Support Twitter conversations autonomously using RAG, Semantic Search, and Confidence-based Human Escalation.

![Final Dashboard Screenshot Placeholder](screenshots/dashboard_placeholder.png)

## Problem Statement

Customer support teams receive repetitive Premium Billing, Account Access, and Subscription issues. Generating unsupported, hallucinated answers creates bad customer experiences.

Our system solves this by retrieving historical Spotify cases using FAISS and synthesizing a grounded response using RAG, instead of just guessing answers.

## Features

- **Intent Detection**: Categorizes queries into Billing, Account Access, Subscription, or General.
- **Semantic Retrieval**: Uses Sentence Transformers to embed queries and FAISS for sub-millisecond retrieval of historical case context.
- **Grounded AI Responses**: OpenRouter LLM uses retrieved context to build helpful, factual replies.
- **Confidence Scoring**: Dynamically calculates confidence based on FAISS L2 distance.
- **Escalation Workflow**: Automatically escalates to a human agent if confidence is low or if the issue is highly sensitive (e.g., Billing).
- **Explainable Evidence**: Surfaces the retrieved historical cases to explain *why* the AI generated a specific reply.

## Architecture

```
Customer Query
      │
Intent Detection
      │
Sentence Transformer
      │
FAISS Retrieval
      │
OpenRouter
      │
Confidence
      │
Escalation
      │
Grounded Response
```

## Tech Stack

| Layer | Technology |
| --- | --- |
| Language | Python |
| Backend | FastAPI |
| Vector Search | FAISS |
| Embeddings | Sentence Transformers |
| LLM | OpenRouter |
| Dataset | Spotify Customer Support |
| Documentation | Markdown |
| Version Control | GitHub |

## Installation

```bash
git clone https://github.com/Rukshana-S/hiver-spotify-ai-support-agent.git
cd hiver-spotify-ai-support-agent
pip install -r backend/requirements.txt
# Ensure your model and data files are in models/ and processed/
# Create a .env file with OPENROUTER_API_KEY
uvicorn backend.app:app --reload
```

## Example Output

Based on our existing Hiver dashboard output format:

```json
{
  "query": "I got double charged for my premium this month!",
  "intent": "Billing Issue",
  "confidence": 88.5,
  "action": "Escalate",
  "reason": "Confidence too low or sensitive intent detected.",
  "reply": "I'm escalating this issue to a human agent who can help you further.",
  "retrieved_cases": [
    {
      "case_id": 142,
      "text": "User was double charged, refund processed...",
      "distance": 1.15
    }
  ]
}
```

## Folder Structure

- `backend/`: Contains the FastAPI application and the core AI engine (`intent.py`, `retrieval.py`, `rag.py`, `escalation.py`).
- `backend/models/`: Internal schemas or Pydantic models (if expanded).
- `notebooks/`: Jupyter notebooks used for initial EDA and pipeline prototyping (e.g., `01_spotify_rag.ipynb`).
- `models/`: Stores the FAISS vector index and serialized embeddings (`spotify_faiss.index`, `spotify_embeddings.npy`).
- `processed/`: Cleaned CSV datasets used for retrieval (`conversation_pairs_cleaned.csv`, `golden_dataset_labeled.csv`).
- `screenshots/`: Images for documentation and demo purposes.

## Future Improvements

- **Multi-brand support**: Expand beyond Spotify to handle context for multiple companies.
- **Better intent classifier**: Train a specialized BERT model instead of heuristic-based classification.
- **Admin dashboard**: A frontend interface for human agents to manage escalated tickets.
- **Conversation analytics**: Track deflection rates and user satisfaction over time.