# Hiver-Spotify-AI

**AI-powered Customer Support Agent using Retrieval-Augmented Generation (RAG)**

A production-style AI support assistant that analyzes customer queries, retrieves similar historical Spotify support conversations using semantic search, generates grounded responses with an LLM, and intelligently decides whether the conversation should be automatically handled or escalated to a human support specialist.

> Built as a Hiver AI Engineering assignment to demonstrate Retrieval-Augmented Generation (RAG), semantic search, confidence-based decision making, and explainable AI workflows.

---

## Features

- Intent Detection – Identifies customer issues such as Premium Billing, Account Access, Playlist Issues, and Subscription Problems.
- Semantic Search – Retrieves the most relevant historical Spotify support conversations using Sentence Transformers and FAISS.
- Grounded AI Responses – Generates responses based on retrieved evidence instead of hallucinating answers.
- Confidence Scoring – Converts retrieval similarity into an interpretable confidence score.
- Smart Escalation – Automatically flags sensitive cases like billing and account verification for human support.
- Explainable Evidence – Shows the historical support cases that influenced the generated response.
- FastAPI Backend – Exposes the complete AI workflow through REST APIs with Swagger documentation.

---

## Problem Statement

Customer support teams repeatedly receive similar questions about Premium subscriptions, billing, account access, and payments. Traditional chatbots often generate generic responses without referencing previous successful resolutions.

This project solves that problem by combining:

- Historical Spotify support conversations
- Semantic retrieval using FAISS
- Large Language Models (LLMs)
- Confidence-based escalation

The result is an AI assistant that behaves more like a real support agent than a generic chatbot.

---

## System Architecture

<svg viewBox="0 0 820 500" width="100%" height="500" xmlns="http://www.w3.org/2000/svg">
  <rect x="210" y="20" width="400" height="54" rx="12" fill="none" stroke="currentColor" stroke-width="2"/>
  <text x="410" y="52" text-anchor="middle" font-size="18" font-weight="600">Customer Query</text>

  <path d="M410 74 L410 110" stroke="currentColor" stroke-width="2"/>

  <rect x="170" y="110" width="480" height="54" rx="12" fill="none" stroke="currentColor" stroke-width="2"/>
  <text x="410" y="142" text-anchor="middle" font-size="15" font-weight="600">Sentence Transformer Embeddings</text>

  <path d="M410 164 L410 200" stroke="currentColor" stroke-width="2"/>

  <rect x="250" y="200" width="320" height="54" rx="12" fill="none" stroke="currentColor" stroke-width="2"/>
  <text x="410" y="232" text-anchor="middle" font-size="15" font-weight="600">FAISS Retrieval</text>

  <path d="M410 254 L410 290" stroke="currentColor" stroke-width="2"/>

  <rect x="160" y="290" width="500" height="54" rx="12" fill="none" stroke="currentColor" stroke-width="2"/>
  <text x="410" y="322" text-anchor="middle" font-size="15" font-weight="600">OpenRouter LLM</text>

  <path d="M410 344 L410 380" stroke="currentColor" stroke-width="2"/>

  <rect x="175" y="380" width="470" height="54" rx="12" fill="none" stroke="currentColor" stroke-width="2"/>
  <text x="410" y="412" text-anchor="middle" font-size="15" font-weight="600">Confidence & Escalation Decision</text>

  <path d="M410 434 L410 470" stroke="currentColor" stroke-width="2"/>

  <rect x="210" y="470" width="400" height="22" rx="10" fill="none" stroke="currentColor" stroke-width="2"/>
  <text x="410" y="486" text-anchor="middle" font-size="12" font-weight="600">Grounded AI Response + Retrieved Evidence</text>
</svg>

---

## Tech Stack

| Layer | Technology |
|--------|------------|
| Programming Language | Python |
| Backend | FastAPI |
| Vector Search | FAISS |
| Embeddings | Sentence Transformers (`all-MiniLM-L6-v2`) |
| LLM | OpenRouter |
| Dataset | Customer Support on Twitter (Spotify) |
| Data Processing | Pandas |
| API Testing | Swagger UI |

---

## Project Structure

```text
Hiver-Spotify-AI/
│
├── backend/
│   ├── app.py
│   ├── rag.py
│   ├── retrieval.py
│   ├── intent.py
│   ├── escalation.py
│   ├── config.py
│   ├── requirements.txt
│   └── models/
│
├── notebooks/
│   └── 01_spotify_rag.ipynb
│
├── models/
│   ├── spotify_faiss.index
│   └── spotify_embeddings.npy
│
├── processed/
│   ├── conversation_pairs_cleaned.csv
│   └── golden_dataset_labeled.csv
│
├── README.md
├── DECISION_LOG.md
├── .gitignore
└── LICENSE
```

---

## Dataset

**Primary Dataset**

**Customer Support on Twitter**

This dataset contains millions of real customer support conversations across multiple brands. This project specifically extracts Spotify conversations and reconstructs customer-agent conversation pairs.

### Dataset Processing

- Extracted Spotify-specific conversations
- Reconstructed conversation threads
- Created customer-agent response pairs
- Cleaned Twitter-specific artifacts
- Generated semantic embeddings
- Built a FAISS vector index

---

## AI Pipeline

### Step 1 — Customer Query

Example:

> "I paid for Premium but my subscription disappeared."

### Step 2 — Intent Detection

The system classifies the query into categories such as:

- Premium Billing
- Account Access
- Playlist Issue
- Subscription Problem

### Step 3 — Semantic Retrieval

The query is converted into an embedding using Sentence Transformers.

FAISS retrieves the three most similar historical Spotify support conversations.

### Step 4 — Response Generation

The retrieved cases become context for the LLM.

Instead of generating unsupported answers, the model produces a grounded response.

### Step 5 — Confidence

Retrieval distance is converted into a confidence score.

Example:

- Distance: `0.185`
- Confidence: `95%`

### Step 6 — Escalation

Billing and account-verification requests are automatically escalated, even when confidence is high.

---

## Example API Response

```json
{
  "request_id": "5421188e-1035-4d5e-8a88-a5f10fd0ae25",
  "timestamp": "2026-09-15T12:45:06Z",
  "status": "success",
  "query": "I got charged twice for my Premium subscription this month!",
  "intent": "Premium Billing",
  "confidence": 96.9,
  "decision": {
    "action": "Escalate",
    "reason": "Billing requests require secure account verification."
  },
  "reply": "Thanks for reaching out. We're sorry you've been charged twice for your Premium subscription. Please continue through private support so a specialist can verify your payment and resolve this securely.",
  "evidence": [
    {
      "case_id": 7844,
      "customer_issue": "Customer was charged twice for Premium.",
      "resolution": "Support requested secure account verification before investigating.",
      "similarity": 96.9
    }
  ]
}
```

---

## FastAPI

### Run locally

```bash
git clone https://github.com/<username>/Hiver-Spotify-AI.git

cd Hiver-Spotify-AI

pip install -r backend/requirements.txt

uvicorn backend.app:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

Swagger UI automatically documents the API.

---

## API Endpoints

### `GET /`

Health Check

Returns the API status.

### `POST /chat`

Generates an AI-assisted support response.

Request

```json
{
  "message": "I got charged twice for my Premium subscription."
}
```

Returns:

- Intent
- Confidence
- Decision
- AI Reply
- Retrieved Evidence

---

## Confidence Scoring

The system converts FAISS retrieval distance into an interpretable confidence score.

| Distance | Confidence |
|----------|------------|
| ≤0.20 | 95% |
| ≤0.35 | 85% |
| ≤0.50 | 70% |
| ≤0.70 | 55% |
| >0.70 | 35% |

This allows support agents to understand how strongly the retrieved evidence matches the customer query.

---

## Escalation Logic

Sensitive intents are intentionally routed to human support.

Examples:

| Intent | Action |
|---------|---------|
| Premium Billing | Escalate |
| Refund Request | Escalate |
| Account Access | Escalate |
| Playlist Issue | Auto Reply |

This mirrors real customer-support workflows where account verification is required.

---

## Engineering Decisions

See `DECISION_LOG.md` for detailed explanations.

Key decisions include:

- Retrieval-Augmented Generation instead of fine-tuning
- FAISS for semantic search
- Confidence-based escalation
- Cleaning Twitter conversations into enterprise knowledge-base entries

---

## Future Improvements

- Multi-brand customer support
- Advanced intent classification using supervised models
- Admin dashboard for support agents
- Conversation analytics
- Real-time conversation memory
- Feedback-driven response optimization

---

## Author

**Rukshana S**

Computer Science & Engineering

Built as part of the **Hiver AI Engineering Assignment**.

---

## License

This project is licensed under the **MIT License**.
