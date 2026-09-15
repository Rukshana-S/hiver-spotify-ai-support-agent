# Engineering Decision Log

This document tracks the major engineering decisions made while building the Hiver AI Support Agent (Spotify Edition).

## Decision 1: Used RAG instead of fine-tuning

**Reason:**
Customer support requires accurate, factual answers. Fine-tuning a model can lead to hallucinations where the AI confidently invents policies or solutions. By using Retrieval-Augmented Generation (RAG) with historical Spotify conversations, we guarantee grounded responses based strictly on actual, resolved support cases.

## Decision 2: Used FAISS for Vector Search

**Reason:**
We required fast, scalable semantic retrieval. FAISS (Facebook AI Similarity Search) is extremely optimized for CPU-based similarity search, making it an excellent choice for sub-millisecond retrieval of the most relevant historical cases directly from the provided embedded dataset without the overhead of spinning up a dedicated vector database server.

## Decision 3: Confidence-based escalation

**Reason:**
Certain issues, particularly related to billing and payments, require strict human verification and cannot be fully automated yet. By mapping the FAISS L2 retrieval distance to a confidence score, we established a clear threshold. If the system isn't highly confident, or if the intent is sensitive (like "Billing Issue"), the system automatically escalates to a human to prevent poor customer experiences.

## Decision 4: Cleaned Twitter conversations dataset

**Reason:**
The raw Twitter dataset contains noise (e.g., emojis, usernames like @SpotifyCares, informal text). To improve the enterprise user experience (UX) and preserve retrieval quality, we meticulously cleaned this dataset. Clean context ensures that Sentence Transformers can generate higher-quality embeddings and the LLM can synthesize much more professional replies.
