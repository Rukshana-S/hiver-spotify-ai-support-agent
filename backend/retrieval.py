import numpy as np
import pandas as pd
import faiss

def retrieve_cases(query_embedding: np.ndarray, index: faiss.Index, df: pd.DataFrame, top_k: int = 3) -> list:
    """
    Retrieves the most similar historical cases using FAISS.
    Note: Replace this with your exact implementation from the notebook.
    """
    # Ensure query_embedding is a 2D float32 array
    query_vector = np.array([query_embedding]).astype('float32')
    distances, indices = index.search(query_vector, top_k)
    
    retrieved = []
    # If the index is empty or didn't find results, indices[0] might be -1
    for i, idx in enumerate(indices[0]):
        if idx != -1 and idx < len(df):
            # Fetch customer message and reply for rich context
            if 'customer_message' in df.columns and 'spotify_reply' in df.columns:
                case_text = f"Customer: {df.iloc[idx]['customer_message']} | Agent: {df.iloc[idx]['spotify_reply']}"
            elif 'clean_customer_message' in df.columns:
                case_text = df.iloc[idx]['clean_customer_message']
            else:
                case_text = f"Sample case {idx}"
            retrieved.append({
                "case_id": int(idx),
                "text": str(case_text),
                "distance": float(distances[0][i])
            })
    return retrieved
