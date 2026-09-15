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
            # Try to fetch text from a likely column name, adjust if your dataset differs
            case_text = df.iloc[idx]['text'] if 'text' in df.columns else f"Sample case {idx}"
            retrieved.append({
                "case_id": int(idx),
                "text": str(case_text),
                "distance": float(distances[0][i])
            })
    return retrieved
