def confidence_from_distance(distance: float) -> float:
    """
    Converts a FAISS L2 distance score into a 0-100 confidence percentage.
    Note: Replace this with your exact formula from the notebook.
    """
    # Example heuristic: smaller distance -> higher confidence
    confidence = max(0.0, 100.0 - (distance * 10))
    return round(confidence, 2)

def decide_action(intent: str, confidence: float) -> str:
    """
    Decides whether to Auto Reply or Escalate based on intent and confidence.
    Note: Replace this with your exact logic from the notebook.
    """
    if confidence < 75.0 or intent in ["Billing Issue", "Account Access"]:
        return "Escalate"
    return "Auto Reply"
