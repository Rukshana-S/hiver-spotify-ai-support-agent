def detect_intent(query: str) -> str:
    """
    Detects the intent of the customer query.
    Note: Replace this with your exact implementation from the notebook if it differs.
    """
    query_lower = query.lower()
    if "billing" in query_lower or "charge" in query_lower or "money" in query_lower or "refund" in query_lower:
        return "Billing Issue"
    elif "password" in query_lower or "login" in query_lower or "account" in query_lower:
        return "Account Access"
    elif "premium" in query_lower or "subscription" in query_lower:
        return "Subscription Issue"
    else:
        return "General Support"
