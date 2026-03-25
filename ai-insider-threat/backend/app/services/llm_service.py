def analyze_text_intent(text: str) -> float:
    """
    Mock LLM service that estimates the "malicious intent" of text.
    In a real system, this would call an LLM API (OpenAI, Anthropic, etc.)
    and use semantic analysis. Here we use keyword matching heuristics to simulate it.
    Returns a score between 0.0 (safe) and 1.0 (highly suspicious).
    """
    if not text:
        return 0.0
        
    text_lower = text.lower()
    
    suspicious_keywords = [
        'resignation', 'confidential', 'source code', 'export', 
        'password', 'wire transfer', 'urgent', 'ip', 'customer database'
    ]
    
    score = 0.0
    for keyword in suspicious_keywords:
        if keyword in text_lower:
            score += 0.3
            
    return min(1.0, score)
