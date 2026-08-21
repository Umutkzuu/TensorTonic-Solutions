import numpy as np

def cosine_similarity(a: list, b: list) -> float:
    va = np.array(a, dtype=float)
    vb = np.array(b, dtype=float)
    
    dot_val = np.dot(va, vb)
    
    norm_a = np.linalg.norm(va)
    norm_b = np.linalg.norm(vb)
    
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    
    similarity = dot_val / (norm_a * norm_b)
    return float(similarity)