import numpy as np

def matrix_transpose(A: list) -> np.ndarray:
    A = np.array(A)
    n, m = A.shape  
    
    B = np.zeros((m, n), dtype=A.dtype)
    
    for i in range(n):
        for j in range(m):
            B[j, i] = A[i, j]
            
    return B