import numpy as np

def candidate_hidden(h_prev: np.ndarray, x_t: np.ndarray, r_t: np.ndarray,
                     W_h: np.ndarray, b_h: np.ndarray) -> np.ndarray:
    r_h = r_t * h_prev
    r_h_x = np.hstack((r_h, x_t))
    
    if r_h_x.ndim == 1:
        linear_transform = W_h @ r_h_x + b_h
    else:
        linear_transform = r_h_x @ W_h.T + b_h
        
    return np.tanh(linear_transform)