import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def reset_gate(h_prev: np.ndarray, x_t: np.ndarray,
               W_r: np.ndarray, b_r: np.ndarray) -> np.ndarray:
    """
    Compute reset gate: r_t = sigmoid(W_r @ [h, x] + b_r)
    """
    
    h_x = np.hstack((h_prev, x_t))
   
    if h_x.ndim == 1:
        linear_transform = W_r @ h_x + b_r
    else:
        linear_transform = h_x @ W_r.T + b_r
        
    return sigmoid(linear_transform)
