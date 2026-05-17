import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def update_gate(h_prev: np.ndarray, x_t: np.ndarray,
                W_z: np.ndarray, b_z: np.ndarray) -> np.ndarray:
    """
    Compute update gate: z_t = sigmoid(W_z @ [h, x] + b_z)
    """
    h_x = np.hstack((h_prev, x_t))
    

    if h_x.ndim == 1:
        linear_transform = W_z @ h_x + b_z
    else:
        linear_transform = h_x @ W_z.T + b_z
        
    return sigmoid(linear_transform)