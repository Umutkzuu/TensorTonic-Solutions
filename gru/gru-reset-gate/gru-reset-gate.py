import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def reset_gate(h_prev: np.ndarray, x_t: np.ndarray,
               W_r: np.ndarray, b_r: np.ndarray) -> np.ndarray:
    """
    Compute reset gate: r_t = sigmoid(W_r @ [h, x] + b_r)
    """
    # np.hstack hem 1D vektörleri ([H] ve [D]) yan yana ekleyip [H+D] yapar,
    # hem de 2D matrisleri ([batch, H] ve [batch, D]) yan yana ekleyip [batch, H+D] yapar.
    h_x = np.hstack((h_prev, x_t))
    
    # Girdinin boyutuna göre dinamik matris çarpımı:
    # Eğer h_x 1 boyutluysa standart W_r @ h_x, 2 boyutluysa (batch ise) h_x @ W_r.T yaparız.
    if h_x.ndim == 1:
        linear_transform = W_r @ h_x + b_r
    else:
        linear_transform = h_x @ W_r.T + b_r
        
    return sigmoid(linear_transform)