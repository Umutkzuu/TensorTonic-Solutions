import numpy as np

def linear_layer_forward(X: list, W: list, b: list) -> list:
    X_arr = np.array(X)
    W_arr = np.array(W)
    b_arr = np.array(b)
    
    Y = np.dot(X_arr, W_arr) + b_arr
    
    return Y.tolist()