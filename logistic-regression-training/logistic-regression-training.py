import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    N, D = X.shape
    
    w = np.zeros(D, dtype=np.float64)
    b = 0.0
    
    y = np.asarray(y, dtype=np.float64)
    
    for _ in range(steps):
        z = np.dot(X, w) + b
        p = _sigmoid(z)
        
        error = p - y
        
      
        dw = np.dot(X.T, error) / N
        db = np.mean(error)
        
        w -= lr * dw
        b -= lr * db
        
    return w, float(b)