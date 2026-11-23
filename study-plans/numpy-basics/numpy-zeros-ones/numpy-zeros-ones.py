import numpy as np

def create_filled_array(shape: list, kind: str) -> np.ndarray:

    
    if kind == 'zeros':
        return np.zeros(shape, dtype=np.float64)
    return np.ones(shape, dtype=np.float64)