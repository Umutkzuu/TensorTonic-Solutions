import numpy as np

def reshape_array(data: list, operation: str) -> np.ndarray:

    
    data = np.array(data, dtype='float64')
    
    if operation == 'flatten':
        return data.flatten()

    elif operation == 'transpose':
        return data.T

    return np.expand_dims(data, axis=0)
        