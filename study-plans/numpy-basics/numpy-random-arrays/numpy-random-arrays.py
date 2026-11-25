import numpy as np

def generate_random_array(shape: list, kind: str, seed: int) -> np.ndarray:

    
    rng = np.random.default_rng(seed)
    if kind == 'uniform':
        return rng.random(tuple(shape), dtype=np.float64)
    return rng.standard_normal(tuple(shape), dtype=np.float64)