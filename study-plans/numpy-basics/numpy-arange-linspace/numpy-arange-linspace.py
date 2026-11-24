import numpy as np

def create_sequence(start: float, stop: float, param: float, kind: str) -> np.ndarray:


    if kind == 'linspace':
        return np.linspace(start,stop,num=int(param))

    return np.arange(start,stop, param, dtype=np.float64)
                                