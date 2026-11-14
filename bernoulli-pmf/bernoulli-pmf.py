
import numpy as np

def bernoulli_pmf_and_moments(x: list, p: float) -> dict:
    x = np.array(x)

    pmf = np.where(x == 1, p, 1 - p)

    return {
        "pmf": pmf,
        "mean": float(p),
        "variance": float(p * (1 - p))
    }
