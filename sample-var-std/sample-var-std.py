import numpy as np

def sample_var_std(x: list) -> dict:
    x_arr = np.array(x, dtype=float)
    n = len(x_arr)
    
    mean = np.mean(x_arr)
    squared_diff_sum = np.sum((x_arr - mean) ** 2)
    
    variance = squared_diff_sum / (n - 1)
    standard_deviation = np.sqrt(variance)
    
    return {
        "variance": float(variance),
        "standard_deviation": float(standard_deviation)
    }