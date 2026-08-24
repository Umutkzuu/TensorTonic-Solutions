from collections import Counter
import numpy as np

def mean_median_mode(x: list) -> dict:

    total = 0
    for num in x:
        total += num

    mean = float(total / len(x))

    sorted_x = sorted(x)

    if len(sorted_x) % 2 == 1:
        median1 = len(sorted_x) // 2 
        median = float(sorted_x[median1])
    else:
        median = float((sorted_x[len(sorted_x) // 2] + sorted_x[(len(sorted_x) // 2) - 1]) / 2.0)

    counts = {}
    for i in range(len(x)):
        eleman = x[i]
        counts[eleman] = counts.get(eleman, 0) + 1

    max_sayac = 0
    mode = None
    for eleman, sayi in counts.items():
        if sayi > max_sayac:
            max_sayac = sayi
            mode = eleman

    return {
        "mean": mean,
        "median": median,
        "mode": float(mode)
    }