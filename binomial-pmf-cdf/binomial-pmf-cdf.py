
import math

def binomial_pmf_cdf(n: int, p: float, k: int) -> dict:

    pmf = math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k))

    cdf = 0

    for i in range(k + 1):
        probability = math.comb(n, i) * (p ** i) * ((1 - p) ** (n - i))
        cdf += probability

    return {
        "pmf": float(pmf),
        "cdf": float(cdf)
    }
