"""
measures and compares the execution time of the old loop based
and new vectorized compute_accelerations function
"""

import numpy as np
import time
from three_body.physics import compute_accelerations, G, SOFTENING

def compute_accelerations_loop(positions, masses):
    """
    Original loop-based implementation for comparison
    """
    accelerations = np.zeros((3, 2))
    for i in range(3):
        for j in range(3):
            if i != j:
                delta = positions[j] - positions[i]
                dist = np.sqrt(np.dot(delta, delta) + SOFTENING ** 2)
                accelerations[i] += G * masses[j] * delta / dist ** 3
    return accelerations


def benchmark(func, positions, masses, n=10000):
    """
    benchmark function that tests the execution time of desired function over n iterations
    """
    start = time.perf_counter()
    for _ in range(n): # repetition only, no real reused variable
        func(positions, masses)
    return time.perf_counter() - start


if __name__ == "__main__":
    positions = np.array([
        [-5.0,  0.0],
        [ 5.0,  0.0],
        [ 0.0,  5.0],
    ])
    masses = np.array([1.0, 2.0, 3.0])
    n = 10000 # no. of test steps

    """
    run both
    """
    t_loop = benchmark(compute_accelerations_loop, positions, masses, n)
    t_vec  = benchmark(compute_accelerations,       positions, masses, n)

    print(f"Loop-based : {t_loop:.4f}s over {n} calls")
    print(f"Vectorised : {t_vec:.4f}s over {n} calls")
    print(f"Comparison    : {t_loop / t_vec:.1f}x faster")