"""
measures and compares the execution time of the old loop based
and new vectorized compute_accelerations function
"""

import numpy as np
import time
from three_body.physics import compute_accelerations, G, SOFTENING

def compute_accelerations_loop(positions, masses):
    """
    Loop-based implementation of compute_accelerations for comparison.

    Parameters
    -
    positions : np.ndarray, shape (3, 2)
        Positions of the three bodies.
    masses : np.ndarray, shape (3,)
        Masses of the three bodies.

    Returns
    -
    np.ndarray, shape (3, 2)
        Acceleration vectors [ax, ay] for each body.
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
    Run a function n times and return total elapsed time in seconds.

    Parameters
    -
    func : callable
        Function to benchmark. Must accept (positions, masses) as arguments.
    positions : np.ndarray, shape (3, 2)
        Positions passed to func on each call.
    masses : np.ndarray, shape (3,)
        Masses passed to func on each call.
    n : int
        Number of times to call func. Default is 10000.

    Returns
    -
    float
        Total elapsed time in seconds over all n calls.
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

    # run both configs and compare
    t_loop = benchmark(compute_accelerations_loop, positions, masses, n)
    t_vec  = benchmark(compute_accelerations,       positions, masses, n)

    print(f"Loop-based : {t_loop:.4f}s over {n} calls")
    print(f"Vectorised : {t_vec:.4f}s over {n} calls")
    print(f"Speedup    : {t_loop / t_vec:.1f}x faster")