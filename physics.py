"""
Pure physics functions for the three-body simulator.
All functions operate on numpy arrays.
"""

import numpy as np

# ----- Constants -----

G = 1.0          # Gravitational constant (simulation units)

# ----- Physics core -----

def gravitational_force(pos1: np.ndarray, pos2: np.ndarray,
                        mass1: float, mass2: float) -> np.ndarray:
    """
    Compute the gravitational force vector exerted on body 1 by body 2.

    Parameters
    ----------
    pos1 : np.ndarray, shape (2,)
        Position of body 1 [x, y].
    pos2 : np.ndarray, shape (2,)
        Position of body 2 [x, y].
    mass1 : float
        Mass of body 1.
    mass2 : float
        Mass of body 2.

    Returns
    -------
    """

def compute_accelerations(positions: np.ndarray,
                           masses: np.ndarray) -> np.ndarray:
    accelerations = np.zeros((3, 2))
    for i in range(3):
        for j in range(3):
            if i != j:
                delta = positions[j] - positions[i]
                dist = np.sqrt(np.dot(delta, delta) + SOFTENING ** 2)
                accelerations[i] += G * masses[j] * delta / dist ** 3
    return accelerations

def rk4_step(positions: np.ndarray, velocities: np.ndarray,
             masses: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Advance positions and velocities by one timestep using RK4 integration.

    Parameters
    ----------
    positions : np.ndarray, shape (3, 2)
        Current positions of the three bodies.
    velocities : np.ndarray, shape (3, 2)
        Current velocities of the three bodies.
    masses : np.ndarray, shape (3,)
        Masses of the three bodies.
    dt : float
        Timestep size.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (new_positions, new_velocities), each shape (3, 2).
    """


# ----- Diagnostics -----

def compute_total_energy(positions: np.ndarray, velocities: np.ndarray,
                         masses: np.ndarray) -> float:
    """
    Compute the total mechanical energy (kinetic + potential) of the system.
    """
# compute_center_of_mass


# ----- Termination checks -----
# check collision
# check out of bounds