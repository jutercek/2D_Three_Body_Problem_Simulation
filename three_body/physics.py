"""
Pure physics functions for the three-body simulator.
All functions operate on numpy arrays.
"""

import numpy as np

# ----- Constants -----

G = 1.0          # Gravitational constant in simulation units
SOFTENING = 0.02  # Softening to avoid singularities at close range
COLLISION_RADIUS = 0.5 # For collision check
BOUNDARY         = 500.0 # For escape check

# ----- Physics core -----

def gravitational_force(pos1: np.ndarray, pos2: np.ndarray,
                        mass1: float, mass2: float) -> np.ndarray:
    """
    Compute the gravitational force vector exerted on body 1 by body 2.

    Parameters
    -
    pos1 : np.ndarray, shape (2,)
        Position of body 1 [x, y].
    pos2 : np.ndarray, shape (2,)
        Position of body 2 [x, y].
    mass1 : float
        Mass of body 1.
    mass2 : float
        Mass of body 2.

    Returns
    -
    np.ndarray, shape (2,)
        Force vector [Fx, Fy] acting on body 1 toward body 2.
    """

    delta = pos2 - pos1
    dist = np.sqrt(np.dot(delta, delta) + SOFTENING ** 2)
    force_magnitude = G * mass1 * mass2 / dist ** 2
    return force_magnitude * (delta / dist)


def compute_accelerations(positions: np.ndarray,
                           masses: np.ndarray) -> np.ndarray:
    """
    Compute the acceleration of each body due to all others.

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

    diff = positions[np.newaxis, :, :] - positions[:, np.newaxis, :]
    # compute softened distances for pairs
    dist_sq = np.sum(diff ** 2, axis=2) + SOFTENING ** 2
    dist_cu = dist_sq ** 1.5

    # compute accelerations as mass weighted sum over all other bodies
    # divide by dist_cu to get direction and magnitude in one step
    accelerations = G * np.sum(
        masses[np.newaxis, :, np.newaxis] * diff / dist_cu[:, :, np.newaxis],
        axis=1
    )
    return accelerations

def rk4_step(positions: np.ndarray, velocities: np.ndarray,
             masses: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Advance positions and velocities by one timestep using
    4th-order Runge-Kutta integration.

    Parameters
    -
    positions : np.ndarray, shape (3, 2)
        Current positions of the three bodies.
    velocities : np.ndarray, shape (3, 2)
        Current velocities of the three bodies.
    masses : np.ndarray, shape (3,)
        Masses of the three bodies.
    dt : float
        Timestep size.

    Returns
    -
    tuple[np.ndarray, np.ndarray]
        (new_positions, new_velocities), each shape (3, 2).
    """

    # k1
    k1_v = compute_accelerations(positions, masses)
    k1_x = velocities

    # k2
    k2_v = compute_accelerations(positions + 0.5 * dt * k1_x, masses)
    k2_x = velocities + 0.5 * dt * k1_v

    # k3
    k3_v = compute_accelerations(positions + 0.5 * dt * k2_x, masses)
    k3_x = velocities + 0.5 * dt * k2_v

    # k4
    k4_v = compute_accelerations(positions + dt * k3_x, masses)
    k4_x = velocities + dt * k3_v

    new_positions  = positions  + (dt / 6.0) * (k1_x + 2*k2_x + 2*k3_x + k4_x)
    new_velocities = velocities + (dt / 6.0) * (k1_v + 2*k2_v + 2*k3_v + k4_v)

    return new_positions, new_velocities


# ----- Diagnostics -----

def compute_total_energy(positions: np.ndarray, velocities: np.ndarray,
                         masses: np.ndarray) -> float:
    """
    Compute the total mechanical energy of the system.

    Parameters
    -
    positions : np.ndarray, shape (3, 2)
        Positions of the three bodies.
    velocities : np.ndarray, shape (3, 2)
        Velocities of the three bodies.
    masses : np.ndarray, shape (3,)
        Masses of the three bodies.

    Returns
    -
    float
        Total mechanical energy (kinetic + potential) of the system.
    """

    # Kinetic energy
    ke = 0.0
    for i in range(3):
        ke += 0.5 * masses[i] * np.dot(velocities[i], velocities[i])

    # Potential energy
    pe = 0.0
    for i in range(3):
        for j in range(i + 1, 3):
            delta = positions[j] - positions[i]
            dist = np.sqrt(np.dot(delta, delta) + SOFTENING ** 2)
            pe -= G * masses[i] * masses[j] / dist

    return ke + pe


def compute_center_of_mass(positions: np.ndarray,
                           masses: np.ndarray) -> np.ndarray:
    """
    Compute the center of mass position of the system.

    Parameters
    -
    positions : np.ndarray, shape (3, 2)
        Positions of the three bodies.
    masses : np.ndarray, shape (3,)
        Masses of the three bodies.

    Returns
    -
    np.ndarray, shape (2,)
        Center of mass position [x, y].
    """
    return np.sum(masses[:, np.newaxis] * positions, axis=0) / np.sum(masses)


# ----- Termination checks -----

def check_collision(positions: np.ndarray,
                    collision_radius=COLLISION_RADIUS) -> tuple[bool, int, int]:
    """
    Check whether any two bodies are within collision_radius of each other.

    Parameters
    -
    positions : np.ndarray, shape (3, 2)
        Positions of the three bodies.
    collision_radius : float
        Distance threshold below which a collision is declared.

    Returns
    -
    tuple[bool, int, int]
        (collision_occurred, index_of_body_a, index_of_body_b).
        Returns (False, -1, -1) if no collision is detected.
    """

    for i in range(3):
        for j in range(i + 1, 3):
            delta = positions[j] - positions[i]
            dist = np.sqrt(np.dot(delta, delta))
            if dist < collision_radius:
                return True, i, j
    return False, -1, -1


def check_escape(positions: np.ndarray,
                 boundary=BOUNDARY) -> tuple[bool, int]:
    """
    Check whether a body has drifted outside the simulation boundary.

    Parameters
    -
    positions : np.ndarray, shape (3, 2)
        Positions of the three bodies.
    boundary : float
        Maximum allowed distance from the origin on any axis.

    Returns
    -
    tuple[bool, int]
        (escaped, index_of_escaped_body).
        Returns (False, -1) if no body has escaped.
    """

    for i in range(3):
        if np.any(np.abs(positions[i]) > boundary):
            return True, i
    return False, -1