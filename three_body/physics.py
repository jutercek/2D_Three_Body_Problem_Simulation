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

    delta = pos2 - pos1
    dist = np.sqrt(np.dot(delta, delta) + SOFTENING ** 2)
    force_magnitude = G * mass1 * mass2 / dist ** 2
    return force_magnitude * (delta / dist)


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
    return np.sum(masses[:, np.newaxis] * positions, axis=0) / np.sum(masses)


# ----- Termination checks -----

def check_collision(positions: np.ndarray,
                    collision_radius=COLLISION_RADIUS) -> tuple[bool, int, int]:

    for i in range(3):
        for j in range(i + 1, 3):
            delta = positions[j] - positions[i]
            dist = np.sqrt(np.dot(delta, delta))
            if dist < collision_radius:
                return True, i, j
    return False, -1, -1


def check_escape(positions: np.ndarray,
                 boundary=BOUNDARY) -> tuple[bool, int]:

    for i in range(3):
        if np.any(np.abs(positions[i]) > boundary):
            return True, i
    return False, -1