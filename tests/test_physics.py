"""
Assert-based tests for physics.py file functions
"""

import numpy as np

from three_body.physics import (
    gravitational_force,
    compute_accelerations,
    rk4_step,
    compute_total_energy,
    compute_center_of_mass,
    G,
    SOFTENING,
    check_collision,
    check_escape
)


def test_gravitational_force_direction():
    """Force on body 1 must point toward 2."""
    pos1 = np.array([0.0, 0.0])
    pos2 = np.array([1.0, 0.0])
    force = gravitational_force(pos1, pos2, mass1=1.0, mass2=1.0)

    assert force[0] > 0, "Force x-component should be positive, pointing right toward 2."
    assert abs(force[1]) < 1e-10, "Force y-component should be zero"
    print("PASSED test_gravitational_force_direction")


def test_gravitational_force_newtons_third_law():
    """Force on body 1 from 2 must be equal and opposite to force on 2 from 1."""
    pos1 = np.array([0.0, 0.0])
    pos2 = np.array([3.0, 4.0])
    mass1, mass2 = 2.0, 5.0
    f12 = gravitational_force(pos1, pos2, mass1, mass2)
    f21 = gravitational_force(pos2, pos1, mass2, mass1)

    assert np.allclose(f12, -f21, atol=1e-10), (
        f"Newton's third law is violated: f12={f12}, f21={f21}"
    )
    print("PASSED test_gravitational_force_newtons_third_law")


def test_gravitational_force_magnitude():
    """Force magnitude should match F."""
    pos1 = np.array([0.0, 0.0])
    pos2 = np.array([3.0, 0.0])
    mass1, mass2 = 1.0, 1.0
    force = gravitational_force(pos1, pos2, mass1, mass2)
    r = 3.0
    dist_softened = np.sqrt(r ** 2 + SOFTENING ** 2)
    expected_magnitude = G * mass1 * mass2 * r / dist_softened ** 3

    assert abs(np.linalg.norm(force) - expected_magnitude) < 1e-10, (
        f"Force magnitude mismatch: got {np.linalg.norm(force)}, expected {expected_magnitude}"
    )
    print("PASSED test_gravitational_force_magnitude")


def test_gravitational_force_no_singularity():
    """Force must remain finite when two bodies occupy the same position."""
    pos1 = np.array([0.0, 0.0])
    pos2 = np.array([0.0, 0.0])
    force = gravitational_force(pos1, pos2, mass1=1.0, mass2=1.0)

    assert np.all(np.isfinite(force)), (
        f"Force is not finite at zero separation: {force}"
    )
    print("PASSED test_gravitational_force_no_singularity")


def test_rk4_step_returns_correct_shape():
    """RK4 must return two arrays of shape (3, 2)."""
    positions  = np.random.rand(3, 2)
    velocities = np.random.rand(3, 2)
    masses     = np.array([1.0, 1.0, 1.0])
    new_pos, new_vel = rk4_step(positions, velocities, masses, dt=0.01)

    assert new_pos.shape == (3, 2), f"new_positions shape wrong: {new_pos.shape}"
    assert new_vel.shape == (3, 2), f"new_velocities shape wrong: {new_vel.shape}"
    print("PASSED test_rk4_step_returns_correct_shape")


def test_rk4_energy_conservation():
    """Total energy must remain approximately constant over a short simulation."""
    positions = np.array([
        [-5.0,  0.0],
        [ 5.0,  0.0],
        [ 0.0,  5.0],
    ])
    velocities = np.array([
        [ 0.0,  0.5],
        [ 0.0, -0.5],
        [ 0.5,  0.0],
    ])
    masses = np.array([1.0, 1.0, 1.0])
    e0 = compute_total_energy(positions, velocities, masses)
    for _ in range(500):
        positions, velocities = rk4_step(positions, velocities, masses, dt=0.01)

    e1 = compute_total_energy(positions, velocities, masses)
    relative_drift = abs(e1 - e0) / abs(e0)

    assert relative_drift < 0.01, (
        f"Energy drift too large: {relative_drift:.4%} (initial={e0:.4f}, final={e1:.4f})"
    )
    print("PASSED test_rk4_energy_conservation")


def test_compute_accelerations_symmetry():
    """
    Three equal masses in an equilateral triangle must produce accelerations
    that are equal in magnitude and point toward the center of mass.
    """
    s = 10.0  # side length
    positions = np.array([
        [0.0,         0.0],
        [s,           0.0],
        [s / 2.0,     s * np.sqrt(3) / 2.0],
    ])
    masses = np.array([1.0, 1.0, 1.0])
    accels = compute_accelerations(positions, masses)
    magnitudes = np.linalg.norm(accels, axis=1)

    assert np.allclose(magnitudes, magnitudes[0], rtol=1e-6), (
        f"Acceleration magnitudes not equal for symmetric config: {magnitudes}"
    )
    print("PASSED test_compute_accelerations_symmetry")


def test_compute_accelerations_total_force_zero():
    """
    Net force on the system must be zero.
    Sum of all accelerations weighted by mass should be zero.
    """
    positions = np.array([
        [0.0,  0.0],
        [5.0,  2.0],
        [-3.0, 4.0],
    ])
    masses = np.array([1.0, 2.0, 3.0])

    accels = compute_accelerations(positions, masses)
    net_force = np.sum(masses[:, np.newaxis] * accels, axis=0)

    assert np.allclose(net_force, 0.0, atol=1e-10), (
        f"Net force on system is not zero: {net_force}"
    )
    print("PASSED test_compute_accelerations_total_force_zero")


def test_check_collision_detected():
    """A collision must be detected when two bodies are within collision radius."""
    positions = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [50.0, 50.0],
    ])
    collided, i, j = check_collision(positions, collision_radius=2.0)

    assert collided, "Collision should have been detected"
    assert set([i, j]) == {0, 1}, f"Wrong bodies reported: {i}, {j}" # Check warning
    print("PASSED test_check_collision_detected")


def test_check_collision_not_detected():
    """No collision when all bodies are well separated."""
    positions = np.array([
        [ 0.0,  0.0],
        [10.0,  0.0],
        [ 5.0, 10.0],
    ])
    collided, i, j = check_collision(positions, collision_radius=2.0)

    assert not collided, f"False collision reported between bodies {i} and {j}"
    print("PASSED test_check_collision_not_detected")


def test_check_escape_detected():
    """Escape must be detected when a body exceeds the boundary."""
    positions = np.array([
        [  0.0,   0.0],
        [600.0,   0.0],
        [  0.0,  10.0],
    ])
    escaped, idx = check_escape(positions, boundary=500.0)

    assert escaped, "Escape should have been detected"
    assert idx == 1, f"Wrong body reported as escaped: {idx}"
    print("PASSED test_check_escape_detected")


def test_check_escape_not_detected():
    """No escape when all bodies are within the boundary."""
    positions = np.array([
        [  0.0,   0.0],
        [100.0,  50.0],
        [-80.0, 200.0],
    ])
    escaped, idx = check_escape(positions, boundary=500.0)

    assert not escaped, f"False escape reported for body {idx}"
    print("PASSED test_check_escape_not_detected")


def test_center_of_mass_equal_masses():
    """Center of mass of equal masses must be their geometric centroid."""
    positions = np.array([
        [0.0, 0.0],
        [6.0, 0.0],
        [3.0, 3.0],
    ])
    masses = np.array([1.0, 1.0, 1.0])
    com = compute_center_of_mass(positions, masses)
    expected = np.array([3.0, 1.0])

    assert np.allclose(com, expected, atol=1e-10), (
        f"Center of mass wrong: got {com}, expected {expected}"
    )
    print("PASSED test_center_of_mass_equal_masses")


def test_center_of_mass_weighted():
    """A very heavy body should pull the center of mass close to itself"""
    positions = np.array([
        [0.0,  0.0],
        [100.0, 0.0],
        [50.0,  0.0],
    ])
    masses = np.array([1e6, 1.0, 1.0])
    com = compute_center_of_mass(positions, masses)

    assert com[0] < 1.0, (
        f"Center of mass should be near the heavy body at x=0, got x={com[0]:.4f}"
    )
    print("PASSED test_center_of_mass_weighted")


# ----- Run tests -----

if __name__ == "__main__":
    test_gravitational_force_direction()
    test_gravitational_force_newtons_third_law()
    test_gravitational_force_magnitude()
    test_gravitational_force_no_singularity()
    test_compute_accelerations_symmetry()
    test_compute_accelerations_total_force_zero()
    test_rk4_step_returns_correct_shape()
    test_rk4_energy_conservation()
    test_center_of_mass_equal_masses()
    test_center_of_mass_weighted()
    test_check_collision_detected()
    test_check_collision_not_detected()
    test_check_escape_detected()
    test_check_escape_not_detected()

    print("\nAll tests passed.")
