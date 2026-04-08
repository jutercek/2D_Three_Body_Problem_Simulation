"""
Assert-based tests for physics.py file functions.
"""

import numpy as np

from physics import (
    gravitational_force,
    compute_accelerations,
    rk4_step,
    compute_total_energy,
    compute_center_of_mass,
    G,
    SOFTENING,
)


def test_gravitational_force_direction():
    """Force on body 1 must point toward 2."""
    pos1 = np.array([0.0, 0.0])
    pos2 = np.array([1.0, 0.0])
    force = gravitational_force(pos1, pos2, mass1=1.0, mass2=1.0)

    assert force[0] > 0, "Force x-component should be positive, pointing right toward 2"
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
    """Force magnitude should match F = G*m1*m2*r / (r^2 + softening^2)^(3/2)."""
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

# def test_compute_accelerations_symmetry():
# def test_compute_accelerations_total_force_zero()
# def test_rk4():
# test_rk4_energy_cons():
# test_center_of_mass_equal_masses():
# test_check_collision():
# test_check_escape():


"""
if __name__ == "__main__":
    test1
    test2
    test3

    print("\nAll tests passed.")
"""