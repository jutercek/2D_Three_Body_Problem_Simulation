"""
pytest version of testing version 1
"""

import numpy as np
import pytest
from three_body.physics import (
    gravitational_force,
    compute_accelerations,
    rk4_step,
    compute_total_energy,
    compute_center_of_mass,
    check_collision,
    check_escape,
    G,
    SOFTENING,
)

# ----- Fix -----
"""
equal_masses and collinear_horizontal defined here so they can be called later without repeating code
"""
@pytest.fixture
def equal_masses():
    """
    Three equal masses in a triangle
    """
    positions = np.array([
        [-10.0,  0.0],
        [ 10.0,  0.0],
        [  0.0, 10.0],
    ])
    velocities = np.array([
        [ 0.0,  0.5],
        [ 0.0, -0.5],
        [ 0.5,  0.0],
    ])
    masses = np.array([1.0, 1.0, 1.0])
    return positions, velocities, masses


@pytest.fixture
def collinear_horizontal():
    """
    Two bodies separated by 3 units and unit masses
    """
    pos1 = np.array([0.0, 0.0])
    pos2 = np.array([3.0, 0.0])
    return pos1, pos2


# ----- Gravitational_force -----

class TestGravitationalForce:

    def test_force_points_toward_attractor(self, collinear_horizontal):
        """
        Force on body 1 must point toward body 2
        For a body to the right, the x-component must be positive
        and the y-component zero
        """
        pos1, pos2 = collinear_horizontal
        force = gravitational_force(pos1, pos2, mass1=1.0, mass2=1.0)

        assert force[0] > 0
        assert abs(force[1]) < 1e-10

    def test_newtons_third_law(self):
        """
        Force on body A from body B must be equal and opposite to
        force on body B from body A
        Verified for an arbitrary configuration with unequal masses
        """
        pos1 = np.array([0.0, 0.0])
        pos2 = np.array([3.0, 4.0])

        f12 = gravitational_force(pos1, pos2, mass1=2.0, mass2=5.0)
        f21 = gravitational_force(pos2, pos1, mass2=2.0, mass1=5.0)

        assert np.allclose(f12, -f21, atol=1e-10)

    def test_force_scales_with_mass(self, collinear_horizontal):
        """
        Doubling one mass must double the force magnitude
        This directly tests F proportional to m1*m2
        """
        pos1, pos2 = collinear_horizontal
        f1 = gravitational_force(pos1, pos2, mass1=1.0, mass2=1.0)
        f2 = gravitational_force(pos1, pos2, mass1=2.0, mass2=1.0)

        assert np.allclose(np.linalg.norm(f2), 2 * np.linalg.norm(f1), rtol=1e-10)

    def test_force_finite_at_zero_separation(self):
        """
        Force must remain finite when two bodies occupy the same position
        Softening prevents division by zero
        """
        pos = np.array([0.0, 0.0])
        force = gravitational_force(pos, pos, mass1=1.0, mass2=1.0)

        assert np.all(np.isfinite(force))

    def test_force_decreases_with_distance(self):
        """
        Force magnitude must decrease as separation increases
        Verified by comparing force at r=3 vs r=6
        """
        pos1 = np.array([0.0, 0.0])
        pos_near = np.array([3.0, 0.0])
        pos_far  = np.array([6.0, 0.0])

        f_near = np.linalg.norm(gravitational_force(pos1, pos_near, 1.0, 1.0))
        f_far  = np.linalg.norm(gravitational_force(pos1, pos_far,  1.0, 1.0))

        assert f_near > f_far

# ----- ComputeAccelerations -----

class TestComputeAccelerations:

    def test_net_force_on_system_is_zero(self, equal_masses):
        """
        The sum of mass times acceleration over all bodies must be zero
        """
        positions, _, masses = equal_masses
        accels = compute_accelerations(positions, masses)
        net_force = np.sum(masses[:, np.newaxis] * accels, axis=0)

        assert np.allclose(net_force, 0.0, atol=1e-10)

    def test_equal_acceleration_magnitudes_in_symmetric_config(self):
        """
        Three equal masses at the corners of an equilateral triangle
        must experience equal acceleration magnitudes
        """
        s = 10.0
        positions = np.array([
            [0.0,              0.0],
            [s,                0.0],
            [s / 2.0, s * np.sqrt(3) / 2.0],
        ])
        masses = np.array([1.0, 1.0, 1.0])
        accels = compute_accelerations(positions, masses)
        magnitudes = np.linalg.norm(accels, axis=1)

        assert np.allclose(magnitudes, magnitudes[0], rtol=1e-6)

    def test_output_shape(self, equal_masses):
        """
        Output must be shape (3, 2)
        """
        positions, _, masses = equal_masses
        accels = compute_accelerations(positions, masses)

        assert accels.shape == (3, 2)

# ----- RK4 steps -----

class TestRK4Step:

    def test_output_shapes(self, equal_masses):
        """
        RK4 must return two arrays of shape (3, 2), one position and one velocity
        """
        positions, velocities, masses = equal_masses
        new_pos, new_vel = rk4_step(positions, velocities, masses, dt=0.01)

        assert new_pos.shape == (3, 2)
        assert new_vel.shape == (3, 2)

    def test_energy_conservation_over_short_run(self, equal_masses):
        """
        Total mechanical energy must remain approximately constant
        over 500 steps with dt=0.01. RK4 is not perfectly symplectic
        so some drift is expected
        """
        positions, velocities, masses = equal_masses
        e0 = compute_total_energy(positions, velocities, masses)

        for _ in range(500):
            positions, velocities = rk4_step(positions, velocities, masses, dt=0.01)

        e1 = compute_total_energy(positions, velocities, masses)
        relative_drift = abs(e1 - e0) / abs(e0)

        assert relative_drift < 0.01

    def test_positions_change_after_step(self, equal_masses):
        """
        Positions must change after a timestep
        """
        positions, velocities, masses = equal_masses
        new_pos, _ = rk4_step(positions, velocities, masses, dt=0.01)

        assert not np.allclose(new_pos, positions)

# ----- Total energy -----

class TestComputeTotalEnergy:

    def test_energy_is_scalar(self, equal_masses):
        """
        Total energy must be a float
        """
        positions, velocities, masses = equal_masses
        energy = compute_total_energy(positions, velocities, masses)

        assert isinstance(energy, float)

    def test_kinetic_energy_only_for_distant_bodies(self):
        """
        For two bodies separated by a large distance, potential energy
        goes to zero and total energy approaches kinetic energy
        """
        positions = np.array([
            [  0.0, 0.0],
            [1e6,   0.0],
            [2e6,   0.0],
        ])
        velocities = np.array([
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
        ])
        masses = np.array([1.0, 1.0, 1.0])

        energy = compute_total_energy(positions, velocities, masses)
        expected_ke = 0.5 * (1.0 + 4.0 + 9.0)  # 0.5 * m * v^2 for each

        assert abs(energy - expected_ke) < 1e-3

    def test_stationary_bodies_have_negative_energy(self):
        """
        Three stationary bodies have zero kinetic energy and negative
        potential energy
        Total energy must be negative
        """
        positions  = np.array([[0.0, 0.0], [5.0, 0.0], [0.0, 5.0]])
        velocities = np.zeros((3, 2))
        masses     = np.array([1.0, 1.0, 1.0])

        energy = compute_total_energy(positions, velocities, masses)

        assert energy < 0