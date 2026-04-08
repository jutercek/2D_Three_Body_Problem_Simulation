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