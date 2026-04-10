# add test_simulation.py for coverage?

import numpy as np

from physics import (
    rk4_step,
    compute_center_of_mass,
    check_collision,
    check_escape
)

# ----- Constants -----
DT = 0.01 # Time step magnitude
MAX_STEPS = 5000 # Put hard cap on number of simulation steps

# ----- Body -----

class Body:
    def __init__(self, mass: float, position: np.ndarray,
                 velocity: np.ndarray, name: str = "Body"):
        self.mass = mass
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.name = name

# ----- Simulation -----

# def run_simulation():
""""
Run 3 Body simulation from set initial conditions until termination condition is met
"""


# ----- Preset code -----

""""
def get_preset()
This will fetch the preset config from stable options

"""