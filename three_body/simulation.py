# add test_simulation.py for coverage?

import numpy as np
from three_body.physics import (
    rk4_step,
    compute_center_of_mass,
    check_collision,
    check_escape
)

# ----- Constants -----
DT = 0.01 # Time step magnitude
MAX_STEPS = 5000 # Put hard cap on number of simulation steps
G_preset = 1.0 # Gravitational constant for preset calc

# ----- Body -----

class Body:
    """A class to represent a single body in the simulation."""
    def __init__(self, mass: float, position: np.ndarray,
                 velocity: np.ndarray, name: str = "Body"):
        self.mass = mass
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.name = name


# ----- Simulation -----

def run_simulation(bodies: list[Body]) -> dict:
    """"
    Run 3 Body simulation from set initial conditions
    until termination condition is met.
    Positions stored in center of mass frame so animation stays centered.
    """

    bodies: list[Body]
    assert len(bodies) == 3, "Simulation requires exactly 3 bodies."
    positions  = np.array([b.position for b in bodies])
    velocities = np.array([b.velocity for b in bodies])
    masses     = np.array([b.mass     for b in bodies])
    names      = [b.name for b in bodies]
    assert np.all(masses > 0), "All masses must be positive."
    trajectory = np.zeros((MAX_STEPS, 3, 2))
    status  = "completed"
    message = "Simulation completed normally."

    for step in range(MAX_STEPS):

        # to center-of-mass frame
        com = compute_center_of_mass(positions, masses)
        trajectory[step] = positions - com

        # termination conditions before step
        collided, i, j = check_collision(positions)
        if collided:
            status  = "collision"
            message = (f"Bodies {names[i]} and {names[j]} "
                       f"collided at t = {step * DT:.2f}s.")
            step += 1
            break

        escaped, idx = check_escape(positions)
        if escaped:
            status  = "escape"
            message = (f"{names[idx]} escaped the system "
                       f"at t = {step * DT:.2f}s.")
            step += 1
            break

        positions, velocities = rk4_step(positions, velocities, masses, DT)

    else:
        # completed without break
        step = MAX_STEPS

    return {
        "positions": trajectory[:step],
        "masses":    masses,
        "names":     names,
        "steps":     step,
        "status":    status,
        "message":   message
    }


# ----- Preset code -----

def get_preset(name: str) -> list[Body]:
    if name == "figure8":
        # Chenciner and Montgomery figure-8 solution
        # Three equal masses chasing each other along a figure-8 curve

        p = 0.97000436
        q = 0.24308753
        vx = 0.93240737 / 2.0
        vy = 0.86473146 / 2.0
        return [
            Body(1.0, [ p, -q], [-vx, -vy], "Body 1"),
            Body(1.0, [-p,  q], [-vx, -vy], "Body 2"),
            Body(1.0, [ 0.0, 0.0], [2*vx, 2*vy], "Body 3"),
        ]

    elif name == "lagrange":
        # Equilateral triangle configuration
        # Three equal masses at triangle corners with tangential velocities
        r = 10.0

        omega = np.sqrt(3.0 * G_preset / r ** 3)
        angles = [np.pi / 2, np.pi / 2 + 2 * np.pi / 3,
                  np.pi / 2 + 4 * np.pi / 3]
        bodies = []
        for k, angle in enumerate(angles):
            pos = np.array([r * np.cos(angle), r * np.sin(angle)])
            # Tangential velocity perpendicular to radius
            vel = omega * r * np.array([-np.sin(angle), np.cos(angle)])
            bodies.append(Body(1.0, pos, vel, f"Body {k + 1}"))
        return bodies

    elif name == "hierarchical":
        # Two heavy bodies in a close binary orbit,
        # one light body in a wide orbit around the pair
        m_heavy = 10.0
        r_inner = 2.0
        v_inner = np.sqrt(G_preset * m_heavy / (2 * r_inner))

        m_light = 1.0
        r_outer = 30.0
        m_total_inner = 2 * m_heavy
        v_outer = np.sqrt(G_preset * m_total_inner / r_outer)

        return [
            Body(m_heavy, [-r_inner,  0.0], [0.0, -v_inner], "Body 1"),
            Body(m_heavy, [ r_inner,  0.0], [0.0,  v_inner], "Body 2"),
            Body(m_light, [ r_outer,  0.0], [0.0,  v_outer], "Body 3"),
        ]

    else:
        raise ValueError(
            f"Unknown preset '{name}'. "
            f"Choose from: 'figure8', 'lagrange', 'hierarchical'."
        )