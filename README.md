# 2D_Three_Body_Problem_Simulation
## Introduction

The three-body problem describes the motion of three masses interacting through gravitational attraction. Unlike the two-body problem, it has no general solution so the trajectories must be computed numerically. Small differences in starting conditions can lead to completely different outcomes, making it one of the most well known examples of chaotic behaviour in classical mechanics.

This program simulates the three-body problem in 2D. The user inputs initial masses, positions and velocities for three bodies, and the program computes their equations of motion using a 4th order Runge-Kutta method and outputs an animation of the trajectories.

As custom inputs of starting conditions can lead to non stable solutions, three stable preset configurations are also included for reference.

## Program structure
```
three_body_sim/
  ├── assets/
     └── figure8.gif
  ├── three_body/
     ├── __init__.py
     ├── physics.py
     ├── simulation.py
     └── visualizer.py
  ├── tests/
     ├── __init__.py
     ├── test_physics.py
     └── test_simulation.py
  ├── main.py
  ├── benchmark.py
  └── README.md
```
The project uses dimensionless units G = 1 for simplicity, and not SI. The program assumes point masses and solely gravitational interactions wihtout relativistic effects.

`physics.py` contains all core physics functions. It handles gravitational force calculations between body pairs, numerical integration using RK4, total energy computation, and collision and escape checks. It does not store any simulation states. Functions take arrays in and return arrays out, making them easier to test in isolation.

`benchmark.py` tests execution time difference of old loop based compute_accelerations function and new vectorized version. 

`simulation.py` holds the time evolution of the system. It defines the `Body` class for storing initial conditions, calls functions from `physics.py`, runs the integration loop and checks termination conditions after each step. It also contains the three preset configurations. When the simulation finishes it returns the trajectory data as a dictionary for the visualizer.

`visualizer.py` takes the trajectory data produced by `simulation.py` and renders it as a matplotlib animation. It handles axis scaling, trail rendering, body marker sizing, and saving the output to `.gif` or `.mp4`. 

`main.py` is the entry point of the program. It handles all user interaction. Prompting for mode selection, collecting and validating custom initial conditions, and choosing output format. It contains no physics or rendering logic, it only connects other modules.

`test_physics.py` and `test_simulation.py` contain the pytest test suites for `physics.py` and `simulation.py`. Tests verify physical correctness using known analytical results and fundamental laws rather than reimplementing the code being tested.

### Simulation Parameters

These constants are defined at the top of their respective files. Tunable parameters can be adjusted directly in the source code.

| Parameter | File | Default | Tunable | Description |
|---|---|---|---|---|
| `G` | `physics.py` | `1.0` | No | Gravitational constant in simulation units |
| `SOFTENING` | `physics.py` | `0.02` | Yes | Softening length to prevent singularities at close range |
| `COLLISION_RADIUS` | `physics.py` | `0.5` | Yes | Distance threshold below which a collision is declared |
| `BOUNDARY` | `physics.py` | `500.0` | Yes | Distance from origin beyond which a body is considered escaped |
| `DT` | `simulation.py` | `0.01` | Yes | Timestep size (smaller is more accurate but slower) |
| `MAX_STEPS` | `simulation.py` | `5000` | Yes | Maximum number of integration steps per simulation |
| `TRAIL_LENGTH` | `visualizer.py` | `200` | Yes | Number of past positions shown as trail per body |
| `BODY_SCALE` | `visualizer.py` | `100` | Yes | Controls scale of bodies rendered in animation |

## Program usage tutorial
### Requirements
- Python (tested for 3.10.5)
- numpy
- matplotlib
- pytest
- Pillow
- ffmpeg  (if exporting .mp4)
- tkinter
    - Windows: ships with Python
    - Linux: sudo apt install python3-tk
### Using the 3-Body Problem Simulator
- Run the program with `python main.py` from the root of the project
- Select option 1 for a stable preset or option 2 to enter custom initial conditions
- If using a preset, select one of the three available configurations: Figure-8, Lagrange triangle, or Hierarchical
- If using custom input, enter mass, position (x, y) and velocity (vx, vy) for each of the three bodies when prompted
- Select output option:
   - Display interactively
   - Save as `.gif`
   - Save as `.mp4`
- If saving, enter the desired file name when prompted

### Running tests
pytest tests/ -v

## Examples
Examples of animation outputs

### Example of the results
Below is shown an example of the animation output for the case of the Figure-8 setting.
![Figure-8 preset](assets/figure8.gif)

