# 2D_Three_Body_Problem_Simulation
## Introduction

The three-body problem describes the motion of three masses interacting through gravitational attraction. Unlike the two-body problem, it has no general solution so the trajectories must be computed numerically. Small differences in starting conditions can lead to completely different outcomes, making it one of the most well known examples of chaotic behaviour in classical mechanics.

This program simulates the three-body problem in 2D. The user inputs initial masses, positions and velocities for three bodies, and the program computes their equations of motion using a 4th order Runge-Kutta method and outputs an animation of the trajectories.

As custom inputs of starting conditions can lead to non stable solutions, three stable preset configurations are also included for reference.
## Program structure
```
three_body_sim/
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
  ├── README.md
  └── DOCUMENTATION.md
```

`physics.py` contains all core physics functions. It handles gravitational force calculations between body pairs, numerical integration using RK4, total energy computation, and collision and escape checks. It does not sotre any simulation states. Functions take arrays in and return arrays out, making them easier to test in isolation.

`simulation.py` holds the time evolution of the system. It defines the `Body` class for storing initial conditions, calls funcitons from `physics.py`, runs the integration loop and checks termination conditions after each step. It also contains the three preset configurations. When the simulation finishes it returns the trajectory data as a dictionary for the visualizer.

`visualizer.py` takes the trajectory data produced by `simulation.py` and renders it as a matplotlib animation. It handles axis scaling, trail rendering, body marker sizing, and saving the output to `.gif` or `.mp4`. 

`main.py` is the entry point of the program. It handles all user interaction. Prompting for mode selection, collecting and validating custom initial conditions, and choosing output format. It contains no physics or rendering logic, it only connects other moduels..

`test_physics.py` contains the pytest test suites for `physics.py`. Tests verify physical correctness using known analytical results and fundamental laws rather than reimplementing the code being tested.

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
   - Sisplay interactively
   - Save as `.gif`
   - Save as `.mp4`
- If saving, enter the desired file name when prompted

### Running tests
Run `pytest_physics.py`

## Examples
Examples of animation outputs

A interactive 2D simulator for the gravitational three-body problem, written in Python as part of a Software and Computing for Applied Physics course. The program numerically integrates Newton's equations of motion using a 4th-order Runge-Kutta (RK4) method and outputs an animation of the resulting trajectories.

Example of the results:
Below is shown an example of the animation output for the case of the Figure-8 setting.
<img width="583" height="433" alt="Preset1" src="https://github.com/user-attachments/assets/4e48c404-4372-4a67-ae68-ddaecc697900" />

