# 2D_Three_Body_Problem_Simulation
A interactive 2D simulator for the gravitational three-body problem, written in Python as part of a Software and Computing for Applied Physics course. The program numerically integrates Newton's equations of motion using a 4th-order Runge-Kutta (RK4) method and outputs an animation of the resulting trajectories.

How It Works
1. Input
When launched, the program asks the user to choose between:

A preset — one of three known stable configurations with hardcoded initial conditions
Custom input — the user manually enters mass, position (x, y) and velocity (v_x, v_y) for each of the three bodies

Presets:
Figure-8 = Three equal masses chasing each other along a figure-eight curve. Based on the Chenciner-Montgomery (2000) solution.
Lagrange triangle= Three equal masses at the corners of an equilateral triangle, rotating rigidly about their center of mass.
Hierarchical = Two heavy bodies in a close binary orbit with a lighter third body orbiting the pair from far away. The most physically common configuration in real stellar systems.

Custom inputs are validated against safe ranges before the simulation runs.

2. Physics
At each timestep the program computes the gravitational force between every pair of bodies using Newton's law. A small softening term is added inside the distance calculation to prevent the force from blowing up when two bodies pass very close to each other.
The equations of motion are then integrated using 4th-order Runge-Kutta (RK4), which evaluates the force at four intermediate points per timestep and combines them into a weighted average step. This gives better energy conservation than an Euler integrator.

4. Termination
The simulation runs for up to 2000 timesteps and stops early if:

Collision — two bodies come within 0.5 units of each other
Escape — a body travels more than 500 units from the origin

In either case the animation plays up to the point of termination and a message is printed explaining what happened and at what time.

4. Animation
Trajectories are rendered in the center-of-mass frame so the system stays centered on screen regardless of net momentum. Each body is drawn as a colored dot scaled by its mass, with a fading trail showing its recent path.

The animation can be:

Displayed interactively in a window
Saved as a .gif (requires Pillow)
Saved as a .mp4 (requires ffmpeg)

Running tests
Run tests_physics.py file.

Example of the results:
Below is shown an example of the animation output for the case of the Figure-8 setting.
<img width="583" height="433" alt="Preset1" src="https://github.com/user-attachments/assets/4e48c404-4372-4a67-ae68-ddaecc697900" />

