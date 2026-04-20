# Code Documentation
## Overview

The project is split into five Python files, each with a single responsibility. "main.py" calls "simulation.py" and "visualizer.py", "simulation.py" calls "physics.py", and the test files import from "physics.py". Nothing imports from "main.py".

main.py
├── simulation.py

│   └── physics.py

└── visualizer.py

test_physics.py    → physics.py
test_simulation.py → simulation.py # not yet implemented


## physics.py

Contains all the pure physics — force calculations, numerical integration, energy computation, and termination checks. No simulation state is stored here and there is no user I/O. Every function takes arrays in and returns arrays out, which makes them straightforward to test in isolation.

### Constants

G = 1.0
SOFTENING = 0.01

"G" is the gravitational constant set to 1.0, meaning the simulation runs in dimensionless units rather than SI. This keeps the numbers manageable and lets the preset initial conditions use clean values.

"SOFTENING" is a small value added inside the distance calculation to prevent the gravitational force from becoming infinite when two bodies get very close. Without it, a collision would produce a force so large it would make the integrator unstable. The value 0.01 was chosen by trial and error. 0.1 was too coarse for the figure-8 preset which requires bodies to pass close to each other.

### gravitational_force(pos1, pos2, mass1, mass2)

Computes the gravitational force vector acting on body 1 due to body 2. Uses the softened distance:

r_softened = sqrt(|pos2 - pos1|² + ε²)
F = G * m1 * m2 * (pos2 - pos1) / r_softened³

The direction of the force is always from body 1 toward body 2, which is handled automatically by the "delta = pos2 - pos1" vector. The "r_softened³" in the denominator comes from combining the "1/r²" magnitude term with the "1/r" unit vector normalization.

Returns a numpy array of shape "(2,)" representing the force vector "[Fx, Fy]".

### compute_accelerations(positions, masses)

Loops over all pairs of bodies and accumulates the total force on each one, then divides by mass to get acceleration (Newton's second law: "a = F/m"). The inner calculation is done inline rather than calling "gravitational_force" as a separate function, which reduces function call overhead over thousands of timesteps.

Takes positions of shape "(3, 2)" and masses of shape "(3,)". Returns accelerations of shape "(3, 2)".

### rk4_step(positions, velocities, masses, dt)

Advances the full system by one timestep using 4th-order Runge-Kutta integration. RK4 was chosen over simpler methods like Euler because it gives much better energy conservation for the same timestep size. Euler integration would cause the orbits to slowly spiral outward over time, which would make the simulation physically wrong.

RK4 works by sampling the derivative of the system at four points within the timestep and combining them:


k1 — derivative at the start of the step
k2 — derivative at the midpoint, using k1 to estimate position
k3 — derivative at the midpoint, using k2 to estimate position
k4 — derivative at the end of the step, using k3 to estimate position

new_state = old_state + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

Because both position and velocity are evolving simultaneously, each "k" is actually a pair: "k_x" (the position derivative, which is just velocity) and "k_v" (the velocity derivative, which is acceleration). All three bodies are updated simultaneously since positions and velocities are "(3, 2)" arrays.

Returns "(new_positions, new_velocities)", both shape "(3, 2)".

### compute_total_energy(positions, velocities, masses)

Computes the total mechanical energy of the system as kinetic energy plus gravitational potential energy:

KE = sum(0.5 * m * |v|²)
PE = sum(-G * mi * mj / r_ij)   for each unique pair i < j


This is used in "test_physics.py" to verify that energy is approximately conserved over a short simulation. RK4 is not a symplectic integrator so some drift is expected, but it should stay below 1% over 500 steps with "DT = 0.01".

### compute_center_of_mass(positions, masses)

Returns the mass-weighted average position of the system:

com = sum(m_i * pos_i) / sum(m_i)

Used in "simulation.py" to shift trajectories into the center-of-mass frame before storing them. The "masses[:, np.newaxis]" reshape is needed to broadcast the "(3,)" mass array against the "(3, 2)" positions array element-wise.

### check_collision(positions, collision_radius=0.5)

Checks all three pairs of bodies and returns "(True, i, j)" if any two are closer than "collision_radius". Uses raw geometric distance with no softening — this is intentional since softening is a physics approximation, not a physical distance.

### check_escape(positions, boundary=500.0)

Returns "(True, i)" if any body has an x or y coordinate exceeding the boundary. This catches bodies that have been gravitationally ejected from the system.

## simulation.py

Owns the time evolution of the system. Imports from "physics.py" and runs the integration loop. Returns trajectory data as a dictionary. It does not display or save anything itself.

### Constants

DT = 0.01
MAX_STEPS = 2000

"DT = 0.01" was chosen as a balance between accuracy and speed. Smaller timesteps give better energy conservation but take longer to compute. At "DT = 0.01" the figure-8 preset completes several full loops within 2000 steps with less than 1% energy drift.

"MAX_STEPS = 2000" is a hard cap to prevent the simulation from running indefinitely on stable configurations.

### class Body

A simple class that holds the initial conditions for one body. No methods beyond "__init__".

Body(mass, position, velocity, name)

"position" and "velocity" are converted to "numpy" arrays on construction so the rest of the code can always assume they are arrays, never plain lists.

### run_simulation(bodies)

The main simulation loop. Takes a list of three "Body" instances and runs the integration until a termination condition is met or "MAX_STEPS" is reached.

At each step it:
1. Shifts positions into the center-of-mass frame and stores them in the trajectory array
2. Checks for collision — stops if two bodies are within "collision_radius"
3. Checks for escape — stops if a body exceeds the boundary
4. Calls "rk4_step" to advance positions and velocities

The trajectory array is pre-allocated to "(MAX_STEPS, 3, 2)" before the loop for efficiency, then trimmed to "[:step]" at the end so the visualizer receives a clean array with no trailing zeros.

The "for/else" Python construct is used so the "else" block only runs if the loop finishes without hitting a "break" — this cleanly handles the "completed normally" case without a separate flag variable.

Returns a dictionary with keys: "positions", "masses", "names", "steps", "status", "message".

### get_preset(name)

Returns a list of three "Body" instances for a named stable configuration. Accepts ""figure8"", ""lagrange"", or ""hierarchical"". Raises "ValueError" for any other input.

**Figure-8** uses the exact Chenciner-Montgomery (2000) initial conditions. These are well-known values from the literature and produce the characteristic figure-eight orbit for three equal masses.

**Lagrange triangle** derives velocities analytically from the circular orbit condition. The angular velocity "ω = sqrt(G*M/r³)" gives each body a tangential velocity that keeps the triangle rotating rigidly.

**Hierarchical** sets up an inner binary pair with a circular orbit velocity "v = sqrt(G*m/2r)", and an outer body orbiting the combined mass of the binary at a much larger radius.

## visualizer.py

Handles all rendering. Takes the trajectory dictionary from "run_simulation" and produces a matplotlib animation. No physics or simulation logic is here.

### Constants

COLORS = ["#e74c3c", "#3498db", "#2ecc71"]
TRAIL_LENGTH = 200
BODY_SCALE = 100


"TRAIL_LENGTH = 200" means each body shows the last 200 positions as a trail. Showing the full trajectory from the start made the plot cluttered — a rolling trail is cleaner and gives a better sense of current motion.

"BODY_SCALE" is the base matplotlib scatter marker size. Actual marker size is computed as "max(BODY_SCALE, BODY_SCALE * log(1 + mass))" so heavier bodies appear visually larger. "log1p" grows meaningfully without becoming absurdly large for high-mass bodies.

### compute_axis_limits(positions, padding=5.0)

Computes axis limits from the full trajectory before the animation starts so the view does not jump around during playback. Forces a square aspect ratio by applying the larger of the x and y ranges to both axes — without this, circular orbits render as ellipses.

### animate_simulation(result, save_path=None)

Creates the matplotlib figure and runs "FuncAnimation". The dark background ("#0d0d0d") was chosen because orbital trails are much more readable against dark backgrounds, and it gives the output a more simulation-like appearance.

At each frame the "update" function redraws the trail for the last "TRAIL_LENGTH" steps and moves each body marker to its current position. Using "blit=True" means matplotlib only redraws the artists that changed, which keeps the animation smooth.

If "save_path" is "None" the animation is displayed interactively. Otherwise it is passed to "_get_writer" and saved to file.

### _get_writer(path)

Internal helper that returns the appropriate matplotlib animation writer based on file extension. ".gif" uses "PillowWriter", ".mp4" uses "FFMpegWriter". The leading underscore signals that this function is not part of the public interface and should not be called directly from outside the module.

## main.py

The entry point. Handles all user interaction and validation, then calls "run_simulation" and "animate_simulation" in sequence. Contains no physics or rendering logic.

### Validation constants

MASS_MIN       =   1e-3
MASS_MAX       =   1e6
POSITION_MIN   = -100.0
POSITION_MAX   =  100.0
VELOCITY_MAX   =   50.0
MIN_SEPARATION =    5.0

These ranges were chosen to keep the simulation well-behaved. Very large velocities cause bodies to escape immediately. Very close starting positions cause near-infinite forces on the first step. The minimum separation is a soft warning rather than a hard block. The user is informed of the risk but allowed to proceed.

### prompt_float(prompt, min_val, max_val)

The core input function. Loops until the user enters a valid float within the allowed range. All numeric input in the program goes through this function so bad input is handled in one place consistently. Raises no exceptions. All errors are caught and the user is re-prompted.

### prompt_mode(), prompt_preset(), prompt_save()

Menu functions that present numbered options and return the user's choice as a string. Each loops until a valid option is entered.

### prompt_custom_bodies()

Calls "prompt_float" for each of the 7 parameters per body (mass, x, y, vx, vy) across all three bodies. Returns a list of three fully initialised "Body" instances.

### validate_separations(bodies)

Checks the starting distance between every pair of bodies and prints a warning if any are closer than "MIN_SEPARATION". Does not block execution — the warning is informational. Only called for custom input since the presets are known to be safe.

### main()

Chains everything together: prompt mode → get bodies → prompt save → run simulation → animate. Prints a progress message before the simulation and again before the animation so the user knows the program is working and has not frozen.


## test_physics.py

14 assert-based tests covering every function in "physics.py". Each test is independent and tests one specific property. Tests are run by executing the file "test_physics.py".

| Test | What it checks |
|---|---|
| "test_gravitational_force_direction" | Force points toward the attracting body |
| "test_gravitational_force_newtons_third_law" | Force on A from B equals minus force on B from A |
| "test_gravitational_force_magnitude" | Magnitude matches the softened formula analytically |
| "test_gravitational_force_no_singularity" | Force is finite at zero separation |
| "test_compute_accelerations_symmetry" | Equal accelerations for symmetric configuration |
| "test_compute_accelerations_total_force_zero" | Net force on system sums to zero |
| "test_rk4_step_returns_correct_shape" | Output arrays are shape (3, 2) |
| "test_rk4_energy_conservation" | Energy drift below 1% over 500 steps |
| "test_center_of_mass_equal_masses" | COM equals geometric centroid for equal masses |
| "test_center_of_mass_weighted" | Heavy body dominates the COM position |
| "test_check_collision_detected" | Collision found when bodies are close |
| "test_check_collision_not_detected" | No false positive when bodies are far apart |
| "test_check_escape_detected" | Escape found when body exceeds boundary |
| "test_check_escape_not_detected" | No false positive when all bodies are in bounds |
