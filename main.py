"""
Handles all user interaction, input validation, and orchestrates
the simulation and visualization pipeline.
Does not contain any physics or render logic.
"""
import matplotlib
matplotlib.use("TkAgg")

import numpy as np
from three_body.simulation import Body, get_preset, run_simulation
from three_body.visualizer import animate_simulation

# ----- Constants -----

MASS_MIN       =   1e-3
MASS_MAX       =   1e6
POSITION_MIN   = -100.0
POSITION_MAX   =  100.0
VELOCITY_MAX   =   50.0
MIN_SEPARATION =    5.0


# ----- Input functions -----

def prompt_float(prompt: str, min_val: float = None,
                    max_val: float = None) -> float:
    """
    Prompt the user for a float, until a valid value is entered.
    Optionally enforces min/max bounds.

    Parameters
    -
    prompt : str
        Message displayed to the user.
    min_val : float, optional
        Minimum allowed value.
    max_val : float, optional
        Maximum allowed value.

    Returns
    -
    float
        Validated float value within the specified bounds.
    """

    while True:
        try:
            value = float(input(prompt))
        except ValueError:
            print("  Invalid input — please enter a number.")
            continue
        if min_val is not None and value < min_val:
            print(f"  Value must be >= {min_val}.")
            continue
        if max_val is not None and value > max_val:
            print(f"  Value must be <= {max_val}.")
            continue
        return value


def prompt_mode() -> str:
    """
    Ask the user to choose between preset and custom mode.
    """
    print("\n--- Three-Body Problem Simulator ---\n")
    print("  [1] Use a stable preset configuration")
    print("  [2] Enter custom initial conditions")

    while True:
        choice = input("\nSelect mode (1 or 2): ").strip()
        if choice == "1":
            return "preset"
        elif choice == "2":
            return "custom"
        else:
            print("  Please enter 1 or 2.")


def prompt_preset() -> str:
    """
    Display the three preset options and return the user's choice as a string.

    Returns
    -
    str
        One of: figure8, lagrange, hierarchical.
    """

    print("\nAvailable presets:\n")
    print("  [1] Figure-8       — three equal masses chasing each other")
    print("  [2] Lagrange       — equilateral triangle, rigid rotation")
    print("  [3] Hierarchical   — close binary pair orbited by a third body")
    options = {"1": "figure8", "2": "lagrange", "3": "hierarchical"}

    while True:
        choice = input("\nSelect preset (1, 2 or 3): ").strip()
        if choice in options:
            return options[choice]
        else:
            print("  Please enter 1, 2 or 3.")


def prompt_custom_bodies() -> list[Body]:
    """
    Interactively collect mass, position and velocity for each of the three
    bodies from the user.

    Returns
    -
    list[Body]
        Three fully initialized Body instances.
    """

    bodies = []
    print("\nEnter initial conditions for each body.")
    print(f"  Mass      : {MASS_MIN} to {MASS_MAX}")
    print(f"  Position  : {POSITION_MIN} to {POSITION_MAX} (each axis)")
    print(f"  Velocity  : -{VELOCITY_MAX} to {VELOCITY_MAX} (each axis)\n")

    for i in range(3):
        print(f"  -- Body {i + 1} --")
        mass = prompt_float(
            f"  Mass: ",
            min_val=MASS_MIN, max_val=MASS_MAX
        )
        x = prompt_float(
            f"  Position x: ",
            min_val=POSITION_MIN, max_val=POSITION_MAX
        )
        y = prompt_float(
            f"  Position y: ",
            min_val=POSITION_MIN, max_val=POSITION_MAX
        )
        vx = prompt_float(
            f"  Velocity vx: ",
            min_val=-VELOCITY_MAX, max_val=VELOCITY_MAX
        )
        vy = prompt_float(
            f"  Velocity vy: ",
            min_val=-VELOCITY_MAX, max_val=VELOCITY_MAX
        )
        # still works with list?
        bodies.append(Body(mass, [x, y], [vx, vy], f"Body {i + 1}"))
        print()
    return bodies


def prompt_save() -> str | None:
    """
    Ask the user whether to save the animation and if so, to what path.

    Returns
    -
    str or None
        File path string (e.g. output.gif) or None to display interactively.
    """

    print("  [1] Display animation interactively")
    print("  [2] Save as .gif  (requires Pillow)")
    print("  [3] Save as .mp4  (requires ffmpeg)")

    while True:
        choice = input("\nOutput option (1, 2 or 3): ").strip()
        if choice == "1":
            return None
        elif choice == "2":
            path = input("  Save path (e.g. output.gif): ").strip()
            if not path.endswith(".gif"):
                path += ".gif"
            return path
        elif choice == "3":
            path = input("  Save path (e.g. output.mp4): ").strip()
            if not path.endswith(".mp4"):
                path += ".mp4"
            return path
        else:
            print("  Please enter 1, 2 or 3.")


def validate_separations(bodies: list[Body]) -> None:
    """
    Check that no two bodies start closer than MIN_SEPARATION.
    Prints a warning but does not block execution.

    Parameters
    -
    bodies : list[Body]
        List of three Body instances with positions set.
    """

    for i in range(3):
        for j in range(i + 1, 3):
            delta = bodies[j].position - bodies[i].position
            dist  = np.linalg.norm(delta)
            if dist < MIN_SEPARATION:
                print(
                    f"\n  WARNING: {bodies[i].name} and {bodies[j].name} "
                    f"are only {dist:.2f} units apart (minimum recommended: "
                    f"{MIN_SEPARATION}). This may cause numerical instability."
                )


# ----- Main -----

def main() -> None:
    """
    Run the full user interaction → simulation → animation pipeline.
    """
    mode = prompt_mode()

    if mode == "preset":
        preset_name = prompt_preset()
        bodies = get_preset(preset_name)
        print(f"\nLoaded preset: {preset_name}")
    else:
        bodies = prompt_custom_bodies()
        validate_separations(bodies)

    save_path = prompt_save()

    print("\nRunning simulation ...")
    result = run_simulation(bodies)
    print(f"  {result['message']}")
    print(f"  Steps run: {result['steps']}")
    print("\nPreparing animation ...")
    animate_simulation(result, save_path=save_path)


if __name__ == "__main__":
    main()

