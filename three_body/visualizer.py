"""
Takes trajectory data from simulation.py and produces a matplotlib animation.
Only rendering here.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ----- Constants -----
COLORS = ["#e74c3c", "#3498db", "#2ecc71"]  # one color per body
TRAIL_LENGTH = 200 # steps of trail to show
BODY_SCALE = 100 # base scatter marker size

# ----- Internal -----

def _get_writer(path: str):
    """
    Return an appropriate animation writer based on the file extension.

    Parameters
    -
    path : str
        Output file path. Extension determines the writer used.

    Returns
    -
    animation.PillowWriter or animation.FFMpegWriter
        Writer instance for .gif or .mp4 respectively.

    Raises
    -
    ValueError
        If the file extension is not .gif or .mp4.
    """

    ext = path.rsplit(".", 1)[-1].lower()

    if ext == "gif":
        return animation.PillowWriter(fps=50)
    elif ext == "mp4":
        return animation.FFMpegWriter(fps=50, bitrate=1800)
    else:
        raise ValueError(
            f"Unsupported file extension '.{ext}'. Use '.gif' or '.mp4'."
        )


# ----- Visualizer -----

def compute_axis_limits(positions: np.ndarray,
                        padding: float = 5.0) -> tuple[float, float, float, float]:
    """
    Compute symmetric axis limits from the full trajectory.

    Parameters
    -
    positions : np.ndarray, shape (n_steps, 3, 2)
        Full trajectory in center-of-mass frame.
    padding : float
        Extra space added beyond the trajectory extents on all sides.

    Returns
    -
    tuple[float, float, float, float]
        (x_min, x_max, y_min, y_max)
    """
    x_all = positions[:, :, 0]
    y_all = positions[:, :, 1]
    x_min = x_all.min() - padding
    x_max = x_all.max() + padding
    y_min = y_all.min() - padding
    y_max = y_all.max() + padding
    # make plot square
    x_range = x_max - x_min
    y_range = y_max - y_min
    max_range = max(x_range, y_range)
    x_mid = (x_min + x_max) / 2
    y_mid = (y_min + y_max) / 2

    return (x_mid - max_range / 2, x_mid + max_range / 2,
            y_mid - max_range / 2, y_mid + max_range / 2)


def animate_simulation(result: dict, save_path: str = None) -> None:
    """
    Render and display the animation.
    Optionally save the file.

    Parameters
    -
    result : dict
        Dictionary returned by run_simulation(). Expected keys:
        positions, masses, names, steps, status, message.
    save_path : str or None
        File path to save the animation (e.g. output.gif or output.mp4).
        If None the animation is displayed interactively.
    """
    positions = result["positions"]   # shape (n_steps, 3, 2)
    masses    = result["masses"]
    names     = result["names"]
    message   = result["message"]
    n_steps   = result["steps"]

    assert positions.shape[1] == 3, "Expected exactly 3 bodies in trajectory."
    assert positions.shape[2] == 2, "Expected 2D positions."

    # fig. step
    fig, ax = plt.subplots(figsize=(7, 7), facecolor="#0d0d0d")
    ax.set_facecolor("#0d0d0d")
    ax.tick_params(colors="gray")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")

    x_min, x_max, y_min, y_max = compute_axis_limits(positions)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.set_xlabel("x", color="gray")
    ax.set_ylabel("y", color="gray")

    # status at top
    title = ax.set_title(message, color="white", fontsize=9, pad=8)

    # art. initialization
    marker_sizes = [max(BODY_SCALE, BODY_SCALE * np.log1p(m)) for m in masses]

    trails  = [ax.plot([], [], "-", color=COLORS[i],
                       alpha=0.4, linewidth=0.8, label=names[i])[0]
               for i in range(3)]

    dots    = [ax.scatter([], [], s=marker_sizes[i],
                          color=COLORS[i], zorder=5)
               for i in range(3)]

    ax.legend(loc="upper right", fontsize=8,
              facecolor="#1a1a1a", labelcolor="white", edgecolor="#333333")

    # Update()
    def update(frame: int):
        """Update trail and body marker position for each body at given frame."""
        trail_start = max(0, frame - TRAIL_LENGTH)

        for i in range(3):
            # Trail
            trail_x = positions[trail_start:frame + 1, i, 0]
            trail_y = positions[trail_start:frame + 1, i, 1]
            trails[i].set_data(trail_x, trail_y)

            # Body marker
            dots[i].set_offsets(positions[frame, i])

        return trails + dots

    # build and show
    interval_ms = 20  # ms between frames → ~50 fps

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=n_steps,
        interval=interval_ms,
        blit=True,
    )

    if save_path is not None:
        print(f"Saving animation to {save_path} ...")
        writer = _get_writer(save_path)
        anim.save(save_path, writer=writer, dpi=120,
                  savefig_kwargs={"facecolor": "#0d0d0d"})
        print("Saved.")
    else:
        plt.tight_layout()
        plt.show()

    plt.close(fig)

