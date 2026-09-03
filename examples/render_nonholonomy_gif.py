"""Render the README's hero animation: a closed joint-space loop.

Two joints trace a circle and return to exactly the angles they started
at. The base does not return. This is the same run verify_nonholonomy.py
measures, driven through the same simulate_loop, so the figure and the
number cannot drift apart.

The left panel is a fixed world camera, so what rotates on screen is the
spacecraft, not the viewpoint. The right panels carry the claim
numerically: the two driven joints come back to zero deviation, and the
base attitude does not come back to zero.

Rendered offscreen through PyBullet's software rasteriser, so it needs no
display.

Usage:
    python examples/render_nonholonomy_gif.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
import pybullet_data
from PIL import Image

from src.freeflight import (JointLoop, disable_damping, rotation_angle,
                            simulate_loop)

FIGURE = Path(__file__).resolve().parent.parent / "docs" / "closed_loop.gif"
LOOP = JointLoop()
TIMESTEP = 2.5e-4           # the step the README's 10.02 deg is quoted at
CAPTURE_EVERY = 100         # 2 s at 0.25 ms -> 80 frames
RENDER = 520                # square, pixels
FRAME_MS = 55
HOLD_MS = 1600              # dwell on the final state so the point lands
BACKGROUND = np.array([30, 34, 41], dtype=np.uint8)   # the Panda is white
GHOST = np.array([86, 96, 116], dtype=np.uint8)       # where it started


def camera():
    """A fixed world view, tilted well over so that base yaw is obvious."""
    view = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=[0.05, 0.0, 0.42], distance=1.55,
        yaw=48, pitch=-32, roll=0, upAxisIndex=2)
    projection = p.computeProjectionMatrixFOV(
        fov=50, aspect=1.0, nearVal=0.05, farVal=12.0)
    return view, projection


def grab(view, projection) -> np.ndarray:
    """Render, then paint the background dark.

    The Panda's meshes are near-white and the rasteriser returns an opaque
    white background, so the arm is nearly invisible against it. The alpha
    channel is no help; the segmentation buffer is, since it marks every
    background pixel -1.
    """
    image = p.getCameraImage(RENDER, RENDER, view, projection,
                             shadow=1, lightDirection=[0.7, 0.5, 1.0],
                             renderer=p.ER_TINY_RENDERER)
    rgb = np.reshape(np.asarray(image[2], dtype=np.uint8),
                     (RENDER, RENDER, 4))[:, :, :3].copy()
    segmentation = np.reshape(np.asarray(image[4], dtype=np.int32),
                              (RENDER, RENDER))
    rgb[segmentation < 0] = BACKGROUND
    return rgb, segmentation >= 0


def compose(render, occupied, ghost, phases, deviations, rotations,
            final_angle, peak) -> Image.Image:
    """One GIF frame: the view, the joint deviations, the base attitude.

    `ghost` is where the robot stood at the start. Painting it wherever the
    robot is not standing now makes the base rotation visible in the render
    itself: a 10 degree yaw leaves a nearly identical silhouette, and
    without the reference it is easy to miss.
    """
    render = render.copy()
    render[ghost & ~occupied] = GHOST
    figure = plt.figure(figsize=(8.6, 3.9), dpi=88)
    grid = figure.add_gridspec(2, 2, width_ratios=[1.02, 1.0],
                               hspace=0.34, wspace=0.22)

    scene = figure.add_subplot(grid[:, 0])
    scene.imshow(render)
    scene.set_axis_off()
    scene.set_title("free-floating base, zero gravity, no contact",
                    fontsize=9, color="0.25", pad=6)
    scene.text(0.5, -0.035, "grey shows where the servicer started",
               transform=scene.transAxes, ha="center", va="top",
               fontsize=7.5, color="0.45")

    first, second = LOOP.joints
    joints = figure.add_subplot(grid[0, 1])
    joints.plot(phases, deviations[:, 0], color="#1f77b4",
                label=f"joint {first + 1}")
    joints.plot(phases, deviations[:, 1], color="#2ca02c",
                label=f"joint {second + 1}")
    joints.axhline(0.0, color="0.6", lw=0.8, ls=":")
    joints.set_xlim(0, 360)
    joints.set_ylim(-0.75, 1.35)
    joints.set_xticks([0, 90, 180, 270, 360])
    joints.set_ylabel("joint angle,\nfrom start (rad)", fontsize=8)
    joints.tick_params(labelsize=7.5)
    joints.legend(fontsize=7.5, loc="upper right", frameon=False)
    joints.set_title("the joints come back", fontsize=9, color="0.25")
    joints.grid(alpha=0.25)

    base = figure.add_subplot(grid[1, 1])
    base.plot(phases, rotations, color="#d62728")
    base.axhline(0.0, color="0.6", lw=0.8, ls=":")
    base.set_xlim(0, 360)
    base.set_ylim(-peak * 0.06, peak * 1.16)
    base.set_xticks([0, 90, 180, 270, 360])
    base.set_xlabel("position around the closed loop (deg)", fontsize=8)
    base.set_ylabel("base rotation\nfrom start (deg)", fontsize=8)
    base.tick_params(labelsize=7.5)
    base.set_title("the base does not", fontsize=9, color="0.25")
    base.grid(alpha=0.25)

    if len(phases) and phases[-1] > 358:
        base.plot([360], [final_angle], "o", color="#d62728", ms=4.5)
        base.annotate(f"{final_angle:.2f} deg", xy=(360, final_angle),
                      xytext=(-6, 13), textcoords="offset points",
                      ha="right", fontsize=9.5, color="#d62728", weight="bold")

    figure.canvas.draw()
    frame = Image.fromarray(
        np.asarray(figure.canvas.buffer_rgba())[:, :, :3].copy())
    plt.close(figure)
    return frame


def main() -> None:
    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, 0)
    body = p.loadURDF("franka_panda/panda.urdf", useFixedBase=False,
                      basePosition=[0, 0, 0])
    disable_damping(body)
    view, projection = camera()

    renders, masks, phases, deviations, rotations = [], [], [], [], []
    first, second = LOOP.joints

    def watch(step, steps, phase):
        if step % CAPTURE_EVERY and step != steps:
            return
        orientation = p.getBasePositionAndOrientation(body)[1]
        matrix = np.asarray(
            p.getMatrixFromQuaternion(orientation)).reshape(3, 3)
        rgb, occupied = grab(view, projection)
        renders.append(rgb)
        masks.append(occupied)
        phases.append(np.degrees(step / steps * 2.0 * np.pi))
        deviations.append([
            p.getJointState(body, first)[0] - LOOP.home[first],
            p.getJointState(body, second)[0] - LOOP.home[second],
        ])
        rotations.append(np.degrees(rotation_angle(matrix)))

    result = simulate_loop(body, LOOP, dt=TIMESTEP, on_step=watch)
    final_angle = np.degrees(result["angle"])
    print(f"loop closed to {result['closure']:.2e} rad; "
          f"base ended {final_angle:.4f} deg rotated, "
          f"peaking at {max(rotations):.2f} deg mid-loop; "
          f"{len(renders)} frames captured")

    deviations = np.asarray(deviations)
    rotations = np.asarray(rotations)
    phases = np.asarray(phases)

    peak = float(rotations.max())
    ghost = masks[0]
    frames = [compose(renders[i], masks[i], ghost, phases[:i + 1],
                      deviations[:i + 1], rotations[:i + 1], final_angle, peak)
              for i in range(len(renders))]
    FIGURE.parent.mkdir(exist_ok=True)
    palette = [frame.quantize(colors=128, method=Image.MEDIANCUT)
               for frame in frames]
    # Hold the last frame by lengthening it rather than repeating it, which
    # would only add zero-delta frames to the file.
    durations = [FRAME_MS] * len(palette)
    durations[-1] = HOLD_MS
    durations[0] = FRAME_MS * 6
    palette[0].save(FIGURE, save_all=True, append_images=palette[1:],
                    duration=durations, loop=0, optimize=True)
    print(f"wrote {FIGURE.relative_to(Path.cwd())} "
          f"({FIGURE.stat().st_size / 1e6:.2f} MB, {len(frames)} frames)")
    p.disconnect()


if __name__ == "__main__":
    main()
