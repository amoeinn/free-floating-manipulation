"""Render a tumbling target under a fixed sun, with ground truth attitude.

The camera and the sun are fixed in the world and the target turns, which is
the geometry of a station keeping observation of a derelict. Rendering it is
easier in the equivalent frame where the target stands still and the world
turns around it: a body rotation `R` becomes a camera pre-multiplied by `R`
and a sun carried into the body frame as `R^T s`. The images are identical and
the exact ray tracer needs no modification.

The tracker sees the other formulation, a fixed camera and a model rotated by
`R`, so both rotate about the target frame origin rather than about any
centroid. Mixing those two centres would put a translation into every attitude
error and would look exactly like a tracking bias.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.raytrace import Camera, Scene, look_at
from src.target import client_satellite, client_scene
from src.tumble import (angular_momentum, assert_triaxial, integrate,
                        kinetic_energy)

ROOT = Path(__file__).resolve().parent.parent
INERTIA = np.array([1.0, 1.9, 2.6])
RATE_DEG_S = 2.7                      # ENVISAT class
OMEGA_DIRECTION = np.array([0.35, 0.22, 0.16])
STANDOFF = 5.0
RESOLUTION = 128


def render_sequence(name, duration, fps, check_triaxial):
    omega0 = OMEGA_DIRECTION / np.linalg.norm(OMEGA_DIRECTION) * np.deg2rad(RATE_DEG_S)
    dt = 1.0 / fps
    n = int(round(duration * fps))
    attitudes, rates = integrate(INERTIA, omega0, dt=dt, steps=n)

    L = angular_momentum(attitudes, rates, INERTIA)
    T = kinetic_energy(rates, INERTIA)
    drift_L = np.abs(np.linalg.norm(L, axis=1) - np.linalg.norm(L[0])).max()
    drift_T = np.abs(T - T[0]).max()
    travel = np.linalg.norm(rates - rates[0], axis=1).max() / np.linalg.norm(rates[0])

    base = client_scene()
    R_cw, t_cw = look_at(np.array([STANDOFF, 0.0, 0.0]), np.zeros(3),
                         np.array([0.0, 0.0, 1.0]))
    cam0 = Camera.with_fov(RESOLUTION, RESOLUTION, 40.0, R_cw, t_cw)

    images, masks, sun_angles = [], [], []
    for R in attitudes:
        scene = Scene(primitives=client_satellite(),
                      light_direction=R.T @ base.light_direction,
                      light_intensity=base.light_intensity)
        cam = Camera(RESOLUTION, RESOLUTION, cam0.fx, cam0.fy, cam0.cx, cam0.cy,
                     R_cw @ R, t_cw)
        out = scene.render(cam)
        images.append(out["image"].astype(np.float32))
        masks.append(out["hit"])
        cosang = np.clip(float((R.T @ base.light_direction) @ base.light_direction),
                         -1.0, 1.0)
        sun_angles.append(np.degrees(np.arccos(cosang)))

    out = ROOT / "data" / f"tumble_{name}.npz"
    np.savez_compressed(out, images=np.asarray(images), masks=np.asarray(masks),
                        attitudes=attitudes, rates=rates,
                        sun_angle_deg=np.asarray(sun_angles), dt=dt,
                        inertia=INERTIA,
                        camera=np.concatenate([R_cw.ravel(), t_cw]),
                        fx=cam0.fx, fy=cam0.fy, cx=cam0.cx, cy=cam0.cy,
                        width=RESOLUTION, height=RESOLUTION)

    print(f"  {name}: {duration:.0f} s at {fps} fps, {len(images)} frames")
    print(f"    body frame sun sweeps 0.0 to {max(sun_angles):.1f} deg")
    print(f"    polhode travel {travel:.3f}"
          f"{'  (above the 0.20 the guard wants)' if travel >= 0.20 else '  (below 0.20)'}")
    print(f"    |L| drift {drift_L:.2e}, energy drift {drift_T:.2e}")
    print(f"    wrote {out.name} ({out.stat().st_size/1e6:.1f} MB)")
    if check_triaxial:
        report = assert_triaxial(INERTIA, rates)
        print(f"    tri-axial guard passes, axis share "
              f"{np.round(report['axis_share'], 2)}")
    return travel


def main():
    print(f"tumbling target, {RATE_DEG_S} deg/s, inertia {INERTIA}, "
          f"camera fixed at {STANDOFF} m")
    print("  the short window is where the baked model is still valid; the long "
          "one is where\n  the Euler coupling actually shows\n")
    render_sequence("short", duration=8.0, fps=5.0, check_triaxial=False)
    print()
    render_sequence("long", duration=48.0, fps=2.0, check_triaxial=True)


if __name__ == "__main__":
    main()
