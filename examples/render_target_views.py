"""Phase 4 step 1: the approach imagery, and the checks that it is trustworthy.

Writes a dataset of views of the client satellite with ground truth camera
poses, then verifies the pieces that everything downstream will lean on. The
splatting forward pass reuses this camera model exactly, so a convention error
here would reappear as a reconstruction error there and be much harder to see.

Every check below is against something independent of the code it checks, or
against a closed form, rather than against a second call to the same function.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.raytrace import Camera, approach_trajectory, look_at
from src.target import bounding_radius, client_satellite, client_scene

OUT = Path(__file__).resolve().parent.parent / "data" / "approach"
N_VIEWS = 60
RESOLUTION = 128


def check_camera_round_trip(scene, camera):
    """`project` inverts `rays`, on points neither of them chose.

    `rays` builds a direction from a pixel and `project` builds a pixel from a
    point. They are separate code paths through the same convention, so
    running a surface point found by one back through the other is a real
    check on the convention rather than a restatement of it.
    """
    origin, direction = camera.rays()
    distance, _, _, _ = scene.intersect(origin, direction)
    hit = np.isfinite(distance)
    points = origin + distance[hit][:, None] * direction[hit]

    v, u = np.nonzero(hit)
    uv, z = camera.project(points)
    pixel_error = np.abs(uv - np.stack([u, v], axis=-1)).max()
    depth_error = np.abs(z - (points - camera.centre) @ camera.R[2]).max()
    return pixel_error, depth_error, int(hit.sum())


def check_depth_against_closed_form():
    """The centre pixel's depth, where the answer is known without rendering.

    A camera at range r on the target's +x axis, looking back at the origin,
    sees the bus face at x = 0.55 dead centre. The depth there must be
    r - 0.55 with nothing else in the way.
    """
    from src.target import BUS_HALF
    worst = 0.0
    for r in (4.0, 6.0, 9.0):
        R, t = look_at(np.array([r, 0.0, 0.0]), np.zeros(3), np.array([0.0, 0.0, 1.0]))
        cam = Camera.with_fov(65, 65, 40.0, R, t)      # odd size puts a pixel dead centre
        out = client_scene().render(cam, shadows=False)
        centre = out["depth"][32, 32]
        worst = max(worst, abs(centre - (r - BUS_HALF[0])))
    return worst


def check_normals(scene, camera):
    """Unit length, and facing the camera on every visible surface.

    A sign error in the box normals is invisible in an image, because the
    shading clamps at zero and a flipped face just goes black. It is not
    invisible here.
    """
    origin, direction = camera.rays()
    distance, normal, _, _ = scene.intersect(origin, direction)
    hit = np.isfinite(distance)
    lengths = np.linalg.norm(normal[hit], axis=-1)
    facing = np.einsum("ij,ij->i", normal[hit], -direction[hit])
    return abs(lengths - 1.0).max(), facing.min()


def main():
    scene = client_scene()
    radius = bounding_radius()
    cameras = approach_trajectory(N_VIEWS, start_range=9.0, end_range=4.2,
                                  sweep_deg=150.0, elevation_deg=35.0,
                                  fov_deg=40.0, width=RESOLUTION, height=RESOLUTION)

    print(f"target: {len(client_satellite())} primitives, bounding radius {radius:.3f} m")
    print(f"cameras: {N_VIEWS} views, {RESOLUTION}x{RESOLUTION}, fov 40 deg, "
          f"range 9.0 -> 4.2 m over a 150 deg sweep\n")

    print("camera model, checked against itself only where the two paths are separate")
    probe = cameras[N_VIEWS // 3]
    pixel_error, depth_error, n_points = check_camera_round_trip(scene, probe)
    print(f"  project inverts rays over {n_points} surface points")
    print(f"    max pixel error            {pixel_error:.3e} px")
    print(f"    max depth error            {depth_error:.3e} m")
    print(f"  centre depth vs closed form  {check_depth_against_closed_form():.3e} m")
    unit_error, facing_min = check_normals(scene, probe)
    print(f"  normals unit length, error   {unit_error:.3e}")
    print(f"  worst normal.view (want > 0) {facing_min:+.4f}")
    if facing_min <= 0:
        raise SystemExit("a visible surface faces away from the camera")

    print("\nrendering")
    images, depths, poses, hits = [], [], [], []
    for cam in cameras:
        out = scene.render(cam)
        images.append(out["image"])
        depths.append(out["depth"])
        hits.append(out["hit"])
        poses.append(np.concatenate([cam.R.ravel(), cam.t]))
    images = np.asarray(images, np.float32)
    depths = np.asarray(depths, np.float32)
    hits = np.asarray(hits)

    coverage = hits.reshape(N_VIEWS, -1).mean(axis=1)
    print(f"  target covers {100*coverage.mean():.1f}% of frame on average, "
          f"{100*coverage.min():.1f}% to {100*coverage.max():.1f}%")

    lit = (images.max(axis=-1) > 1e-6) & hits
    shadowed = hits & ~lit
    print(f"  {100*shadowed.sum()/hits.sum():.1f}% of surface pixels are unlit, "
          "either facing away or in shadow")
    if shadowed.sum() == 0:
        raise SystemExit("no pixel is in shadow, so the hard light is doing nothing "
                         "and this dataset does not test what it exists to test")

    print("\n  per primitive, views in which it is visible at all")
    prims = client_satellite()
    for i, prim in enumerate(prims):
        seen = 0
        pixels = 0
        for cam in cameras:
            out = scene.render(cam, shadows=False)
            n = int((out["primitive"] == i).sum())
            pixels += n
            seen += n > 0
        print(f"    {prim.name:16s} {seen:3d} of {N_VIEWS} views, "
              f"{pixels/N_VIEWS:7.1f} px per view")

    centres = np.array([cam.centre for cam in cameras])
    baseline = np.linalg.norm(centres[:, None] - centres[None], axis=-1).max()
    print(f"\n  widest baseline between views {baseline:.2f} m, "
          f"against a target {2*radius:.2f} m across")

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT / "views.npz", images=images, depths=depths, hits=hits,
        poses=np.asarray(poses), width=RESOLUTION, height=RESOLUTION,
        fx=cameras[0].fx, fy=cameras[0].fy, cx=cameras[0].cx, cy=cameras[0].cy,
        bounding_radius=radius)
    print(f"\nwrote {OUT / 'views.npz'} "
          f"({(OUT / 'views.npz').stat().st_size / 1e6:.1f} MB)")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        pick = np.linspace(0, N_VIEWS - 1, 12).astype(int)
        fig, axes = plt.subplots(3, 4, figsize=(8, 6))
        for ax, k in zip(axes.ravel(), pick):
            ax.imshow(images[k])
            ax.set_title(f"view {k}", fontsize=8)
            ax.axis("off")
        fig.suptitle("Client satellite over the approach, single hard light, no fill")
        fig.tight_layout()
        docs = Path(__file__).resolve().parent.parent / "docs"
        docs.mkdir(exist_ok=True)
        fig.savefig(docs / "approach_views.png", dpi=130)
        print(f"wrote {docs / 'approach_views.png'}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
