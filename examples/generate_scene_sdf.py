"""Generate the Gazebo scene from the definitions already in this repository.

Two descriptions of the same satellite that can drift apart is a defect
waiting to happen, so there is one description and this emits the other.
`src/target.py` stays the source for the client and the Panda URDF stays the
source for the arm; nothing here restates a dimension.

Run with `--check` to fail when the committed SDF no longer matches what the
sources produce. That is what makes "generated" mean something a month from
now rather than "generated once".

Every rigid body is its own static model, which is not a stylistic choice.
Gazebo's `set_pose` service accepts a link name, returns success, and does
nothing: measured, a link left at its SDF pose while the service reported
true. Only model level poses actually move, so anything this scene needs to
move has to be a model.
"""

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.raytrace import Box, Cylinder
from src.target import client_satellite, client_scene

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "ws" / "src" / "free_floating_manipulation" / "models"
OUT = MODELS / "servicing_scene.sdf"
MANIFEST = MODELS / "servicing_scene.json"
PANDA_URDF = ROOT / "ws" / "build" / "panda_identified.urdf"

# The servicer's own body, the same box the phase 3 demonstration used, so the
# two scenes describe the same vehicle.
BUS_HALF = np.array([0.30, 0.30, 0.18])
BUS_CENTRE = np.array([0.0, 0.0, -0.32])
BUS_ALBEDO = np.array([0.55, 0.55, 0.58])
CLIENT_START = np.array([0.0, 0.0, 6.0])


def rgba(albedo, alpha=1.0):
    r, g, b = (float(c) for c in albedo)
    return f"{r:.4f} {g:.4f} {b:.4f} {alpha:.1f}"


def geometry_xml(prim):
    if isinstance(prim, Box):
        s = 2.0 * np.asarray(prim.half_extents, float)
        return f"<box><size>{s[0]:.6f} {s[1]:.6f} {s[2]:.6f}</size></box>"
    if isinstance(prim, Cylinder):
        return (f"<cylinder><radius>{prim.radius:.6f}</radius>"
                f"<length>{2.0 * prim.half_height:.6f}</length></cylinder>")
    raise TypeError(f"no SDF mapping for {type(prim).__name__}")


def visual(name, geom, albedo, pose="0 0 0 0 0 0"):
    return f"""      <visual name="{name}">
        <pose>{pose}</pose>
        <geometry>{geom}</geometry>
        <material>
          <ambient>{rgba(np.asarray(albedo) * 0.3)}</ambient>
          <diffuse>{rgba(albedo)}</diffuse>
          <specular>0.1 0.1 0.1 1.0</specular>
        </material>
      </visual>"""


def static_model(name, pose, visuals):
    body = "\n".join(visuals)
    return f"""    <model name="{name}">
      <static>true</static>
      <pose>{pose}</pose>
      <link name="body">
{body}
      </link>
    </model>"""


def client_models():
    """One model per primitive, from `src/target.py` and nothing else."""
    models = []
    for prim in client_satellite():
        p, r = np.asarray(prim.position, float), np.asarray(prim.rpy, float)
        pose = (f"{p[0] + CLIENT_START[0]:.6f} {p[1] + CLIENT_START[1]:.6f} "
                f"{p[2] + CLIENT_START[2]:.6f} {r[0]:.9f} {r[1]:.9f} {r[2]:.9f}")
        models.append(static_model(f"client_{prim.name}", pose,
                                   [visual("v", geometry_xml(prim), prim.albedo)]))
    return models


def panda_models():
    """One model per Panda link, carrying the URDF's own visual mesh."""
    root = ET.parse(PANDA_URDF).getroot()
    models, names = [], []
    for link in root.findall("link"):
        vis = link.find("visual")
        if vis is None:
            continue
        mesh = vis.find("geometry/mesh")
        if mesh is None:
            continue
        origin = vis.find("origin")
        xyz = [float(v) for v in (origin.get("xyz", "0 0 0").split()
                                  if origin is not None else "0 0 0".split())]
        rpy = [float(v) for v in (origin.get("rpy", "0 0 0").split()
                                  if origin is not None else "0 0 0".split())]
        pose = (f"{xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f} "
                f"{rpy[0]:.9f} {rpy[1]:.9f} {rpy[2]:.9f}")
        geom = f"<mesh><uri>{mesh.get('filename')}</uri></mesh>"
        name = link.get("name")
        names.append(name)
        models.append(f"""    <model name="{name}">
      <static>true</static>
      <pose>0 0 0 0 0 0</pose>
      <link name="body">
        <visual name="v">
          <pose>{pose}</pose>
          <geometry>{geom}</geometry>
        </visual>
      </link>
    </model>""")
    return models, names


def build():
    sun = client_scene().light_direction
    models = [static_model("servicer_bus",
                           f"{BUS_CENTRE[0]:.6f} {BUS_CENTRE[1]:.6f} "
                           f"{BUS_CENTRE[2]:.6f} 0 0 0",
                           [visual("v", geometry_xml(
                               Box(position=BUS_CENTRE, rpy=[0, 0, 0],
                                   albedo=BUS_ALBEDO, name="bus",
                                   half_extents=BUS_HALF)), BUS_ALBEDO)])]
    arm, arm_names = panda_models()
    models += arm
    models += client_models()
    client_names = [f"client_{p.name}" for p in client_satellite()]

    header = ("<!-- GENERATED by examples/generate_scene_sdf.py from src/target.py\n"
              "     and ws/build/panda_identified.urdf. Do not edit by hand: run\n"
              "     the generator with --check to confirm this file is current.\n\n"
              "     Every body is a separate static model on purpose. Gazebo's\n"
              "     set_pose accepts a link name, returns success and moves\n"
              "     nothing, so only models can be driven. Static also means the\n"
              "     physics engine integrates none of this, which is the point:\n"
              "     the motion comes from the verified dynamics in src/. -->")

    return f"""<?xml version="1.0" ?>
{header}
<sdf version="1.9">
  <world name="servicing">
    <physics name="default" type="dart">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
    </physics>

    <!-- The Physics system is present and integrates nothing, which is not a
         contradiction. Every model below is static, so the engine owns none of
         them, and the check in the node reads dynamic_pose/info to confirm
         that rather than trusting this comment.

         It cannot simply be left out. Declaring plugins explicitly replaces
         Gazebo's default set, and a world without the Physics system accepts a
         set_pose, answers true, and never applies it: measured, a model sat at
         its SDF pose while every call reported success. UserCommands stages
         the change and something in the physics update has to commit it. -->
    <plugin filename="gz-sim-physics-system"
            name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-scene-broadcaster-system"
            name="gz::sim::systems::SceneBroadcaster"/>
    <plugin filename="gz-sim-user-commands-system"
            name="gz::sim::systems::UserCommands"/>

    <light type="directional" name="sun">
      <cast_shadows>true</cast_shadows>
      <direction>{sun[0]:.6f} {sun[1]:.6f} {sun[2]:.6f}</direction>
      <diffuse>1.0 1.0 1.0 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
    </light>
    <scene>
      <ambient>0.0 0.0 0.0 1</ambient>
      <background>0.0 0.0 0.0</background>
      <grid>false</grid>
    </scene>

{chr(10).join(models)}
  </world>
</sdf>
""", arm_names, client_names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="fail if the committed SDF is not what the sources produce")
    args = ap.parse_args()

    sdf, arm_names, client_names = build()
    if args.check:
        if not OUT.exists():
            raise SystemExit(f"{OUT} does not exist; run the generator without --check")
        if not MANIFEST.exists() or MANIFEST.read_text() != json.dumps({"world": "servicing", "arm": arm_names, "client": client_names, "bus": "servicer_bus"}, indent=2) + "\n":
            raise SystemExit(f"{MANIFEST.name} is stale or missing; regenerate it "
                             "with `python examples/generate_scene_sdf.py`.")
        current = OUT.read_text()
        if current != sdf:
            raise SystemExit(
                f"{OUT.name} is stale: it does not match what src/target.py and the "
                "Panda URDF now produce.\nRegenerate it with "
                "`python examples/generate_scene_sdf.py`.")
        print(f"{OUT.name} is current: {len(arm_names)} arm bodies, "
              f"{len(client_names)} client bodies")
        return

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(sdf)
    # The manifest lists the models this file actually contains, and the node
    # drives exactly these and nothing else. It exists because the node used
    # to build its own list from PyBullet's links, which include panda_link8,
    # a flange frame with no visual that the generator therefore never
    # emitted. One unknown name voids the whole set_pose_vector batch while
    # the service still returns success, so the entire scene froze silently
    # and every call reported true.
    MANIFEST.write_text(json.dumps({"world": "servicing", "arm": arm_names, "client": client_names, "bus": "servicer_bus"}, indent=2) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)} ({len(sdf)/1000:.1f} kB)")
    print(f"wrote {MANIFEST.relative_to(ROOT)}, the only models the node may drive")
    print(f"  {len(arm_names)} arm bodies: {', '.join(arm_names[:4])} ...")
    print(f"  {len(client_names)} client bodies: {', '.join(client_names)}")
    print(f"  every one static, so Gazebo integrates none of it")


if __name__ == "__main__":
    main()
