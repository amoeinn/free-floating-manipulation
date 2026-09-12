"""What the Gazebo scene must be true of, checked without starting Gazebo.

Phase 5 is a renderer and a ROS host, not a physics engine: the dynamics stay
in code that has been verified, and Gazebo is handed poses rather than asked
to compute them. Several of the things that has to be true of the scene can be
established from the files alone, and those are here.

What cannot be established from the files is deliberately absent. Whether the
running world actually integrates nothing, whether a pose command is applied
rather than merely acknowledged, whether the node's startup guards fire
against a live server: all of that needs a Gazebo process and a ROS graph, and
a unit test that mocked them would be testing the mock. That is named future
work as a `ros` marked integration tier, not an omission.

The first case here is the one that has caught something. `look_at` returned a
singular rotation whenever the up vector lay along the line of sight, and the
symptom was a uniformly black render, which reads as an empty scene rather
than a broken camera.
"""

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))

import numpy as np
import pybullet as p
import pytest

import generate_scene_sdf as generator
from src.raytrace import Camera, Scene, look_at
from src.target import client_satellite, client_scene


@pytest.fixture(scope="module")
def generated():
    """What the generator produces from the sources, right now."""
    return generator.build()


def test_look_at_refuses_a_view_direction_along_the_up_vector():
    """The guard that has caught something.

    With the up vector on the line of sight there is no way to orient the
    image, and the arithmetic does not complain: the cross product is zero,
    the rotation is singular, every ray degenerates and the render comes out
    uniformly black. That looks like an empty scene, not a broken camera, and
    it cost a debugging session before the guard existed.
    """
    with pytest.raises(ValueError, match="parallel to the view direction"):
        look_at(np.array([0.0, 0.0, 0.25]), np.array([0.0, 0.0, 6.0]),
                np.array([0.0, 0.0, 1.0]))


def test_a_degenerate_camera_would_have_rendered_a_black_frame():
    """Why the guard above has to exist rather than be a style preference.

    Built by hand, because `look_at` now refuses to produce one. A camera
    whose rotation is singular renders nothing at all, and nothing in the
    image says why.
    """
    scene = client_scene()
    upright = look_at(np.array([4.0, -3.0, 2.0]), np.zeros(3),
                      np.array([0.0, 0.0, 1.0]))
    good = scene.render(Camera.with_fov(32, 32, 40.0, *upright))
    assert good["hit"].any(), "the control camera sees nothing, so this proves nothing"

    singular = np.zeros((3, 3))
    singular[2, 2] = 1.0
    blind = Camera(32, 32, 40.0, 40.0, 15.5, 15.5, singular,
                   np.array([0.0, 0.0, -5.0]))
    assert not scene.render(blind)["hit"].any(), (
        "a singular camera rotation is supposed to render an empty frame; if "
        "it does not, this case no longer demonstrates the failure mode")


def test_the_world_declares_the_physics_system_it_needs_to_apply_poses():
    """Declaring plugins explicitly replaces Gazebo's defaults, and a world
    without the Physics system accepts a set_pose, answers true, and never
    applies it. Measured: a model sat at its SDF pose while every call
    reported success."""
    root = ET.fromstring(generator.build()[0])
    plugins = {plugin.get("filename") for plugin in root.iter("plugin")}
    assert "gz-sim-physics-system" in plugins, (
        "without the Physics system every pose command is silently ignored")
    for required in ("gz-sim-scene-broadcaster-system", "gz-sim-user-commands-system"):
        assert required in plugins


def test_every_model_is_static_so_gazebo_integrates_none_of_the_scene():
    """The whole justification for phase 5 is that the motion is the verified
    dynamics and not a second solver's opinion. A model left dynamic would add
    Gazebo's integration on top of ours and still look plausible."""
    root = ET.fromstring(generator.build()[0])
    models = list(root.iter("model"))
    assert len(models) >= 17, f"only {len(models)} models; the scene is incomplete"
    for model in models:
        static = model.find("static")
        assert static is not None and static.text.strip() == "true", (
            f"model {model.get('name')!r} is not static, so the physics engine "
            "owns it and its motion is no longer ours")


def test_the_committed_world_is_what_the_sources_produce(generated):
    """Two descriptions of the same satellite that can drift apart is a defect
    waiting to happen, so there is one description and the other is
    generated. This is what makes "generated" still mean something later."""
    sdf, arm_names, client_names = generated
    assert generator.OUT.exists(), f"{generator.OUT} is missing; run the generator"
    assert generator.OUT.read_text() == sdf, (
        f"{generator.OUT.name} no longer matches what src/target.py and the "
        "Panda URDF produce; regenerate it")
    expected = json.dumps({"world": "servicing", "arm": arm_names,
                           "client": client_names, "bus": "servicer_bus"},
                          indent=2) + "\n"
    assert generator.MANIFEST.read_text() == expected


def test_the_installed_world_is_the_one_the_launch_file_will_load(generated):
    """The other guard that has caught something.

    colcon installs models/ into the share directory and the launch file loads
    the installed copy, so regenerating the source without rebuilding leaves
    the launch path running an older world. That is how a scene ran without
    the Physics system while the source had it, and the symptom was a
    launch that came up, published every topic, and drove nothing.
    """
    if not generator.INSTALLED.exists():
        pytest.skip("nothing installed yet; build the workspace with colcon")
    assert generator.INSTALLED.read_text() == generated[0], (
        "the installed world is stale, so `ros2 launch` will run a different "
        "scene from the one in ws/src. Rebuild with colcon.")


def test_the_manifest_names_no_body_the_robot_model_does_not_have(panda_free,
                                                                 generated):
    """The node drives exactly the manifest and nothing else.

    It used to build its list from PyBullet's links, which include
    panda_link8, a flange frame with no visual that the generator never
    emits. One unknown name voids the whole set_pose_vector batch while the
    service still answers true, so the scene froze while every call reported
    success.
    """
    _, arm_names, _ = generated
    links = {p.getJointInfo(panda_free, j)[12].decode()
             for j in range(p.getNumJoints(panda_free))}
    links.add("panda_link0")
    missing = sorted(set(arm_names) - links)
    assert not missing, f"the scene declares {missing}, which the robot lacks"
    skipped = sorted(links - set(arm_names))
    assert "panda_link8" in skipped, (
        "panda_link8 is in the scene after all; this case was written because "
        "it is the link with no visual that the generator must skip, and if "
        "that changed the guard it protects needs revisiting")


def test_the_client_in_the_scene_is_the_one_in_the_source(generated):
    """The SDF carries the target's geometry, and it is generated rather than
    restated. If the two ever disagree the reconstruction would be fitted to
    one satellite and rendered against another."""
    root = ET.fromstring(generated[0])
    boxes = {}
    for model in root.iter("model"):
        name = model.get("name")
        if not name.startswith("client_"):
            continue
        box = model.find(".//box/size")
        if box is not None:
            boxes[name] = np.array([float(v) for v in box.text.split()])
    for prim in client_satellite():
        key = f"client_{prim.name}"
        if key in boxes:
            assert np.allclose(boxes[key], 2.0 * np.asarray(prim.half_extents)), (
                f"{key} in the SDF is not twice the half extents in src/target.py")
    assert boxes, "no client boxes found in the world; the scene is not complete"
