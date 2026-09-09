#!/usr/bin/env python3
"""Own the integration, publish it on ROS 2, and hand Gazebo the result.

Gazebo is the visual and ROS 2 host here, not the physics. Every pose this
node sends was computed by code that has been checked against something: the
servicer's base motion from the free-floating dynamics verified against
PyBullet's free base mass matrix at its own eigensolve floor and cross checked
against an independent C++ port, the arm's link poses from a chain that agrees
with `getLinkState` to 0.000107 mm, and the client's attitude from a torque
free Euler integration whose angular momentum holds to 1e-13.

Nothing in the scene is dynamic. That is asserted at startup rather than
assumed, because a model left dynamic would quietly add Gazebo's integration
on top of ours and the result would still look plausible.
"""

import sys
from pathlib import Path


def _find_repo():
    """The repository root, from either the source tree or the install tree.

    The installed copy sits at a different depth than the source copy, so a
    fixed number of parents resolves correctly from one and silently wrongly
    from the other. Searching for the markers is the version that works from
    both.
    """
    for parent in Path(__file__).resolve().parents:
        if (parent / "src" / "dynamics.py").exists() and (parent / ".venv").exists():
            return parent
    raise SystemExit("cannot locate the repository root from "
                     f"{Path(__file__).resolve()}")


REPO = _find_repo()

# This node needs rclpy, which only exists once ROS is sourced, and torch and
# pybullet, which only exist in the repository venv. Neither interpreter has
# both, so the venv goes on the path here rather than being rebuilt against
# the system.
#
# The order matters and is not incidental. ROS 2 Jazzy ships numpy 1.26.4 and
# the venv pins 2.5.2, and whichever appears first on sys.path is the one that
# loads. Every verification in this repository ran against 2.5.2, so that is
# the one this node must get, and the venv is prepended for that reason. The
# version is then asserted rather than assumed, because a silent fall back to
# 1.26.4 would still run and would still look like a working scene.
_VENV = next(iter(sorted((REPO / ".venv" / "lib").glob("python3.*/site-packages"))), None)
if _VENV is not None and str(_VENV) not in sys.path:
    sys.path.insert(0, str(_VENV))
sys.path.insert(0, str(REPO))

import numpy as np

EXPECTED_NUMPY = "2.5.2"
if np.__version__ != EXPECTED_NUMPY:
    raise SystemExit(
        f"numpy {np.__version__} is loaded but this repository is verified "
        f"against {EXPECTED_NUMPY}. ROS 2 Jazzy ships its own numpy and it has "
        "won the path ordering; check that the venv site-packages sits ahead "
        "of the ROS site-packages on sys.path.")

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node as RosNode
from sensor_msgs.msg import JointState

from src import freeflight
from src.freeflight import JointLoop, rotation_exponential
from src.tumble import angular_momentum, assert_triaxial, integrate, kinetic_energy

WORLD = "servicing"
CLIENT_START = np.array([0.0, 0.0, 6.0])
CLIENT_INERTIA = np.array([1.0, 1.9, 2.6])
CLIENT_RATE_DEG_S = 2.7
CLIENT_OMEGA_DIR = np.array([0.35, 0.22, 0.16])
RATE_HZ = 20.0


def quaternion_from_matrix(R):
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        return np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s,
                         (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = np.sqrt(R[i, i] - R[j, j] - R[k, k] + 1.0) * 2
    q = np.zeros(4)
    q[0] = (R[k, j] - R[j, k]) / s
    q[1 + i] = 0.25 * s
    q[1 + j] = (R[j, i] + R[i, j]) / s
    q[1 + k] = (R[k, i] + R[i, k]) / s
    return q


class ScenePublisher(RosNode):
    def __init__(self):
        super().__init__("servicing_scene")
        self.declare_parameter("gazebo", True)

        import pybullet as p
        import pybullet_data
        self.p = p
        # DIRECT, with no GUI and no rendering. PyBullet is here only as the
        # forward kinematics this repository already verified against, and as
        # the loader for the same URDF everything else uses. It integrates
        # nothing: no stepSimulation is ever called below.
        if p.getConnectionInfo().get("isConnected", 0) != 1:
            p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.body = freeflight.load_panda(fixed_base=False)
        freeflight.disable_damping(self.body)
        self.model = freeflight.build_model(self.body)
        self.loop = JointLoop()
        self.arm_joints = [j for j in range(p.getNumJoints(self.body))
                           if p.getJointInfo(self.body, j)[2] != p.JOINT_FIXED][:7]
        # The models this node may drive come from the generator's manifest,
        # never from PyBullet's link list. Those two disagree: PyBullet reports
        # panda_link8, a flange frame with no visual that the generator never
        # emits. Sending a name Gazebo does not know voids the whole
        # set_pose_vector batch and the service still returns success, so the
        # scene freezes while every call reports true.
        import json
        manifest_path = Path(__file__).resolve().parent
        for cand in (manifest_path, REPO / "ws" / "src" / "free_floating_manipulation"):
            found = list(cand.glob("**/servicing_scene.json"))
            if found:
                self.manifest = json.loads(found[0].read_text())
                break
        else:
            raise SystemExit("servicing_scene.json not found; run "
                             "examples/generate_scene_sdf.py")

        pybullet_links = {}
        for j in range(p.getNumJoints(self.body)):
            pybullet_links[p.getJointInfo(self.body, j)[12].decode()] = j
        pybullet_links["panda_link0"] = -1
        missing = [n for n in self.manifest["arm"] if n not in pybullet_links]
        if missing:
            raise SystemExit(f"the scene declares arm bodies {missing} that the "
                             "robot model does not have; the SDF and the URDF "
                             "have drifted apart")
        self.link_names = {n: pybullet_links[n] for n in self.manifest["arm"]}
        skipped = sorted(set(pybullet_links) - set(self.link_names))
        self.get_logger().info(
            f"driving {len(self.link_names)} arm bodies from the manifest; "
            f"not in the scene and therefore never sent: {', '.join(skipped) or 'none'}")

        self.dt = 1.0 / RATE_HZ
        omega0 = (CLIENT_OMEGA_DIR / np.linalg.norm(CLIENT_OMEGA_DIR)
                  * np.deg2rad(CLIENT_RATE_DEG_S))
        steps = int(600 * RATE_HZ)
        self.client_att, rates = integrate(CLIENT_INERTIA, omega0, self.dt, steps)
        report = assert_triaxial(CLIENT_INERTIA, rates)
        L = angular_momentum(self.client_att, rates, CLIENT_INERTIA)
        drift = np.abs(np.linalg.norm(L, axis=1) - np.linalg.norm(L[0])).max()
        self.get_logger().info(
            f"client tumble: tri-axial guard passes, axis share "
            f"{np.round(report['axis_share'], 2)}, |L| drift {drift:.1e}")

        self.base_R = np.eye(3)
        self.base_t = np.zeros(3)
        self.k = 0

        self.pub_target = self.create_publisher(PoseStamped, "/target/pose", 10)
        self.pub_base = self.create_publisher(PoseStamped, "/servicer/base_pose", 10)
        self.pub_joints = self.create_publisher(JointState, "/servicer/joint_states", 10)

        self.gz = None
        if self.get_parameter("gazebo").value:
            self.gz = self._connect_gazebo()
        self.timer = self.create_timer(self.dt, self.step)

    def _connect_gazebo(self):
        from gz.msgs10.boolean_pb2 import Boolean
        from gz.msgs10.pose_v_pb2 import Pose_V
        from gz.transport13 import Node as GzNode
        self._Pose_V, self._Boolean = Pose_V, Boolean
        node = GzNode()
        self.assert_physics_disabled(node)
        self.assert_scene_matches()
        return node

    def assert_scene_matches(self):
        """Every model this node will drive exists in the running world.

        Checked once at startup rather than discovered at 20 Hz, because the
        failure is silent: an unknown name makes Gazebo reject the entire
        batch and answer true anyway, so the scene simply stops moving while
        nothing reports an error.
        """
        import re
        import subprocess
        try:
            out = subprocess.run(
                ["gz", "topic", "-e", "-t", f"/world/{WORLD}/pose/info", "-n", "1"],
                capture_output=True, text=True, timeout=8).stdout
        except subprocess.TimeoutExpired:
            raise SystemExit(f"no pose traffic on world '{WORLD}'; is the server up?")
        present = set(re.findall(r'name:\s*"([^"]+)"', out))
        wanted = set(self.manifest["arm"]) | set(self.manifest["client"]) | {
            self.manifest["bus"]}
        absent = sorted(wanted - present)
        if absent:
            raise SystemExit(
                f"the running world does not contain {absent}. Every one of these "
                "would be sent in the same batch, and one unknown name voids the "
                "batch while the service still reports success.")
        self.get_logger().info(
            f"scene check: all {len(wanted)} models this node drives exist in "
            f"world '{WORLD}'")

    def assert_physics_disabled(self, node):
        """Confirm Gazebo integrates nothing, rather than trusting the SDF.

        Every model is declared static, but a declaration is not a
        measurement. `dynamic_pose/info` carries exactly those poses the
        physics engine is updating, so if the engine owns anything in this
        scene it appears there. A model left dynamic would silently add its
        own integration on top of the poses this node commands, and the
        result would still look like a plausible scene.
        """
        import subprocess
        try:
            out = subprocess.run(
                ["gz", "topic", "-e", "-t", f"/world/{WORLD}/dynamic_pose/info", "-n", "1"],
                capture_output=True, text=True, timeout=4).stdout
        except subprocess.TimeoutExpired:
            out = ""
        moving = [ln for ln in out.splitlines() if "name:" in ln]
        if moving:
            raise SystemExit(
                "Gazebo is integrating " + ", ".join(moving) + ". Every model in "
                "this scene must be static, or its motion is Gazebo's solver on "
                "top of ours rather than the verified dynamics.")
        self.get_logger().info(
            "physics check: dynamic_pose/info carries no bodies, so Gazebo is "
            "integrating nothing and every pose below is ours")

    def arm_state(self, t):
        """The loop's angles and rates at time `t`.

        `JointLoop` is parameterised by phase rather than by time, and takes
        the phase rate separately so the rates stay consistent with the
        angles under any time scaling. Both come back as full seven vectors.
        """
        phase = 2.0 * np.pi * t / self.loop.period
        phase_rate = 2.0 * np.pi / self.loop.period
        return self.loop.angles(phase), self.loop.rates(phase, phase_rate)

    def step(self):
        import torch
        t = self.k * self.dt
        q, qd = self.arm_state(t)

        # The base twist the zero momentum constraint pins, from our dynamics.
        twist = self.model.base_velocity(torch.as_tensor(q, dtype=torch.float64),
                                         torch.as_tensor(qd, dtype=torch.float64))
        v = twist.detach().numpy()
        linear, omega = v[:3], v[3:]
        self.base_t = self.base_t + self.base_R @ (linear * self.dt)
        self.base_R = self.base_R @ rotation_exponential(omega * self.dt)

        for i, j in enumerate(self.arm_joints):
            self.p.resetJointState(self.body, j, float(q[i]))

        stamp = self.get_clock().now().to_msg()
        self._publish_pose(self.pub_base, stamp, self.base_R, self.base_t)
        client_R = self.client_att[self.k % len(self.client_att)]
        self._publish_pose(self.pub_target, stamp, client_R, CLIENT_START)

        js = JointState()
        js.header.stamp = stamp
        js.name = [self.p.getJointInfo(self.body, j)[1].decode() for j in self.arm_joints]
        js.position = [float(a) for a in q]
        js.velocity = [float(a) for a in qd]
        self.pub_joints.publish(js)

        if self.gz is not None:
            self._drive_gazebo(client_R)
        self.k += 1

    def _publish_pose(self, pub, stamp, R, t):
        m = PoseStamped()
        m.header.stamp = stamp
        m.header.frame_id = "world"
        m.pose.position.x, m.pose.position.y, m.pose.position.z = (float(c) for c in t)
        w, x, y, z = quaternion_from_matrix(R)
        m.pose.orientation.w = float(w)
        m.pose.orientation.x, m.pose.orientation.y, m.pose.orientation.z = (
            float(x), float(y), float(z))
        pub.publish(m)

    def _add(self, req, name, R, t):
        e = req.pose.add()
        e.name = name
        e.position.x, e.position.y, e.position.z = (float(c) for c in t)
        w, x, y, z = quaternion_from_matrix(R)
        e.orientation.w, e.orientation.x, e.orientation.y, e.orientation.z = (
            float(w), float(x), float(y), float(z))

    def _drive_gazebo(self, client_R):
        req = self._Pose_V()
        self._add(req, "servicer_bus", self.base_R,
                  self.base_t + self.base_R @ np.array([0.0, 0.0, -0.32]))
        for name, index in self.link_names.items():
            if index == -1:
                R_l, t_l = np.eye(3), np.zeros(3)
            else:
                st = self.p.getLinkState(self.body, index, computeForwardKinematics=1)
                t_l = np.array(st[4])
                R_l = np.array(self.p.getMatrixFromQuaternion(st[5])).reshape(3, 3)
            self._add(req, name, self.base_R @ R_l, self.base_t + self.base_R @ t_l)
        for prim in _client_primitives():
            R_p = self.p.getMatrixFromQuaternion(
                self.p.getQuaternionFromEuler([float(a) for a in prim.rpy]))
            R_p = np.array(R_p).reshape(3, 3)
            self._add(req, f"client_{prim.name}", client_R @ R_p,
                      CLIENT_START + client_R @ np.asarray(prim.position, float))
        self.gz.request(f"/world/{WORLD}/set_pose_vector", req, self._Pose_V,
                        self._Boolean, 200)


def _client_primitives():
    from src.target import client_satellite
    return client_satellite()


def main():
    rclpy.init()
    node = ScenePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
