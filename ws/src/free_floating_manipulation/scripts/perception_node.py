#!/usr/bin/env python3
"""Perception for the mission executive: acquire a pose, verify it, then track.

This is the only thing in the mission that knows where the client is, and it
knows it from imagery alone. It never subscribes to /truth. That separation is
the whole reason the audit means anything: if the executive could read the
answer, a mission that returns SUCCESS would prove nothing.

The three capabilities are the ones phase 4 measured, wrapped rather than
rewritten. Acquisition draws restarts uniformly over SO(3) and keeps the
lowest final loss, at 22 percent per attempt and 87 percent at eight restarts.
The flip test is one extra render and is only meaningful near the body frame
sun angle the splat model was fitted at. Tracking is silhouette registration,
which holds indefinitely and cannot acquire.

The order matters and is not interchangeable: photometric acquires because the
baked lighting breaks the 180 degree degeneracy, and silhouette holds because
it ignores the lighting that decays. Neither alone is a tracker.
"""

import sys
import threading
import time
from pathlib import Path


def _find_repo():
    for parent in Path(__file__).resolve().parents:
        if (parent / "src" / "dynamics.py").exists() and (parent / ".venv").exists():
            return parent
    raise SystemExit(f"cannot locate the repository root from {Path(__file__)}")


REPO = _find_repo()
_VENV = next(iter(sorted((REPO / ".venv" / "lib").glob("python3.*/site-packages"))), None)
if _VENV is not None and str(_VENV) not in sys.path:
    sys.path.insert(0, str(_VENV))
sys.path.insert(0, str(REPO))

import numpy as np

EXPECTED_NUMPY = "2.5.2"
if np.__version__ != EXPECTED_NUMPY:
    raise SystemExit(f"numpy {np.__version__} loaded, expected {EXPECTED_NUMPY}; "
                     "the venv site-packages must sit ahead of the ROS ones on "
                     "sys.path; see CONTRIBUTING.md")

import rclpy
import torch
from geometry_msgs.msg import PoseStamped
from rclpy.action import ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node as RosNode
from sensor_msgs.msg import Image

from free_floating_manipulation.action import AcquireTarget
from free_floating_manipulation.srv import VerifyFlip

from src.acquisition import (acquire_pose, body_frame_sun_angle_deg, flip_matrix,
                             flip_test, track_step)
from src.fitting import load_model
from src.splatting import TorchCamera
from src.target import client_scene
from src.tracking import matrix_to_axis_angle

DT = torch.float32
CAMERA_RES = 128
CAMERA_FOV_DEG = 40.0
# Where the client sits relative to the servicer. Not arbitrary: the splat
# model carries the lighting it was fitted under, so the servicer has to view
# an aspect the sun actually lights. Measured over six placements, this one
# lights 23.5 percent of the frame against the model's 40.3 and gives the
# smallest image difference at 0.0117; directly overhead lights 4.4 percent
# and acquisition has almost nothing to lock onto.
CLIENT_NOMINAL = np.array([4.0, -3.0, 2.0])


def matrix_to_quaternion(R):
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


class Perception(RosNode):
    def __init__(self):
        super().__init__("perception")
        group = ReentrantCallbackGroup()
        self.model = load_model(REPO / "data" / "model.pt", dtype=DT)
        self.sun_world = client_scene().light_direction
        self.lock = threading.Lock()
        self.image = None
        self.mask = None
        self.base_R = np.eye(3)
        self.base_t = np.zeros(3)
        self.estimate = None          # rotation matrix, the current belief
        self.tracking = False
        self.pending_flip = False
        self.busy = False
        self.last_skip = None
        self.published = 0

        self.create_subscription(Image, "/servicer/camera/image", self.on_image, 2,
                                 callback_group=group)
        self.create_subscription(Image, "/servicer/camera/mask", self.on_mask, 2,
                                 callback_group=group)
        self.create_subscription(PoseStamped, "/servicer/base_pose", self.on_base, 10,
                                 callback_group=group)
        self.pub_estimate = self.create_publisher(PoseStamped, "/target/pose_estimate", 10)

        self.acquire_server = ActionServer(
            self, AcquireTarget, "acquire_target", self.on_acquire,
            callback_group=group)
        self.create_service(VerifyFlip, "verify_flip", self.on_verify,
                            callback_group=group)
        self.create_timer(0.5, self.on_track, callback_group=group)
        self.get_logger().info("perception up: acquire_target, verify_flip, "
                               "and /target/pose_estimate once tracking")

    def on_image(self, msg):
        a = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3)
        with self.lock:
            self.image = torch.as_tensor(a.astype(np.float32) / 255.0)

    def on_mask(self, msg):
        a = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width)
        with self.lock:
            self.mask = torch.as_tensor((a > 127).astype(np.float32))

    def on_base(self, msg):
        q = msg.pose.orientation
        w, x, y, z = q.w, q.x, q.y, q.z
        self.base_R = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
        self.base_t = np.array([msg.pose.position.x, msg.pose.position.y,
                                msg.pose.position.z])

    def camera(self):
        """The scene's camera, expressed in the client's own frame.

        The splat model was fitted in the client's frame and sits at the
        origin there, while the scene renders the client at its world
        position. Registering the model against the scene's image therefore
        needs the camera moved into the model's frame, not the model moved
        into the world: shifting the eye by the client's nominal position
        gives exactly the relative geometry the scene rendered, which is all
        registration depends on.

        The nominal position is what approach navigation supplies. It is a
        position, not an attitude, and attitude is the whole thing perception
        is here to work out.
        """
        from src.raytrace import Camera, look_at
        eye = self.base_t + self.base_R @ np.array([0.0, 0.0, 0.25]) - CLIENT_NOMINAL
        # Same line of sight as the scene camera and the same up vector, or
        # perception would be registering against a rolled image.
        R_cw, t_cw = look_at(eye, np.zeros(3), np.array([0.0, 0.0, 1.0]))
        return TorchCamera.from_raytrace(
            Camera.with_fov(CAMERA_RES, CAMERA_RES, CAMERA_FOV_DEG, R_cw, t_cw),
            dtype=DT)

    def on_acquire(self, goal_handle):
        request = goal_handle.request
        result = AcquireTarget.Result()
        # Stop tracking while re-acquiring. Besides being the obvious
        # behaviour, it closes a race that silently undid the mutation test: a
        # tracking step reads the belief, spends 3.48 s registering, then
        # writes its result back, so anything that changed the belief in that
        # window was overwritten by a pose derived from the old one. The
        # injected flip was being erased that way and the audit then saw a
        # correct pose and called the mission right.
        self.tracking = False
        # Wait for the first frame rather than abort on it. The scene runs at
        # a small fraction of real time, so a mission that starts promptly can
        # easily get here before the first image, and "no imagery yet" is a
        # startup race rather than a failed acquisition.
        image = None
        for _ in range(120):
            with self.lock:
                image = None if self.image is None else self.image.clone()
            if image is not None:
                break
            time.sleep(0.5)
        if image is None:
            goal_handle.abort()
            result.acquired = False
            result.reason = "no imagery after 60 s of waiting"
            return result

        def progress(done, best):
            fb = AcquireTarget.Feedback()
            fb.restarts_done = int(done)
            fb.best_loss = float(best)
            goal_handle.publish_feedback(fb)

        R, loss, used = acquire_pose(
            self.model, self.camera(), image,
            restarts=max(int(request.restarts), 1),
            iterations=max(int(request.iterations), 1),
            seed=31, on_restart=progress)

        # Test hook, never used in flight, and it fires after verification
        # rather than before. Flipping the acquisition itself would simply be
        # caught and repaired by the flip test, which is that test working.
        # The hazard phase 4 actually documented is a belief that goes wrong
        # where nothing can catch it: the flip test is valid only at handover,
        # silhouette tracking holds a wrong pose exactly as stably as a right
        # one, and nothing downstream re-examines it. So the injection is
        # armed here and applied once the verdict has been given.
        self.pending_flip = bool(request.inject_flip)
        if self.pending_flip:
            self.get_logger().warn(
                "inject_flip armed: the belief will be flipped after the flip "
                "test has passed, which is where nothing can catch it")

        with self.lock:
            self.estimate = R
        result.acquired = True
        result.final_loss = float(loss)
        result.restarts_used = int(used)
        result.pose = self.pose_message(R)
        result.reason = "acquired"
        goal_handle.succeed()
        self.get_logger().info(f"acquired after {used} restarts, loss {loss:.4e}")
        return result

    def on_verify(self, request, response):
        with self.lock:
            image = None if self.image is None else self.image.clone()
            R = None if self.estimate is None else self.estimate.copy()
        if image is None or R is None:
            response.valid = False
            response.reason = "nothing acquired yet"
            return response

        angle = body_frame_sun_angle_deg(R, self.sun_world)
        response.sun_angle_deg = float(angle)
        limit = request.max_sun_angle_deg if request.max_sun_angle_deg > 0 else 15.0
        if angle > limit:
            response.valid = False
            response.reason = (
                f"body frame sun at {angle:.1f} deg exceeds the {limit:.1f} deg "
                "window; the flip test is ambiguous by 28 deg and inverts past "
                "40, so it would answer rather than decide")
            self.get_logger().warn(response.reason)
            return response

        repaired, ratio = flip_test(self.model, self.camera(), image, R)
        if repaired:
            R = R @ flip_matrix()
            with self.lock:
                self.estimate = R
        response.valid = True
        response.repaired = bool(repaired)
        response.ratio = float(ratio)
        response.reason = "repaired to the flip" if repaired else "pose held"
        if self.pending_flip:
            with self.lock:
                self.estimate = self.estimate @ flip_matrix()
            self.pending_flip = False
            self.get_logger().warn(
                "inject_flip applied after the verdict; from here nothing in "
                "the mission can detect it, which is the point of the test")
        self.tracking = True
        self.get_logger().info(
            f"flip test at {angle:.1f} deg sun: ratio {ratio:.2f}x, "
            f"{response.reason}; tracking now armed")
        return response

    def on_track(self):
        """One tracking step, and say plainly when it cannot take one.

        A tracker that silently does nothing is the failure this whole phase
        is about, so the reasons it declines are logged once each rather than
        left as an absence of output.
        """
        if not self.tracking:
            return
        if self.busy:
            return
        with self.lock:
            mask = None if self.mask is None else self.mask.clone()
            R = None if self.estimate is None else self.estimate.copy()
        if mask is None or R is None:
            why = "no mask yet" if mask is None else "no pose belief yet"
            if why != self.last_skip:
                self.last_skip = why
                self.get_logger().warn(f"tracking idle: {why}")
            return
        self.busy = True
        try:
            R = track_step(self.model, self.camera(), mask, R)
            with self.lock:
                self.estimate = R
            self.pub_estimate.publish(self.pose_message(R))
            self.published += 1
            if self.published == 1:
                self.get_logger().info("tracking is publishing /target/pose_estimate")
        except Exception as exc:                      # noqa: BLE001
            self.get_logger().error(f"tracking step failed: {exc!r}")
        finally:
            self.busy = False

    def pose_message(self, R):
        m = PoseStamped()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = "world"
        m.pose.position.x, m.pose.position.y, m.pose.position.z = (
            float(c) for c in CLIENT_NOMINAL)
        w, x, y, z = matrix_to_quaternion(R)
        m.pose.orientation.w = float(w)
        m.pose.orientation.x, m.pose.orientation.y, m.pose.orientation.z = (
            float(x), float(y), float(z))
        return m


def main():
    rclpy.init()
    node = Perception()
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
