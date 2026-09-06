"""Differentiable forward kinematics for a serial chain.

PyBullet already computes forward kinematics, but it is a C++ call that
returns a number with no gradient attached. An optimizer needs to ask how
the end effector moves when a joint moves, and answering that by finite
differences costs one full kinematics evaluation per joint per step.
Writing the chain in torch gives the derivative for free through autograd.

This is what makes Cartesian objectives expressible. Project B could only
penalize acceleration in joint space, so the gripper bobbed while the
joint angles looked smooth. It matters more here: a free flying servicer's
arm motion reacts on the spacecraft base, and reasoning about that coupling
means working in Cartesian quantities inside the optimization.

Geometry is read from the URDF, not from PyBullet. That is a deliberate
correction rather than a preference.

getJointInfo reports a joint's origin relative to the parent link's
INERTIAL frame, and an earlier version of this file composed that offset
forward to recover link frames. It agreed with getLinkState to 5.7e-8 and
was wrong anyway: it is only correct when every inertial frame is axis
aligned with its link frame, which is true of the Panda URDF that ships
with PyBullet and of nothing else in particular. PyBullet rotates a link's
inertial frame as soon as that link's inertia tensor has off-diagonal
terms, because it stores principal moments and the rotation that
diagonalises them. Given such a model the old chain was exact through link
1 and 137 mm out at link 2.

The URDF states joint origins relative to the parent LINK frame, which is
what a forward chain wants, and says nothing about inertial frames at all.
Reading them from there removes the whole class of error. PyBullet is still
used for the tree structure, link index to name and joint type, none of
which is a frame convention.

Joint axes on the Panda are always (0, 0, 1) and that is correct rather
than a bug: the origin rotations carry each joint frame so its axis lies
along z.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence
from xml.etree import ElementTree

import numpy as np
import pybullet as p
import torch


@dataclass
class JointSpec:
    """One joint's fixed geometry, read from the model.

    Attributes:
        index: the joint's index in PyBullet's numbering.
        name: the joint's name, for diagnostics.
        parent: the parent link index, -1 for the base.
        origin_position: joint origin, relative to the parent LINK frame,
            as the URDF states it.
        origin_rotation: the same, as a 3x3 matrix built from the URDF's
            roll pitch yaw.
        axis: the unit vector the joint rotates about, in its own frame.
        movable: False for fixed joints, which contribute only the origin.
    """

    index: int
    name: str
    parent: int
    origin_position: torch.Tensor
    origin_rotation: torch.Tensor
    axis: torch.Tensor
    movable: bool


def rpy_to_matrix(rpy: Sequence[float], dtype: torch.dtype) -> torch.Tensor:
    """Rotation matrix from a URDF roll pitch yaw triple, composed Rz Ry Rx."""
    roll, pitch, yaw = (torch.tensor(float(v), dtype=dtype) for v in rpy)
    one = torch.ones((), dtype=dtype)
    zero = torch.zeros((), dtype=dtype)

    def about(angle, which):
        c, s = torch.cos(angle), torch.sin(angle)
        if which == "x":
            rows = [[one, zero, zero], [zero, c, -s], [zero, s, c]]
        elif which == "y":
            rows = [[c, zero, s], [zero, one, zero], [-s, zero, c]]
        else:
            rows = [[c, -s, zero], [s, c, zero], [zero, zero, one]]
        return torch.stack([torch.stack(row) for row in rows])

    return about(yaw, "z") @ about(pitch, "y") @ about(roll, "x")


def read_urdf_joints(path: str) -> dict:
    """{joint name: (origin xyz, origin rpy, axis)} straight from the URDF.

    The URDF is the authority on kinematics. PyBullet re-expresses these
    relative to the parent's inertial frame, which is a storage detail of
    Bullet and not a fact about the robot.
    """
    joints = {}
    for joint in ElementTree.parse(str(path)).getroot().findall("joint"):
        origin = joint.find("origin")
        xyz = [float(v) for v in ((origin.get("xyz") if origin is not None else None)
                                  or "0 0 0").split()]
        rpy = [float(v) for v in ((origin.get("rpy") if origin is not None else None)
                                  or "0 0 0").split()]
        axis = joint.find("axis")
        direction = [float(v) for v in ((axis.get("xyz") if axis is not None else None)
                                        or "1 0 0").split()]
        joints[joint.get("name")] = (xyz, rpy, direction)
    return joints


def quaternion_to_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    """Rotation matrix from a quaternion in xyzw order.

    PyBullet uses xyzw; most maths texts write wxyz. Mixing them produces
    a rotation that looks plausible and is wrong, so the order is stated
    here rather than assumed.
    """
    x, y, z, w = quaternion
    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w),
                     2 * (x * z + y * w)]),
        torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z),
                     2 * (y * z - x * w)]),
        torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w),
                     1 - 2 * (x * x + y * y)]),
    ])


def axis_angle_to_matrix(axis: torch.Tensor,
                         angle: torch.Tensor) -> torch.Tensor:
    """Rotation matrix for a rotation about an axis, by Rodrigues' formula."""
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    x, y, z = axis

    zero = torch.zeros((), dtype=axis.dtype)
    skew = torch.stack([
        torch.stack([zero, -z, y]),
        torch.stack([z, zero, -x]),
        torch.stack([-y, x, zero]),
    ])
    outer = torch.outer(axis, axis)
    identity = torch.eye(3, dtype=axis.dtype)

    return cos * identity + sin * skew + (1 - cos) * outer


def transform(rotation: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
    """Assemble a 4x4 homogeneous transform."""
    matrix = torch.zeros((4, 4), dtype=rotation.dtype)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation
    matrix[3, 3] = 1.0
    return matrix


def invert(matrix: torch.Tensor) -> torch.Tensor:
    """Inverse of a homogeneous transform, exploiting its structure.

    The rotation block is orthogonal, so its inverse is its transpose and
    no general matrix inversion is needed.
    """
    rotation = matrix[:3, :3]
    translation = matrix[:3, 3]
    return transform(rotation.T, -rotation.T @ translation)


class ForwardKinematics:
    """Differentiable forward kinematics for a chain read from a model."""

    def __init__(self, body: int, end_effector_link: int,
                 movable_joints: Optional[Sequence[int]] = None,
                 urdf: Optional[str] = None,
                 dtype: torch.dtype = torch.float64):
        """
        Args:
            body: a PyBullet body id, already loaded.
            end_effector_link: the link whose pose is wanted.
            movable_joints: joints treated as inputs. Defaults to every
                non-fixed joint on the path to the end effector.
            urdf: path to the URDF the body was loaded from, which is where
                the joint geometry is read from. Required, because
                PyBullet's own report of it depends on how it chose to
                store inertial frames.
            dtype: float64 by default. Verification is against a C++
                implementation and float32 would put the round trip error
                near the tolerance being measured.
        """
        if urdf is None:
            raise ValueError(
                "ForwardKinematics needs the URDF path: joint origins are read "
                "from the file, not from getJointInfo, which reports them "
                "relative to the parent's inertial frame")
        self.body = body
        self.end_effector_link = end_effector_link
        self.dtype = dtype
        self.urdf = str(urdf)
        self._urdf_joints = read_urdf_joints(self.urdf)
        self.joints = self._read_chain(end_effector_link)

        available = [j.index for j in self.joints if j.movable]
        self.movable_joints = (list(movable_joints)
                               if movable_joints is not None else available)

    def _read_chain(self, link: int) -> List[JointSpec]:
        """Walk from the end effector back to the base, then reverse.

        A URDF is a tree, so the chain is found by following parent links
        rather than assuming joint indices are contiguous. On the Panda the
        path to the grasp target passes through three fixed joints after
        the last revolute one.
        """
        chain = []
        current = link
        while current != -1:
            info = p.getJointInfo(self.body, current)
            parent = info[16]
            name = info[1].decode()
            if name not in self._urdf_joints:
                raise KeyError(f"joint {name} is not in {self.urdf}")
            xyz, rpy, axis = self._urdf_joints[name]

            chain.append(JointSpec(
                index=current,
                name=name,
                parent=parent,
                origin_position=torch.tensor(xyz, dtype=self.dtype),
                origin_rotation=rpy_to_matrix(rpy, self.dtype),
                axis=torch.tensor(axis, dtype=self.dtype),
                movable=info[2] != p.JOINT_FIXED,
            ))
            current = parent
        chain.reverse()
        return chain

    def _parent_to_joint(self, joint: JointSpec) -> torch.Tensor:
        """Parent link frame to this joint's frame, before the joint moves.

        Straight from the URDF, which states exactly this transform. No
        inertial frame appears anywhere in the chain, which is the point:
        the previous version composed the parent's inertial offset forward
        and was correct only while every such frame was axis aligned.
        """
        return transform(joint.origin_rotation, joint.origin_position)

    def _link_to_link(self, joint: JointSpec,
                      angle: Optional[torch.Tensor]) -> torch.Tensor:
        """Transform from the parent link frame to this joint's link frame:
        the fixed parent-to-joint transform, then the joint rotation."""
        pose = self._parent_to_joint(joint)
        if angle is not None:
            rotation = axis_angle_to_matrix(joint.axis, angle)
            pose = pose @ transform(rotation,
                                    torch.zeros(3, dtype=self.dtype))
        return pose

    def joint_frames(self, configuration: torch.Tensor):
        """World point and axis of each input joint, plus the end pose.

        Read in the joint frame before that joint's own rotation, which is
        what a geometric Jacobian column needs: for a revolute joint,
        column i is [axis_i x (target - point_i); axis_i]. Returns
        ({joint index: (point (3,), axis (3,))}, end pose (4, 4)).
        """
        angles = {joint: configuration[i]
                  for i, joint in enumerate(self.movable_joints)}
        pose = torch.eye(4, dtype=self.dtype)
        frames = {}
        for joint in self.joints:
            pose = pose @ self._parent_to_joint(joint)
            angle = angles.get(joint.index) if joint.movable else None
            if joint.index in angles:
                frames[joint.index] = (pose[:3, 3].clone(),
                                       pose[:3, :3] @ joint.axis)
            if angle is not None:
                pose = pose @ transform(
                    axis_angle_to_matrix(joint.axis, angle),
                    torch.zeros(3, dtype=self.dtype))
        return frames, pose

    def __call__(self, configuration: torch.Tensor) -> torch.Tensor:
        """Pose of the end effector for a configuration.

        Args:
            configuration: shape (n,) or (batch, n) joint angles.

        Returns:
            A 4x4 transform, or (batch, 4, 4) for a batched input.
        """
        if configuration.dim() == 1:
            return self._single(configuration)
        return torch.stack([self._single(row) for row in configuration])

    def _single(self, configuration: torch.Tensor) -> torch.Tensor:
        angles = {joint: configuration[i]
                  for i, joint in enumerate(self.movable_joints)}

        pose = torch.eye(4, dtype=self.dtype)
        for joint in self.joints:
            angle = angles.get(joint.index) if joint.movable else None
            pose = pose @ self._link_to_link(joint, angle)
        return pose

    def position(self, configuration: torch.Tensor) -> torch.Tensor:
        """End effector position only, shape (3,) or (batch, 3)."""
        pose = self(configuration)
        if pose.dim() == 2:
            return pose[:3, 3]
        return pose[:, :3, 3]

    def jacobian(self, configuration: torch.Tensor) -> torch.Tensor:
        """Position Jacobian: how the end effector moves per joint.

        Shape (3, n). Column i is the end effector velocity produced by a
        unit rate on joint i. This is what autograd was for.
        """
        configuration = configuration.detach().clone().requires_grad_(True)
        return torch.autograd.functional.jacobian(
            lambda q: self.position(q), configuration)


def describe_chain(kinematics: ForwardKinematics) -> str:
    """Human readable summary of the chain that was read."""
    lines = [f"{len(kinematics.joints)} joints to link "
             f"{kinematics.end_effector_link}:"]
    for joint in kinematics.joints:
        kind = "movable" if joint.movable else "fixed  "
        offset = ", ".join(f"{v:+.3f}" for v in joint.origin_position)
        lines.append(f"  {joint.index:>3} {kind} {joint.name:<24} "
                     f"parent {joint.parent:>3}  "
                     f"urdf origin ({offset})")
    lines.append(f"inputs: {kinematics.movable_joints}")
    return "\n".join(lines)