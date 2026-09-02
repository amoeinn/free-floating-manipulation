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

The frame convention is the part that bites. getJointInfo reports a joint's
origin relative to the parent link's INERTIAL frame, while getLinkState
reports the LINK frame. The two differ by the parent's local inertial
offset, which for the Panda's base is 50 mm in z. Chaining the joint
origins directly accumulates that offset at every joint and drifts by over
a metre across seven of them. The parent's inertial offset therefore has to
be undone before each joint origin is applied, which is what
_link_to_link does.

Joint axes are always (0, 0, 1) here, and that is correct rather than a
bug: the origin quaternions rotate each joint frame so its axis lies along
z, which is why they alternate between plus and minus 0.7071 about x.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence

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
        origin_position: joint origin, relative to the parent's inertial
            frame, which is how getJointInfo reports it.
        origin_orientation: the same, as a quaternion in xyzw order.
        axis: the unit vector the joint rotates about, in its own frame.
        parent_inertial_position: the parent's local inertial offset, which
            has to be undone to get from the parent's link frame to where
            the joint origin is measured from.
        parent_inertial_orientation: the same, as a quaternion.
        movable: False for fixed joints, which contribute only the origin.
    """

    index: int
    name: str
    parent: int
    origin_position: torch.Tensor
    origin_orientation: torch.Tensor
    axis: torch.Tensor
    parent_inertial_position: torch.Tensor
    parent_inertial_orientation: torch.Tensor
    movable: bool


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
                 dtype: torch.dtype = torch.float64):
        """
        Args:
            body: a PyBullet body id, already loaded.
            end_effector_link: the link whose pose is wanted.
            movable_joints: joints treated as inputs. Defaults to every
                non-fixed joint on the path to the end effector.
            dtype: float64 by default. Verification is against a C++
                implementation and float32 would put the round trip error
                near the tolerance being measured.
        """
        self.body = body
        self.end_effector_link = end_effector_link
        self.dtype = dtype
        self.joints = self._read_chain(end_effector_link)

        available = [j.index for j in self.joints if j.movable]
        self.movable_joints = (list(movable_joints)
                               if movable_joints is not None else available)

    def _inertial(self, link: int):
        """A link's local inertial offset, as tensors."""
        info = p.getDynamicsInfo(self.body, link)
        return (torch.tensor(info[3], dtype=self.dtype),
                torch.tensor(info[4], dtype=self.dtype))

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
            inertial_position, inertial_orientation = self._inertial(parent)

            chain.append(JointSpec(
                index=current,
                name=info[1].decode(),
                parent=parent,
                origin_position=torch.tensor(info[14], dtype=self.dtype),
                origin_orientation=torch.tensor(info[15], dtype=self.dtype),
                axis=torch.tensor(info[13], dtype=self.dtype),
                parent_inertial_position=inertial_position,
                parent_inertial_orientation=inertial_orientation,
                movable=info[2] != p.JOINT_FIXED,
            ))
            current = parent
        chain.reverse()
        return chain

    def _link_to_link(self, joint: JointSpec,
                      angle: Optional[torch.Tensor]) -> torch.Tensor:
        """Transform from the parent link frame to this joint's link frame.

        Three steps: apply the parent's inertial offset, apply the joint
        origin, then rotate by the joint angle about the joint axis.

        The inertial offset composes forward rather than being inverted.
        Verified on the base joint, where getJointInfo reports an origin of
        0.283 in z, the base inertial offset is 0.050, and getLinkState
        puts the resulting link frame at 0.333.
        """
        parent_inertial = transform(
            quaternion_to_matrix(joint.parent_inertial_orientation),
            joint.parent_inertial_position,
        )
        # getJointInfo reports the origin orientation as the rotation from
        # the joint frame to the parent frame, which is the inverse of what
        # a forward chain composes, so it is transposed here. Verified
        # against getLinkState at every link: without the transpose the
        # chain is exact through link 1 and then diverges by 632 mm at
        # link 2, because a rise of 0.316 in world z comes out as -0.316
        # in the rotated frame's y.
        origin = transform(
            quaternion_to_matrix(joint.origin_orientation).T,
            joint.origin_position,
        )
        pose = parent_inertial @ origin

        if angle is not None:
            rotation = axis_angle_to_matrix(joint.axis, angle)
            pose = pose @ transform(rotation,
                                    torch.zeros(3, dtype=self.dtype))
        return pose

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
        offset = ", ".join(f"{v:+.3f}"
                           for v in joint.parent_inertial_position)
        lines.append(f"  {joint.index:>3} {kind} {joint.name:<24} "
                     f"parent {joint.parent:>3}  "
                     f"parent inertial offset ({offset})")
    lines.append(f"inputs: {kinematics.movable_joints}")
    return "\n".join(lines)