"""Invariants of the differentiable forward kinematics (phase 1).

Every case here exists because a specific frame convention in PyBullet is
not what it looks like, and getting it wrong produced a plausible number
rather than an error.
"""

import numpy as np
import pybullet as p
import pytest
import torch

from src.freeflight import ARM_JOINTS, end_effector_index, panda_spec

URDF_PATH = panda_spec()["urdf_path"]
from src.kinematics import ForwardKinematics

DTYPE = torch.float64

# getLinkState comes back through PyBullet's single-precision state, so a
# micrometre is the floor on any comparison against it, not a slack
# tolerance. Phase 1 measured 5.7e-8 m and 4.2e-7 on rotation elements.
TOLERANCE = 1e-6


def _pybullet_link_pose(body, link, angles):
    for joint, angle in zip(ARM_JOINTS, angles):
        p.resetJointState(body, joint, float(angle))
    state = p.getLinkState(body, link, computeForwardKinematics=True)
    rotation = np.asarray(p.getMatrixFromQuaternion(state[5])).reshape(3, 3)
    return np.asarray(state[4]), rotation


def test_analytic_chain_reproduces_every_link_pose(panda_fixed, configurations):
    """The chain in torch must agree with getLinkState everywhere, not just
    at the end effector.

    An error in the origin/inertial composition shows at one link and then
    propagates, so checking only the tip hides which link introduced it.
    """
    for link in range(p.getNumJoints(panda_fixed)):
        kinematics = ForwardKinematics(panda_fixed, link,
                                       movable_joints=ARM_JOINTS, urdf=URDF_PATH,
                                       dtype=DTYPE)
        for angles in configurations:
            truth_position, truth_rotation = _pybullet_link_pose(
                panda_fixed, link, angles)
            pose = kinematics(torch.tensor(angles, dtype=DTYPE))
            assert np.abs(pose[:3, 3].numpy()
                          - truth_position).max() < TOLERANCE, f"link {link}"
            assert np.abs(pose[:3, :3].numpy()
                          - truth_rotation).max() < TOLERANCE, f"link {link}"


def _rotated_inertia(principal, rpy):
    """A valid inertia tensor that is not diagonal.

    Built by rotating a diagonal one rather than by inventing six numbers:
    PyBullet rejects a tensor that is not positive definite or that breaks
    the triangle inequality on its principal moments, silently zeroing it,
    which would make the test below vacuous.
    """
    roll, pitch, yaw = rpy
    def about(angle, axis):
        c, s = np.cos(angle), np.sin(angle)
        if axis == "x":
            return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        if axis == "y":
            return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    rotation = about(yaw, "z") @ about(pitch, "y") @ about(roll, "x")
    return rotation @ np.diag(principal) @ rotation.T


def _rotated_inertia_urdf() -> str:
    """A three link chain whose every link has off-diagonal inertia."""
    links = [
        ("base", 2.0, (0.01, -0.02, 0.03), (0.20, 0.25, 0.30), (0.3, -0.4, 0.5)),
        ("upper", 1.5, (-0.04, 0.05, 0.06), (0.16, 0.18, 0.22), (-0.6, 0.2, 0.9)),
        ("lower", 1.0, (0.02, 0.03, -0.05), (0.11, 0.12, 0.15), (0.7, 0.5, -0.3)),
    ]
    body = ['<?xml version="1.0"?>', '<robot name="rotated">']
    for name, mass, com, principal, rpy in links:
        i = _rotated_inertia(np.array(principal), rpy)
        body += [
            f'  <link name="{name}">', '    <inertial>',
            f'      <origin xyz="{com[0]} {com[1]} {com[2]}" rpy="0 0 0"/>',
            f'      <mass value="{mass}"/>',
            f'      <inertia ixx="{float(i[0,0]):.17g}" ixy="{float(i[0,1]):.17g}"'
            f' ixz="{float(i[0,2]):.17g}" iyy="{float(i[1,1]):.17g}"'
            f' iyz="{float(i[1,2]):.17g}" izz="{float(i[2,2]):.17g}"/>',
            '    </inertial>', '  </link>']
    body += [
        '  <joint name="shoulder" type="revolute">',
        '    <parent link="base"/><child link="upper"/>',
        '    <origin xyz="0 0 0.4" rpy="0.3 -0.2 0.5"/>',
        '    <axis xyz="0 0 1"/>',
        '    <limit lower="-3" upper="3" effort="10" velocity="1"/>',
        '  </joint>',
        '  <joint name="elbow" type="revolute">',
        '    <parent link="upper"/><child link="lower"/>',
        '    <origin xyz="0.1 -0.05 0.3" rpy="-0.4 0.6 0.1"/>',
        '    <axis xyz="0 1 0"/>',
        '    <limit lower="-3" upper="3" effort="10" velocity="1"/>',
        '  </joint>', '</robot>']
    return "\n".join(body)


def test_chain_holds_when_the_inertial_frames_are_rotated(physics, tmp_path):
    """The chain must not depend on how PyBullet stores inertial frames.

    An earlier version composed the parent's inertial offset forward to
    recover link frames, reading it out of getJointInfo. That agreed with
    getLinkState to 5.7e-8 on the Panda and was wrong anyway: it is correct
    only while every inertial frame is axis aligned with its link frame,
    which is true of the Panda URDF PyBullet ships and is not a property of
    robots in general. PyBullet rotates a link's inertial frame as soon as
    that link's inertia tensor has off-diagonal terms, because it stores
    principal moments plus the rotation that diagonalises them, and the old
    chain then went 137 mm wrong at the second joint.

    The gap was coverage, not code. This model has off-diagonal terms on
    every link precisely so the rotation is there to be got wrong.
    """
    path = tmp_path / "rotated.urdf"
    path.write_text(_rotated_inertia_urdf())
    # At the origin, because the chain is built from the base frame outward
    # and getLinkState reports world coordinates. This model declares no
    # collision geometry, so sharing the origin with the fixtures is safe.
    body = p.loadURDF(str(path), useFixedBase=True, basePosition=[0, 0, 0],
                      flags=p.URDF_USE_INERTIA_FROM_FILE)
    joints = [0, 1]

    # Vacuous unless PyBullet really did rotate them, which it will not do if
    # it rejected the tensors as unphysical and quietly zeroed them.
    rotations = [np.abs(np.asarray(p.getDynamicsInfo(body, link)[4])
                        - np.array([0.0, 0.0, 0.0, 1.0])).max()
                 for link in (-1, 0, 1)]
    assert max(rotations) > 0.05, (
        f"no inertial frame is rotated ({rotations}), so this would pass "
        "against the very bug it exists to catch")

    rng = np.random.default_rng(3)
    for link in joints:
        kinematics = ForwardKinematics(body, link, movable_joints=joints,
                                       urdf=str(path), dtype=DTYPE)
        for _ in range(8):
            angles = rng.uniform(-2.5, 2.5, size=len(joints))
            for joint, angle in zip(joints, angles):
                p.resetJointState(body, joint, float(angle))
            truth = np.asarray(p.getLinkState(
                body, link, computeForwardKinematics=True)[4])
            mine = kinematics(torch.tensor(angles, dtype=DTYPE))[:3, 3].numpy()
            assert np.abs(mine - truth).max() < TOLERANCE, f"link {link}"

    p.removeBody(body)


def test_joint_frames_end_pose_agrees_with_the_composed_chain(
        panda_fixed, configurations):
    """joint_frames walks the chain a second way, to pick up each joint's
    world axis. Its end pose must still be the pose the chain composes, or
    the geometric Jacobian is anchored to frames the FK does not agree with.
    """
    kinematics = ForwardKinematics(panda_fixed, end_effector_index(panda_fixed),
                                   movable_joints=ARM_JOINTS, urdf=URDF_PATH,
                                       dtype=DTYPE)
    for angles in configurations:
        q = torch.tensor(angles, dtype=DTYPE)
        _, end_pose = kinematics.joint_frames(q)
        assert torch.allclose(end_pose, kinematics(q), atol=1e-12)


def test_joint_axes_are_unit_and_reach_the_world_through_the_origins(
        panda_fixed, configurations):
    """Every Panda joint reads its axis as (0,0,1) and that is correct, not
    a bug: the origin quaternions rotate each joint frame so its axis lies
    along z. The world axes must still come out unit length and distinct.
    """
    kinematics = ForwardKinematics(panda_fixed, end_effector_index(panda_fixed),
                                   movable_joints=ARM_JOINTS, urdf=URDF_PATH,
                                       dtype=DTYPE)
    frames, _ = kinematics.joint_frames(
        torch.tensor(configurations[0], dtype=DTYPE))
    assert sorted(frames) == list(ARM_JOINTS)
    axes = np.array([frames[j][1].numpy() for j in ARM_JOINTS])
    assert np.allclose(np.linalg.norm(axes, axis=1), 1.0, atol=1e-12)
    assert np.linalg.matrix_rank(axes, tol=1e-9) == 3


def test_position_jacobian_matches_a_finite_difference(panda_fixed,
                                                       configurations):
    """Autograd's derivative of the chain must equal a finite difference of
    the same chain. Catches a chain that is right at the sampled poses and
    differentiates wrongly."""
    kinematics = ForwardKinematics(panda_fixed, end_effector_index(panda_fixed),
                                   movable_joints=ARM_JOINTS, urdf=URDF_PATH,
                                       dtype=DTYPE)
    step = 1e-6
    for angles in configurations[:2]:
        analytic = kinematics.jacobian(torch.tensor(angles, dtype=DTYPE)).numpy()
        numeric = np.zeros_like(analytic)
        for column in range(len(ARM_JOINTS)):
            ahead, behind = angles.copy(), angles.copy()
            ahead[column] += step
            behind[column] -= step
            numeric[:, column] = (
                kinematics.position(torch.tensor(ahead, dtype=DTYPE)).numpy()
                - kinematics.position(torch.tensor(behind, dtype=DTYPE)).numpy()
            ) / (2 * step)
        assert np.abs(analytic - numeric).max() < 1e-8
