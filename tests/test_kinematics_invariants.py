"""Invariants of the differentiable forward kinematics (phase 1).

Every case here exists because a specific frame convention in PyBullet is
not what it looks like, and getting it wrong produced a plausible number
rather than an error.
"""

import numpy as np
import pybullet as p
import pytest
import torch

from src.freeflight import ARM_JOINTS, END_EFFECTOR
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
                                       movable_joints=ARM_JOINTS, dtype=DTYPE)
        for angles in configurations:
            truth_position, truth_rotation = _pybullet_link_pose(
                panda_fixed, link, angles)
            pose = kinematics(torch.tensor(angles, dtype=DTYPE))
            assert np.abs(pose[:3, 3].numpy()
                          - truth_position).max() < TOLERANCE, f"link {link}"
            assert np.abs(pose[:3, :3].numpy()
                          - truth_rotation).max() < TOLERANCE, f"link {link}"


def test_link_two_is_exact_where_an_untransposed_joint_origin_would_not_be(
        panda_fixed, configurations):
    """getJointInfo field 15 is the rotation from joint frame to parent
    frame -- the inverse of what a forward chain composes.

    Without the transpose the chain stays exact through link 1 and then
    goes 632 mm wrong at link 2, because a rise of 0.316 in world z comes
    out as -0.316 in the rotated frame's y. Link 2 is therefore the link
    that decides whether the transpose is there.
    """
    kinematics = ForwardKinematics(panda_fixed, 2, movable_joints=ARM_JOINTS,
                                   dtype=DTYPE)
    for angles in configurations:
        truth_position, _ = _pybullet_link_pose(panda_fixed, 2, angles)
        pose = kinematics(torch.tensor(angles, dtype=DTYPE))
        assert np.linalg.norm(pose[:3, 3].numpy() - truth_position) < TOLERANCE


def test_joint_frames_end_pose_agrees_with_the_composed_chain(
        panda_fixed, configurations):
    """joint_frames walks the chain a second way, to pick up each joint's
    world axis. Its end pose must still be the pose the chain composes, or
    the geometric Jacobian is anchored to frames the FK does not agree with.
    """
    kinematics = ForwardKinematics(panda_fixed, END_EFFECTOR,
                                   movable_joints=ARM_JOINTS, dtype=DTYPE)
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
    kinematics = ForwardKinematics(panda_fixed, END_EFFECTOR,
                                   movable_joints=ARM_JOINTS, dtype=DTYPE)
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
    kinematics = ForwardKinematics(panda_fixed, END_EFFECTOR,
                                   movable_joints=ARM_JOINTS, dtype=DTYPE)
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
