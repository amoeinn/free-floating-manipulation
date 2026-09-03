"""Invariants of the floating-base mass matrix (phase 2a)."""

import numpy as np
import pybullet as p
import pytest
import torch

from src.dynamics import FloatingBaseModel
from src.freeflight import ARM_JOINTS, FINGER_JOINTS

DTYPE = torch.float64
ARM = len(ARM_JOINTS)

# PyBullet orders a free base's six DoF [angular, linear]; this project
# uses [linear, angular]. This permutation maps one onto the other.
SWAP = np.zeros((6, 6))
SWAP[:3, 3:] = np.eye(3)
SWAP[3:, :3] = np.eye(3)


def _full(angles):
    """Arm angles padded with the two locked finger joints."""
    return list(angles) + [0.0] * len(FINGER_JOINTS)


def test_base_link_carries_mass_only_when_the_base_is_free(panda_fixed,
                                                           panda_free):
    """PyBullet zeroes the base link's mass and inertia under
    useFixedBase=True and fills in the URDF values under False.

    The Panda's panda_link0 is really 2.9 kg. A model built from a
    fixed-base load silently loses it, and every H_b that follows is wrong
    by that mass with no error raised anywhere.
    """
    assert p.getDynamicsInfo(panda_fixed, -1)[0] == 0.0
    assert p.getDynamicsInfo(panda_free, -1)[0] == pytest.approx(2.9)
    assert max(p.getDynamicsInfo(panda_free, -1)[2]) > 0.0


def test_com_jacobian_is_taken_at_the_inertial_frame_not_the_link_frame(
        panda_fixed, model, configurations):
    """The CoM Jacobian is the Jacobian of the link's inertial frame.

    calculateJacobian's localPosition is in the link (URDF) frame, so the
    reference call needs localPosition = getDynamicsInfo field 3. Passing
    [0,0,0] gives the link-frame origin instead, which on the wrist links
    is up to 110 mm of lever arm away. The second assertion is what makes
    the first one mean something.
    """
    zeros = [0.0] * (ARM + len(FINGER_JOINTS))
    angles = configurations[0]
    link = next(entry for entry in model.links if entry.index == 4)
    info = p.getDynamicsInfo(panda_fixed, link.index)

    mine, _ = model.link_jacobian(link, torch.tensor(angles, dtype=DTYPE))

    at_inertial, _ = p.calculateJacobian(panda_fixed, link.index,
                                         list(info[3]), _full(angles),
                                         zeros, zeros)
    at_link_origin, _ = p.calculateJacobian(panda_fixed, link.index,
                                            [0.0, 0.0, 0.0], _full(angles),
                                            zeros, zeros)

    assert np.abs(mine.numpy()
                  - np.asarray(at_inertial)[:, :ARM]).max() < 1e-9
    assert np.abs(mine.numpy()
                  - np.asarray(at_link_origin)[:, :ARM]).max() > 1e-3


def test_proximal_link_has_zero_columns_for_joints_that_do_not_move_it(
        model, configurations):
    """Turning joint 5 cannot move link 1.

    The geometric Jacobian formula, applied blindly to every arm joint,
    returns a non-zero axis x (com - point) for joints distal to the link
    and quietly corrupts the mass matrix. Columns past a link's own
    ancestors must be exactly zero.
    """
    q = torch.tensor(configurations[0], dtype=DTYPE)
    for link in model.links:
        if link.kinematics is None:
            continue
        linear, angular = model.link_jacobian(link, q)
        ancestors = link.arm_ancestors
        for column, joint in enumerate(ARM_JOINTS):
            if joint in ancestors:
                continue
            assert np.all(linear[:, column].numpy() == 0.0), (
                f"link {link.index} moves with distal joint {joint}")
            assert np.all(angular[:, column].numpy() == 0.0), (
                f"link {link.index} rotates with distal joint {joint}")

    # The check above is vacuous unless some link really is proximal.
    shallow = next(entry for entry in model.links if entry.index == 0)
    assert shallow.arm_ancestors == (0,)


def test_mass_matrix_is_symmetric(model, configurations):
    """M is a quadratic form in the generalized velocity, so it is
    symmetric by construction. An asymmetry means a block was assembled
    with mismatched Jacobians."""
    for angles in configurations:
        matrix = model.mass_matrix(torch.tensor(angles, dtype=DTYPE)).numpy()
        assert np.abs(matrix - matrix.T).max() < 1e-12


def test_translational_block_is_total_mass_times_identity(model,
                                                          configurations):
    """M[:3,:3] is the whole system's mass resisting a base translation:
    total_mass . I, independent of pose and of the reference point.

    It is also what pins the base DoF order down: in [angular, linear]
    order this block would be the rotational inertia instead.
    """
    total = sum(link.mass for link in model.links)
    assert total == pytest.approx(17.96)
    for angles in configurations:
        matrix = model.mass_matrix(torch.tensor(angles, dtype=DTYPE)).numpy()
        assert np.abs(matrix[:3, :3] - total * np.eye(3)).max() < 1e-12


def test_base_inertia_and_coupling_match_free_base_mass_matrix(
        panda_free, model, configurations):
    """H_b and H_bm must equal PyBullet's own free-base mass matrix.

    This is the check that does not depend on our derivation at all. It
    only holds with the base twist written about the base link's inertial
    frame and permuted out of PyBullet's [angular, linear] order; both were
    found by sweeping candidates, and the wrong choices are off by O(1) to
    O(10), not by a rounding error.
    """
    for angles in configurations:
        base_inertia, coupling = model.coupling(
            torch.tensor(angles, dtype=DTYPE))
        truth = np.asarray(p.calculateMassMatrix(panda_free, _full(angles)))

        assert np.abs(SWAP @ base_inertia.numpy() @ SWAP
                      - truth[:6, :6]).max() < 1e-10
        assert np.abs(SWAP @ coupling.numpy()
                      - truth[:6, 6:6 + ARM]).max() < 1e-10


def test_the_wrong_base_reference_point_does_not_also_match(panda_free,
                                                            configurations):
    """base_com is not one of three equally good choices.

    If the other two were nearly as good the agreement above would be weak
    evidence. They are not: they miss by order 1 to 10.
    """
    angles = configurations[0]
    truth = np.asarray(p.calculateMassMatrix(panda_free, _full(angles)))
    for reference in ("base_link_origin", "system_com"):
        other = FloatingBaseModel(panda_free, ARM_JOINTS,
                                  base_reference=reference, dtype=DTYPE)
        base_inertia, _ = other.coupling(torch.tensor(angles, dtype=DTYPE))
        assert np.abs(SWAP @ base_inertia.numpy() @ SWAP
                      - truth[:6, :6]).max() > 0.1


def test_arm_block_is_independent_of_the_base_reference_point(
        panda_free, configurations):
    """H_m involves no base lever arm, so moving the reference point must
    not touch it. Isolates a bug in the link Jacobians from a bug in the
    base-frame bookkeeping."""
    angles = torch.tensor(configurations[0], dtype=DTYPE)
    blocks = [FloatingBaseModel(panda_free, ARM_JOINTS,
                                base_reference=reference,
                                dtype=DTYPE).mass_matrix(angles)[6:, 6:].numpy()
              for reference in ("base_com", "base_link_origin", "system_com")]
    for block in blocks[1:]:
        assert np.abs(block - blocks[0]).max() < 1e-10


def test_locked_fingers_keep_their_mass_at_their_own_centre(model,
                                                            configurations):
    """Locking the fingers drops their joint columns, not their mass.

    fingers="folded" lumps the 0.2 kg onto the hand instead. It is a stated
    approximation, so it must differ from the exact sum -- but only at the
    1e-2 level, against H_b entries of order 18.
    """
    q = torch.tensor(configurations[0], dtype=DTYPE)
    exact, exact_coupling = model.coupling(q, fingers="exact")
    folded, folded_coupling = model.coupling(q, fingers="folded")
    difference = np.abs(exact.numpy() - folded.numpy()).max()
    assert 1e-4 < difference < 1e-1
    assert np.abs(exact_coupling.numpy()
                  - folded_coupling.numpy()).max() < 1e-1


def test_mass_matrix_is_differentiable_in_the_joint_angles(model,
                                                           configurations):
    """The trajectory optimiser differentiates through M. Autograd must
    survive the whole path: geometric Jacobians, the inertia rotation, and
    the linear solve in base_velocity."""
    q = torch.tensor(configurations[0], dtype=DTYPE, requires_grad=True)
    rates = torch.ones(ARM, dtype=DTYPE)

    def speed(configuration):
        return model.base_velocity(configuration, rates).pow(2).sum()

    speed(q).backward()
    analytic = q.grad.clone()

    step = 1e-6
    numeric = torch.zeros_like(analytic)
    for index in range(ARM):
        ahead, behind = q.detach().clone(), q.detach().clone()
        ahead[index] += step
        behind[index] -= step
        numeric[index] = (speed(ahead) - speed(behind)) / (2 * step)

    assert (analytic - numeric).abs().max().item() < 1e-6
