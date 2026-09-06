"""Invariants of the generalized Jacobian J_g = J_m - J_b H_b^-1 H_bm
(phase 2b).

No PyBullet call returns J_g, so these lean on the pieces that do have a
reference (J_m), on a free-floating simulation, and on the large-bus limit
where J_g has to collapse back to J_m.
"""

import numpy as np
import pybullet as p
import pytest
import torch

from src.freeflight import (ARM_JOINTS, BUS_GYRATION_SQUARED, build_model,
                            end_effector_index, to_base_link_axes,
                            FINGER_JOINTS, simulate_held_rates)

DTYPE = torch.float64
ARM = len(ARM_JOINTS)


def _bus_model(body, mass):
    inertia = mass * BUS_GYRATION_SQUARED
    return build_model(body, base_mass=mass,
                       base_inertia_diagonal=[inertia] * 3, dtype=DTYPE)


def test_manipulator_jacobian_matches_pybullet_on_a_fixed_base(
        panda_fixed, model, configurations):
    """J_m is the one term with an independent reference. Check it first,
    so a bug there is never mistaken for a bug in the coupling term."""
    zeros = [0.0] * (ARM + len(FINGER_JOINTS))
    for angles in configurations:
        mine = model.manipulator_jacobian(
            torch.tensor(angles, dtype=DTYPE)).numpy()
        linear, angular = p.calculateJacobian(
            panda_fixed, end_effector_index(panda_fixed), [0.0, 0.0, 0.0],
            list(angles) + [0.0, 0.0], zeros, zeros)
        truth = np.vstack([np.asarray(linear)[:, :ARM],
                           np.asarray(angular)[:, :ARM]])
        assert np.abs(mine - truth).max() < 1e-9


def test_base_jacobian_shares_the_base_twist_layout_with_the_mass_matrix(
        model, configurations):
    """J_b consumes exactly what H_b^-1 H_bm produces.

    A swapped layout gives a J_g of the right shape that is wrong
    everywhere, which is why the model checks it rather than trusting it.
    In [linear, angular] order a pure base translation carries the end
    effector one for one and adds no rotation.
    """
    base = model.base_jacobian(torch.tensor(configurations[0], dtype=DTYPE))
    assert np.allclose(base[:3, :3].numpy(), np.eye(3), atol=1e-12)
    assert np.allclose(base[3:, 3:].numpy(), np.eye(3), atol=1e-12)
    assert np.allclose(base[3:, :3].numpy(), np.zeros((3, 3)), atol=1e-12)


def test_a_swapped_base_twist_layout_is_rejected_not_silently_used(
        model, configurations, monkeypatch):
    """The guard has to actually fire.

    Flip _twist_transport into [angular, linear] and J_g must raise, not
    return a plausible matrix.
    """
    q = torch.tensor(configurations[0], dtype=DTYPE)
    model.generalized_jacobian(q)          # sanity: it passes as built

    original = model._twist_transport

    def swapped(point, reference):
        matrix = original(point, reference)
        permutation = torch.zeros((6, 6), dtype=DTYPE)
        permutation[:3, 3:] = torch.eye(3, dtype=DTYPE)
        permutation[3:, :3] = torch.eye(3, dtype=DTYPE)
        return permutation @ matrix @ permutation

    monkeypatch.setattr(model, "_twist_transport", swapped)
    with pytest.raises(AssertionError):
        model.generalized_jacobian(q)


def test_a_mismatched_base_reference_point_is_rejected(panda_free,
                                                       configurations):
    """H_b is written about the base link's inertial frame. Asking for J_g
    from a model built about a different point must fail loudly, since the
    two terms would be referenced to different places."""
    other = build_model(panda_free, base_reference="system_com", dtype=DTYPE)
    with pytest.raises(ValueError, match="base_com"):
        other.generalized_jacobian(torch.tensor(configurations[0], dtype=DTYPE))


def test_generalized_jacobian_predicts_the_simulated_end_effector_twist(
        panda_free, model, configurations):
    """The claim J_g exists to make: on a free base at zero momentum, the
    end effector moves at J_g qdot, not J_m qdot.

    Ground truth is a zero-gravity simulation driven at known joint rates;
    the residual is reported against the magnitude of the velocity because
    an absolute tolerance would be meaningless across configurations.
    """
    rng = np.random.default_rng(7)
    for angles in configurations[:4]:
        rates = rng.uniform(-0.6, 0.6, size=ARM)
        measured = simulate_held_rates(panda_free, angles, rates)
        # The simulation reports twists in world axes; the model produces
        # them in base link axes. The same rotation only for a diagonal
        # base inertia tensor.
        measured["ee_twist"] = to_base_link_axes(panda_free, measured["ee_twist"])

        q = torch.tensor(measured["q"], dtype=DTYPE)
        qdot = torch.tensor(measured["qdot"], dtype=DTYPE)
        predicted = (model.generalized_jacobian(q) @ qdot).numpy()

        scale = np.linalg.norm(measured["ee_twist"])
        assert np.linalg.norm(predicted - measured["ee_twist"]) / scale < 2e-3

        # And the fixed-base Jacobian, the thing J_g replaces, is wrong.
        naive = (model.manipulator_jacobian(q) @ qdot).numpy()
        assert np.linalg.norm(naive - measured["ee_twist"]) / scale > 1e-2


def test_base_twist_predicts_the_simulated_base_twist(panda_free, model,
                                                      configurations):
    """-H_b^-1 H_bm qdot is the base reaction, in [linear, angular] about
    the base CoM, which is exactly what getBaseVelocity reports."""
    rng = np.random.default_rng(11)
    for angles in configurations[:3]:
        rates = rng.uniform(-0.6, 0.6, size=ARM)
        measured = simulate_held_rates(panda_free, angles, rates)
        measured["base_twist"] = to_base_link_axes(panda_free,
                                                   measured["base_twist"])

        predicted = model.base_velocity(
            torch.tensor(measured["q"], dtype=DTYPE),
            torch.tensor(measured["qdot"], dtype=DTYPE)).numpy()
        scale = np.linalg.norm(measured["base_twist"])
        assert np.linalg.norm(predicted - measured["base_twist"]) / scale < 2e-3


def test_generalized_jacobian_approaches_the_fixed_base_jacobian_as_the_bus_grows(
        panda_free, configurations):
    """A heavy enough bus makes the free-flyer a fixed base again.

    H_b^-1 H_bm falls as 1/bus inertia, so a decade of bus mass has to buy
    a decade of convergence. If it converged at some other rate the
    coupling term would be picking up something that does not scale with
    the base inertia.
    """
    masses = [1.0e3, 1.0e4, 1.0e5]
    errors = []
    for mass in masses:
        model = _bus_model(panda_free, mass)
        per_configuration = []
        for angles in configurations[:3]:
            q = torch.tensor(angles, dtype=DTYPE)
            manipulator = model.manipulator_jacobian(q)
            generalized = model.generalized_jacobian(q)
            per_configuration.append(
                (torch.linalg.norm(generalized - manipulator)
                 / torch.linalg.norm(manipulator)).item())
        errors.append(float(np.mean(per_configuration)))

    assert errors[0] < 0.05          # already small at a 1 tonne bus
    slope = np.polyfit(np.log10(masses), np.log10(errors), 1)[0]
    assert -1.1 < slope < -0.9, f"converges as bus^{slope:.2f}, expected -1"


@pytest.mark.slow
def test_bus_convergence_holds_over_five_decades(panda_free, configurations):
    """The same limit, swept wide enough to be a figure rather than a
    check. Slow: a full sweep across every configuration."""
    masses = np.array([1.0e2, 3.0e2, 1.0e3, 3.0e3, 1.0e4, 3.0e4, 1.0e5])
    errors = []
    for mass in masses:
        model = _bus_model(panda_free, float(mass))
        errors.append(float(np.mean([
            (torch.linalg.norm(model.generalized_jacobian(
                torch.tensor(angles, dtype=DTYPE))
                - model.manipulator_jacobian(torch.tensor(angles, dtype=DTYPE)))
             / torch.linalg.norm(model.manipulator_jacobian(
                 torch.tensor(angles, dtype=DTYPE)))).item()
            for angles in configurations])))
    slope = np.polyfit(np.log10(masses[-4:]), np.log10(errors[-4:]), 1)[0]
    assert -1.05 < slope < -0.95


def test_generalized_jacobian_is_differentiable_in_the_joint_angles(
        model, configurations):
    """J_g runs through torch.linalg.solve. Autograd has to come out the
    other side, or the trajectory optimiser cannot use it."""
    q = torch.tensor(configurations[0], dtype=DTYPE, requires_grad=True)
    rates = torch.ones(ARM, dtype=DTYPE)

    def speed(configuration):
        return (model.generalized_jacobian(configuration) @ rates).pow(2).sum()

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
