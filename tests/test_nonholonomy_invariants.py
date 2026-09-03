"""Invariants of the closed-loop base rotation (phase 2c).

The headline result is that a closed joint-space loop leaves the base
rotated. Most of these cases exist to keep that from being an artifact:
of the stepper, of the default damping, or of integrating the base twist
in the wrong frame.
"""

import numpy as np
import pybullet as p
import pytest
import torch

from src.dynamics import FloatingBaseModel
from src.freeflight import (ARM_JOINTS, JointLoop, disable_damping,
                            integrate_base_rotation, momentum_residual,
                            rotation_angle, set_bus, simulate_loop)

DTYPE = torch.float64
URDF = "franka_panda/panda.urdf"


@pytest.fixture(scope="module")
def analytic_rotation(panda_free, loop):
    """Net base rotation from integrating the model around the loop, in
    degrees. Module-scoped because it costs a few seconds."""
    model = FloatingBaseModel(panda_free, ARM_JOINTS, dtype=DTYPE)
    return {
        frame: np.degrees(rotation_angle(
            integrate_base_rotation(model, loop, samples=80, frame=frame)))
        for frame in ("body", "world")
    }


def test_closed_joint_loop_leaves_the_base_rotated(panda_free, loop):
    """The whole point of the project. Every joint comes back to where it
    started and the base does not.

    On a fixed base this rotation would be exactly zero, so any clearly
    non-zero value with the joints closed is the nonholonomy.
    """
    result = simulate_loop(panda_free, loop, dt=5e-4)
    assert result["closure"] < 2e-3, "the loop did not close in joint space"
    assert np.degrees(result["angle"]) > 5.0


def test_net_base_rotation_reverses_under_backward_traversal(panda_free, loop):
    """Running the same loop backward has to undo the rotation exactly.

    Integration drift accumulates the same way whichever direction the
    path is walked, so it would add on the reverse pass rather than cancel.
    A geometric phase cancels: composing the two rotations returns to the
    identity.
    """
    forward = simulate_loop(panda_free, loop, dt=2.5e-4, direction=1)
    backward = simulate_loop(panda_free, loop, dt=2.5e-4, direction=-1)

    ahead, behind = forward["vector"], backward["vector"]
    alignment = (np.dot(ahead, behind)
                 / (np.linalg.norm(ahead) * np.linalg.norm(behind)))
    assert alignment < -0.99, "backward rotation is not about the opposite axis"

    residual = rotation_angle(forward["rotation"] @ backward["rotation"])
    assert residual < 0.02 * forward["angle"], (
        "forward then backward did not return to the identity")


def test_net_base_rotation_converges_as_the_timestep_shrinks(panda_free, loop):
    """A drift artifact would not survive refinement; a geometric phase
    does.

    Each halving of dt must change the answer by less than the last, and
    the sequence must settle on a value that is clearly not zero.
    """
    angles = [np.degrees(simulate_loop(panda_free, loop, dt=dt)["angle"])
              for dt in (2e-3, 1e-3, 5e-4, 2.5e-4)]
    increments = np.abs(np.diff(angles))

    assert np.all(increments[1:] < increments[:-1]), (
        f"refinement is not converging: {angles}")
    assert abs(angles[-1] - angles[0]) / abs(angles[-1]) < 0.02
    assert angles[-1] > 5.0


def test_momentum_is_conserved_when_link_damping_is_zeroed(panda_free, loop):
    """Motor torques act between links, so total momentum stays at zero
    through the whole maneuver.

    What is left is the stepper, not a leak, and the way to tell is that
    the residual falls in proportion to dt. A real leak would not.
    """
    coarse = simulate_loop(panda_free, loop, dt=1e-3, watch_momentum=True)
    fine = simulate_loop(panda_free, loop, dt=2.5e-4, watch_momentum=True)

    assert fine["momentum"] < 1e-2
    ratio = coarse["momentum"] / fine["momentum"]
    assert 3.0 < ratio < 5.0, (
        f"residual fell {ratio:.1f}x for a 4x smaller step, so it is not "
        "first-order discretisation")


def test_link_damping_must_be_zeroed_explicitly_not_left_to_the_default(
        panda_free, loop, analytic_rotation):
    """PyBullet gives every link 0.04 linear and 0.04 angular damping
    unless told otherwise, and getDynamicsInfo reports -1.0 either way, so
    it cannot be read back.

    Damping is an external force. Left in, it puts the closed-loop base
    rotation about 2.5% below the zero-momentum value -- a plausible
    number, quietly wrong. This case fails if disable_damping ever stops
    being called on the free-flight path.
    """
    damped = p.loadURDF(URDF, useFixedBase=False, basePosition=[0, 0, 100])
    try:
        with_default_damping = np.degrees(
            simulate_loop(damped, loop, dt=2.5e-4,
                          base_position=(0, 0, 100))["angle"])
    finally:
        p.removeBody(damped)

    zeroed = np.degrees(simulate_loop(panda_free, loop, dt=2.5e-4)["angle"])
    truth = analytic_rotation["body"]

    assert abs(with_default_damping - zeroed) / zeroed > 0.01, (
        "damping made no difference, so this test no longer guards anything")
    assert abs(zeroed - truth) / truth < abs(
        with_default_damping - truth) / truth, (
        "the damped run agreed with the zero-momentum model as well as the "
        "undamped one, which cannot be right")
    assert abs(zeroed - truth) / truth < 0.01


def test_base_twist_integrates_in_the_body_frame_not_the_world_frame(
        panda_free, loop, analytic_rotation):
    """omega_b from -H_b^-1 H_bm qdot is the base's angular velocity in its
    own frame, so it integrates as Rdot = R skew(omega).

    The world-frame form Rdot = skew(omega) R is the natural thing to
    write and gives about 77 degrees where the truth is 10: wrong by a
    factor of eight, but not obviously wrong on sight.
    """
    simulated = np.degrees(simulate_loop(panda_free, loop, dt=2.5e-4)["angle"])

    assert abs(analytic_rotation["body"] - simulated) / simulated < 0.01
    assert abs(analytic_rotation["world"] - simulated) / simulated > 1.0


def test_a_heavier_bus_rotates_less(panda_free, loop):
    """The base reaction scales as 1/bus inertia, so the same arm motion on
    a heavier bus must produce a proportionally smaller excursion."""
    excursions = []
    for mass in (1.0e3, 1.0e4):
        set_bus(panda_free, mass)
        excursions.append(
            np.degrees(simulate_loop(panda_free, loop, dt=1e-3)["angle"]))

    ratio = excursions[0] / excursions[1]
    assert 8.0 < ratio < 12.0, (
        f"ten times the bus gave {ratio:.1f}x less rotation, expected ~10")


@pytest.mark.slow
def test_attitude_excursion_follows_enclosed_area_and_inverse_bus_inertia(
        panda_free, loop):
    """The full sweep behind docs/attitude_excursion.png.

    A geometric phase is a line integral around the loop, so for small
    loops the excursion follows the enclosed area, which goes as amplitude
    squared. Slow: a grid of simulations.
    """
    amplitudes = np.array([0.1, 0.2, 0.3, 0.45, 0.6])
    bus_masses = np.array([1.0e3, 3.0e3, 1.0e4])

    excursion = np.zeros((len(bus_masses), len(amplitudes)))
    for row, mass in enumerate(bus_masses):
        set_bus(panda_free, float(mass))
        for column, amplitude in enumerate(amplitudes):
            excursion[row, column] = np.degrees(simulate_loop(
                panda_free, JointLoop(amplitude=float(amplitude)),
                dt=1e-3)["angle"])

    amplitude_slope = np.polyfit(np.log(amplitudes[:3]),
                                 np.log(excursion[-1, :3]), 1)[0]
    mass_slope = np.polyfit(np.log(bus_masses), np.log(excursion[:, 2]), 1)[0]

    assert 1.8 < amplitude_slope < 2.2, (
        f"small loops scaled as A^{amplitude_slope:.2f}, expected enclosed area")
    assert -1.1 < mass_slope < -0.9
