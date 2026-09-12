"""What a torque free tumble must satisfy, and what makes one worth using.

Two different things are protected here and they are worth separating.

The first is physics: angular momentum and rotational energy are conserved for
a torque free body, and neither needs ground truth to check. They are the
tracking equivalent of the momentum residual in phase 2, and they are what
lets a filter be judged later without being told the answer.

The second is that the tumble is able to expose a wrong dynamics model at all.
A constant axis spin is predicted correctly by a filter carrying no inertia
model whatsoever, so grading against one would measure nothing while looking
like a result. `assert_triaxial` refuses a tumble that has gone degenerate in
any of three ways, and each refusal has its own case below, because a guard
nobody has watched fail is a guard nobody should trust.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))

import numpy as np
import pytest

from src.tumble import (angular_momentum, assert_triaxial, integrate,
                        kinetic_energy, skew)

# The parameters the scene and the phase 4 sequences actually run on, so these
# cases protect the tumble in use rather than a convenient one.
INERTIA = np.array([1.0, 1.9, 2.6])
RATE_DEG_S = 2.7
DIRECTION = np.array([0.35, 0.22, 0.16])


def omega(rate_deg_s=RATE_DEG_S, direction=DIRECTION):
    d = np.asarray(direction, float)
    return d / np.linalg.norm(d) * np.deg2rad(rate_deg_s)


@pytest.fixture(scope="module")
def tumble():
    """Thirty seconds of the tumble the client actually performs."""
    return integrate(INERTIA, omega(), dt=0.02, steps=1500)


def test_angular_momentum_is_conserved_in_the_world_frame(tumble):
    """No external torque means `R I w` is a fixed vector, in magnitude and in
    direction. Drift here is the integrator leaking, and it would corrupt
    every conclusion drawn from the tumble downstream."""
    attitudes, rates = tumble
    L = angular_momentum(attitudes, rates, INERTIA)
    magnitude = np.linalg.norm(L, axis=1)
    assert np.abs(magnitude - magnitude[0]).max() < 1e-10

    direction = L @ L[0] / (magnitude * magnitude[0])
    assert np.abs(direction - 1.0).max() < 1e-12, "L changed direction"


def test_rotational_energy_is_conserved(tumble):
    """The second invariant, and it is not implied by the first: a body rate
    can move along the momentum sphere while changing energy if the
    integration is wrong."""
    _, rates = tumble
    T = kinetic_energy(rates, INERTIA)
    assert np.abs(T - T[0]).max() < 1e-10


def test_the_attitude_stays_on_the_rotation_group(tumble):
    """The attitude must remain a rotation, or nothing derived from it means
    anything.

    What this catches and what it does not was established by breaking both,
    because the distinction is not obvious. It catches a corrupted
    re-orthonormalisation: scaling one singular value to 1.05 fails here, and
    in the two conservation cases as well.

    It does not catch the re-orthonormalisation being removed. That was
    measured: RK4 at this timestep holds orthonormality to 2.5e-15 over 30 s
    and 4.2e-14 over 150000 steps with no projection at all, against 1.3e-15
    with one. The projection is cheap insurance for longer runs rather than
    something the tumble currently depends on, and a case here claiming to
    protect it would be claiming more than it can deliver.
    """
    attitudes, _ = tumble
    for R in attitudes[::100]:
        assert np.abs(R @ R.T - np.eye(3)).max() < 1e-12
        assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-12)


def test_conservation_holds_as_the_timestep_shrinks():
    """Refinement, which separates a conserved quantity from one that merely
    looks conserved at the step that happened to be chosen."""
    drift = {}
    for dt in (0.08, 0.02, 0.005):
        attitudes, rates = integrate(INERTIA, omega(), dt=dt, steps=int(30 / dt))
        L = np.linalg.norm(angular_momentum(attitudes, rates, INERTIA), axis=1)
        drift[dt] = float(np.abs(L - L[0]).max())
    assert max(drift.values()) < 1e-9, drift


def test_the_tumble_in_use_is_accepted_by_the_guard(tumble):
    """The guard has to pass on the real parameters, or the scene and the
    phase 4 sequences are running something the project's own standard
    rejects."""
    _, rates = tumble
    report = assert_triaxial(INERTIA, rates)
    assert min(report["inertia_ratios"]) > 1.15
    assert report["axis_share"].min() > 0.05
    assert report["polhode_travel"] > 0.2


def test_a_symmetric_inertia_is_rejected_because_euler_stops_coupling():
    """With two equal principal moments Euler's equations reduce to a
    constant rate about a fixed axis, and no dynamics model can be wrong
    about it."""
    with pytest.raises(SystemExit, match="too close to symmetric"):
        symmetric = np.array([2.0, 2.0, 2.0])
        _, rates = integrate(symmetric, omega(), dt=0.02, steps=500)
        assert_triaxial(symmetric, rates)


def test_a_single_axis_spin_is_rejected_however_asymmetric_the_body():
    """The inertia can be perfectly tri-axial and the motion still be a plain
    spin, if the rate starts on a principal axis. That is the case a filter
    with no inertia model predicts exactly, so it must not count."""
    spin = np.array([0.0, 0.0, np.deg2rad(RATE_DEG_S)])
    _, rates = integrate(INERTIA, spin, dt=0.02, steps=500)
    with pytest.raises(SystemExit, match="single axis spin"):
        assert_triaxial(INERTIA, rates)


def test_a_polhode_that_barely_moves_is_rejected():
    """The third way a tumble goes degenerate: the rate vector is spread over
    all three axes and the inertia is asymmetric, but the motion is so close
    to a principal axis that the body rate hardly evolves over the sequence.
    A short enough window does this to any tumble."""
    attitudes, rates = integrate(INERTIA, omega(), dt=0.02, steps=20)
    travel = np.linalg.norm(rates - rates[0], axis=1).max() / np.linalg.norm(rates[0])
    assert travel < 0.2, f"this window is not short enough to be degenerate: {travel:.3f}"
    with pytest.raises(SystemExit, match="polhode"):
        assert_triaxial(INERTIA, rates)


def test_the_polhode_travel_grows_with_the_window_and_crosses_the_threshold():
    """The positive counterpart to the case above, and the reason phase 4 has
    two windows rather than one.

    Whether a tumble exercises the dynamics is a property of the window, not
    of the tumble: the same motion is degenerate over a few seconds and
    genuinely tri-axial over a minute. Measured at 2.7 deg/s, travel is about
    0.03 at 2 s, 0.11 at 8 s and 0.47 at 48 s, so the 8 s photometric window
    sits below the guard's threshold and the 48 s silhouette window sits well
    above it. That is why the filter could only be graded on the long one.
    """
    travel = {}
    for seconds in (2.0, 8.0, 48.0):
        _, rates = integrate(INERTIA, omega(), dt=0.02, steps=int(seconds / 0.02))
        travel[seconds] = float(
            np.linalg.norm(rates - rates[0], axis=1).max() / np.linalg.norm(rates[0]))
    assert travel[2.0] < travel[8.0] < travel[48.0], travel
    assert travel[8.0] < 0.2, (
        f"the 8 s window now travels {travel[8.0]:.3f}, above the guard's "
        "threshold; phase 4's reason for splitting the windows has changed")
    assert travel[48.0] > 0.2, (
        f"the 48 s window travels only {travel[48.0]:.3f}, so the long window "
        "no longer exercises the Euler coupling it was chosen for")


def test_the_skew_matrix_is_the_cross_product_it_stands_for():
    """`skew(w) v` has to equal `w x v`, because the attitude derivative is
    built on it. A transposed skew integrates the tumble backwards and still
    conserves both invariants above, so neither of them would catch it."""
    rng = np.random.default_rng(0)
    for _ in range(8):
        w, v = rng.normal(size=3), rng.normal(size=3)
        assert np.allclose(skew(w) @ v, np.cross(w, v))
