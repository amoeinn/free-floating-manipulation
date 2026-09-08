"""A torque free tri-axial tumble, integrated from Euler's equations.

Step 4 tracks a target that is tumbling, and the tumble has to be one that a
filter's dynamics model can get wrong. A constant axis spin would not be: any
filter, including one with no inertia model at all, predicts a fixed axis
rotation correctly, so grading a filter on that measures nothing. This
integrates the real thing.

In the body frame, with no external torque,

    I1 w1' = (I2 - I3) w2 w3
    I2 w2' = (I3 - I1) w3 w1
    I3 w3' = (I1 - I2) w1 w2

and the attitude follows R' = R skew(w), with R mapping body to world.

Two quantities are conserved and neither needs ground truth to check:
angular momentum `R I w` is constant as a vector in the world frame, and
rotational energy `0.5 w . I w` is constant. They are the tracking equivalent
of the momentum residual in phase 2, and they are how the filter gets checked
without being told the answer.
"""

from __future__ import annotations

import numpy as np


def skew(w):
    return np.array([[0.0, -w[2], w[1]],
                     [w[2], 0.0, -w[0]],
                     [-w[1], w[0], 0.0]])


def euler_derivative(state, inertia):
    """`(R, w)` flattened to 12, and its time derivative."""
    R = state[:9].reshape(3, 3)
    w = state[9:]
    I1, I2, I3 = inertia
    dw = np.array([(I2 - I3) * w[1] * w[2] / I1,
                   (I3 - I1) * w[2] * w[0] / I2,
                   (I1 - I2) * w[0] * w[1] / I3])
    return np.concatenate([(R @ skew(w)).ravel(), dw])


def integrate(inertia, omega0, dt, steps, R0=None):
    """RK4 over `steps` of `dt`, returning attitudes and body rates.

    The attitude is re-orthonormalised each step. Without it the rotation
    matrix drifts off SO(3) and the conserved quantities below stop being a
    check on the physics and become a check on the integrator instead.
    """
    R = np.eye(3) if R0 is None else np.asarray(R0, float)
    state = np.concatenate([R.ravel(), np.asarray(omega0, float)])
    inertia = np.asarray(inertia, float)
    attitudes, rates = [R.copy()], [np.asarray(omega0, float).copy()]
    for _ in range(steps):
        k1 = euler_derivative(state, inertia)
        k2 = euler_derivative(state + 0.5 * dt * k1, inertia)
        k3 = euler_derivative(state + 0.5 * dt * k2, inertia)
        k4 = euler_derivative(state + dt * k3, inertia)
        state = state + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        u, _, vt = np.linalg.svd(state[:9].reshape(3, 3))
        state[:9] = (u @ vt).ravel()
        attitudes.append(state[:9].reshape(3, 3).copy())
        rates.append(state[9:].copy())
    return np.array(attitudes), np.array(rates)


def angular_momentum(attitudes, rates, inertia):
    """`R I w` per step, in the world frame. Constant for a torque free body."""
    inertia = np.asarray(inertia, float)
    return np.einsum("kij,j,kj->ki", attitudes, inertia, rates)


def kinetic_energy(rates, inertia):
    return 0.5 * np.einsum("kj,j,kj->k", rates, np.asarray(inertia, float), rates)


def assert_triaxial(inertia, rates, min_ratio=1.15, min_share=0.05):
    """The tumble must be able to express what the filter could get wrong.

    Three separate ways this could go vacuous, all checked. The inertia must
    actually be asymmetric, or Euler's equations reduce to a constant rate.
    Every body axis must carry a real share of the rate, or the motion is a
    spin about one axis whatever the inertia says. And the rate vector must
    move in the body frame, or the polhode has collapsed to a point and a
    filter with no dynamics at all would score perfectly.
    """
    I = np.sort(np.asarray(inertia, float))
    ratios = (I[1] / I[0], I[2] / I[1])
    share = np.abs(rates).max(axis=0)
    share = share / share.sum()
    travel = np.linalg.norm(rates - rates[0], axis=1).max() / np.linalg.norm(rates[0])
    report = {"inertia_ratios": ratios, "axis_share": share, "polhode_travel": travel}
    if min(ratios) < min_ratio:
        raise SystemExit(f"inertia is too close to symmetric {ratios}, so Euler's "
                         "equations barely couple the axes and this tumble cannot "
                         "test a dynamics model")
    if share.min() < min_share:
        raise SystemExit(f"one body axis carries {100*share.min():.1f} percent of the "
                         "rate, so this is nearly a single axis spin")
    if travel < 0.2:
        raise SystemExit(f"the body rate moves only {100*travel:.1f} percent over the "
                         "sequence, so the polhode is nearly a point and a filter with "
                         "no dynamics model would predict it correctly")
    return report
