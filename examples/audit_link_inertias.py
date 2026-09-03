"""Audit the Panda's link inertias before any free-floating dynamics.

A free-floating simulation is driven entirely by mass and inertia. A zero or
missing value on any link makes every downstream result — coupling inertia,
generalized Jacobian, base attitude excursion — meaningless, and it fails
silently rather than raising. So print every link's mass, diagonal inertia and
local inertial offset, total the mass, and flag anything non-positive before
trusting the model.

Load with useFixedBase=False. PyBullet zeroes the base link's mass and inertia
when the base is fixed and only fills in the URDF values when it is free, so a
fixed-base audit reports panda_link0 as massless when it is really 2.9 kg. The
free-base numbers are the ones phase 2 runs on.

Second, a bookkeeping question for the validation plan: does
calculateMassMatrix widen to (6+n)x(6+n) when the base is free, giving a direct
numerical check on H_b and H_bm, or does it always return n x n? Load the same
URDF both ways and report the shape each returns.

Usage:
    python examples/audit_link_inertias.py
"""

import numpy as np
import pybullet as p
import pybullet_data

ARM_JOINTS = [0, 1, 2, 3, 4, 5, 6]


def audit_inertias() -> None:
    """One row per link, base link included, with a non-positive flag."""
    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    robot = p.loadURDF("franka_panda/panda.urdf", useFixedBase=False)

    print("link inertia audit — franka_panda/panda.urdf, useFixedBase=False")
    print(f"{'idx':>3}  {'name':<22} {'mass (kg)':>10}  "
          f"{'Ixx':>10} {'Iyy':>10} {'Izz':>10}  "
          f"{'inertial offset (m)':>28}  flag")

    total_mass = 0.0
    flagged = []

    # Fixed frames carrying no mass (a mounting flange, a grasp target) are
    # expected to read zero and are not a problem. A movable link, or one
    # between two masses, reading zero is.
    fixed_frames = {"panda_link8", "panda_grasptarget"}

    for link in range(-1, p.getNumJoints(robot)):
        if link == -1:
            name = p.getBodyInfo(robot)[0].decode()
        else:
            name = p.getJointInfo(robot, link)[12].decode()

        dynamics = p.getDynamicsInfo(robot, link)
        mass = dynamics[0]
        inertia_diag = np.asarray(dynamics[2])
        inertial_pos = np.asarray(dynamics[3])

        total_mass += mass

        problems = []
        if mass <= 0.0:
            problems.append("mass<=0")
        if np.any(inertia_diag <= 0.0):
            problems.append("inertia<=0")
        expected_zero = name in fixed_frames
        flag = ("expected (fixed frame)" if problems and expected_zero
                else ",".join(problems))
        if problems and not expected_zero:
            flagged.append((link, name, ",".join(problems)))

        offset = (f"({inertial_pos[0]:+.3f}, {inertial_pos[1]:+.3f}, "
                  f"{inertial_pos[2]:+.3f})")
        print(f"{link:>3}  {name:<22} {mass:>10.5f}  "
              f"{inertia_diag[0]:>10.6f} {inertia_diag[1]:>10.6f} "
              f"{inertia_diag[2]:>10.6f}  {offset:>28}  {flag}")

    print(f"\n  links: {p.getNumJoints(robot) + 1}  (base link + "
          f"{p.getNumJoints(robot)} joint links)")
    print(f"  total mass: {total_mass:.5f} kg")

    print(f"  arm, hand and fingers (links 0-11): "
          f"{total_mass - p.getDynamicsInfo(robot, -1)[0]:.5f} kg")

    if flagged:
        print(f"\n  FLAGGED {len(flagged)} unexpected link(s) with "
              f"non-positive mass or inertia:")
        for link, name, flag in flagged:
            print(f"    link {link:>2} {name:<22} {flag}")
        print("  a free-floating run on this model is not trustworthy "
              "until these are fixed")
    else:
        print("\n  every load-bearing link has positive mass and inertia; "
              "the only zeros are the two fixed frames, which is expected. "
              "model is usable for free-floating dynamics")

    p.disconnect()


def mass_matrix_shape() -> None:
    """Does calculateMassMatrix widen when the base is free?"""
    print("\n\ncalculateMassMatrix shape, fixed base vs free base")

    for fixed in (True, False):
        p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        robot = p.loadURDF("franka_panda/panda.urdf", useFixedBase=fixed)

        movable = [j for j in range(p.getNumJoints(robot))
                   if p.getJointInfo(robot, j)[2] != p.JOINT_FIXED]
        m = len(movable)

        # PyBullet sizes the matrix from the model's own DOF count and pads
        # or truncates the position argument, so the input length does not
        # decide the shape.
        matrix = np.asarray(p.calculateMassMatrix(robot, [0.0] * m))
        label = "useFixedBase=True " if fixed else "useFixedBase=False"
        expected = f"{m}x{m}" if fixed else f"{6 + m}x{6 + m} if it widens"
        print(f"  {label}: {m} movable joints, returns "
              f"{matrix.shape[0]}x{matrix.shape[1]}   (expected {expected})")

        p.disconnect()

    print("\n  a free-base (6+m)x(6+m) return means the top-left 6x6 is a "
          "direct PyBullet check on H_b and the 6xm block a check on H_bm, "
          "independent of our own mass-matrix implementation")


if __name__ == "__main__":
    audit_inertias()
    mass_matrix_shape()
