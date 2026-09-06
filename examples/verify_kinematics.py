"""Check the differentiable kinematics against PyBullet.

Three questions, in order of how badly a wrong answer would hurt.

Does the chain match? Sample configurations across the joint limits and
compare position and orientation against getLinkState. Sub-millimetre
agreement or the chain is wrong somewhere, most likely in the order of
origin and rotation, the quaternion component order, or the fixed joints
between the last revolute joint and the grasp target.

Is the Jacobian right? Compare autograd's derivative against a finite
difference of the same function. This catches a chain that happens to be
correct at the sampled poses but differentiates wrongly.

Is it fast enough to sit inside an optimizer? Time a batch against the
equivalent PyBullet calls.

Usage:
    python examples/verify_kinematics.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pybullet as p
import pybullet_data
import torch

from src.freeflight import end_effector_index, load_panda, panda_spec
from src.kinematics import ForwardKinematics, describe_chain

ARM_JOINTS = [0, 1, 2, 3, 4, 5, 6]
END_EFFECTOR = None  # resolved from the selected model


def pybullet_pose(body, link, configuration):
    """Ground truth pose from PyBullet, for one configuration."""
    for joint, angle in zip(ARM_JOINTS, configuration):
        p.resetJointState(body, joint, float(angle))
    state = p.getLinkState(body, link, computeForwardKinematics=True)
    return np.asarray(state[4]), np.asarray(state[5])


def main() -> None:
    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    robot = load_panda(fixed_base=True)
    end_effector = end_effector_index(robot)

    limits = [(p.getJointInfo(robot, j)[8], p.getJointInfo(robot, j)[9])
              for j in ARM_JOINTS]

    kinematics = ForwardKinematics(robot, end_effector,
                                   movable_joints=ARM_JOINTS,
                                   urdf=panda_spec()["urdf_path"])
    print(describe_chain(kinematics))

    # ------------------------------------------------------- pose agreement
    print("\nposition agreement over 500 random configurations")
    rng = np.random.default_rng(0)
    position_errors = []
    orientation_errors = []

    for _ in range(500):
        configuration = np.array([rng.uniform(lo, hi) for lo, hi in limits])
        truth_position, truth_orientation = pybullet_pose(
            robot, end_effector, configuration)

        pose = kinematics(torch.tensor(configuration, dtype=torch.float64))
        mine_position = pose[:3, 3].numpy()
        position_errors.append(np.linalg.norm(mine_position - truth_position))

        # Compare rotation matrices rather than quaternions, which are
        # only defined up to sign.
        truth_rotation = np.asarray(
            p.getMatrixFromQuaternion(truth_orientation)).reshape(3, 3)
        orientation_errors.append(
            np.abs(pose[:3, :3].numpy() - truth_rotation).max())

    position_errors = np.asarray(position_errors)
    orientation_errors = np.asarray(orientation_errors)

    print(f"  position:    mean {position_errors.mean() * 1000:.6f} mm, "
          f"max {position_errors.max() * 1000:.6f} mm")
    print(f"  orientation: max element difference "
          f"{orientation_errors.max():.3e}")

    passed = position_errors.max() < 1e-6
    print(f"  {'agrees to under a micrometre' if passed else 'DISAGREES'}")

    if not passed:
        worst = int(np.argmax(position_errors))
        print(f"\n  worst case was sample {worst}, "
              f"{position_errors[worst] * 1000:.3f} mm out")
        print("  the usual causes are the order of origin and rotation, "
              "the quaternion component order, or a missing fixed joint")
        p.disconnect()
        return

    # ----------------------------------------------------------- Jacobian
    print("\nJacobian against a finite difference, 20 configurations")
    step = 1e-6
    jacobian_errors = []

    for _ in range(20):
        configuration = np.array([rng.uniform(lo, hi) for lo, hi in limits])
        tensor = torch.tensor(configuration, dtype=torch.float64)

        analytic = kinematics.jacobian(tensor).numpy()

        numeric = np.zeros((3, len(ARM_JOINTS)))
        for joint in range(len(ARM_JOINTS)):
            forward = configuration.copy()
            backward = configuration.copy()
            forward[joint] += step
            backward[joint] -= step
            ahead = kinematics.position(
                torch.tensor(forward, dtype=torch.float64)).numpy()
            behind = kinematics.position(
                torch.tensor(backward, dtype=torch.float64)).numpy()
            numeric[:, joint] = (ahead - behind) / (2 * step)

        jacobian_errors.append(np.abs(analytic - numeric).max())

    jacobian_errors = np.asarray(jacobian_errors)
    print(f"  max difference: {jacobian_errors.max():.3e}")
    print(f"  {'autograd matches finite differences' if jacobian_errors.max() < 1e-6 else 'JACOBIAN DISAGREES'}")

    # -------------------------------------------------------------- timing
    print("\nspeed")
    batch = torch.tensor(
        np.array([[rng.uniform(lo, hi) for lo, hi in limits]
                  for _ in range(200)]), dtype=torch.float64)

    began = time.perf_counter()
    kinematics(batch)
    mine = time.perf_counter() - began

    began = time.perf_counter()
    for row in batch.numpy():
        pybullet_pose(robot, end_effector, row)
    theirs = time.perf_counter() - began

    print(f"  200 poses in torch:    {mine * 1000:.1f} ms")
    print(f"  200 poses in PyBullet: {theirs * 1000:.1f} ms")
    print(f"  ratio: {mine / theirs:.1f}x")

    began = time.perf_counter()
    for row in batch[:20]:
        kinematics.jacobian(row)
    print(f"  20 Jacobians: {(time.perf_counter() - began) * 1000:.0f} ms")

    p.disconnect()


if __name__ == "__main__":
    main()