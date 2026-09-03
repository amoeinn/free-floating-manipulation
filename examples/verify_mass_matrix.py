"""Check the floating-base mass matrix against PyBullet, in three stages.

Each stage checks the one before it, and each reports error per item rather
than as one aggregate, because a per-link or per-block number says where a
bug is and a norm does not.

  1. Link CoM Jacobians. For every link, compare our geometric Jt and Jr
     against calculateJacobian. A probe first pins down two PyBullet
     conventions by finite difference: how long objPositions must be, and
     what frame localPosition is in.

  2. Mass matrix assembly. Symmetry, the translational block against
     total mass, and the arm-arm block H_m against both the free-base and
     the fixed-base PyBullet mass matrix. H_m does not depend on the base
     reference point, so it isolates the link Jacobians and inertias from
     the base-frame bookkeeping.

  3. Base/arm partition. H_b and H_bm against the free-base
     calculateMassMatrix, each block reported separately, swept over the
     three candidate base reference points to find which one PyBullet
     uses. Then the fingers="folded" approximation against fingers="exact".

Usage:
    python examples/verify_mass_matrix.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pybullet as p
import pybullet_data
import torch

from src.dynamics import FloatingBaseModel
from src.kinematics import quaternion_to_matrix

ARM_JOINTS = [0, 1, 2, 3, 4, 5, 6]
MOVABLE_JOINTS = [0, 1, 2, 3, 4, 5, 6, 9, 10]
DTYPE = torch.float64


def load_both() -> tuple:
    """One DIRECT client, one fixed-base body and one free-base body.

    Separate p.connect calls would need every later call to carry a
    physicsClientId; two bodies in one world do not, and calculateJacobian
    and calculateMassMatrix both take a bodyUniqueId anyway.
    """
    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    fixed_body = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)
    free_body = p.loadURDF("franka_panda/panda.urdf", useFixedBase=False,
                           basePosition=[10.0, 0.0, 0.0])
    return fixed_body, free_body


def full_configuration(arm_angles: np.ndarray) -> list:
    """Arm angles padded with the two locked finger joints."""
    return list(arm_angles) + [0.0, 0.0]


def set_arm(body: int, arm_angles: np.ndarray) -> None:
    for joint, angle in zip(ARM_JOINTS, arm_angles):
        p.resetJointState(body, joint, float(angle))
    for joint in (9, 10):
        p.resetJointState(body, joint, 0.0)


def com_world_position(body: int, link: int) -> np.ndarray:
    return np.asarray(p.getLinkState(body, link, computeForwardKinematics=True)[0])


# ---------------------------------------------------------------- convention probe

def probe_conventions(body: int, model: FloatingBaseModel,
                      arm_angles: np.ndarray) -> list:
    """Finite-difference the CoM position to learn what localPosition means.

    calculateJacobian's doc says localPosition is 'in the link frame'. The
    link frame and the inertial frame differ by up to 120 mm on the wrist,
    so a wrong assumption here shows up as a large Jacobian error on
    exactly those links. Try three candidates against a finite difference
    of getLinkState's CoM position and report which one is exact.
    """
    print("probe: what frame is calculateJacobian's localPosition in?")
    step = 1e-6
    n = len(MOVABLE_JOINTS)
    zeros = [0.0] * n
    base = full_configuration(arm_angles)

    # Length of objPositions PyBullet will accept.
    for length in (len(ARM_JOINTS), len(MOVABLE_JOINTS)):
        try:
            p.calculateJacobian(body, 2, [0, 0, 0], base[:length],
                                [0.0] * length, [0.0] * length)
            print(f"  objPositions length {length}: accepted")
        except Exception as exception:  # noqa: BLE001 - reporting a probe
            print(f"  objPositions length {length}: rejected ({exception})")

    candidates = {"link_frame_origin": lambda info: [0.0, 0.0, 0.0],
                  "inertial_offset": lambda info: list(info[3]),
                  "negated_inertial_offset": lambda info: [-c for c in info[3]]}

    rows = []
    for link in range(p.getNumJoints(body)):
        info = p.getDynamicsInfo(body, link)
        if info[0] == 0.0:
            continue

        numeric = np.zeros((3, len(ARM_JOINTS)))
        for column, joint in enumerate(ARM_JOINTS):
            ahead = arm_angles.copy()
            behind = arm_angles.copy()
            ahead[column] += step
            behind[column] -= step
            set_arm(body, ahead)
            forward = com_world_position(body, link)
            set_arm(body, behind)
            backward = com_world_position(body, link)
            numeric[:, column] = (forward - backward) / (2 * step)

        errors = {}
        for name, local_of in candidates.items():
            linear, _ = p.calculateJacobian(body, link, local_of(info),
                                            base, zeros, zeros)
            analytic = np.asarray(linear)[:, :len(ARM_JOINTS)]
            errors[name] = np.abs(analytic - numeric).max()
        rows.append((link, info[3], errors))

    print(f"  {'link':>4}  {'inertial offset':>26}   "
          + "  ".join(f"{name:>22}" for name in candidates))
    for link, offset, errors in rows:
        offset_text = f"({offset[0]:+.3f}, {offset[1]:+.3f}, {offset[2]:+.3f})"
        cells = "  ".join(f"{errors[name]:>22.3e}" for name in candidates)
        print(f"  {link:>4}  {offset_text:>26}   {cells}")

    best = min(candidates,
               key=lambda name: max(row[2][name] for row in rows))
    print(f"  -> localPosition is the {best}\n")
    return [c for c in candidates if c == best] or ["link_frame_origin"]


# --------------------------------------------------------- stage 1: Jacobians

def stage_one(body: int, model: FloatingBaseModel,
              local_frame: str, configurations: np.ndarray) -> bool:
    print("stage 1: link CoM Jacobians vs calculateJacobian")
    n = len(MOVABLE_JOINTS)
    zeros = [0.0] * n
    local_of = {
        "link_frame_origin": lambda info: [0.0, 0.0, 0.0],
        "inertial_offset": lambda info: list(info[3]),
        "negated_inertial_offset": lambda info: [-c for c in info[3]],
    }[local_frame]

    worst_linear = 0.0
    worst_angular = 0.0
    per_link = []

    for link in model.links:
        if link.kinematics is None:
            continue
        info = p.getDynamicsInfo(body, link.index)
        link_linear = 0.0
        link_angular = 0.0

        for arm_angles in configurations:
            q = torch.tensor(arm_angles, dtype=DTYPE)
            mine_linear, mine_angular = model.link_jacobian(link, q)

            base = full_configuration(arm_angles)
            linear, angular = p.calculateJacobian(body, link.index,
                                                  local_of(info), base,
                                                  zeros, zeros)
            truth_linear = np.asarray(linear)[:, :len(ARM_JOINTS)]
            truth_angular = np.asarray(angular)[:, :len(ARM_JOINTS)]

            link_linear = max(link_linear,
                              np.abs(mine_linear.numpy() - truth_linear).max())
            link_angular = max(link_angular,
                               np.abs(mine_angular.numpy() - truth_angular).max())

        per_link.append((link.index, link.name, link.mass,
                         link_linear, link_angular))
        worst_linear = max(worst_linear, link_linear)
        worst_angular = max(worst_angular, link_angular)

    print(f"  {'idx':>3}  {'name':<20} {'mass':>7}  "
          f"{'max |dJt|':>12} {'max |dJr|':>12}")
    for index, name, mass, linear_error, angular_error in per_link:
        print(f"  {index:>3}  {name:<20} {mass:>7.3f}  "
              f"{linear_error:>12.3e} {angular_error:>12.3e}")

    passed = worst_linear < 1e-9 and worst_angular < 1e-9
    print(f"  worst: dJt {worst_linear:.3e}, dJr {worst_angular:.3e}  "
          f"-> {'matches calculateJacobian' if passed else 'DISAGREES'}\n")
    return passed


# -------------------------------------------------------- stage 2: assembly

def stage_two(free_body: int, fixed_body: int, model: FloatingBaseModel,
              configurations: np.ndarray) -> bool:
    print("stage 2: mass matrix assembly")
    total_mass = sum(link.mass for link in model.links)
    n = len(ARM_JOINTS)

    worst_symmetry = 0.0
    worst_translation = 0.0
    worst_hm_free = 0.0
    worst_hm_fixed = 0.0

    for arm_angles in configurations:
        q = torch.tensor(arm_angles, dtype=DTYPE)
        matrix = model.mass_matrix(q).numpy()

        worst_symmetry = max(worst_symmetry,
                             np.abs(matrix - matrix.T).max())
        translation_block = matrix[:3, :3]
        worst_translation = max(
            worst_translation,
            np.abs(translation_block - total_mass * np.eye(3)).max())

        full = full_configuration(arm_angles)
        free = np.asarray(p.calculateMassMatrix(free_body, full))
        fixed = np.asarray(p.calculateMassMatrix(fixed_body, full))

        mine_hm = matrix[6:, 6:]
        worst_hm_free = max(worst_hm_free,
                            np.abs(mine_hm - free[6:6 + n, 6:6 + n]).max())
        worst_hm_fixed = max(worst_hm_fixed,
                             np.abs(mine_hm - fixed[:n, :n]).max())

    print(f"  symmetry              max |M - M^T|      {worst_symmetry:.3e}")
    print(f"  translational block   max |M[:3,:3] - m.I|  "
          f"{worst_translation:.3e}   (m = {total_mass:.3f} kg)")
    print(f"  arm-arm block H_m     vs free-base PyBullet  {worst_hm_free:.3e}")
    print(f"  arm-arm block H_m     vs fixed-base PyBullet {worst_hm_fixed:.3e}")

    passed = max(worst_symmetry, worst_translation,
                 worst_hm_free, worst_hm_fixed) < 1e-9
    print(f"  -> {'assembly consistent' if passed else 'ASSEMBLY INCONSISTENT'}\n")
    return passed


# ------------------------------------------------- stage 3: base/arm partition

# PyBullet orders the base DoF [angular, linear]; PLAN.md and this module
# use [linear, angular]. This permutation maps ours onto PyBullet's.
SWAP = np.zeros((6, 6))
SWAP[:3, 3:] = np.eye(3)
SWAP[3:, :3] = np.eye(3)


def stage_three(free_body: int, body_for_model: int,
                configurations: np.ndarray) -> bool:
    print("stage 3: H_b and H_bm vs free-base calculateMassMatrix")
    print("  (PyBullet base DoF order [angular, linear] permuted to "
          "[linear, angular])")
    n = len(ARM_JOINTS)
    references = ["base_link_origin", "base_com", "system_com"]

    print(f"  {'reference':>18}   {'max |dH_b|':>12}   {'max |dH_bm|':>12}")
    results = {}
    for reference in references:
        model = FloatingBaseModel(body_for_model, ARM_JOINTS,
                                  base_reference=reference, dtype=DTYPE)
        worst_hb = 0.0
        worst_hbm = 0.0
        for arm_angles in configurations:
            q = torch.tensor(arm_angles, dtype=DTYPE)
            base_inertia, coupling = model.coupling(q)
            mine_hb = SWAP @ base_inertia.numpy() @ SWAP
            mine_hbm = SWAP @ coupling.numpy()

            full = full_configuration(arm_angles)
            truth = np.asarray(p.calculateMassMatrix(free_body, full))
            truth_hb = truth[:6, :6]
            truth_hbm = truth[:6, 6:6 + n]

            worst_hb = max(worst_hb, np.abs(mine_hb - truth_hb).max())
            worst_hbm = max(worst_hbm, np.abs(mine_hbm - truth_hbm).max())
        results[reference] = (worst_hb, worst_hbm)
        print(f"  {reference:>18}   {worst_hb:>12.3e}   {worst_hbm:>12.3e}")

    best = min(results, key=lambda name: max(results[name]))
    worst_hb, worst_hbm = results[best]
    passed = max(worst_hb, worst_hbm) < 1e-9
    print(f"  -> PyBullet's base twist is written about {best}")
    print(f"     {'H_b and H_bm match' if passed else 'NO REFERENCE MATCHES'}\n")

    # ---- fingers folded vs exact, on the matching reference
    print("stage 3b: fingers=\"folded\" approximation vs \"exact\"")
    model = FloatingBaseModel(body_for_model, ARM_JOINTS,
                              base_reference=best, dtype=DTYPE)
    worst_hb = 0.0
    worst_hbm = 0.0
    for arm_angles in configurations:
        q = torch.tensor(arm_angles, dtype=DTYPE)
        exact_hb, exact_hbm = model.coupling(q, fingers="exact")
        folded_hb, folded_hbm = model.coupling(q, fingers="folded")
        worst_hb = max(worst_hb,
                       np.abs(exact_hb.numpy() - folded_hb.numpy()).max())
        worst_hbm = max(worst_hbm,
                        np.abs(exact_hbm.numpy() - folded_hbm.numpy()).max())
    print(f"  lumping 0.2 kg of finger onto the hand shifts")
    print(f"    H_b  by up to {worst_hb:.3e}  (kg, kg.m)")
    print(f"    H_bm by up to {worst_hbm:.3e}  (kg.m)")
    print(f"  relative to H_b entries of order {np.abs(exact_hb.numpy()).max():.2f}\n")

    return passed


# ------------------------------------------------------- differentiability

def check_gradient(body: int, configuration: np.ndarray) -> None:
    print("gradient: M(q) is differentiable end to end")
    model = FloatingBaseModel(body, ARM_JOINTS, base_reference="system_com",
                              dtype=DTYPE)
    q = torch.tensor(configuration, dtype=DTYPE, requires_grad=True)

    def base_speed(configuration: torch.Tensor) -> torch.Tensor:
        rates = torch.ones(len(ARM_JOINTS), dtype=configuration.dtype)
        return model.base_velocity(configuration, rates).pow(2).sum().sqrt()

    value = base_speed(q)
    value.backward()
    analytic = q.grad.clone()

    numeric = torch.zeros_like(analytic)
    step = 1e-6
    for i in range(len(ARM_JOINTS)):
        ahead = q.detach().clone()
        behind = q.detach().clone()
        ahead[i] += step
        behind[i] -= step
        numeric[i] = (base_speed(ahead) - base_speed(behind)) / (2 * step)

    error = (analytic - numeric).abs().max().item()
    print(f"  d||v_b||/dq: autograd vs finite difference  {error:.3e}  "
          f"-> {'differentiable' if error < 1e-6 else 'GRADIENT WRONG'}\n")


def main() -> None:
    rng = np.random.default_rng(0)
    fixed_body, free_body = load_both()
    limits = [(p.getJointInfo(fixed_body, j)[8], p.getJointInfo(fixed_body, j)[9])
              for j in ARM_JOINTS]
    configurations = np.array([[rng.uniform(lo, hi) for lo, hi in limits]
                               for _ in range(10)])

    # Stage 1 needs only the arm Jacobians, so the fixed-base body is fine
    # and simpler. Stages 2 and 3 need the base link's real mass, which
    # PyBullet only fills in under useFixedBase=False.
    fixed_model = FloatingBaseModel(fixed_body, ARM_JOINTS, dtype=DTYPE)
    free_model = FloatingBaseModel(free_body, ARM_JOINTS, dtype=DTYPE)

    local_frame = probe_conventions(fixed_body, fixed_model, configurations[0])[0]
    ok1 = stage_one(fixed_body, fixed_model, local_frame, configurations)
    if not ok1:
        print("stopping: fix the link Jacobians before assembling M")
        p.disconnect()
        return

    ok2 = stage_two(free_body, fixed_body, free_model, configurations)
    if not ok2:
        print("stopping: fix assembly before partitioning")
        p.disconnect()
        return

    stage_three(free_body, free_body, configurations)
    check_gradient(free_body, configurations[0])

    p.disconnect()


if __name__ == "__main__":
    main()
