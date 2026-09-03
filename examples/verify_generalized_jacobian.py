"""Check the generalized Jacobian J_g = J_m - J_b H_b^-1 H_bm.

There is no PyBullet call that returns J_g, so it is validated two ways
that do not share a failure mode:

  0. J_m alone. The fixed-base end-effector Jacobian does have a PyBullet
     equivalent; check it against calculateJacobian first so a bug there
     is not mistaken for a bug in the coupling term.

  1. Ground truth by simulation. Load free-floating at zero gravity with
     no ground plane, hold the arm at a chosen joint rate with velocity
     control, step, and read the actual end-effector velocity. Total
     momentum stays zero because the motor torques are internal, so this
     is exactly the J_g scenario. Compare J_g qdot against the measured
     velocity, and -H_b^-1 H_bm qdot against the measured base twist,
     both as a fraction of the measured magnitude.

  2. Limit test. As the bus inertia grows, H_b^-1 H_bm -> 0 and J_g must
     converge to J_m. Sweep the bus mass across five decades and report
     the convergence. This is also the first data for the bus sweep
     figure.

Usage:
    python examples/verify_generalized_jacobian.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
import pybullet_data
import torch

from src.dynamics import FloatingBaseModel

ARM_JOINTS = [0, 1, 2, 3, 4, 5, 6]
FINGER_JOINTS = [9, 10]
END_EFFECTOR = 11
DTYPE = torch.float64
FREE_BASE_POSITION = [0.0, 0.0, 20.0]
# docs/ is the one PNG location .gitignore keeps.
FIGURE = Path(__file__).resolve().parent.parent / "docs" / "bus_sweep_jacobian.png"


def connect() -> tuple:
    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, 0)
    fixed_body = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)
    # Well clear of the fixed body: two Pandas at the same origin collide and
    # the free one is flung off at 100 m/s.
    free_body = p.loadURDF("franka_panda/panda.urdf", useFixedBase=False,
                           basePosition=FREE_BASE_POSITION)
    return fixed_body, free_body


def random_configurations(body: int, count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    limits = [(p.getJointInfo(body, j)[8], p.getJointInfo(body, j)[9])
              for j in ARM_JOINTS]
    # Back off the hard stops so the velocity-controlled sim does not fight
    # a limit during the step.
    return np.array([[rng.uniform(0.8 * lo, 0.8 * hi) for lo, hi in limits]
                     for _ in range(count)])


# ------------------------------------------------------ part 0: J_m vs PyBullet

def check_manipulator_jacobian(fixed_body: int, model: FloatingBaseModel,
                               configurations: np.ndarray) -> bool:
    print("part 0: J_m (fixed base) vs calculateJacobian")
    zeros = [0.0] * (len(ARM_JOINTS) + len(FINGER_JOINTS))
    worst = 0.0
    for arm_angles in configurations:
        q = torch.tensor(arm_angles, dtype=DTYPE)
        mine = model.manipulator_jacobian(q).numpy()

        full = list(arm_angles) + [0.0, 0.0]
        linear, angular = p.calculateJacobian(fixed_body, END_EFFECTOR,
                                              [0.0, 0.0, 0.0], full,
                                              zeros, zeros)
        truth = np.vstack([np.asarray(linear)[:, :len(ARM_JOINTS)],
                           np.asarray(angular)[:, :len(ARM_JOINTS)]])
        worst = max(worst, np.abs(mine - truth).max())

    passed = worst < 1e-9
    print(f"  max |J_m - calculateJacobian|: {worst:.3e}  "
          f"-> {'match' if passed else 'MISMATCH'}\n")
    return passed


# ------------------------------------------------- part 1: J_g vs simulation

def simulate(free_body: int, arm_angles: np.ndarray, joint_rates: np.ndarray,
             dt: float = 1e-4, steps: int = 12) -> dict:
    """Drive the arm at joint_rates on the free base and read what moves."""
    p.resetBasePositionAndOrientation(free_body, FREE_BASE_POSITION,
                                      [0, 0, 0, 1])
    p.resetBaseVelocity(free_body, [0, 0, 0], [0, 0, 0])
    for joint, angle in zip(ARM_JOINTS, arm_angles):
        p.resetJointState(free_body, joint, float(angle), targetVelocity=0.0)
    for joint in FINGER_JOINTS:
        p.resetJointState(free_body, joint, 0.0, targetVelocity=0.0)

    # Velocity control holds the commanded joint rates; the base is free to
    # react. Motor torques are internal so total momentum stays at zero.
    p.setJointMotorControlArray(free_body, ARM_JOINTS, p.VELOCITY_CONTROL,
                                targetVelocities=list(joint_rates),
                                forces=[1.0e3] * len(ARM_JOINTS))
    p.setJointMotorControlArray(free_body, FINGER_JOINTS, p.VELOCITY_CONTROL,
                                targetVelocities=[0.0, 0.0],
                                forces=[1.0e3, 1.0e3])
    p.setTimeStep(dt)
    for _ in range(steps):
        p.stepSimulation()

    states = p.getJointStates(free_body, ARM_JOINTS)
    link = p.getLinkState(free_body, END_EFFECTOR, computeLinkVelocity=1,
                          computeForwardKinematics=1)
    base_linear, base_angular = p.getBaseVelocity(free_body)
    return {
        "q": np.array([s[0] for s in states]),
        "qdot": np.array([s[1] for s in states]),
        "ee_twist": np.concatenate([np.array(link[6]), np.array(link[7])]),
        "base_twist": np.concatenate([np.array(base_linear),
                                      np.array(base_angular)]),
    }


def check_against_simulation(free_body: int, model: FloatingBaseModel,
                             configurations: np.ndarray) -> bool:
    print("part 1: J_g qdot vs a free-floating simulation")
    rng = np.random.default_rng(7)
    print(f"  {'config':>6}  {'|v_ee|':>9}  {'ee rel err':>11}  "
          f"{'base rel err':>12}")

    worst_ee = 0.0
    worst_base = 0.0
    for index, arm_angles in enumerate(configurations):
        joint_rates = rng.uniform(-0.6, 0.6, size=len(ARM_JOINTS))
        measured = simulate(free_body, arm_angles, joint_rates)

        q = torch.tensor(measured["q"], dtype=DTYPE)
        rates = torch.tensor(measured["qdot"], dtype=DTYPE)

        predicted_ee = (model.generalized_jacobian(q) @ rates).numpy()
        predicted_base = model.base_velocity(q, rates).numpy()

        ee_scale = np.linalg.norm(measured["ee_twist"])
        base_scale = np.linalg.norm(measured["base_twist"])
        ee_error = np.linalg.norm(predicted_ee - measured["ee_twist"]) / ee_scale
        base_error = (np.linalg.norm(predicted_base - measured["base_twist"])
                      / base_scale)

        worst_ee = max(worst_ee, ee_error)
        worst_base = max(worst_base, base_error)
        print(f"  {index:>6}  {ee_scale:>9.4f}  {ee_error:>11.3e}  "
              f"{base_error:>12.3e}")

    passed = worst_ee < 2e-3 and worst_base < 2e-3
    print(f"  worst: end-effector {worst_ee:.3e}, base {worst_base:.3e}")
    print(f"  -> {'J_g matches the simulation' if passed else 'J_g DISAGREES'}")
    print("     (a few 1e-4 is the velocity-control and single-step "
          "discretisation, not the model)\n")
    return passed


# --------------------------------------------------- part 2: bus mass limit

def bus_sweep(free_body: int, configurations: np.ndarray) -> bool:
    print("part 2: J_g -> J_m as the bus grows")
    # A real servicing bus is metres across, not the Panda pedestal's dense
    # puck, so model it with a fixed radius of gyration: I_bus = m k^2 I_3
    # with k = 0.5 m (a ~1.5 m box). Using the pedestal's tiny k^2 for a
    # 30-tonne bus leaves the arm's own O(5 kg m^2) in the rotational block
    # and the limit never cleans up.
    gyration_squared = 0.25
    bus_masses = np.array([100.0, 300.0, 1000.0, 3000.0, 10000.0, 30000.0,
                           100000.0])
    relative_error = []
    coupling_norm = []
    base_rate = []  # |omega_b| for a unit-norm joint rate, an attitude proxy

    print(f"  {'bus mass':>10}  {'||Hb^-1 Hbm||':>14}  "
          f"{'||Jg-Jm||/||Jm||':>16}  {'|omega_b| / |qdot|':>18}")
    for mass in bus_masses:
        inertia = mass * gyration_squared
        model = FloatingBaseModel(
            free_body, ARM_JOINTS, base_mass=float(mass),
            base_inertia_diagonal=[inertia, inertia, inertia], dtype=DTYPE)

        errors = []
        couplings = []
        rates = []
        for arm_angles in configurations:
            q = torch.tensor(arm_angles, dtype=DTYPE)
            manipulator = model.manipulator_jacobian(q)
            generalized = model.generalized_jacobian(q)
            base_inertia, coupling = model.coupling(q)

            errors.append((torch.linalg.norm(generalized - manipulator)
                           / torch.linalg.norm(manipulator)).item())
            couplings.append(torch.linalg.norm(
                torch.linalg.solve(base_inertia, coupling)).item())

            unit_rate = torch.ones(len(ARM_JOINTS), dtype=DTYPE)
            unit_rate = unit_rate / torch.linalg.norm(unit_rate)
            twist = model.base_velocity(q, unit_rate)
            rates.append(torch.linalg.norm(twist[3:]).item())

        relative_error.append(np.mean(errors))
        coupling_norm.append(np.mean(couplings))
        base_rate.append(np.mean(rates))
        print(f"  {mass:>10.1f}  {coupling_norm[-1]:>14.4e}  "
              f"{relative_error[-1]:>16.4e}  {base_rate[-1]:>18.4e}")

    # H_b^-1 H_bm ~ 1/bus_inertia once the bus dominates the arm, so the tail
    # of the log-log curve should have slope about -1. Fit the last four
    # points (bus from 1000 kg up, well past the 15 kg arm).
    tail = slice(-4, None)
    slope = np.polyfit(np.log10(bus_masses[tail]),
                       np.log10(relative_error[tail]), 1)[0]
    passed = -1.1 < slope < -0.9
    print(f"  log-log slope over the last four points: {slope:.3f}  "
          f"(expect -1)  -> "
          f"{'converges to J_m as 1/bus inertia' if passed else 'WRONG RATE'}\n")

    _write_figure(bus_masses, relative_error, base_rate)
    return passed


def _write_figure(bus_masses, relative_error, base_rate) -> None:
    FIGURE.parent.mkdir(exist_ok=True)
    figure, left = plt.subplots(figsize=(7, 4.5))
    right = left.twinx()

    jacobian_line, = left.loglog(
        bus_masses, relative_error, "o-", color="#1f77b4",
        label=r"$\|J_g - J_m\| / \|J_m\|$ (Frobenius)")
    rate_line, = right.loglog(
        bus_masses, base_rate, "s--", color="#d62728",
        label=r"$\|\omega_b\| / \|\dot q\|$  (rad/s per rad/s)")

    left.set_xlabel("bus mass (kg), inertia = m (0.5 m)$^2$")
    left.set_ylabel("generalized vs fixed-base Jacobian, relative")
    right.set_ylabel("base angular rate per unit joint rate")
    left.set_title("Free-flying coupling vanishes as the bus grows")
    left.grid(True, which="both", alpha=0.3)

    for mass, label in [(180.0, "Otter class"), (2300.0, "MEV-1")]:
        left.axvline(mass, color="grey", lw=0.8, ls=":")
        left.text(mass, relative_error[0], f"  {label}", rotation=90,
                  va="top", ha="left", fontsize=8, color="grey")

    left.legend(handles=[jacobian_line, rate_line], loc="lower left",
                fontsize=9)
    figure.tight_layout()
    figure.savefig(FIGURE, dpi=130)
    plt.close(figure)
    print(f"  wrote {FIGURE.relative_to(Path.cwd())}")


def check_gradient(model: FloatingBaseModel, arm_angles: np.ndarray) -> bool:
    """J_g runs through torch.linalg.solve; confirm autograd survives it."""
    print("part 3: J_g is differentiable in q")
    q = torch.tensor(arm_angles, dtype=DTYPE, requires_grad=True)
    rates = torch.ones(len(ARM_JOINTS), dtype=DTYPE)

    def ee_speed(configuration: torch.Tensor) -> torch.Tensor:
        return (model.generalized_jacobian(configuration) @ rates).pow(2).sum()

    ee_speed(q).backward()
    analytic = q.grad.clone()

    numeric = torch.zeros_like(analytic)
    step = 1e-6
    for i in range(len(ARM_JOINTS)):
        ahead = q.detach().clone()
        behind = q.detach().clone()
        ahead[i] += step
        behind[i] -= step
        numeric[i] = (ee_speed(ahead) - ee_speed(behind)) / (2 * step)

    error = (analytic - numeric).abs().max().item()
    passed = error < 1e-6
    print(f"  d||J_g qdot||^2/dq: autograd vs finite difference  {error:.3e}  "
          f"-> {'differentiable' if passed else 'GRADIENT WRONG'}\n")
    return passed


def main() -> None:
    fixed_body, free_body = connect()
    configurations = random_configurations(fixed_body, count=8, seed=0)
    model = FloatingBaseModel(free_body, ARM_JOINTS, dtype=DTYPE)

    ok0 = check_manipulator_jacobian(fixed_body, model, configurations)
    if not ok0:
        print("stopping: J_m is wrong, the coupling term cannot be judged")
        p.disconnect()
        return

    ok1 = check_against_simulation(free_body, model, configurations)
    ok2 = bus_sweep(free_body, configurations)
    ok3 = check_gradient(model, configurations[0])

    print(f"summary: J_m {'ok' if ok0 else 'FAIL'}, "
          f"simulation {'ok' if ok1 else 'FAIL'}, "
          f"bus limit {'ok' if ok2 else 'FAIL'}, "
          f"gradient {'ok' if ok3 else 'FAIL'}")
    p.disconnect()


if __name__ == "__main__":
    main()
