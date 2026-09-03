"""Driving a free-flying servicer and measuring what its base does.

The pieces the phase 2 demonstrations and the test suite share: a closed
joint-space loop, a zero-gravity PyBullet run that tracks it, a direct
momentum readout, and the analytic integration of the base attitude along
the same path. Keeping them here rather than in the example scripts means
the tests protect the code the figures were made with.

Two things bite and are handled here rather than left to the caller:

  - PyBullet gives every link 0.04 linear and 0.04 angular damping by
    default, and `getDynamicsInfo` fields 8/9 report -1.0 whether or not
    it is in force, so it cannot be read back. Damping is an external
    force: it leaks momentum and puts the closed-loop base rotation 2.5%
    off. `disable_damping` must be called on any body used for free
    flight, and the test suite has a case that fails if it is not.
  - the base twist from -H_b^-1 H_bm qdot is in the base BODY frame, so it
    integrates as Rdot = R skew(omega). The world-frame form gives 77
    degrees where the truth is 10.
"""

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pybullet as p
import torch

# Panda layout. The two loop joints are the arm's heavy pair; swinging
# them moves the most inertia and so produces the largest base reaction.
ARM_JOINTS = (0, 1, 2, 3, 4, 5, 6)
FINGER_JOINTS = (9, 10)
END_EFFECTOR = 11
HOME = np.array([0.0, -0.3, 0.0, -1.8, 0.0, 1.5, 0.0])

# A servicing bus is metres across, not the Panda pedestal's dense puck.
# I_bus = m k^2 with k = 0.5 m, so the bus dominates the arm's own
# O(5 kg m^2) about the base and the large-bus limits come out clean.
BUS_GYRATION_SQUARED = 0.25


@dataclass(frozen=True)
class JointLoop:
    """A closed circuit in joint space: out and back to the same angles.

    One joint traces a sine and the other a raised cosine, so the pair
    walks a circle in its own plane and the path encloses area. Both the
    angles and the rates are periodic, so the loop closes in position and
    in velocity, and the base reaction is a clean geometric phase.
    """

    amplitude: float = 0.6
    period: float = 2.0
    joints: tuple = (1, 2)
    home: np.ndarray = field(default_factory=lambda: HOME.copy())
    arm_joints: tuple = ARM_JOINTS

    def angles(self, phase: float) -> np.ndarray:
        first, second = self.joints
        angles = self.home.copy()
        angles[first] += self.amplitude * np.sin(phase)
        angles[second] += self.amplitude * (1.0 - np.cos(phase))
        return angles

    def rates(self, phase: float, phase_rate: float) -> np.ndarray:
        first, second = self.joints
        rates = np.zeros(len(self.arm_joints))
        rates[first] = self.amplitude * np.cos(phase) * phase_rate
        rates[second] = self.amplitude * np.sin(phase) * phase_rate
        return rates


# ------------------------------------------------------------------ rotations

def rotation_angle(rotation: np.ndarray) -> float:
    """Angle of a rotation matrix, in radians."""
    return float(np.arccos(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)))


def rotation_vector(rotation: np.ndarray) -> np.ndarray:
    """Axis times angle. Zero for a rotation too small to have an axis."""
    angle = rotation_angle(rotation)
    if angle < 1e-12:
        return np.zeros(3)
    axis = np.array([rotation[2, 1] - rotation[1, 2],
                     rotation[0, 2] - rotation[2, 0],
                     rotation[1, 0] - rotation[0, 1]])
    return axis / np.linalg.norm(axis) * angle


def rotation_exponential(vector: np.ndarray) -> np.ndarray:
    """Rotation matrix for an axis-times-angle vector, by Rodrigues."""
    angle = np.linalg.norm(vector)
    if angle < 1e-15:
        return np.eye(3)
    unit = vector / angle
    cross = np.array([[0.0, -unit[2], unit[1]],
                      [unit[2], 0.0, -unit[0]],
                      [-unit[1], unit[0], 0.0]])
    return (np.eye(3) + np.sin(angle) * cross
            + (1.0 - np.cos(angle)) * cross @ cross)


# ------------------------------------------------------------- body preparation

def disable_damping(body: int) -> None:
    """Zero the per-link damping PyBullet applies by default.

    Not optional for free flight. The default 0.04 linear and 0.04
    angular damping is an external force; it bleeds momentum and the
    closed loop stops returning a clean geometric phase. It also cannot
    be read back, so there is no way to check it after the fact --
    only to call this, and to test the behaviour it fixes.
    """
    for link in range(-1, p.getNumJoints(body)):
        p.changeDynamics(body, link, linearDamping=0.0, angularDamping=0.0,
                         jointDamping=0.0)


def set_bus(body: int, mass: float,
            gyration_squared: float = BUS_GYRATION_SQUARED) -> None:
    """Replace the base link's mass and inertia with a spacecraft bus."""
    p.changeDynamics(body, -1, mass=mass,
                     localInertiaDiagonal=[mass * gyration_squared] * 3)


# ------------------------------------------------------------------- momentum

def _world_inertia(diagonal, orientation) -> np.ndarray:
    rotation = np.array(p.getMatrixFromQuaternion(orientation)).reshape(3, 3)
    return rotation @ np.diag(diagonal) @ rotation.T


def total_momentum(body: int):
    """(linear, angular) momentum about the world origin, with a scale for
    each so the residual can be reported as a fraction of what is moving.

    Both should be zero throughout a free-flight maneuver: the motors act
    between links, so their torques are internal.
    """
    linear = np.zeros(3)
    angular = np.zeros(3)
    linear_scale = 0.0
    angular_scale = 0.0

    position, orientation = p.getBasePositionAndOrientation(body)
    velocity, omega = p.getBaseVelocity(body)
    info = p.getDynamicsInfo(body, -1)
    position, velocity, omega = (np.array(position), np.array(velocity),
                                 np.array(omega))
    inertia = _world_inertia(info[2], orientation)
    linear += info[0] * velocity
    angular += info[0] * np.cross(position, velocity) + inertia @ omega
    linear_scale += info[0] * np.linalg.norm(velocity)
    angular_scale += np.linalg.norm(inertia @ omega)

    for link in range(p.getNumJoints(body)):
        info = p.getDynamicsInfo(body, link)
        if info[0] == 0.0:
            continue
        state = p.getLinkState(body, link, computeLinkVelocity=1)
        centre, velocity, omega = (np.array(state[0]), np.array(state[6]),
                                   np.array(state[7]))
        inertia = _world_inertia(info[2], state[1])
        linear += info[0] * velocity
        angular += info[0] * np.cross(centre, velocity) + inertia @ omega
        linear_scale += info[0] * np.linalg.norm(velocity)
        angular_scale += (np.linalg.norm(inertia @ omega)
                          + info[0] * np.linalg.norm(np.cross(centre, velocity)))

    return linear, angular, max(linear_scale, 1e-12), max(angular_scale, 1e-12)


def momentum_residual(body: int) -> float:
    """Worse of the two momentum residuals, as a fraction of its scale."""
    linear, angular, linear_scale, angular_scale = total_momentum(body)
    return max(np.linalg.norm(linear) / linear_scale,
               np.linalg.norm(angular) / angular_scale)


# ----------------------------------------------------------------- simulation

def _reset(body: int, angles: Sequence[float], arm_joints: Sequence[int],
           position=(0.0, 0.0, 0.0)) -> None:
    p.resetBasePositionAndOrientation(body, list(position), [0, 0, 0, 1])
    p.resetBaseVelocity(body, [0, 0, 0], [0, 0, 0])
    for joint, angle in zip(arm_joints, angles):
        p.resetJointState(body, joint, float(angle), targetVelocity=0.0)
    for joint in FINGER_JOINTS:
        p.resetJointState(body, joint, 0.0, targetVelocity=0.0)


def _hold_fingers(body: int) -> None:
    p.setJointMotorControlArray(body, list(FINGER_JOINTS), p.POSITION_CONTROL,
                                targetPositions=[0.0, 0.0],
                                forces=[1.0e3, 1.0e3])


def simulate_loop(body: int, loop: JointLoop, dt: float,
                  direction: int = 1, revolutions: int = 1,
                  watch_momentum: bool = False,
                  base_position=(0.0, 0.0, 0.0)) -> dict:
    """Track the loop with position control on a free base.

    Position control rather than a kinematic reset: resetJointState
    teleports and applies no force, so the base would never react. Motor
    torques are internal, so total momentum stays at zero and this is the
    zero-momentum case the analytic model describes.
    """
    _reset(body, loop.home, loop.arm_joints, base_position)
    p.setTimeStep(dt)
    steps = int(round(revolutions * loop.period / dt))
    phase_rate = direction * 2.0 * np.pi / loop.period
    worst_momentum = 0.0

    for step in range(1, steps + 1):
        phase = phase_rate * (step * dt)
        p.setJointMotorControlArray(
            body, list(loop.arm_joints), p.POSITION_CONTROL,
            targetPositions=list(loop.angles(phase)),
            targetVelocities=list(loop.rates(phase, phase_rate)),
            forces=[5.0e3] * len(loop.arm_joints),
            positionGains=[1.0] * len(loop.arm_joints))
        _hold_fingers(body)
        p.stepSimulation()
        if watch_momentum and step % 25 == 0:
            worst_momentum = max(worst_momentum, momentum_residual(body))

    final = np.array([p.getJointState(body, j)[0] for j in loop.arm_joints])
    orientation = p.getBasePositionAndOrientation(body)[1]
    rotation = np.array(p.getMatrixFromQuaternion(orientation)).reshape(3, 3)
    return {
        "rotation": rotation,
        "angle": rotation_angle(rotation),
        "vector": rotation_vector(rotation),
        "closure": float(np.abs(final - loop.home).max()),
        "momentum": worst_momentum,
    }


def simulate_held_rates(body: int, angles: np.ndarray, rates: np.ndarray,
                        dt: float = 1e-4, steps: int = 12,
                        arm_joints: Sequence[int] = ARM_JOINTS,
                        end_effector: int = END_EFFECTOR,
                        base_position=(0.0, 0.0, 0.0)) -> dict:
    """Hold the arm at a fixed set of joint rates and read what moves.

    The instantaneous case the generalized Jacobian describes: joints
    turning at a known rate, base free, total momentum zero.
    """
    _reset(body, angles, arm_joints, base_position)
    p.setJointMotorControlArray(body, list(arm_joints), p.VELOCITY_CONTROL,
                                targetVelocities=list(rates),
                                forces=[1.0e3] * len(arm_joints))
    p.setJointMotorControlArray(body, list(FINGER_JOINTS), p.VELOCITY_CONTROL,
                                targetVelocities=[0.0, 0.0],
                                forces=[1.0e3, 1.0e3])
    p.setTimeStep(dt)
    for _ in range(steps):
        p.stepSimulation()

    states = p.getJointStates(body, list(arm_joints))
    link = p.getLinkState(body, end_effector, computeLinkVelocity=1,
                          computeForwardKinematics=1)
    base_linear, base_angular = p.getBaseVelocity(body)
    return {
        "q": np.array([s[0] for s in states]),
        "qdot": np.array([s[1] for s in states]),
        "ee_twist": np.concatenate([np.array(link[6]), np.array(link[7])]),
        "base_twist": np.concatenate([np.array(base_linear),
                                      np.array(base_angular)]),
    }


# --------------------------------------------------------- analytic integration

def integrate_base_rotation(model, loop: JointLoop, samples: int,
                            frame: str = "body",
                            dtype: torch.dtype = torch.float64) -> np.ndarray:
    """Net base rotation from integrating omega_b once around the loop.

    omega_b = [-H_b^-1 H_bm qdot]_ang, RK4 in the loop phase. `frame` is
    "body" for the correct Rdot = R skew(omega); "world" exists only so
    the test suite can show it gives the wrong answer.
    """
    step = 2.0 * np.pi / samples
    scale = loop.period / (2.0 * np.pi)   # omega per unit phase, not per second

    def omega(phase: float) -> np.ndarray:
        twist = model.base_velocity(
            torch.tensor(loop.angles(phase), dtype=dtype),
            torch.tensor(loop.rates(phase, 2.0 * np.pi / loop.period),
                         dtype=dtype))
        return twist[3:].numpy() * scale

    rotation = np.eye(3)
    for index in range(samples):
        phase = index * step
        increment = (omega(phase) + 4 * omega(phase + step / 2)
                     + omega(phase + step)) * step / 6
        turn = rotation_exponential(increment)
        rotation = rotation @ turn if frame == "body" else turn @ rotation
    return rotation
