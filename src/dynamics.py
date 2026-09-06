"""Floating-base mass matrix, coupling inertia, and the generalized Jacobian.

Phases 2a and 2b. With zero initial momentum and no external force, the
base twist is fixed at every instant by the joint rates:

    H_b v_b + H_bm qdot = 0        ->        v_b = -H_b^-1 H_bm qdot

and the end-effector velocity, J_m qdot on a fixed base, picks up the base
reaction:

    xdot = J_m qdot + J_b v_b = (J_m - J_b H_b^-1 H_bm) qdot = J_g qdot

J_g (Umetani and Yoshida, 1989) is what the planner works against. It
depends on the mass distribution, not only the geometry. J_b, H_b and H_bm
must share one base-twist layout ([linear, angular]) and one reference
point (the base link inertial frame); `_assert_base_convention` checks
that numerically, because a swapped layout gives a J_g of the right shape
that is wrong everywhere.

H_b (6x6) and H_bm (6x7) are the leading blocks of the floating-base mass
matrix

    M(q) = sum_i [ m_i Jt_i^T Jt_i + Jr_i^T (R_i I_i R_i^T) Jr_i ]

summed over every link's centre-of-mass Jacobian. The Jacobians are the
geometric form, `[axis x (com - point); axis]` per revolute ancestor,
built from the joint frames `kinematics.py` produces — all torch ops, so
M stays differentiable in q for the trajectory optimiser, at about a
quarter the cost of differentiating the pose through autograd. Nothing
here converts to numpy on the compute path.

Frame conventions, measured not assumed (see CONTRIBUTING.md and
verify_mass_matrix.py):

  - `getDynamicsInfo` field 3/4 is a link's inertial frame relative to its
    URDF link frame. The CoM Jacobian is the Jacobian of that frame; the
    link frame is 40-120 mm away on the wrist links.
  - field 2 is the diagonal inertia in the inertial frame. It enters the
    world as R_i diag(I_i) R_i^T, with R_i the world orientation of the
    inertial frame.
  - the two prismatic fingers are locked at their zero position. Their
    0.1 kg each still rides at its true CoM; only the finger joint columns
    are dropped, which is what leaves H_bm at 6x7. `mass_matrix` also
    offers a `fingers="folded"` mode that lumps the finger mass onto the
    hand, and verify_mass_matrix.py reports what that approximation costs.
  - the reference point for the base 6 DoF is the base link's inertial
    frame ("base_com"), found by measurement: verify_mass_matrix.py sweeps
    the candidates and only that one matches PyBullet. PyBullet also orders
    the base DoF [angular, linear]; this module keeps the project's
    [linear, angular] and the verify script permutes before comparing.
  - PyBullet zeroes the base link's mass and inertia under
    useFixedBase=True and restores the URDF values (2.9 kg, non-trivial
    inertia for the Panda) under useFixedBase=False. Build this model from
    a free-base load, or the base link contributes nothing. The inertia
    audit, run fixed-base, is misleading on link -1 for this reason.
  - `base_mass` and `base_inertia_diagonal` override the base link, which
    is how the bus mass sweep swaps the Panda's 2.9 kg pedestal for a
    real spacecraft bus.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence

import pybullet as p
import torch

from src.kinematics import ForwardKinematics, quaternion_to_matrix, transform


def _rpy_to_matrix(rpy, dtype: torch.dtype) -> torch.Tensor:
    """Rotation matrix from URDF roll pitch yaw, applied Z then Y then X."""
    roll, pitch, yaw = (torch.tensor(float(v), dtype=dtype) for v in rpy)
    def axis(c, s, which):
        one = torch.ones((), dtype=dtype)
        zero = torch.zeros((), dtype=dtype)
        if which == "x":
            rows = [[one, zero, zero], [zero, c, -s], [zero, s, c]]
        elif which == "y":
            rows = [[c, zero, s], [zero, one, zero], [-s, zero, c]]
        else:
            rows = [[c, -s, zero], [s, c, zero], [zero, zero, one]]
        return torch.stack([torch.stack(r) for r in rows])
    return (axis(torch.cos(yaw), torch.sin(yaw), "z")
            @ axis(torch.cos(pitch), torch.sin(pitch), "y")
            @ axis(torch.cos(roll), torch.sin(roll), "x"))


def skew(vector: torch.Tensor) -> torch.Tensor:
    """The 3x3 matrix S with S w = vector x w."""
    x, y, z = vector
    zero = torch.zeros((), dtype=vector.dtype)
    return torch.stack([
        torch.stack([zero, -z, y]),
        torch.stack([z, zero, -x]),
        torch.stack([-y, x, zero]),
    ])


@dataclass
class Link:
    """One link's fixed inertial data plus a chain that reaches it.

    Attributes:
        index: PyBullet link index, -1 for the base link.
        name: link name, for the per-link error tables.
        mass: link mass in kg, `getDynamicsInfo` field 0.
        inertia: the full symmetric inertia tensor about the centre of
            mass, expressed in the inertial frame. PyBullet reports
            principal moments and a rotation that diagonalises them, so
            reading it back through `getDynamicsInfo` always yields a
            diagonal here; a URDF that declares off-diagonal terms does
            not, which is why this is a matrix rather than three numbers.
        inertial_transform: 4x4, the link frame -> inertial frame transform,
            constant, from `getDynamicsInfo` fields 3 and 4.
        kinematics: forward kinematics to this link's URDF frame as a
            function of the arm joints, or None for the base link.
        arm_ancestors: the arm joints that actually move this link, in
            order. A proximal link is moved by only the first few, and the
            Jacobian columns for the rest must be zero, not the geometric
            formula's non-zero garbage.
    """

    index: int
    name: str
    mass: float
    inertia: torch.Tensor
    inertial_transform: torch.Tensor
    kinematics: Optional[ForwardKinematics]
    arm_ancestors: tuple


class FloatingBaseModel:
    """Floating-base mass matrix and coupling inertia for a serial arm.

    The arm joints are the generalized coordinates; the base is free. Every
    link on the body contributes, including the ones whose joints are not
    arm joints (the locked fingers), because their mass still moves when
    the arm moves.
    """

    # Panda-specific: prismatic fingers on link 8 (the hand). Used only by
    # the fingers="folded" approximation.
    _FINGER_LINKS = (9, 10)
    _HAND_LINK = 8

    def __init__(self, body: int, arm_joints: Sequence[int],
                 links: Optional[Sequence[int]] = None,
                 end_effector: Optional[int] = None,
                 base_reference: str = "base_com",
                 base_mass: Optional[float] = None,
                 base_inertia_diagonal: Optional[Sequence[float]] = None,
                 base_inertia: Optional[Sequence[Sequence[float]]] = None,
                 inertials: Optional[dict] = None,
                 urdf: Optional[str] = None,
                 dtype: torch.dtype = torch.float64):
        """
        Args:
            body: a PyBullet body id, already loaded. Load it with
                useFixedBase=False, or the base link reads as massless.
            arm_joints: the joints treated as generalized coordinates.
            links: link indices to sum over. Defaults to every link on the
                body, base link included.
            end_effector: the link the generalized Jacobian is written for.
                Defaults to the last link (the grasp target on the Panda).
            base_reference: which point the base twist is written about.
                "base_com" matches PyBullet and is the default; verify_
                mass_matrix.py shows "base_link_origin" and "system_com"
                do not.
            base_mass: if given, replaces the base link's mass. This is the
                bus mass for a spacecraft, swept from tens to thousands of
                kilograms while the Panda arm stays fixed.
            base_inertia_diagonal: if given, replaces the base link's
                principal moments (3 values, about its inertial frame).
                Equivalent to passing a diagonal `base_inertia`.
            base_inertia: if given, replaces the base link's full inertia
                tensor (3x3, about its inertial frame). Takes precedence
                over `base_inertia_diagonal`.
            inertials: if given, {link name: (mass, com xyz, com rpy,
                3x3 tensor)} read from the URDF, used in place of
                `getDynamicsInfo`. PyBullet is not a faithful reader of
                declared inertias: it substitutes a unit mass and inertia
                for a link with no inertial block, and it diagonalises the
                tensor, which costs about 6e-9. Both matter once the model
                carries real identified parameters.
            dtype: float64. The check is against a C++ mass matrix and
                float32 would sit near the tolerance.
        """
        self.body = body
        self.dtype = dtype
        self.arm_joints = list(arm_joints)
        self.n = len(self.arm_joints)
        self.base_reference = base_reference
        self._base_mass = base_mass
        self._base_inertia_diagonal = base_inertia_diagonal
        self._base_inertia = base_inertia
        self._inertials = inertials
        self.urdf = urdf

        # The geometric Jacobian columns assume every arm joint is a
        # revolute ancestor of every link on the arm (a plain serial chain).
        for joint in self.arm_joints:
            if p.getJointInfo(body, joint)[2] != p.JOINT_REVOLUTE:
                raise ValueError(
                    f"arm joint {joint} is not revolute; the geometric "
                    "Jacobian in this module is revolute-serial only")

        if end_effector is None:
            end_effector = p.getNumJoints(body) - 1
        self.end_effector = end_effector
        self._end_effector_kinematics = ForwardKinematics(
            body, end_effector, movable_joints=self.arm_joints, urdf=urdf,
            dtype=dtype)
        self._end_effector_ancestors = self._arm_ancestors(
            self._end_effector_kinematics)

        if links is None:
            links = list(range(-1, p.getNumJoints(body)))
        self.links: List[Link] = [self._read_link(index) for index in links]

    def _read_link(self, index: int) -> Link:
        info = p.getDynamicsInfo(self.body, index)
        inertial_transform = transform(
            quaternion_to_matrix(torch.tensor(info[4], dtype=self.dtype)),
            torch.tensor(info[3], dtype=self.dtype),
        )
        mass = info[0]
        inertia = torch.diag(torch.tensor(info[2], dtype=self.dtype))

        link_name = (p.getBodyInfo(self.body)[0].decode() if index == -1
                     else p.getJointInfo(self.body, index)[12].decode())
        if self._inertials is not None and link_name in self._inertials:
            declared_mass, xyz, rpy, tensor = self._inertials[link_name]
            mass = float(declared_mass)
            inertia = torch.tensor(tensor, dtype=self.dtype)
            inertial_transform = transform(
                _rpy_to_matrix(rpy, self.dtype),
                torch.tensor(xyz, dtype=self.dtype))

        if index == -1:
            name = p.getBodyInfo(self.body)[0].decode()
            kinematics = None
            if self._base_mass is not None:
                mass = self._base_mass
            if self._base_inertia_diagonal is not None:
                inertia = torch.diag(torch.tensor(self._base_inertia_diagonal,
                                                  dtype=self.dtype))
            if self._base_inertia is not None:
                inertia = torch.tensor(self._base_inertia, dtype=self.dtype)
        else:
            name = p.getJointInfo(self.body, index)[12].decode()
            kinematics = ForwardKinematics(self.body, index,
                                           movable_joints=self.arm_joints,
                                           urdf=self.urdf, dtype=self.dtype)
        return Link(
            index=index,
            name=name,
            mass=mass,
            inertia=inertia,
            inertial_transform=inertial_transform,
            kinematics=kinematics,
            arm_ancestors=self._arm_ancestors(kinematics),
        )

    def _arm_ancestors(self, kinematics: Optional[ForwardKinematics]) -> tuple:
        """Which arm joints lie on the chain to this link, in order."""
        if kinematics is None:
            return ()
        on_chain = {joint.index for joint in kinematics.joints if joint.movable}
        return tuple(joint for joint in self.arm_joints if joint in on_chain)

    # ------------------------------------------------------------------ poses

    def com_pose(self, link: Link, q: torch.Tensor) -> torch.Tensor:
        """4x4 world pose of a link's inertial frame."""
        if link.kinematics is None:
            return link.inertial_transform
        return link.kinematics(q) @ link.inertial_transform

    def com_frame(self, link: Link, q: torch.Tensor):
        """(position (3,), rotation (3,3)) of a link's inertial frame."""
        pose = self.com_pose(link, q)
        return pose[:3, 3], pose[:3, :3]

    def system_com(self, q: torch.Tensor) -> torch.Tensor:
        """Whole-body centre of mass at configuration q, world frame."""
        total = torch.zeros(3, dtype=self.dtype)
        mass = 0.0
        for link in self.links:
            if link.mass == 0.0:
                continue
            position, _ = self.com_frame(link, q)
            total = total + link.mass * position
            mass += link.mass
        return total / mass

    def _reference_point(self, q: torch.Tensor) -> torch.Tensor:
        if self.base_reference == "base_link_origin":
            return torch.zeros(3, dtype=self.dtype)
        if self.base_reference == "base_com":
            return self.links[0].inertial_transform[:3, 3]
        if self.base_reference == "system_com":
            return self.system_com(q)
        raise ValueError(f"unknown base_reference {self.base_reference!r}")

    # -------------------------------------------------------------- Jacobians

    def _twist_transport(self, point: torch.Tensor,
                         reference: torch.Tensor) -> torch.Tensor:
        """6x6 mapping a base twist at `reference` to the twist at `point`.

        [v; w] at reference  ->  [v + w x (point - reference); w] at point,
        in [linear, angular] block order. This is the base-velocity block of
        every link's velocity Jacobian, and evaluated at the end-effector it
        is J_b itself, so routing both through here keeps their convention
        identical by construction.
        """
        lever = point - reference
        eye3 = torch.eye(3, dtype=self.dtype)
        zero3 = torch.zeros((3, 3), dtype=self.dtype)
        return torch.cat([
            torch.cat([eye3, -skew(lever)], dim=1),
            torch.cat([zero3, eye3], dim=1),
        ], dim=0)

    def _arm_joint_frames(self, q: torch.Tensor) -> dict:
        """{joint index: (world point, world axis)} for the arm joints.

        One pass down the end-effector chain, shared by every link's
        Jacobian since every link is distal to all of them.
        """
        frames, _ = self._end_effector_kinematics.joint_frames(q)
        return frames

    def _geometric_columns(self, target: torch.Tensor, joint_frames: dict,
                           ancestors: tuple):
        """(linear (3, n), angular (3, n)) for a point rigidly fixed to a
        link, base held fixed.

        For a revolute ancestor the column is [axis x (target - point);
        axis]; for an arm joint that is not an ancestor the link does not
        move, so the column is zero.
        """
        zero = torch.zeros(3, dtype=self.dtype)
        linear = []
        angular = []
        for joint in self.arm_joints:
            if joint in ancestors:
                point, axis = joint_frames[joint]
                linear.append(torch.linalg.cross(axis, target - point))
                angular.append(axis)
            else:
                linear.append(zero)
                angular.append(zero)
        return torch.stack(linear, dim=1), torch.stack(angular, dim=1)

    def link_jacobian(self, link: Link, q: torch.Tensor):
        """(Jt, Jr), each (3, n): world linear and angular velocity of the
        link's inertial frame per arm joint rate, base held fixed."""
        if link.kinematics is None:
            zero = torch.zeros((3, self.n), dtype=self.dtype)
            return zero, zero
        position = self.com_pose(link, q)[:3, 3]
        return self._geometric_columns(position, self._arm_joint_frames(q),
                                       link.arm_ancestors)

    def manipulator_jacobian(self, q: torch.Tensor) -> torch.Tensor:
        """J_m, (6, n): the fixed-base end-effector Jacobian, [linear; angular]."""
        frames, end_pose = self._end_effector_kinematics.joint_frames(q)
        linear, angular = self._geometric_columns(
            end_pose[:3, 3], frames, self._end_effector_ancestors)
        return torch.cat([linear, angular], dim=0)

    def base_jacobian(self, q: torch.Tensor) -> torch.Tensor:
        """J_b, (6, 6): end-effector twist produced by a base twist, joints
        fixed. Same [linear, angular] layout and reference point as H_b."""
        end_effector = self._end_effector_kinematics(q)[:3, 3]
        return self._twist_transport(end_effector, self._reference_point(q))

    # ------------------------------------------------------------ mass matrix

    def _effective_mass(self, link: Link, fingers: str) -> float:
        if fingers == "exact":
            return link.mass
        if fingers == "folded":
            if link.index in self._FINGER_LINKS:
                return 0.0
            if link.index == self._HAND_LINK:
                return link.mass + sum(
                    other.mass for other in self.links
                    if other.index in self._FINGER_LINKS)
            return link.mass
        raise ValueError(f"unknown fingers mode {fingers!r}")

    def mass_matrix(self, q: torch.Tensor,
                    fingers: str = "exact") -> torch.Tensor:
        """The (6 + n) x (6 + n) floating-base mass matrix at q.

        Args:
            q: (n,) arm joint angles.
            fingers: "exact" keeps each finger's mass at its own CoM;
                "folded" lumps it onto the hand (a stated approximation).
        """
        size = 6 + self.n
        matrix = torch.zeros((size, size), dtype=self.dtype)
        reference = self._reference_point(q)
        joint_frames = self._arm_joint_frames(q)

        for link in self.links:
            mass = self._effective_mass(link, fingers)
            if mass == 0.0 and link.mass == 0.0:
                continue

            position, rotation = self.com_frame(link, q)
            linear_arm, angular_arm = self._geometric_columns(
                position, joint_frames, link.arm_ancestors)

            # Full link CoM Jacobian: base block from _twist_transport (the
            # same map J_b uses), arm block from autograd.
            transport = self._twist_transport(position, reference)
            jacobian = torch.cat(
                [transport, torch.cat([linear_arm, angular_arm], dim=0)], dim=1)
            linear, angular = jacobian[:3], jacobian[3:]

            inertia_world = rotation @ link.inertia @ rotation.T

            matrix = (matrix
                      + mass * linear.T @ linear
                      + angular.T @ inertia_world @ angular)
        return matrix

    def coupling(self, q: torch.Tensor, fingers: str = "exact"):
        """(H_b (6x6), H_bm (6xn)) at q."""
        matrix = self.mass_matrix(q, fingers=fingers)
        return matrix[:6, :6], matrix[:6, 6:]

    def base_velocity(self, q: torch.Tensor, joint_rates: torch.Tensor,
                      fingers: str = "exact") -> torch.Tensor:
        """Base twist implied by joint rates at zero total momentum:
        v_b = -H_b^-1 H_bm qdot, in [linear, angular] about the base
        reference point."""
        base_inertia, coupling = self.coupling(q, fingers=fingers)
        return -torch.linalg.solve(base_inertia, coupling @ joint_rates)

    # -------------------------------------------------- generalized Jacobian

    def _assert_base_convention(self, q: torch.Tensor, fingers: str) -> None:
        """Guard the one mistake that makes J_g the right shape and wrong
        everywhere: J_b and H_b disagreeing on the base-twist layout.

        Both must be [linear, angular] about the same point. Checked by
        value, not by reading the code:

          - H_b's leading 3x3 block is the translational inertia, exactly
            total_mass . I. That holds only in [linear, angular] order; in
            [angular, linear] order it would be the rotational inertia.
          - J_b's leading 3x3 block is I and its lower-left block is 0: a
            pure base translation carries the end-effector one for one and
            adds no rotation. Again only true in [linear, angular] order.
        """
        if self.base_reference != "base_com":
            raise ValueError(
                "generalized_jacobian needs base_reference='base_com' to "
                f"match the mass matrix; got {self.base_reference!r}")

        eye3 = torch.eye(3, dtype=self.dtype)
        zero3 = torch.zeros((3, 3), dtype=self.dtype)
        total_mass = sum(self._effective_mass(link, fingers)
                         for link in self.links)

        base_inertia, _ = self.coupling(q, fingers=fingers)
        if not torch.allclose(base_inertia[:3, :3], total_mass * eye3,
                              atol=1e-6):
            raise AssertionError(
                "H_b leading block is not total_mass.I; the base DoF are not "
                "in the [linear, angular] order J_b assumes")

        base_jacobian = self.base_jacobian(q)
        if not torch.allclose(base_jacobian[:3, :3], eye3, atol=1e-9):
            raise AssertionError(
                "J_b leading block is not I; base-twist order disagrees "
                "with H_b")
        if not torch.allclose(base_jacobian[3:, :3], zero3, atol=1e-9):
            raise AssertionError(
                "J_b maps base translation into end-effector rotation; "
                "base-twist order is wrong")

    def generalized_jacobian(self, q: torch.Tensor,
                             fingers: str = "exact") -> torch.Tensor:
        """J_g = J_m - J_b H_b^-1 H_bm, (6, n), [linear; angular].

        The map from joint rates to end-effector velocity on the free
        floating base at zero total momentum.
        """
        self._assert_base_convention(q, fingers)
        manipulator = self.manipulator_jacobian(q)
        base = self.base_jacobian(q)
        base_inertia, coupling = self.coupling(q, fingers=fingers)
        return manipulator - base @ torch.linalg.solve(base_inertia, coupling)
