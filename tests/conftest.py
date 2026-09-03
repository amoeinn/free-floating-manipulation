"""Shared PyBullet setup for the invariant tests.

One DIRECT client and two bodies for the whole session: connecting and
loading the URDF costs 150 ms and nothing in these tests needs a private
world. The free body is the one that gets stepped, so it is the one whose
damping is zeroed; the fixed body only ever answers calculateJacobian and
calculateMassMatrix.

Both sit at the origin, because `ForwardKinematics` builds its chain from
the base frame outward and only agrees with `getLinkState` for a body
whose base is at the identity. Two overlapping Pandas would collide and
fling the free one off at 100 m/s, so the fixed body's collision mask is
cleared: it is an oracle, not a physical object.

Anything a test mutates on a shared body (the bus mass, above all) is put
back by the autouse `pristine_bus` fixture, so test order cannot matter.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pybullet as p
import pybullet_data
import pytest
import torch

from src.dynamics import FloatingBaseModel
from src.freeflight import ARM_JOINTS, JointLoop, disable_damping

DTYPE = torch.float64
URDF = "franka_panda/panda.urdf"


@pytest.fixture(scope="session")
def physics():
    """A zero-gravity world with no ground plane.

    Both matter: a single contact leaks momentum and the conservation law
    under test stops holding.
    """
    client = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, 0)
    yield client
    p.disconnect()


@pytest.fixture(scope="session")
def panda_fixed(physics):
    """A fixed-base Panda at the origin, for the PyBullet reference calls.

    Collision cleared: it shares the origin with the free body so that
    both agree with ForwardKinematics, and it must not push it around.
    """
    body = p.loadURDF(URDF, useFixedBase=True, basePosition=[0, 0, 0])
    for link in range(-1, p.getNumJoints(body)):
        p.setCollisionFilterGroupMask(body, link, 0, 0)
    return body


@pytest.fixture(scope="session")
def panda_free(physics, panda_fixed):
    """A free-base Panda, damping zeroed, ready for free flight."""
    body = p.loadURDF(URDF, useFixedBase=False, basePosition=[0, 0, 0])
    disable_damping(body)
    return body


@pytest.fixture(scope="session")
def urdf_bus(panda_free):
    """The base link's as-loaded mass and inertia, to restore after a sweep."""
    info = p.getDynamicsInfo(panda_free, -1)
    return info[0], list(info[2])


@pytest.fixture(autouse=True)
def pristine_bus(request, urdf_bus):
    """Put the base link back after any test that swaps in a bus."""
    yield
    body = request.getfixturevalue("panda_free")
    mass, inertia = urdf_bus
    p.changeDynamics(body, -1, mass=mass, localInertiaDiagonal=inertia)


@pytest.fixture
def model(panda_free):
    return FloatingBaseModel(panda_free, ARM_JOINTS, dtype=DTYPE)


@pytest.fixture(scope="session")
def loop():
    return JointLoop()


@pytest.fixture(scope="session")
def configurations(panda_fixed):
    """Random joint angles, backed off the hard stops so a velocity-driven
    step is not fighting a limit."""
    rng = np.random.default_rng(0)
    limits = [(p.getJointInfo(panda_fixed, j)[8],
               p.getJointInfo(panda_fixed, j)[9]) for j in ARM_JOINTS]
    return np.array([[rng.uniform(0.8 * lo, 0.8 * hi) for lo, hi in limits]
                     for _ in range(6)])
