"""Does the Gazebo scene render the motion phase 2 verified?

The scene's whole justification is that Gazebo is a renderer and the dynamics
stay in code that has been checked. That is only true if the integration the
scene node performs actually reproduces the verified result, and the node
integrates at its publish rate rather than at the timestep the verification
used, so the two are not identical by construction.

`verify_nonholonomy.py` part 4 integrates the analytic base twist around one
closed joint space loop and gets 18.9823 degrees at 400 samples, against a
simulation that gives 18.9808. This repeats that integration at the rates the
scene node might publish at, so the cost of the publish rate is a measured
number rather than an assumption.
"""

import sys
from pathlib import Path

import numpy as np
import pybullet as p
import pybullet_data
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import freeflight
from src.freeflight import JointLoop, rotation_angle, rotation_exponential

REFERENCE_DEG = 18.9823          # verify_nonholonomy.py part 4, body frame, 400 samples
NODE_RATE_HZ = 20.0


def net_rotation(model, loop, hz):
    dt = 1.0 / hz
    R = np.eye(3)
    for k in range(int(round(loop.period * hz))):
        phase = 2 * np.pi * (k * dt) / loop.period
        q = loop.angles(phase)
        qd = loop.rates(phase, 2 * np.pi / loop.period)
        twist = model.base_velocity(torch.as_tensor(q, dtype=torch.float64),
                                    torch.as_tensor(qd, dtype=torch.float64))
        R = R @ rotation_exponential(twist.detach().numpy()[3:] * dt)
    return np.degrees(rotation_angle(R))


def main():
    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    body = freeflight.load_panda(fixed_base=False)
    freeflight.disable_damping(body)
    model = freeflight.build_model(body)
    loop = JointLoop()

    print("the scene node's own integration, against the phase 2 analytic result")
    print(f"  reference: {REFERENCE_DEG} deg, verify_nonholonomy.py part 4\n")
    print(f"  {'publish rate':>13} {'steps/loop':>11} {'net rotation':>15} {'error':>10}")
    results = {}
    for hz in (5.0, 10.0, NODE_RATE_HZ, 50.0, 200.0, 1000.0):
        value = net_rotation(model, loop, hz)
        results[hz] = value
        mark = "   <- the node" if hz == NODE_RATE_HZ else ""
        print(f"  {hz:10.0f} Hz {int(round(loop.period*hz)):11d} "
              f"{value:12.4f} deg {100*(value-REFERENCE_DEG)/REFERENCE_DEG:8.2f}%{mark}")

    fine = results[1000.0]
    if abs(fine - REFERENCE_DEG) / REFERENCE_DEG > 1e-3:
        raise SystemExit(
            f"refined to {fine:.4f} deg against the verified {REFERENCE_DEG}, "
            "so the scene is not integrating the same dynamics")
    node_error = 100 * abs(results[NODE_RATE_HZ] - REFERENCE_DEG) / REFERENCE_DEG
    print(f"\n  refined, the scene integration lands on {fine:.4f} deg against the "
          f"verified {REFERENCE_DEG}")
    print(f"  at the node's {NODE_RATE_HZ:.0f} Hz the publish rate costs "
          f"{node_error:.2f} percent, which is quadrature and not a modelling "
          f"difference")


if __name__ == "__main__":
    main()
