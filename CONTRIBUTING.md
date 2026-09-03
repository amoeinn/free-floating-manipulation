# Contributing

Manipulation planning for a free-flying satellite servicer, where arm motion
reacts on the spacecraft base. The layout:

- `src/kinematics.py` — differentiable forward kinematics for a serial chain.
- `src/dynamics.py` — floating-base mass matrix, coupling inertia, generalized
  Jacobian. Its module docstring is the reference for the frame conventions
  the rest of the code depends on.
- `src/freeflight.py` — driving a free-flying body and measuring what its base
  does: the joint loop, the zero-gravity run, the momentum readout.
- `examples/` — one script per result. Each prints a per-item table and
  re-derives its own figures into `docs/`.
- `tests/` — the invariants, one case per invariant.

## Environment

- Python 3 with a virtualenv at `.venv/`. Activate it before running anything,
  and leave `PYTHONPATH` unset — the test suite and the example scripts both
  put the repository root on `sys.path` themselves.
- torch 2.14.0+cpu, pybullet 3.2.7, numpy 2.5.2, matplotlib 3.11.1.
  Runtime dependencies are in `requirements.txt`; pytest is in
  `requirements-dev.txt`.
- CPU only. There is no CUDA in the development environment, so cuRobo and
  Isaac Lab are unavailable and everything must run without them.

## Running things

- `pytest` runs the invariant suite. The two parameter sweeps are marked `slow`
  and left out of the default run; `pytest -m slow` runs those and
  `pytest -m "slow or not slow"` runs everything.
- The scripts in `examples/` are the reports behind the numbers in the module
  docstrings. Run one and read its table before trusting a change to the code
  it covers.

## Working rules

- **Write the measurement script before the fix.** If a value is wrong, print
  what is actually happening — per link, per pair, per step — rather than
  reasoning about what should be happening. Reasoning first has not resolved a
  single non-trivial bug here; measuring first has resolved all of them.
- **Prefer a per-item error table to an aggregate error.** "Exact through link
  1, then 632 mm out at link 2" points at a specific composition. "Mean error
  0.4 m" points at nothing. Localisation is the whole value.
- **A result better than a known bound is a defect report, not a success.**
  Investigate it before celebrating it.
- **Verify against exact geometry, never against a learned or approximate
  model.** Approximations belong in cost terms; ground truth decides validity.
  Where an approximation is used deliberately, measure what it costs and say so.
- **Every test names the bug or invariant it protects**, in its name and in its
  docstring. Do not add tests to raise a coverage number. A test that cannot
  fail for a reason you can state is not protecting anything — check it by
  breaking the thing it guards and confirming it is the case that fails.
- **State limitations in documentation rather than tuning a scenario until a
  method looks good.** Where a method loses is part of the result.

## PyBullet frame conventions

Assume nothing about which frame a PyBullet call reports in. Every item below
was settled by measurement, and every one of them returns a plausible number
when you guess wrong, so a mistake here surfaces as a subtly bad result rather
than an exception. Check a new call against a finite difference or against
`getLinkState` before building on it.

**Link and joint frames**

- `getJointInfo` fields 14/15 give a joint's origin relative to the parent
  link's *inertial* frame; `getLinkState` field 4 gives the *link* frame. They
  differ by the parent's local inertial offset, `getDynamicsInfo` field 3 — on
  the Panda base, 50 mm in z. Compose that offset forward, do not invert it.
- `getJointInfo` field 15 is the rotation from the joint frame to the parent
  frame, the inverse of what a forward chain composes, so transpose it. The
  symptom of omitting it is a chain that is exact through link 1 and then
  hundreds of millimetres out at link 2.
- Joint axes reading `(0,0,1)` on every joint is correct, not a bug. The origin
  quaternions rotate each joint frame so its axis lies along z, which is why
  they alternate between plus and minus 0.7071 about x.
- `getBasePositionAndOrientation` returns the base link's *inertial* frame, not
  its link frame — the same class of error as the 50 mm offset above, and it
  will silently corrupt any comparison between analytic and simulated base
  motion.
- `calculateJacobian`'s `localPosition` is in the link (URDF) frame, as its
  docstring says. A centre-of-mass Jacobian therefore needs
  `localPosition = getDynamicsInfo field 3`, not `[0,0,0]`; `[0,0,0]` gives the
  link-frame origin's Jacobian, which on wrist links is over 100 mm of lever
  arm away.

**Mass, inertia and the free base**

- `useFixedBase=True` zeroes the base link's mass and inertia; `useFixedBase=
  False` restores the URDF values. Audit inertias from a free-base load, and
  build any floating-base model from one, or the base link silently contributes
  nothing.
- `calculateMassMatrix` on a free base returns `(6+m)x(6+m)` for `m` movable
  joints, and sizes the matrix from the model's own DoF count — it pads or
  truncates the position argument, so the argument length does not change the
  shape.
- That free-base matrix orders the base six as **[angular, linear]** and writes
  the base twist about the **base link inertial frame** — not the link-frame
  origin, and not the system centre of mass. This project uses [linear,
  angular] internally and permutes at the comparison. If you introduce a new
  base-referenced quantity, sweep the candidate reference points and orderings
  rather than picking one: the wrong choices are off by O(1) to O(10), which is
  large enough to see and small enough to rationalise.

**Free-flight simulation**

- Load no ground plane and set gravity to zero. A single contact leaks momentum
  and the conservation law being validated stops holding.
- **Zero the damping.** Every link gets 0.04 linear and 0.04 angular damping by
  default, and `getDynamicsInfo` fields 8/9 report `-1.0` — meaning "unset" —
  rather than the value in force, so it cannot be read back and is invisible
  unless you know to look. Damping is an external force: left in, it leaks
  momentum and biases a closed-loop base rotation by a few percent. Call
  `changeDynamics(body, link, linearDamping=0, angularDamping=0)` on every
  link including `-1`, and use `src/freeflight.py`'s `disable_damping`.
- Drive joints with a motor, never with `resetJointState`. A reset teleports
  and applies no force, so the base never reacts; motor torques are internal,
  so total momentum stays at zero and the run is the zero-momentum case the
  analytic model describes.
- The base twist from `-H_b^-1 H_bm qdot` is in the **base body frame**, so it
  integrates as `Rdot = R skew(omega)`. The world-frame form
  `Rdot = skew(omega) R` is the natural thing to write and is wrong by nearly
  an order of magnitude, while still producing a plausible-looking angle.

## Finishing a phase

Write the phase's definition of done before starting it, and make every line
of it something with a number attached. A phase is finished when:

- The module it promised exists, and is differentiable end to end wherever a
  later optimiser will need a gradient through it.
- Every analytic result is checked against exact ground truth — a PyBullet
  reference call, a finite difference, or a simulation — with **both the
  tolerance and the number of trials stated**. "Agrees well" is not a
  definition of done; "to within X relative, over N random configurations" is.
- Where no reference call exists for a quantity, it is validated at least two
  ways that do not share a failure mode: against a simulation, and against a
  limit where it must collapse to something already verified.
- Any headline claim is a measured, reported number or figure rather than a
  description of one.
- Where a result depends on a modelling choice, the alternative was measured
  and its cost written down, so the choice is a stated approximation and not an
  assumption.
- Tests cover the invariants by name, and each was mutation-checked: break the
  thing it guards and confirm that case is the one that fails.
- Limitations are in the documentation. Where the method loses is part of the
  result.

## Commits

Keep them small and local. A commit message states what was *verified*, not
only what was added: the tolerance, the number of configurations, the reference
the result was checked against.
