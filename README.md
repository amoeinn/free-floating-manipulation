# free-floating-manipulation

Manipulation planning for an arm on an uncontrolled free-floating base, with on-orbit satellite servicing as the driving case: a servicer with its thrusters off and its wheels idle, moving its arm toward a target it has not yet grappled. A fixed-base arm pushes against the world for free, because every reaction torque a joint generates is absorbed by the floor. Take the floor away and the reaction has nowhere to go but the spacecraft, so the base translates and rotates in response to every joint motion and base attitude ends up depending on the entire history of arm motion rather than on where the joints currently are.

![A free-floating Panda walks a closed loop in joint space; the joints return to their starting angles and the base ends 10.02 degrees rotated](docs/closed_loop.gif)

Two joints trace a circle in joint space and return to exactly the angles they started at. The base does not return. The consequence is not a correction term: the map from configuration to end-effector pose stops being a function of configuration at all, because two paths that start and end at the same seven joint angles put the gripper in two different places. Every fixed-base planner assumes that map exists. This repository builds the objects that replace it, and then measures how large the effect is that they exist to capture.

## What is here

**Differentiable forward kinematics.** An analytic chain in torch, reading joint origins and axes from the URDF, with the Jacobian through autograd. PyBullet already computes forward kinematics, but it returns a number with no gradient attached, and a Cartesian objective needs to know which way to move.

**The floating-base mass matrix and coupling inertia.** With zero initial momentum and no external wrench, the base twist is not a free variable. It is pinned at every instant by the joint rates:

```
H_b v_b + H_bm qdot = 0        →        v_b = -H_b^-1 H_bm qdot
```

`H_b` (6x6) is the locked inertia of the whole system about the base and `H_bm` (6x7) is the block that couples arm motion into base motion. Both are partitions of the floating-base mass matrix, assembled from every link's centre-of-mass Jacobian:

```
M(q) = sum_i [ m_i Jt_i^T Jt_i + Jr_i^T (R_i I_i R_i^T) Jr_i ]
```

It is built in torch and stays differentiable in `q` throughout, because a trajectory optimiser will need gradients through it.

**The generalized Jacobian.** End-effector velocity on a fixed base is `J_m qdot`. On a free base it picks up the base reaction, and substituting the line above gives the object a planner actually works against (Umetani and Yoshida, 1989):

```
xdot = J_m qdot + J_b v_b = ( J_m - J_b H_b^-1 H_bm ) qdot = J_g qdot
```

**What makes that different from a fixed-base Jacobian.** `J_m` is pure geometry: link lengths and joint axes. `J_g` contains `H_b^-1 H_bm`, so it depends on the mass distribution, and two arms with identical link geometry but different link masses have different generalized Jacobians and reach different places from the same joint trajectory. It is also configuration-dependent in a way that does not integrate. There is no free-floating forward kinematics that `J_g` is the derivative of, and the closed-loop result below is what that absence looks like when you measure it.

Everything is checked against exact references rather than against each other:

| quantity | checked against | agreement |
| --- | --- | --- |
| `H_b`, `H_bm` | PyBullet free-base `calculateMassMatrix` | 4e-15, 10 configurations |
| `H_m`, the arm block | free- and fixed-base `calculateMassMatrix` | 1e-15 |
| link centre-of-mass Jacobians | `calculateJacobian`, per link | 7e-16 |
| `J_m` | `calculateJacobian`, fixed base | 8e-16 |
| `J_g qdot` | zero-g free-floating simulation | 2e-4 to 7e-4 relative |
| `-H_b^-1 H_bm qdot` | measured base twist in simulation | 3e-5 to 1e-4 relative |
| `M(q)` and `J_g` gradients | central differences | 1e-9 |

`J_g` has no PyBullet equivalent to check against, so it is validated two ways that do not share a failure mode: against simulation, and against the large-bus limit where `H_b^-1 H_bm` goes to zero and it must collapse back to `J_m`.

## Results

### A closed loop in joint space leaves the base rotated

Two joints trace a circle of amplitude 0.6 rad over 2 s and return every angle to its starting value. On a fixed base the arm would be exactly where it started. On a 17.96 kg free-floating base it ends 10.02 degrees rotated, about `[-0.091, 0.000, -0.149]`. The base swings out to 64 degrees at the halfway point and comes most of the way back, but not all of it.

That is the whole claim in one number, so the work is in showing it is physics and not the integrator sliding:

| evidence | result |
| --- | --- |
| **Timestep independence** | 10.0642 → 10.0409 → 10.0295 → 10.0238 → 10.0210 degrees as `dt` halves from 2e-3 to 1.25e-4. The increments halve each time, which is first-order convergence on 10.018 degrees rather than drift. Coarse to fine spans 0.43%. |
| **Sign reversal** | Traversing the loop backward gives 10.0190 degrees about the exactly opposite axis, alignment -1.000. Composing the two lands 0.0058 degrees from the identity, against 10.02 degrees for one loop. Drift accumulates the same way in either direction; a geometric phase cancels. |
| **Independent integration** | RK4 integration of `omega_b = [-H_b^-1 H_bm qdot]_ang` around the same path gives 10.0191 degrees, 0.02% from the simulation, which shares no code with it. |
| **Momentum conservation** | Peak total momentum along the loop is 9.7e-3 of its own scale at `dt` = 1e-3 and 2.4e-3 at 2.5e-4, falling with `dt`, so it is the stepper and not a leak. |

The rotation is a geometric phase. It depends on the area the loop encloses in joint space, not on how fast the loop is walked.

### How large the effect is, across buses

![Net base rotation after one closed joint-space loop against loop amplitude, for buses from 100 kg to 10 t](docs/attitude_excursion.png)

Net base rotation after one closed loop. Bus inertia is modelled as `I = m (0.5 m)^2`, a metre-scale spacecraft rather than the Panda's dense pedestal puck, so the bus dominates the arm's own 5 kg m^2 about the base. Small loops follow a log-log slope of 1.94 in amplitude, tracking the enclosed area, and -0.99 in bus mass. The curves bend below slope 2 past about 0.5 rad, where the loop stops being small; that limit is on the figure rather than cropped out of it.

A 2,300 kg servicer of roughly MEV-1 scale still turns a few hundredths of a degree per loop. That is small, and it accumulates over every loop of a real manipulation task, in a direction that depends on the path taken.

![Relative difference between the generalized and fixed-base Jacobians against bus mass, falling on a log-log slope of -1](docs/bus_sweep_jacobian.png)

The same limit seen through the Jacobian. `||J_g - J_m|| / ||J_m||` falls on a log-log slope of -0.997 across five decades of bus mass, under the same `I = m (0.5 m)^2` assumption. A heavy enough bus makes the free-flyer a fixed base again, and `J_g` degrades gracefully into `J_m` rather than being a separate regime.

## Bugs found and fixed

**PyBullet's default link damping, which is invisible.** Every link gets 0.04 linear and 0.04 angular damping unless told otherwise, and `getDynamicsInfo` fields 8 and 9 report `-1.0`, meaning unset, whether or not damping is in force. It cannot be read back, so there is nothing to notice. Damping is an external force: it leaked 7e-3 of the system momentum and put the closed-loop base rotation about 2.5% below the zero-momentum value. Not obviously wrong, just quietly wrong in the fourth significant figure of the headline result. Zeroing it on every link, base included, took the analytic-versus-simulation gap from 2.56% to 0.02%. That fix is what made the RK4 agreement above meaningful, and a test now fails if the call is ever dropped.

**Two robots loaded at one origin.** The Jacobian checks need a fixed-base reference body and a free-base body in the same world. Both were loaded at the origin, they interpenetrated, and the contact solver flung the free one off at end-effector speeds of 160 to 300 m/s where 0.6 m/s was expected. The magnitude was the tell: a modelling error gives a plausible number, a collision gives an absurd one. The test fixtures need both bodies at the origin, because the analytic chain builds outward from the base frame and only agrees with `getLinkState` for a body at the identity, so the reference body's collision mask is cleared instead. It is an oracle, not a physical object.

**Ancestor masking, surfaced by a rewrite.** The link Jacobians were first taken by differentiating the pose through autograd. Replacing them with the geometric form `[axis x (com - point); axis]` for a fourfold speedup introduced a bug the autograd version had been structurally immune to. Applied blindly to every arm joint, the formula returns a non-zero column for joints distal to a link, claiming that turning joint 5 moves link 1; differentiating the actual chain cannot produce dependence on a joint the chain does not contain. The per-link error table located it in one run: links 6 through 11 exact to 1e-16, links 0 through 5 wrong by up to 0.72 and progressively less so with depth, which is exactly the signature of columns that should have been zero. An aggregate norm would have said 0.3 and pointed at nothing.

## Tests

32 cases, each named for the invariant it protects rather than the function it calls: `test_proximal_link_has_zero_columns_for_joints_that_do_not_move_it`, not `test_link_jacobian`. Most of them exist because of one of the bugs above. `pytest` runs the 30 fast ones in about 16 s; two parameter sweeps are marked `slow` and excluded by default.

A test that cannot fail for a reason you can state is not protecting anything, so each guard was checked by breaking the thing it guards and confirming that case is the one that fails:

| deliberate break | case that fails |
| --- | --- |
| ancestor masking removed from the geometric Jacobian | `test_proximal_link_has_zero_columns_for_joints_that_do_not_move_it` |
| damping-zeroing made a no-op | `test_link_damping_must_be_zeroed_explicitly_not_left_to_the_default` |
| base twist integrated in the world frame | `test_base_twist_integrates_in_the_body_frame_not_the_world_frame` |
| base reference point moved off the base centre of mass | `test_base_inertia_and_coupling_match_free_base_mass_matrix` |
| the base-twist layout guard removed | `test_a_swapped_base_twist_layout_is_rejected_not_silently_used` |

Two cases exist specifically to keep others from going vacuous. One asserts that some link really is proximal, so the zero-columns loop has something to check, and one asserts that the wrong base reference points genuinely do disagree with PyBullet, so the agreement of the right one is evidence rather than coincidence.

## Limitations

Stated rather than softened. Where a method loses is part of the result.

**The Panda is a ground arm standing in for a space manipulator.** Its link masses, inertias, reach and joint limits are those of a 7-DOF table-mounted research arm. The arm-to-bus inertia ratios swept here are representative of a servicer; the absolute numbers are not, and no conclusion should be transferred to a specific mission from them.

**The bus is a mass and an inertia tensor, not a modelled spacecraft.** There are no solar panels, no flexible modes, no fuel slosh and no attitude control system. A real servicer would fight base motion with reaction wheels or RCS and would budget the propellant to do it. That control problem is precisely what this repository does not address; it quantifies the disturbance such a control loop would have to reject.

**The fingers are locked.** Their mass rides at its true centre of mass in the mass matrix, but they contribute no degrees of freedom. Nothing here models grasping.

**There are no contact dynamics at all.** Everything is free motion in free space. Berthing, grappling and the combined-body dynamics after capture, which is the part of servicing that actually matters, are entirely absent.

**Zero initial momentum is assumed throughout.** A real target tumbles, and a real servicer arrives with residual rates.

**`J_g` is built, not yet planned with.** The generalized Jacobian and its gradients exist and are verified. No planner consumes them yet.

## Requirements

Python 3.10+ and a CPU. No GPU needed; there is no CUDA dependency anywhere, which is deliberate.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
```

The pinned torch build is CPU-only and comes from PyTorch's own index:

```bash
pip install torch==2.14.0+cpu --index-url https://download.pytorch.org/whl/cpu
```

## Running it

Each script is the report behind one result. All of them print a per-item table rather than an aggregate, and the ones that own a figure re-derive it into `docs/`.

```bash
python examples/audit_link_inertias.py         # link masses and inertias, before trusting any of them
python examples/verify_kinematics.py           # the analytic chain against getLinkState
python examples/verify_mass_matrix.py          # H_b and H_bm against calculateMassMatrix
python examples/verify_generalized_jacobian.py # J_g against simulation and the large-bus limit
python examples/verify_nonholonomy.py          # the closed-loop base rotation, and its figure
python examples/render_nonholonomy_gif.py      # the animation at the top of this file
```

Run the tests with `pytest`. The two parameter sweeps are marked `slow` and left out of the default run; `pytest -m slow` runs those and `pytest -m "slow or not slow"` runs everything.

## Structure

```
src/kinematics.py   differentiable forward kinematics for a serial chain
src/dynamics.py     floating-base mass matrix, coupling inertia, generalized Jacobian
src/freeflight.py   driving a free-flying body and measuring what its base does
examples/           one script per result, each printing a per-item table
tests/              32 invariants, each case named for what it protects
docs/               figures, all regenerable from the scripts above
```

`src/dynamics.py`'s module docstring is the reference for the PyBullet frame conventions the rest of the code depends on. `CONTRIBUTING.md` collects those conventions alongside the working rules.

## License

MIT, see [LICENSE](LICENSE).

This covers the code in this repository. The Franka Panda model is not included here: it is loaded at runtime from the data that ships with PyBullet, and the kinematic and inertial values every result depends on are Franka Emika's, under their own terms.
