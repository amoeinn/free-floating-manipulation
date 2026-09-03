# free-floating-manipulation

Manipulation planning for an arm on an **uncontrolled free-floating base**, with
on-orbit satellite servicing as the driving case: a servicer that has grappled
nothing yet, thrusters off, reaction wheels idle, moving its arm toward a
target.

A fixed-base arm pushes against the world for free. Every reaction torque a
joint generates is absorbed by the floor, so joint angles determine gripper
pose and nothing else does. Take the floor away and the reaction has nowhere to
go but the spacecraft. Total momentum is conserved, the base translates and
rotates in response to every joint motion, and **base attitude depends on the
entire history of arm motion rather than on where the joints currently are**.

The consequence is not a correction term. The map from configuration to
end-effector pose stops being a function of configuration at all. Two paths
that start and end at the same seven joint angles put the gripper in two
different places, because the base has turned by different amounts along the
way. Every fixed-base planner assumes that map exists. This repository builds
the objects that replace it, and then measures the size of the effect they
exist to capture.

```
src/kinematics.py   differentiable forward kinematics for a serial chain
src/dynamics.py     floating-base mass matrix, coupling inertia, generalized Jacobian
src/freeflight.py   driving a free-flying body and measuring what its base does
examples/           one script per result; each prints a per-item table
tests/              32 invariants, each case named for what it protects
```

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
pytest                              # 30 invariant cases, ~16 s
python examples/verify_nonholonomy.py   # the headline result, and its figure
```

CPU only; there is no CUDA dependency anywhere.

## Method

**Coupling inertia.** With zero initial momentum and no external wrench, the
base twist is not a free variable — it is pinned at every instant by the joint
rates:

```
H_b v_b + H_bm q̇ = 0        →        v_b = −H_b⁻¹ H_bm q̇
```

`H_b` (6×6) is the locked inertia of the whole system about the base, `H_bm`
(6×7) the block coupling arm motion into base motion. Both are partitions of
the floating-base mass matrix, assembled from every link's centre-of-mass
Jacobian:

```
M(q) = Σᵢ [ mᵢ Jtᵢᵀ Jtᵢ + Jrᵢᵀ (Rᵢ Iᵢ Rᵢᵀ) Jrᵢ ]
```

Built in torch and differentiable in `q` throughout, because a trajectory
optimiser will need gradients through it.

**Generalized Jacobian.** End-effector velocity on a fixed base is `J_m q̇`.
On a free base it picks up the reaction, and substituting the line above gives
the object the planner actually works against (Umetani and Yoshida, 1989):

```
ẋ = J_m q̇ + J_b v_b = ( J_m − J_b H_b⁻¹ H_bm ) q̇ = J_g q̇
```

**What makes this different from a fixed-base Jacobian.** `J_m` is pure
geometry — link lengths and joint axes. `J_g` contains `H_b⁻¹ H_bm`, so it
depends on the *mass distribution*: two arms with identical link geometry but
different link masses have different generalized Jacobians and reach different
places from the same joint trajectory. It is also configuration-dependent in a
way that does not integrate. There is no "free-floating forward kinematics"
that `J_g` is the derivative of; the closed-loop result below is what that
absence looks like when you measure it.

Verification, all against exact references rather than against each other:

| quantity | checked against | agreement |
| --- | --- | --- |
| `H_b`, `H_bm` | PyBullet free-base `calculateMassMatrix` | 4e-15, 10 configurations |
| `H_m` (arm block) | free- and fixed-base `calculateMassMatrix` | 1e-15 |
| link CoM Jacobians | `calculateJacobian`, per link | 7e-16 |
| `J_m` | `calculateJacobian`, fixed base | 8e-16 |
| `J_g q̇` | zero-g free-floating simulation | 2e-4 – 7e-4 relative |
| `−H_b⁻¹H_bm q̇` | measured base twist in simulation | 3e-5 – 1e-4 relative |
| `M(q)`, `J_g` gradients | central differences | 1e-9 |

`J_g` has no PyBullet equivalent to check against, so it is validated two ways
that do not share a failure mode: against simulation, and against the
large-bus limit where `H_b⁻¹H_bm → 0` and it must collapse back to `J_m`.

## Results

### A closed loop in joint space leaves the base rotated

Two joints trace a circle of amplitude 0.6 rad over 2 s and return every angle
to its starting value. On a fixed base the arm would be exactly where it
started. On a 17.96 kg free-floating base it ends **10.02° rotated**, about
`[−0.091, 0.000, −0.149]`.

That is the whole claim in one number, so the work is in showing it is physics
and not the integrator sliding:

| evidence | result |
| --- | --- |
| **Timestep independence** | 10.0642 → 10.0409 → 10.0295 → 10.0238 → 10.0210° as `dt` halves from 2e-3 to 1.25e-4. Increments halve each time — first order, converging on 10.018°, not drifting. Coarse to fine spans 0.43%. |
| **Sign reversal** | Traversing the loop backward gives 10.0190° about the exactly opposite axis (alignment −1.000). Composing the two lands **0.0058° from the identity**, against 10.02° for one loop. Drift accumulates the same way in either direction; a geometric phase cancels. |
| **Independent integration** | RK4 integration of `ω_b = [−H_b⁻¹H_bm q̇]_ang` around the same path gives 10.0191° — **0.02%** from the simulation, which shares no code with it. |
| **Momentum conservation** | Peak total momentum along the loop is 9.7e-3 of its own scale at `dt`=1e-3 and 2.4e-3 at 2.5e-4 — falling with `dt`, so it is the stepper and not a leak. |

The rotation is a geometric phase: it depends on the area the loop encloses in
joint space, not on how fast it is walked.

### How large the effect is, across buses

![Net base rotation after one closed joint-space loop, against loop amplitude, for buses from 100 kg to 10 t](docs/attitude_excursion.png)

*Net base rotation after one closed loop. Bus inertia is modelled as
`I = m·(0.5 m)²` — a metre-scale spacecraft, not the Panda's dense pedestal
puck — so the bus dominates the arm's own ≈5 kg·m² about the base. Small loops
follow a log-log slope of 1.94 in amplitude, tracking the enclosed area, and
−0.99 in bus mass. The curves bend below slope 2 past about 0.5 rad, where the
loop stops being small; that limit is on the figure rather than cropped out
of it.*

A 2,300 kg servicer of roughly MEV-1 scale still turns a few hundredths of a
degree per loop. Small — and it accumulates over every loop of a real
manipulation task, in a direction that depends on the path taken.

![Relative difference between the generalized and fixed-base Jacobians against bus mass, falling on a log-log slope of −1](docs/bus_sweep_jacobian.png)

*The same limit seen through the Jacobian. `‖J_g − J_m‖/‖J_m‖` falls on a
log-log slope of −0.997 across five decades of bus mass (same
`I = m·(0.5 m)²` assumption): a heavy enough bus makes the free-flyer a fixed
base again, and `J_g` degrades gracefully into `J_m` rather than being a
separate regime.*

## Bugs found and fixed

**PyBullet's default link damping, which is invisible.** Every link gets 0.04
linear and 0.04 angular damping unless told otherwise, and `getDynamicsInfo`
fields 8/9 report `-1.0` — meaning "unset" — whether or not damping is in
force, so it cannot be read back and there is nothing to notice. Damping is an
external force: it leaked 7e-3 of the system momentum and put the closed-loop
base rotation about 2.5% below the zero-momentum value. Not obviously wrong —
just quietly wrong, in the fourth significant figure of the headline result.
Zeroing it on every link, base included, took the analytic-versus-simulation
gap from **2.56% to 0.02%**. That fix is what made the RK4 agreement above
meaningful, and the test suite now fails if the call is ever dropped.

**Two robots loaded at one origin.** The Jacobian checks need a fixed-base
reference body and a free-base body in the same world. Both were loaded at the
origin, they interpenetrated, and the contact solver flung the free one off —
end-effector speeds of 160–300 m/s where 0.6 m/s was expected. The magnitude
was the tell: a modelling error gives a plausible number, a collision gives an
absurd one. The test fixtures need both bodies *at* the origin, because the
analytic chain builds outward from the base frame and only agrees with
`getLinkState` for a body at the identity, so the reference body's collision
mask is cleared instead — it is an oracle, not a physical object.

**Ancestor masking, surfaced by a rewrite.** The link Jacobians were first
taken by differentiating the pose through autograd. Replacing them with the
geometric form `[axis × (com − point); axis]` for a 4× speedup introduced a
bug the autograd version had been structurally immune to: applied blindly to
every arm joint, the formula returns a non-zero column for joints *distal* to
a link, claiming that turning joint 5 moves link 1. Differentiating the actual
chain cannot produce dependence on a joint the chain does not contain. The
per-link error table located it in one run — links 6 through 11 exact to
1e-16, links 0 through 5 wrong by up to 0.72 and progressively less so with
depth, which is exactly the signature of columns that should have been zero.
An aggregate norm would have said "0.3" and pointed at nothing.

## Testing

32 cases, each named for the invariant it protects rather than the function it
calls — `test_proximal_link_has_zero_columns_for_joints_that_do_not_move_it`,
not `test_link_jacobian`. `pytest` runs the 30 fast ones in about 16 s; two
parameter sweeps are marked `slow` and excluded by default.

A test that cannot fail for a reason you can state is not protecting anything,
so each guard was checked by breaking the thing it guards and confirming that
case is the one that fails:

| deliberate break | case that fails |
| --- | --- |
| ancestor masking removed from the geometric Jacobian | `test_proximal_link_has_zero_columns_for_joints_that_do_not_move_it` |
| damping-zeroing made a no-op | `test_link_damping_must_be_zeroed_explicitly_not_left_to_the_default` |
| base twist integrated in the world frame | `test_base_twist_integrates_in_the_body_frame_not_the_world_frame` |
| base reference point moved off the base CoM | `test_base_inertia_and_coupling_match_free_base_mass_matrix` |
| the base-twist layout guard removed | `test_a_swapped_base_twist_layout_is_rejected_not_silently_used` |

Two cases exist specifically to keep others from going vacuous: one asserts
that some link really is proximal, so the zero-columns loop has something to
check, and one asserts that the wrong base reference points genuinely *do*
disagree with PyBullet, so the agreement of the right one is evidence rather
than coincidence.

## Limitations

Stated rather than softened. Where a method loses is part of the result.

- **The Panda is a ground arm standing in for a space manipulator.** Its link
  masses, inertias, reach and joint limits are those of a 7-DoF table-mounted
  research arm. The arm-to-bus inertia *ratios* swept here are representative
  of a servicer; the absolute numbers are not, and no conclusion should be
  transferred to a specific mission from them.
- **The bus is a mass and an inertia tensor, not a modelled spacecraft.** There
  are no solar panels, no flexible modes, no fuel slosh, and no attitude
  control system. A real servicer would fight base motion with reaction wheels
  or RCS, and would budget the propellant to do it. That control problem is
  precisely what this repository does not address — it quantifies the
  disturbance that control loop would have to reject.
- **The fingers are locked.** Their mass rides at its true centre of mass in
  the mass matrix, but they contribute no degrees of freedom. Nothing here
  models grasping.
- **There are no contact dynamics at all.** Everything is free motion in free
  space. Berthing, grappling, and the combined-body dynamics after capture —
  the part of servicing that actually matters — are entirely absent.
- **Zero initial momentum is assumed throughout.** A real target tumbles, and
  a real servicer arrives with residual rates.
- **`J_g` is built, not yet planned with.** The generalized Jacobian and its
  gradients exist and are verified; no planner consumes them yet.

## License

MIT, see [LICENSE](LICENSE).

This covers the code in this repository. The Franka Panda URDF and meshes are Franka Emika's, redistributed with PyBullet under their own terms.