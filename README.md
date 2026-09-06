# free-floating-manipulation

Manipulation planning for an arm on an uncontrolled free floating base, with on orbit satellite servicing as the driving case: a servicer with its thrusters off and its wheels idle, moving its arm toward a target it has not yet grappled. A fixed base arm pushes against the world for free, because every reaction torque a joint generates is absorbed by the floor. Take the floor away and the reaction has nowhere to go but the spacecraft, so the base translates and rotates in response to every joint motion and base attitude ends up depending on the entire history of arm motion rather than on where the joints currently are.

![A free-floating Panda walks a closed loop in joint space; the joints return to their starting angles and the base ends 18.98 degrees rotated](docs/closed_loop.gif)

Two joints trace a circle in joint space and return to exactly the angles they started at. The base does not return. The consequence is not a correction term: the map from configuration to end effector pose stops being a function of configuration at all, because two paths that start and end at the same seven joint angles put the gripper in two different places. Every fixed base planner assumes that map exists. This repository builds the objects that replace it, measures how large the effect is, and then builds the whole thing a second time in C++ to find out what the first version had got wrong.

## What is here

**Differentiable forward kinematics.** An analytic chain in torch, with the Jacobian through autograd. PyBullet already computes forward kinematics, but it returns a number with no gradient attached, and every Cartesian objective downstream needs to know which way to move. Joint origins are read from the URDF, which states them relative to the parent link frame. That is not a stylistic choice and the reason is under Bugs.

**The floating base mass matrix and coupling inertia.** With zero initial momentum and no external wrench the base twist is pinned at every instant by the joint rates:

```
H_b v_b + H_bm qdot = 0        →        v_b = -H_b^-1 H_bm qdot
```

H_b is the six by six locked inertia of the whole system about the base and H_bm the six by seven block coupling arm motion into base motion, both partitions of

```
M(q) = sum_i [ m_i Jt_i^T Jt_i + Jr_i^T (R_i I_i R_i^T) Jr_i ]
```

summed over every link's centre of mass Jacobian. Inertia enters as a full tensor, not three principal moments, because the model this runs on has genuinely non-diagonal ones. It is built in torch and stays differentiable in q.

**The generalized Jacobian.** End effector velocity on a fixed base is J_m qdot. On a free base it picks up the base reaction, and substituting the line above gives the object a planner works against, which is Umetani and Yoshida, 1989:

```
xdot = J_m qdot + J_b v_b = ( J_m - J_b H_b^-1 H_bm ) qdot = J_g qdot
```

J_m is pure geometry: link lengths and joint axes. J_g contains H_b^-1 H_bm, so it depends on the mass distribution, and two arms with identical link geometry and different link masses reach different places from the same joint trajectory. J_g is also not the derivative of anything. If it were, integrating it around a closed loop in joint space would return exactly zero net base motion, because that is what integrating an exact differential around a closed path does. The measurement below says 18.98 degrees.

**A second implementation, in C++.** The same objects again, in Eigen, reading the model through MoveIt's RobotModel rather than PyBullet. It exists to disagree with the first one, and it did.

**The model is Franka's published identified parameters**, not the Panda URDF that ships with PyBullet, whose declared inertia is a placeholder. Total mass 17.5869 kg, of which the base link is 0.6298 kg.

Everything is checked against exact references rather than against each other.

| quantity | checked against | agreement |
| --- | --- | --- |
| forward kinematics | getLinkState, 500 configurations | 0.000107 mm, 4.7e-07 on rotation |
| link centre of mass Jacobians | calculateJacobian, per link | 2.1e-15 |
| M symmetry | its own transpose | 8.9e-16 |
| translational block of M | total mass times identity | exact |
| H_m, the arm block | free and fixed base calculateMassMatrix | 1.1e-10 |
| H_b and H_bm | free base calculateMassMatrix | 5.7e-09 |
| J_m | calculateJacobian, fixed base | 1.7e-15 |
| J_g qdot | zero gravity free floating simulation | 1.0e-04 relative |
| minus H_b^-1 H_bm qdot | measured base twist in simulation | 1.1e-04 relative |
| every block | the independent C++ implementation | 3.6e-15 |
| M and J_g gradients | central differences | 2.1e-09 |

The 5.7e-09 on H_b is not our floor. PyBullet stores an inertia tensor as principal moments plus the rotation that diagonalises them, so a tensor with off diagonal terms only survives the round trip as well as its eigensolve does, which is 5.802e-09 on this model. The tolerance is derived from that measurement rather than set as a constant.

J_g has no reference implementation anywhere, so it is validated three ways that do not share a failure mode: against a simulation, against the large bus limit where it must collapse back to J_m, and against the C++ port.

## Results

### A closed loop in joint space leaves the base rotated

Two joints trace a circle of amplitude 0.6 rad over 2 s and return every angle to its starting value, closing to 4.71e-04 rad. On a fixed base the arm would be exactly where it started. On the 17.59 kg free floating servicer it ends 18.9798 degrees rotated, about an axis of [-0.071, 0.006, -0.323]. The base swings well past that during the loop and comes most of the way back, but not all of it.

That is the whole claim in one number, so the work is in showing it is physics and not the integrator sliding.

| evidence | result |
| --- | --- |
| Timestep independence | the rotation converges on 18.982 degrees as dt halves from 2e-3 to 1.25e-4, with the increments shrinking each time. Coarse to fine spans 0.07 percent. |
| Sign reversal | backward traversal gives 18.9890 degrees about the exactly opposite axis, alignment minus 1.000. Composing the two lands 0.0131 degrees from the identity, against 18.98 for one loop. Drift accumulates the same way in either direction; a geometric phase cancels. |
| Independent integration | RK4 integration of omega_b = [-H_b^-1 H_bm qdot]_ang around the same path gives 18.9823 degrees against the simulation's 18.9808, a gap of 0.01 percent, sharing no code with it. |
| Momentum conservation | peak total momentum along the loop is 6.37e-03 of its own scale at dt = 1e-3 and 1.59e-03 at 2.5e-4, falling by 4.0x for a 4x smaller step, so it is the stepper and not a leak. |

Three of those four are arguments a wrong answer could not survive, and they fail in different ways. Timestep refinement catches an integrator artifact. Sign reversal catches anything that accumulates rather than closing. The RK4 comparison catches an error in the simulation rather than in the model.

### How large the effect is, across buses

![Net base rotation after one closed joint-space loop against loop amplitude, for buses from 100 kg to 10 t](docs/attitude_excursion.png)

Net base rotation after one closed loop. Bus inertia is modelled as I = m (0.5 m)^2, a metre scale spacecraft rather than the Panda's dense pedestal, so the bus dominates the arm's own contribution. Small loops follow a log log slope of 1.94 in amplitude, tracking the enclosed area, and minus 0.98 in bus mass. The curves bend below slope 2 past about 0.5 rad, where the loop stops being small, and that limit is on the figure rather than cropped out of it.

A 2,300 kg servicer of roughly MEV-1 scale still turns about a quarter of a degree per loop at 0.6 rad amplitude. That is small, and it accumulates over every loop of a real task, in a direction that depends on the path taken.

![Relative difference between the generalized and fixed-base Jacobians against bus mass, falling on a log-log slope of -1](docs/bus_sweep_jacobian.png)

The same limit seen through the Jacobian. The relative difference between J_g and J_m falls on a log log slope of minus 0.996 across five decades of bus mass. A heavy enough bus makes the free flyer a fixed base again, and J_g degrades gracefully into J_m rather than being a separate regime, which is the sanity property a formulation like this ought to have.

## Bugs found and fixed

Six, and what they have in common is worth naming first. Every one was a silent substitution or an unstated convention that produced a plausible number rather than an error, and not one was found by reading code. Four were found by a second implementation or a second model disagreeing with the first.

### The forward kinematics was verified to 5.7e-08 and wrong

This is the one worth telling. getJointInfo reports a joint's origin relative to the parent link's inertial frame, and the chain composed that offset forward to recover link frames. It agreed with getLinkState to 5.7e-08 over 500 random configurations, which is the number phase 1 shipped on.

It was still wrong. That composition is correct only while every inertial frame is axis aligned with its link frame. It is, on the Panda URDF PyBullet ships: every one of its inertia tensors is diagonal, so PyBullet never has to diagonalise anything, and every inertial rotation is exactly the identity. Point the same chain at a model whose tensors have off diagonal terms, which PyBullet stores as principal moments plus the rotation that diagonalises them, and it is exact through link 1 and 137 mm out at link 2.

Five hundred configurations of a model that cannot express the failure is five hundred configurations of evidence for nothing. The fix is not a better composition, it is to stop depending on the question: joint origins now come from the URDF, which states them relative to the parent link frame and says nothing about inertial frames at all. The regression is a chain built specifically to have rotated inertial frames.

Two more conventions had been pinned on exactly the same insufficient evidence and fell out with it. calculateJacobian's localPosition is in the link's inertial frame, so a centre of mass Jacobian needs R_inertial^T times getDynamicsInfo field 3, not field 3 itself. calculateMassMatrix writes the base twist in the base link's inertial axes, not merely about that point; rotating H_b and H_bm into those axes took the residual from 1.25 down to 5.6e-09. Both reduce to what phase 2 concluded when the rotation is the identity, which on that model it always was.

### A declared inertia that is a placeholder, and a phantom kilogram

Two model faults, both caught by building the C++ port and comparing block by block, neither visible from inside a single implementation.

PyBullet ignores the inertia a URDF declares. By default it derives one from each link's collision geometry, and the file's values are honoured only with URDF_USE_INERTIA_FROM_FILE. The Panda URDF it ships declares ixx = iyy = izz = 0.1 on every single link, a placeholder, so anything that reads that file and believes it gets a physically meaningless robot that still runs. The C++ port read the file and got 0.1 everywhere; the Python read PyBullet and got the mesh derived set. Masses, frames, ancestor sets and both geometric Jacobians agreed to machine precision, and only the inertia dependent blocks disagreed, which is what pointed straight at it.

A link with no inertial block gets mass 1 and a unit inertia rather than zero. panda_link8 is a flange frame and the identified model declares no inertial for it, so PyBullet invents a phantom kilogram in the middle of the arm, warns on stderr, and continues. changeDynamics cannot repair it, because calculateMassMatrix reads load time values, so the fix has to be in the URDF: declare the zero explicitly.

### PyBullet's default link damping, which the API that reports it does not report

Every link gets 0.04 linear and 0.04 angular damping unless told otherwise, and getDynamicsInfo fields 8 and 9 return minus 1.0, meaning unset, whether or not damping is in force, and after changeDynamics has explicitly set them to zero as well. There is no read path.

Damping is an external force, so it violates precisely the assumption the whole derivation rests on. It puts the closed loop base rotation about 2.5 percent below the zero momentum value: large enough to be real, small enough to be blamed on the integrator. The momentum residual barely moves, because at this timestep the stepper's own discretisation error is larger than the leak, so the obvious diagnostic does not detect it. What detects it is comparing the simulation against an independent analytic integration.

### Two robots loaded at one origin

The Jacobian checks need a fixed base reference body and a free base body in the same world. Both were loaded at the origin, they interpenetrated, and the contact solver flung the free one off at end effector speeds of 165 m/s where the correct answer was 0.18. The magnitude is the diagnosis: a modelling error gives a plausible number, a collision gives an absurd one. The fixtures need both bodies at the origin, because the chain builds outward from the base frame and only agrees with getLinkState for a body at the identity, so the reference body's collision mask is cleared instead. It is an oracle, not a physical object.

### Ancestor masking, and what the rewrite actually cost

The link Jacobians were first taken by differentiating the pose through autograd, then replaced with the geometric form for a fourfold speedup. Applied to every arm joint without qualification, that form returns a non zero column for joints distal to a link, claiming that turning joint 5 moves link 1.

The autograd version was structurally immune: differentiating the real chain cannot invent a dependence on a joint the chain does not contain. The bug became possible only when the geometric form began asserting that structure by hand. That is a property of the two implementations, not a latent defect the rewrite uncovered. The rewrite bought speed and paid for it by moving a guarantee out of the representation and into code that has to be right. A per link error table found it in one run, because the proximal links were wrong and the distal ones exact to machine precision, which is not a pattern an aggregate norm would ever show.

## Tests

32 cases across four files, 30 fast and 2 marked slow, each named for the invariant it protects rather than the function it calls: `test_proximal_link_has_zero_columns_for_joints_that_do_not_move_it`, not `test_link_jacobian`. Most exist because of one of the bugs above, which is why this section follows that one. They pass against both models.

A test that cannot fail for a reason you can state is not protecting anything, so each guard was checked by breaking the thing it guards and confirming the case named for it is the one that fails.

| deliberate break | first case to fail |
| --- | --- |
| joint origins read from getJointInfo again | `test_chain_holds_when_the_inertial_frames_are_rotated` |
| ancestor masking removed from the geometric Jacobian | `test_proximal_link_has_zero_columns_for_joints_that_do_not_move_it` |
| damping zeroing made a no op | `test_link_damping_must_be_zeroed_explicitly_not_left_to_the_default` |
| base twist integrated in the world frame | `test_base_twist_integrates_in_the_body_frame_not_the_world_frame` |
| base reference point moved off the base centre of mass | `test_base_inertia_and_coupling_match_free_base_mass_matrix` |
| the base twist layout guard removed | `test_a_swapped_base_twist_layout_is_rejected_not_silently_used` |

Three cases exist to stop others going vacuous. One asserts that some link really is proximal, so the zero columns loop has something to iterate over. One asserts the wrong base reference points genuinely do disagree with PyBullet, so the agreement of the right one is evidence rather than a loose tolerance. And the rotated inertial frame regression asserts that PyBullet really did rotate the frames before testing anything, because a tensor it rejects as unphysical is silently zeroed, which would make that test pass against the very bug it exists to catch.

Tolerances are measured rather than chosen wherever something else sets the floor. The H_b comparison is judged against PyBullet's own eigensolve residual on the same tensors, not against a constant.

## Limitations

Stated rather than softened. Where a method loses is part of the result.

**The Panda is a ground arm standing in for a space manipulator.** Its masses, inertias, reach and joint limits are those of a seven degree of freedom table mounted research arm, now with Franka's published identified parameters rather than a placeholder. The arm to bus inertia ratios swept here are representative of a servicer; the absolute numbers are not, and no conclusion should be transferred to a specific mission from them.

**The bus is a mass and an inertia tensor, not a modelled spacecraft.** No solar panels, no flexible modes, no fuel slosh, no attitude control system. A real servicer would fight base motion with reaction wheels or RCS and would budget propellant to do it. That control problem is exactly what this repository does not address; it quantifies the disturbance such a loop would have to reject.

**There are no contact dynamics at all.** Everything is free motion in free space. Berthing, grappling and the combined body dynamics after capture, which is the part of servicing that actually matters, are absent entirely.

**The fingers are locked.** Their mass rides at its true centre of mass, but they contribute no degrees of freedom, and nothing here models grasping. Lumping that mass onto the hand instead is available and costs 1.08e-02 in H_b, which is measured rather than assumed negligible.

**Zero initial momentum is assumed throughout.** A real target tumbles and a real servicer arrives with residual rates. Every result starts from rest.

**J_g is built, not yet planned with.** The generalized Jacobian and its gradients exist and are verified against every reference available. No planner consumes them yet.

## Requirements

Python 3.10+ and a CPU. No GPU needed; there is no CUDA dependency anywhere, which is deliberate.

The C++ port and the identified model additionally need ROS 2 Jazzy with MoveIt 2 and `moveit_resources_panda_description`, all from the public ROS 2 apt repository. Everything in Python runs without them by setting `FFM_MODEL=pybullet`.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
```

The pinned torch build is CPU only and comes from PyTorch's own index:

```bash
pip install torch==2.14.0+cpu --index-url https://download.pytorch.org/whl/cpu
```

The C++ workspace, if you want it:

```bash
source /opt/ros/jazzy/setup.bash
colcon build --base-paths ws
```

## Running it

Each script is the report behind one result. All print a per item table rather than an aggregate, and the ones that own a figure re-derive it into `docs/`.

```bash
python examples/audit_link_inertias.py         # masses and inertias, before trusting any of them
python examples/verify_kinematics.py           # the analytic chain against getLinkState
python examples/verify_mass_matrix.py          # H_b and H_bm against calculateMassMatrix
python examples/verify_generalized_jacobian.py # J_g against simulation and the large bus limit
python examples/verify_nonholonomy.py          # the closed loop base rotation, and its figure
python examples/verify_cpp_dynamics.py         # the C++ port against the torch one, block by block
python examples/render_nonholonomy_gif.py      # the animation at the top of this file
```

`FFM_MODEL` selects the model: `identified` is the default and is Franka's published parameters, `pybullet` is the Panda that ships with PyBullet and is what phases 1 and 2 were measured on. Every script and every test runs against both.

Run the tests with `pytest`. The two parameter sweeps are marked `slow` and left out of the default run; `pytest -m slow` runs those and `pytest -m "slow or not slow"` runs everything.

## Structure

```
src/kinematics.py   differentiable forward kinematics, joint origins read from the URDF
src/dynamics.py     floating base mass matrix, coupling inertia, generalized Jacobian
src/freeflight.py   model selection, the joint loop, the zero gravity run, momentum
examples/           one script per result, each printing a per item table
tests/              32 invariants, each case named for what it protects
ws/                 the C++ port, a colcon package reading the model through MoveIt
docs/               figures, all regenerable from the scripts above
```

`src/dynamics.py`'s module docstring is the reference for the frame conventions the rest of the code depends on, and `CONTRIBUTING.md` collects them alongside the working rules.

## License

MIT, see [LICENSE](LICENSE).

This covers the code in this repository. The Franka Panda model is not included here: it is loaded at runtime from data that ships with PyBullet or with MoveIt's `moveit_resources_panda_description`, and the kinematic and inertial values every result depends on are Franka Emika's, under their own terms.
