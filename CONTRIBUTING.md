# Contributing

Read `PLAN.md` first. It holds the project's objective, the current phase, the
PyBullet frame conventions that have already cost debugging time, and the
definition of done for the phase in progress.

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
- The scripts in `examples/` are the reports behind the numbers quoted in
  `PLAN.md`. Each one prints a per-item table and re-derives its own figures.

## Working rules

These are carried from two earlier projects in the same portfolio, where they
found every non-trivial bug.

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

## Commits

Keep them small and local. A commit message states what was *verified*, not
only what was added: the tolerance, the number of configurations, the reference
the result was checked against.
