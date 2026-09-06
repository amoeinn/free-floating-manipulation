"""Compare the C++ dynamics against the phase 2 torch implementation.

The two are independent: one reads the model through MoveIt's RobotModel and
assembles the matrices with Eigen, the other reads it through PyBullet and
assembles them with torch. Neither is checked against itself.

Both are pointed at the same URDF on purpose. MoveIt's own Panda description
carries no inertial data at all, and the xacro that does carries different
masses, so using it would compare two different robots and any disagreement
would be uninterpretable. Reading one file removes that confound and leaves
the comparison about the implementations.

Everything is reported per block and per link. A single norm over the whole
mass matrix would say a number and point nowhere, which is exactly how the
ancestor masking bug survived its first pass in phase 2.

Usage:
    python examples/verify_cpp_dynamics.py
"""

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pybullet as p
import pybullet_data
import torch

from src.dynamics import FloatingBaseModel
from src.freeflight import ARM_JOINTS, PANDA_MODEL, panda_spec

ROOT = Path(__file__).resolve().parent.parent
DUMP = ROOT / "ws" / "install" / "free_floating_manipulation" / "lib" / \
    "free_floating_manipulation" / "dump_dynamics"
DTYPE = torch.float64
CONFIGURATIONS = 8


def urdf_path(spec: dict) -> Path:
    """Absolute path to the URDF both implementations read."""
    if Path(spec["urdf"]).is_absolute():
        return Path(spec["urdf"])
    return Path(pybullet_data.getDataPath()) / spec["urdf"]


def parse(path: Path) -> dict:
    """Read the C++ dump into {header fields, per link constants, per config}."""
    tokens = path.read_text().splitlines()
    out = {"links": {}, "configs": []}
    index = 0
    current = None
    while index < len(tokens):
        parts = tokens[index].split()
        index += 1
        if not parts:
            continue
        key = parts[0]
        if key == "MODEL_FRAME":
            out["model_frame"] = parts[1]
        elif key == "TOTAL_MASS":
            out["total_mass"] = float(parts[1])
        elif key == "NCONFIG":
            out["nconfig"] = int(parts[1])
        elif key == "LINK":
            name = parts[1]
            ancestors = parts[parts.index("ancestors") + 1:]
            out["links"][name] = {
                "mass": float(parts[2]),
                "com_in_link": np.array([float(v) for v in parts[3:6]]),
                "inertia": np.array([float(v) for v in parts[6:15]]).reshape(3, 3),
                "ancestors": [int(a) for a in ancestors],
            }
        elif key == "CONFIG":
            current = {"linkframe": {}, "com": {}, "jointframe": {},
                       "jt": {}, "jr": {}}
            out["configs"].append(current)
        elif key == "LINKFRAME":
            current["linkframe"][parts[1]] = np.array([float(v) for v in parts[2:5]])
        elif key == "COM":
            current["com"][parts[1]] = np.array([float(v) for v in parts[2:5]])
        elif key == "JOINTFRAME":
            current["jointframe"][int(parts[1])] = (
                np.array([float(v) for v in parts[2:5]]),
                np.array([float(v) for v in parts[5:8]]))
        elif key in ("M", "HB", "HBM", "JM", "JB", "JG", "JT", "JR"):
            if key in ("JT", "JR"):
                name, rows, cols = parts[1], int(parts[2]), int(parts[3])
            else:
                rows, cols = int(parts[1]), int(parts[2])
            block = np.array([[float(v) for v in tokens[index + r].split()]
                              for r in range(rows)]).reshape(rows, cols)
            index += rows
            if key == "JT":
                current["jt"][name] = block
            elif key == "JR":
                current["jr"][name] = block
            else:
                current[key] = block
    return out


def run_dump(urdf: Path, config_file: Path, dump_file: Path,
             end_effector: str) -> None:
    """Run the C++ dump with the workspace overlay sourced.

    The binary links against libfree_floating_dynamics and MoveIt, and this
    script is normally run from the Python venv rather than from a shell with
    ROS sourced, so the environment has to be established here. colcon's
    install/setup.bash chains to the underlay it was built against, so
    sourcing it alone is enough.
    """
    setup = ROOT / "ws" / "install" / "setup.bash"
    if not setup.exists():
        print(f"build the workspace first; {setup} not found")
        sys.exit(2)
    command = (f'source "{setup}" >/dev/null 2>&1 && '
               f'exec "{DUMP}" "{urdf}" "{config_file}" "{dump_file}" '
               f'"{end_effector}"')
    finished = subprocess.run(["bash", "-c", command],
                              capture_output=True, text=True)
    if finished.returncode != 0:
        print("the C++ dump failed:")
        print(finished.stdout.strip())
        print(finished.stderr.strip())
        sys.exit(1)


def report(label: str, worst: float, tolerance: float, extra: str = "") -> bool:
    passed = worst < tolerance
    mark = "ok" if passed else "DISAGREES"
    print(f"  {label:<46} {worst:>10.3e}   {mark}{extra}")
    return passed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inertia", choices=("declared", "pybullet"), default="declared",
        help="which inertia the Python side reads. 'declared' loads with "
             "URDF_USE_INERTIA_FROM_FILE so both implementations read the "
             "inertia the URDF states, which is the only setting under which "
             "this is a comparison of implementations. 'pybullet' is the "
             "default PyBullet behaviour used in phase 2, where the URDF "
             "inertia is ignored and one is derived from the collision "
             "geometry instead; the C++ side cannot reproduce that, so the "
             "inertia dependent blocks are expected to differ.")
    arguments = parser.parse_args()

    if not DUMP.exists():
        print(f"build the workspace first; {DUMP} not found")
        sys.exit(2)

    spec = panda_spec()
    urdf = urdf_path(spec)
    rng = np.random.default_rng(0)

    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    # With the identified model the inertias come straight from the URDF, so
    # the flag is irrelevant; with the pybullet model it selects between the
    # declared placeholder and PyBullet's mesh derived set.
    flags = (p.URDF_USE_INERTIA_FROM_FILE
             if (arguments.inertia == "declared" and spec["inertials"] is None) else 0)
    body = p.loadURDF(spec["urdf"], useFixedBase=False, flags=flags)
    limits = [(p.getJointInfo(body, j)[8], p.getJointInfo(body, j)[9]) for j in ARM_JOINTS]
    configurations = np.array([[rng.uniform(0.8 * lo, 0.8 * hi) for lo, hi in limits]
                               for _ in range(CONFIGURATIONS)])

    scratch = ROOT / "ws" / "build" / "cpp_comparison"
    scratch.mkdir(parents=True, exist_ok=True)
    config_file = scratch / "configurations.txt"
    dump_file = scratch / "cpp_dynamics.txt"
    config_file.write_text("\n".join(" ".join(f"{v:.17g}" for v in row)
                                     for row in configurations) + "\n")
    run_dump(urdf, config_file, dump_file, spec["end_effector"])
    cpp = parse(dump_file)

    end_effector_index = next(
        j for j in range(p.getNumJoints(body))
        if p.getJointInfo(body, j)[12].decode() == spec["end_effector"])
    model = FloatingBaseModel(body, ARM_JOINTS, end_effector=end_effector_index,
                              inertials=spec["inertials"],
                              urdf=spec["urdf_path"], dtype=DTYPE)
    names = {link.name: link for link in model.links}

    print(f"model: {PANDA_MODEL}")
    print(f"both implementations read {urdf}")
    print(f"C++ model frame: {cpp['model_frame']}, end effector {spec['end_effector']}")
    source = "URDF directly" if spec["inertials"] is not None else arguments.inertia
    print(f"Python inertia source: {source}\n")

    ok = True

    # ---------------------------------------------------------------- model
    print("the model each side read, before any arithmetic")
    worst_mass = worst_com = worst_inertia = 0.0
    mismatched_ancestors = []
    for name, entry in cpp["links"].items():
        mine = names.get(name)
        if mine is None:
            print(f"  link {name} is absent from the Python model")
            ok = False
            continue
        worst_mass = max(worst_mass, abs(entry["mass"] - mine.mass))
        worst_com = max(worst_com, np.abs(
            entry["com_in_link"] - mine.inertial_transform[:3, 3].numpy()).max())
        worst_inertia = max(worst_inertia, np.abs(
            entry["inertia"] - mine.inertia.numpy()).max())
        theirs = [ARM_JOINTS.index(j) for j in mine.arm_ancestors]
        if entry["ancestors"] != theirs:
            mismatched_ancestors.append((name, entry["ancestors"], theirs))
    ok &= report("link masses", worst_mass, 1e-12)
    ok &= report("centre of mass offset in the link frame", worst_com, 1e-12)
    ok &= report("inertia tensor", worst_inertia, 1e-12)
    ok &= report("total mass", abs(cpp["total_mass"] - sum(l.mass for l in model.links)), 1e-12)
    if mismatched_ancestors:
        ok = False
        print("  ancestor sets differ:")
        for name, a, b in mismatched_ancestors:
            print(f"    {name}: C++ {a} vs Python {b}")
    else:
        print(f"  {'arm ancestor sets, all links':<46} {'exact':>10}   ok")

    # ------------------------------------------------------------- frames
    print("\nframes, per link, worst over all configurations")
    print(f"  {'link':<22} {'link frame':>12} {'centre of mass':>16}")
    worst_frame = worst_compos = 0.0
    for name, mine in names.items():
        if name not in cpp["configs"][0]["linkframe"]:
            continue
        frame_error = com_error = 0.0
        for i, angles in enumerate(configurations):
            q = torch.tensor(angles, dtype=DTYPE)
            if mine.kinematics is None:
                py_frame = np.zeros(3)
            else:
                py_frame = mine.kinematics(q)[:3, 3].numpy()
            py_com = model.com_pose(mine, q)[:3, 3].numpy()
            frame_error = max(frame_error,
                              np.abs(cpp["configs"][i]["linkframe"][name] - py_frame).max())
            com_error = max(com_error,
                            np.abs(cpp["configs"][i]["com"][name] - py_com).max())
        worst_frame = max(worst_frame, frame_error)
        worst_compos = max(worst_compos, com_error)
        print(f"  {name:<22} {frame_error:>12.3e} {com_error:>16.3e}")
    ok &= report("worst link frame", worst_frame, 1e-9)
    ok &= report("worst centre of mass", worst_compos, 1e-9)

    # ------------------------------------------------- per link Jacobians
    print("\nlink Jacobians, per link, worst over all configurations")
    print(f"  {'link':<22} {'max |dJt|':>12} {'max |dJr|':>12}")
    worst_jt = worst_jr = 0.0
    for name, mine in names.items():
        if name not in cpp["configs"][0]["jt"]:
            continue
        jt_error = jr_error = 0.0
        for i, angles in enumerate(configurations):
            q = torch.tensor(angles, dtype=DTYPE)
            py_t, py_r = model.link_jacobian(mine, q)
            jt_error = max(jt_error, np.abs(cpp["configs"][i]["jt"][name] - py_t.numpy()).max())
            jr_error = max(jr_error, np.abs(cpp["configs"][i]["jr"][name] - py_r.numpy()).max())
        worst_jt = max(worst_jt, jt_error)
        worst_jr = max(worst_jr, jr_error)
        print(f"  {name:<22} {jt_error:>12.3e} {jr_error:>12.3e}")
    ok &= report("worst link Jacobian", max(worst_jt, worst_jr), 1e-9)

    # ------------------------------------------------------------- blocks
    print("\nthe blocks, worst over all configurations")
    blocks = {"M, whole mass matrix": [], "M[:6,:6], H_b": [], "M[:6,6:], H_bm": [],
              "M[6:,6:], H_m arm block": [], "J_m": [], "J_b": [], "J_g": []}
    for i, angles in enumerate(configurations):
        q = torch.tensor(angles, dtype=DTYPE)
        c = cpp["configs"][i]
        py_m = model.mass_matrix(q).numpy()
        py_hb, py_hbm = (t.numpy() for t in model.coupling(q))
        blocks["M, whole mass matrix"].append(np.abs(c["M"] - py_m).max())
        blocks["M[:6,:6], H_b"].append(np.abs(c["HB"] - py_hb).max())
        blocks["M[:6,6:], H_bm"].append(np.abs(c["HBM"] - py_hbm).max())
        blocks["M[6:,6:], H_m arm block"].append(np.abs(c["M"][6:, 6:] - py_m[6:, 6:]).max())
        blocks["J_m"].append(np.abs(c["JM"] - model.manipulator_jacobian(q).numpy()).max())
        blocks["J_b"].append(np.abs(c["JB"] - model.base_jacobian(q).numpy()).max())
        blocks["J_g"].append(np.abs(c["JG"] - model.generalized_jacobian(q).numpy()).max())
    for label, values in blocks.items():
        ok &= report(label, max(values), 1e-9)

    scale = np.abs(model.mass_matrix(torch.tensor(configurations[0], dtype=DTYPE)).numpy()).max()
    print(f"\n  entries of M are of order {scale:.2f}, so the tolerances above are relative")
    print(f"  1e-9 against that is about {1e-9 / scale:.1e} relative\n")
    print(f"summary: {'C++ and torch agree block for block' if ok else 'THERE IS A REAL DISAGREEMENT'}")
    p.disconnect()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
