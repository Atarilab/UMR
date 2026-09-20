#!/usr/bin/env python3
"""Play an HSI/HOI sequence's object trajectories in MuJoCo, straight from the
source data, with no UMR retargeting or scaling applied.

Purpose: decide whether floor penetration comes from the source sequence itself
or from the retarget pipeline's scaling. Nothing here reads a retarget result,
a robot config, or a correspondence cache.

The scene is just a floor plane at z=0 plus every `<stem>.xml` in the sequence
directory that has a matching `prop_<stem>.csv`, driven by that CSV.

Usage
-----
    # report penetration numbers only, no window
    python scripts/check_source_object_ground.py sample_data/embody/chair_tucking --report

    # same, then open an interactive viewer
    python scripts/check_source_object_ground.py sample_data/embody/chair_tucking

    # reproduce what the retarget pipeline does to the geometry:
    #   uniform  -> both mesh and position scaled (retarget_object_size=scaled)
    #   decoupled-> mesh left at 1.0, position scaled (retarget_object_size=original)
    python scripts/check_source_object_ground.py <seq> --report --mode uniform   --scale 0.72675663
    python scripts/check_source_object_ground.py <seq> --report --mode decoupled --scale 0.72675663
"""
from __future__ import annotations

import argparse
import csv
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

try:
    import mujoco
except ImportError:  # pragma: no cover
    raise SystemExit("mujoco is required; run with the UMR virtualenv interpreter.")

FLOOR_Z = 0.0


def discover(seq_dir: Path) -> list[tuple[str, Path, Path]]:
    """Every object XML in the sequence that has a prop trajectory beside it."""
    found = []
    for xml_path in sorted(seq_dir.glob("*.xml")):
        prop = seq_dir / f"prop_{xml_path.stem}.csv"
        if prop.exists():
            found.append((xml_path.stem, xml_path, prop))
    return found


def read_prop(prop_csv: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (positions Nx3, quats_wxyz Nx4) exactly as stored, no transform."""
    rows = list(csv.DictReader(prop_csv.open(newline="")))
    if not rows:
        raise ValueError(f"empty trajectory: {prop_csv}")
    pos = np.asarray([[float(r["px"]), float(r["py"]), float(r["pz"])] for r in rows], dtype=np.float64)
    xyzw = np.asarray([[float(r["qx"]), float(r["qy"]), float(r["qz"]), float(r["qw"])] for r in rows], dtype=np.float64)
    wxyz = np.concatenate([xyzw[:, 3:4], xyzw[:, :3]], axis=1)
    wxyz /= np.maximum(np.linalg.norm(wxyz, axis=1, keepdims=True), 1e-12)
    return pos, wxyz


def build_model(objects, mesh_scale: float) -> mujoco.MjModel:
    """Merge the object MJCFs into one scene with a reference floor plane."""
    assets, bodies = [], []
    for stem, xml_path, _ in objects:
        root = ET.parse(xml_path).getroot()
        compiler = root.find("compiler")
        meshdir = compiler.get("meshdir") if compiler is not None else None
        base = (xml_path.parent / meshdir).resolve() if meshdir else xml_path.parent.resolve()

        for asset in root.findall("asset"):
            for mesh in asset.findall("mesh"):
                mesh = ET.fromstring(ET.tostring(mesh))
                if mesh.get("file"):
                    mesh.set("file", str((base / mesh.get("file")).resolve()))
                vals = [float(v) for v in mesh.get("scale", "1 1 1").split()]
                if len(vals) == 1:
                    vals *= 3
                mesh.set("scale", " ".join(f"{v * mesh_scale:.8g}" for v in vals[:3]))
                assets.append(ET.tostring(mesh, encoding="unicode"))

        worldbody = root.find("worldbody")
        if worldbody is None:
            continue
        for body in worldbody.findall("body"):
            bodies.append(ET.tostring(body, encoding="unicode"))

    xml = f"""<mujoco model="source_object_check">
  <compiler angle="radian"/>
  <visual><headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3"/></visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.24 0.26 0.30"
             rgb2="0.30 0.32 0.36" width="512" height="512"/>
    <material name="grid" texture="grid" texrepeat="12 12" reflectance="0.05"/>
    {''.join(assets)}
  </asset>
  <worldbody>
    <light pos="0 0 4" dir="0 0 -1" directional="true"/>
    <geom name="floor" type="plane" size="8 8 0.05" pos="0 0 {FLOOR_Z}" material="grid"/>
    {''.join(bodies)}
  </worldbody>
</mujoco>"""
    return mujoco.MjModel.from_xml_string(xml)


def lowest_z_per_frame(model, data, pos, quat, addr, n_frames) -> np.ndarray:
    """Lowest world vertex of every mesh geom, per frame."""
    mesh_geoms = [g for g in range(model.ngeom) if int(model.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_MESH)]
    out = np.empty(n_frames, dtype=np.float64)
    for f in range(n_frames):
        for (p, q, a) in zip(pos, quat, addr):
            i = min(f, len(p) - 1)
            data.qpos[a:a + 3] = p[i]
            data.qpos[a + 3:a + 7] = q[i]
        mujoco.mj_forward(model, data)
        lo = np.inf
        for g in mesh_geoms:
            mid = int(model.geom_dataid[g])
            s, n = int(model.mesh_vertadr[mid]), int(model.mesh_vertnum[mid])
            v = model.mesh_vert[s:s + n].astype(np.float64)
            R = data.geom_xmat[g].reshape(3, 3)
            lo = min(lo, float((v @ R.T + data.geom_xpos[g])[:, 2].min()))
        out[f] = lo
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sequence", type=Path, help="Sequence directory holding <stem>.xml and prop_<stem>.csv")
    ap.add_argument("--mode", choices=("source", "uniform", "decoupled"), default="source",
                    help="source: no scaling at all (default). uniform: mesh and position both scaled. "
                         "decoupled: position scaled, mesh left at true size.")
    ap.add_argument("--scale", type=float, default=1.0, help="smpl_scale, used by uniform and decoupled.")
    ap.add_argument("--report", action="store_true", help="Print numbers and exit without opening a viewer.")
    ap.add_argument("--fps", type=float, default=30.0)
    args = ap.parse_args()

    seq = args.sequence.expanduser().resolve()
    objects = discover(seq)
    if not objects:
        print(f"No <stem>.xml with a matching prop_<stem>.csv in {seq}")
        return 1

    mesh_scale = args.scale if args.mode == "uniform" else 1.0
    pos_scale = args.scale if args.mode in ("uniform", "decoupled") else 1.0

    model = build_model(objects, mesh_scale)
    data = mujoco.MjData(model)

    positions, quats, addrs = [], [], []
    for stem, _, prop in objects:
        p, q = read_prop(prop)
        positions.append(p * pos_scale)
        quats.append(q)
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{stem}_freejoint")
        if jid < 0:
            free = [j for j in range(model.njnt) if int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE)]
            jid = free[len(addrs)]
        addrs.append(int(model.jnt_qposadr[jid]))

    n = max(len(p) for p in positions)
    lows = lowest_z_per_frame(model, data, positions, quats, addrs, n)

    print(f"sequence   {seq}")
    print(f"mode       {args.mode}   mesh x{mesh_scale:.6f}   position x{pos_scale:.6f}")
    print(f"objects    {', '.join(s for s, _, _ in objects)}")
    print(f"frames     {n}")
    print()
    print(f"lowest vertex over trajectory : {lows.min():+.5f} m   (floor z = {FLOOR_Z:.3f})")
    print(f"mean lowest per frame         : {lows.mean():+.5f} m")
    below = int((lows < FLOOR_Z - 1e-6).sum())
    print(f"frames below the floor        : {below} / {n}  ({100.0 * below / n:.1f}%)")
    if below:
        print(f"deepest frame                 : {int(lows.argmin())}")
        print(f"lift needed to rest on floor  : {FLOOR_Z - lows.min():+.5f} m")
    else:
        print("nothing penetrates the floor.")

    if args.report:
        return 0

    try:
        import mujoco.viewer as mj_viewer
    except ImportError:
        print("\nmujoco.viewer unavailable; rerun with --report.")
        return 0

    import time
    print("\nopening viewer; close the window or press Ctrl+C to stop.")
    with mj_viewer.launch_passive(model, data) as viewer:
        f = 0
        while viewer.is_running():
            for (p, q, a) in zip(positions, quats, addrs):
                i = min(f, len(p) - 1)
                data.qpos[a:a + 3] = p[i]
                data.qpos[a + 3:a + 7] = q[i]
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(1.0 / max(args.fps, 1e-3))
            f = (f + 1) % n
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
