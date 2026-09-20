#!/usr/bin/env python3
"""Score an HSI/HOI retarget result against the source sequence it came from.

Answers the questions a real-scale object retarget has to answer, and answers them the same way
for every ``retarget_object_size`` mode so two runs can be compared directly:

* are the objects the size and on the trajectory they were recorded at?
* is the scene's internal geometry (object to object) unchanged?
* does the robot touch an object where the human touched it?
* does anything penetrate the floor or an object?
* do planted feet stay planted, and how smooth is the trajectory?

The object placement the solver used is READ from the result (``object_mesh_scale``,
``object_position_scale``, ``source_warp_offset``) rather than re-derived, so this script cannot
drift from the solver's convention.

    python scripts/evaluate_hsi_hoi_retarget.py \\
        --result output/<robot>_retarget/<seq>_hsi_hoi_<robot>.npz \\
        --sequence sample_data/embody/chair_tucking
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree, Delaunay
from scipy.spatial.transform import Rotation as R

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import mujoco  # noqa: E402
import xml.etree.ElementTree as ET  # noqa: E402

import retarget_smpl_to_humanoid_surface_vector as solver  # noqa: E402

HAND_PATTERN = ("hand", "palm", "thumb", "index", "middle", "ring", "pinky", "wrist")
FOOT_PATTERN = ("ankle", "foot", "toe")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--result", type=Path, required=True, help="Retarget result .npz")
    parser.add_argument("--sequence", type=Path, required=True, help="Source sequence directory")
    parser.add_argument("--contact-threshold", type=float, default=0.03,
                        help="Human-to-object distance that counts as a source contact, in source metres.")
    parser.add_argument("--stride", type=int, default=1, help="Evaluate every Nth frame.")
    parser.add_argument("--object-samples", type=int, default=4096, help="Surface samples per object.")
    parser.add_argument("--human-stride", type=int, default=8, help="Subsample of SMPL-X vertices for contact search.")
    parser.add_argument("--robot-stride", type=int, default=4, help="Subsample of robot mesh vertices.")
    parser.add_argument("--smplx-dir", type=Path, default=ROOT / "smpl")
    parser.add_argument("--baseline", type=Path, default=None, help="Optional second result to compare against.")
    return parser.parse_args()


def scalar(result, key, default=None):
    if key not in result:
        return default
    value = np.asarray(result[key]).reshape(-1)
    return value[0] if value.size else default


def load_objects(seq_dir: Path, samples: int, seed: int = 0):
    """Every object with a prop trajectory, at its recorded size and pose."""
    objects = []
    stems = sorted({p.stem for p in seq_dir.glob("*.obj")} | {p.stem for p in seq_dir.glob("*.xml")})
    for stem in stems:
        prop = seq_dir / f"prop_{stem}.csv"
        obj = seq_dir / f"{stem}.obj"
        if not prop.exists() or not obj.exists():
            continue
        mesh = trimesh.load(obj, force="mesh", process=False)
        points, _ = trimesh.sample.sample_surface(mesh, samples, seed=seed)
        rows = np.loadtxt(prop, delimiter=",", skiprows=1)
        objects.append(
            {
                "name": stem,
                "mesh": mesh,
                "points_local": np.asarray(points, dtype=np.float64),
                "positions": rows[:, :3].astype(np.float64),
                "rot_mats": R.from_quat(rows[:, 3:7]).as_matrix(),
                "xml": seq_dir / f"{stem}.xml",
                "prop": prop,
            }
        )
    return objects


def human_vertices(seq_dir: Path, smplx_dir: Path, n_frames: int, stride: int):
    """SMPL-X surface points per frame in the SOURCE world, subsampled."""
    import torch
    import smplx

    poses = np.load(seq_dir / "poses.npy").astype(np.float32)[:n_frames]
    transl = np.load(seq_dir / "transl.npy").astype(np.float32)[:n_frames]
    betas = np.load(seq_dir / "betas.npy").astype(np.float32).reshape(1, -1)[:, :10]
    gender = str(np.load(seq_dir / "gender.npy", allow_pickle=True))
    direct = Path(smplx_dir) / f"SMPLX_{gender.upper()}.npz"
    if not direct.is_file():
        direct = Path(smplx_dir) / f"SMPLX_{gender.upper()}.pkl"
    model = smplx.SMPLX(str(direct), gender=gender, use_pca=False, flat_hand_mean=True,
                        ext=direct.suffix.lstrip("."), batch_size=len(poses), num_betas=betas.shape[1])
    tensor = lambda a: torch.as_tensor(a, dtype=torch.float32)  # noqa: E731
    with torch.no_grad():
        out = model(
            betas=tensor(np.repeat(betas, len(poses), axis=0)),
            global_orient=tensor(poses[:, 0:3]), body_pose=tensor(poses[:, 3:66]),
            jaw_pose=tensor(poses[:, 66:69]), leye_pose=tensor(poses[:, 69:72]), reye_pose=tensor(poses[:, 72:75]),
            left_hand_pose=tensor(poses[:, 75:120]), right_hand_pose=tensor(poses[:, 120:165]),
            transl=tensor(transl),
        )
    return out.vertices.detach().cpu().numpy().astype(np.float64)[:, ::stride, :]


def robot_geom_vertices(model, data, stride: int):
    """World-space mesh vertices of every visual geom, grouped by body name."""
    chunks = []
    for geom_id in range(model.ngeom):
        mesh_id = int(model.geom_dataid[geom_id])
        if mesh_id < 0:
            continue
        block = slice(int(model.mesh_vertadr[mesh_id]),
                      int(model.mesh_vertadr[mesh_id]) + int(model.mesh_vertnum[mesh_id]))
        local = model.mesh_vert[block][::stride]
        world = data.geom_xpos[geom_id] + local @ data.geom_xmat[geom_id].reshape(3, 3).T
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom_id])) or ""
        chunks.append((body_name.lower(), world))
    return chunks


def build_penetration_model(robot_xml: Path, objects, mesh_scale: float):
    """Robot + every object in one model, built with the solver's own merge helpers."""
    robot_root = ET.parse(robot_xml).getroot()
    solver.make_mjcf_mesh_paths_absolute(robot_root, robot_xml)
    appended = []
    for obj in objects:
        if not obj["xml"].exists():
            continue
        solver.append_object_mjcf_to_robot_root(
            robot_root, obj["xml"], object_mesh_scale=mesh_scale, name_prefix=f"{obj['name']}__"
        )
        appended.append(obj)
    if not appended:
        return None, None, [], None, None
    # Written beside the robot XML so relative includes inside it still resolve.
    tmp = robot_xml.with_name("evaluate_hsi_hoi_penetration.xml")
    tmp.write_text(ET.tostring(robot_root, encoding="unicode"))
    try:
        model = mujoco.MjModel.from_xml_path(str(tmp))
    finally:
        tmp.unlink(missing_ok=True)
    base = mujoco.MjModel.from_xml_path(str(robot_xml))
    qadrs = [int(model.jnt_qposadr[j]) for j in range(model.njnt)
             if int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE)
             and int(model.jnt_qposadr[j]) >= int(base.nq)]
    robot_geoms = solver.robot_object_collision_geom_ids(base)
    robot_geoms = robot_geoms[robot_geoms < model.ngeom].astype(np.int32)
    object_geoms = np.asarray([g for g in range(int(base.ngeom), int(model.ngeom))
                               if int(model.geom_type[g]) != int(mujoco.mjtGeom.mjGEOM_PLANE)
                               and (int(model.geom_group[g]) == 3 or int(model.geom_contype[g]) != 0
                                    or int(model.geom_conaffinity[g]) != 0)], dtype=np.int32)
    return model, mujoco.MjData(model), appended, qadrs, (robot_geoms, object_geoms, int(base.nq))


def object_penetration(model, data, parts, qadrs, appended, qpos, row, position_scale, z_offset, position_offset):
    """Most negative robot-to-object distance at one frame, in metres."""
    robot_geoms, object_geoms, robot_nq = parts
    mujoco.mj_resetData(model, data)
    data.qpos[:robot_nq] = qpos[:robot_nq]
    for qadr, obj in zip(qadrs, appended):
        index = min(row, len(obj["positions"]) - 1)
        pos = (obj["positions"][index] - np.asarray([0.0, 0.0, z_offset])) * position_scale + position_offset
        quat = R.from_matrix(obj["rot_mats"][index]).as_quat()
        data.qpos[qadr:qadr + 3] = pos
        data.qpos[qadr + 3:qadr + 7] = [quat[3], quat[0], quat[1], quat[2]]
    mujoco.mj_forward(model, data)
    saved = (model.geom_contype.copy(), model.geom_conaffinity.copy(), model.geom_margin.copy())
    worst = np.inf
    try:
        model.geom_contype[robot_geoms] = 1
        model.geom_conaffinity[robot_geoms] = 1
        model.geom_contype[object_geoms] = 1
        model.geom_conaffinity[object_geoms] = 1
        model.geom_margin[:] = np.maximum(saved[2], 0.05)
        mujoco.mj_collision(model, data)
        robot_set, object_set = set(robot_geoms.tolist()), set(object_geoms.tolist())
        fromto = np.zeros(6)
        seen = set()
        for contact_id in range(data.ncon):
            g1, g2 = int(data.contact[contact_id].geom1), int(data.contact[contact_id].geom2)
            pair = (g1, g2) if g1 in robot_set and g2 in object_set else (
                (g2, g1) if g2 in robot_set and g1 in object_set else None)
            if pair is None or pair in seen:
                continue
            seen.add(pair)
            worst = min(worst, float(mujoco.mj_geomDistance(model, data, pair[0], pair[1], 0.05, fromto)))
    finally:
        model.geom_contype[:], model.geom_conaffinity[:], model.geom_margin[:] = saved
    return worst


def percentile_row(label, values, unit="m", scale=1.0):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        print(f"  {label:<44s} (no samples)")
        return
    print(f"  {label:<44s} mean {values.mean() * scale:9.3f}  p95 {np.percentile(values, 95) * scale:9.3f}  "
          f"min {values.min() * scale:9.3f}  max {values.max() * scale:9.3f}  [{unit}]")


def main():
    args = parse_args()
    result = np.load(args.result, allow_pickle=True)
    qpos = np.asarray(result["qpos"], dtype=np.float64)
    frame_ids = np.asarray(result["frame_ids"], dtype=np.int32).reshape(-1)
    robot_xml = Path(str(np.asarray(result["robot_xml"])))
    smpl_scale = float(scalar(result, "smpl_scale", 1.0))
    mode = str(np.asarray(result["retarget_object_size"])) if "retarget_object_size" in result else "unknown"
    mesh_scale = float(scalar(result, "object_mesh_scale", smpl_scale))
    position_scale = (
        np.asarray(result["object_position_scale"], dtype=np.float64).reshape(-1)
        if "object_position_scale" in result
        else np.full(3, smpl_scale)
    )
    if position_scale.size == 1:
        position_scale = np.repeat(position_scale, 3)
    position_offset = (
        np.asarray(result["object_position_offset"], dtype=np.float64).reshape(-1)
        if "object_position_offset" in result
        else np.zeros(3)
    )
    warp = np.asarray(result["source_warp_offset"], dtype=np.float64) if "source_warp_offset" in result \
        else np.zeros((len(qpos), 3))
    ground_z = float(scalar(result, "ground_z", 0.0))

    seq_dir = Path(args.sequence)
    objects = load_objects(seq_dir, args.object_samples)
    frames = np.arange(0, len(qpos), max(1, int(args.stride)))

    print(f"\n=== {args.result.name} ===")
    print(f"  mode={mode}  smpl_scale={smpl_scale:.6f}  object_mesh_scale={mesh_scale:.6f}  "
          f"object_position_scale=({position_scale[0]:.4f}, {position_scale[1]:.4f}, {position_scale[2]:.4f})  "
          f"object_position_offset=({position_offset[0]:+.4f}, {position_offset[1]:+.4f})  "
          f"warp |offset| max={np.linalg.norm(warp, axis=1).max():.4f} m")
    print(f"  objects: {', '.join(o['name'] for o in objects)}   frames={len(qpos)}  evaluated={len(frames)}")

    # ---- object size, trajectory and scene geometry --------------------------
    # The solver subtracts the human's ground shift from objects only in the real modes; mirror it
    # so every cloud below is exactly the geometry the solver constrained against.
    object_z_offset = ground_z if mode in {"real", "real_xy"} else 0.0

    print("\n[objects]")
    for obj in objects:
        extents = obj["mesh"].extents * mesh_scale
        print(f"  {obj['name']:<20s} size {extents[0]:.3f} x {extents[1]:.3f} x {extents[2]:.3f} m "
              f"({mesh_scale:.4f}x recorded)")
    if len(objects) >= 2:
        horizontal, vertical, clearance = [], [], []
        for frame in frames:
            row = min(int(frame_ids[frame]), len(objects[0]["positions"]) - 1)
            a_src, b_src = objects[0]["positions"][row], objects[1]["positions"][row]
            if np.linalg.norm((a_src - b_src)[:2]) > 1e-9:
                horizontal.append(
                    np.linalg.norm(((a_src - b_src) * position_scale)[:2]) / np.linalg.norm((a_src - b_src)[:2])
                )
            if abs((a_src - b_src)[2]) > 1e-9:
                vertical.append(abs(((a_src - b_src) * position_scale)[2]) / abs((a_src - b_src)[2]))
            clouds = []
            for obj in objects[:2]:
                index = min(row, len(obj["positions"]) - 1)
                placed = (obj["positions"][index] - np.asarray([0.0, 0.0, object_z_offset])) * position_scale + position_offset
                clouds.append((obj["points_local"] * mesh_scale) @ obj["rot_mats"][index].T + placed)
            clearance.append(float(cKDTree(clouds[1]).query(clouds[0], k=1)[0].min()))
        horizontal, vertical, clearance = map(np.asarray, (horizontal, vertical, clearance))
        print(f"  object-to-object distance vs source: horizontal {horizontal.mean():.6f}x   "
              f"vertical {vertical.mean() if vertical.size else 1.0:.6f}x")
        print(f"  object-to-object surface clearance:  min {clearance.min() * 1000:.1f} mm   "
              f"frames under 10 mm: {int((clearance < 0.010).sum())}/{len(clearance)}"
              + ("   OVERLAP" if clearance.min() < 0 else ""))

    # ---- where the human touched --------------------------------------------
    print("\n[contact]")
    verts = human_vertices(seq_dir, args.smplx_dir, int(frame_ids.max()) + 1, args.human_stride)
    verts[..., 2] -= ground_z
    source_contact = np.zeros(len(qpos), dtype=bool)
    source_targets = {}
    for frame in frames:
        row = min(int(frame_ids[frame]), len(verts) - 1)
        source_cloud, solved_cloud = [], []
        for obj in objects:
            rot = obj["rot_mats"][min(row, len(obj["rot_mats"]) - 1)]
            pos = obj["positions"][min(row, len(obj["positions"]) - 1)]
            source_cloud.append(obj["points_local"] @ rot.T + pos)
            solved_pos = (pos - np.asarray([0.0, 0.0, object_z_offset])) * position_scale + position_offset
            solved_cloud.append((obj["points_local"] * mesh_scale) @ rot.T + solved_pos)
        source_cloud = np.concatenate(source_cloud, axis=0)
        solved_cloud = np.concatenate(solved_cloud, axis=0)
        dist, ids = cKDTree(source_cloud).query(verts[row], k=1)
        near = dist <= args.contact_threshold
        if near.any():
            source_contact[frame] = True
            source_targets[int(frame)] = solved_cloud[ids[near]]
    print(f"  source contact frames (human within {args.contact_threshold * 100:.0f} cm): "
          f"{int(source_contact.sum())}/{len(frames)} evaluated")

    # ---- robot geometry ------------------------------------------------------
    model = mujoco.MjModel.from_xml_path(str(robot_xml))
    data = mujoco.MjData(model)
    hand_to_object, robot_to_object, lowest, hand_frames = [], [], [], []
    foot_points = {}
    for frame in frames:
        data.qpos[:] = qpos[frame]
        mujoco.mj_forward(model, data)
        chunks = robot_geom_vertices(model, data, args.robot_stride)
        all_pts = np.concatenate([w for _n, w in chunks], axis=0)
        lowest.append(all_pts[:, 2].min())
        feet = [w for n, w in chunks if any(k in n for k in FOOT_PATTERN)]
        if feet:
            foot_points[int(frame)] = np.concatenate(feet, axis=0)
        if frame in source_targets:
            targets = source_targets[int(frame)]
            tree = cKDTree(targets)
            hands = [w for n, w in chunks if any(k in n for k in HAND_PATTERN)]
            if hands:
                hand_to_object.append(float(tree.query(np.concatenate(hands, axis=0), k=1)[0].min()))
                hand_frames.append(int(frame))
            robot_to_object.append(float(tree.query(all_pts, k=1)[0].min()))

    percentile_row("hand to the object the human touched", hand_to_object, "mm", 1000.0)
    percentile_row("closest robot point to that object", robot_to_object, "mm", 1000.0)

    # ---- object penetration --------------------------------------------------
    print("\n[object penetration]")
    pen_model, pen_data, appended, qadrs, parts = build_penetration_model(robot_xml, objects, mesh_scale)
    if pen_model is None:
        print("  (no object MJCF to collide against)")
    else:
        worst = []
        for frame in frames:
            value = object_penetration(pen_model, pen_data, parts, qadrs, appended,
                                       qpos[frame], int(frame_ids[frame]), position_scale, object_z_offset,
                                       position_offset)
            if np.isfinite(value):
                worst.append(value)
        if worst:
            worst = np.asarray(worst)
            print(f"  robot-to-object distance: min {worst.min() * 1000:8.2f} mm   "
                  f"frames within 5 cm: {len(worst)}/{len(frames)}   "
                  f"frames penetrating >5 mm: {int((worst < -0.005).sum())}")
        else:
            print("  robot never comes within 5 cm of an object")

    # ---- floor ---------------------------------------------------------------
    print("\n[floor]")
    lowest = np.asarray(lowest)
    print(f"  lowest robot vertex: min {lowest.min() * 1000:8.2f} mm   mean {lowest.mean() * 1000:8.2f} mm")
    print(f"  frames below -5 mm: {int((lowest < -0.005).sum())}/{len(lowest)}   "
          f"frames above +10 mm (airborne): {int((lowest > 0.010).sum())}/{len(lowest)}")

    # ---- foot slide ----------------------------------------------------------
    print("\n[feet]")
    keys = sorted(foot_points)
    slides = []
    for previous, current in zip(keys, keys[1:]):
        before, after = foot_points[previous], foot_points[current]
        if len(before) != len(after):
            continue
        planted = (before[:, 2] < 0.02) & (after[:, 2] < 0.02)
        if planted.any():
            slides.append(np.linalg.norm(after[planted][:, :2] - before[planted][:, :2], axis=1).mean())
    if slides:
        # `frames` may be strided, so normalise the per-step travel back to per-frame.
        slides = np.asarray(slides) / max(1, int(args.stride))
        print(f"  planted-foot XY travel per frame: mean {slides.mean() * 1000:.2f} mm   "
              f"p95 {np.percentile(slides, 95) * 1000:.2f} mm   "
              f"-> {slides.mean() * 18 * 1000:.0f} mm over a 0.6 s stance")

    # ---- self collision ------------------------------------------------------
    # Nothing checks this by default: the HSI/HOI defaults ship robot_self_penetration_cost at 0
    # and the hard constraint off, so an arm may pass straight through the torso. Pairs that
    # already overlap at the robot's neutral pose are a property of the model, not the motion, and
    # are excluded the same way the solver excludes them.
    print("\n[self collision]")
    collidable = {
        g for g in range(model.ngeom)
        if not (model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0)
        and int(model.geom_type[g]) != int(mujoco.mjtGeom.mjGEOM_PLANE)
        and (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g])) or "") != "world"
    }
    neutral = np.zeros(model.nq)
    for joint_id in range(model.njnt):
        if int(model.jnt_type[joint_id]) == int(mujoco.mjtJoint.mjJNT_FREE):
            neutral[int(model.jnt_qposadr[joint_id]) + 3] = 1.0
    data.qpos[:] = neutral
    mujoco.mj_forward(model, data)
    mujoco.mj_collision(model, data)
    structural = {
        (min(int(data.contact[c].geom1), int(data.contact[c].geom2)),
         max(int(data.contact[c].geom1), int(data.contact[c].geom2)))
        for c in range(data.ncon) if float(data.contact[c].dist) < 0.0
    }
    worst_self, worst_pair, bad_frames = 0.0, None, 0
    for frame in frames:
        data.qpos[:] = qpos[frame]
        mujoco.mj_forward(model, data)
        mujoco.mj_collision(model, data)
        deepest = 0.0
        for c in range(data.ncon):
            contact = data.contact[c]
            g1, g2 = int(contact.geom1), int(contact.geom2)
            if g1 not in collidable or g2 not in collidable:
                continue
            if (min(g1, g2), max(g1, g2)) in structural:
                continue
            if float(contact.dist) < deepest:
                deepest = float(contact.dist)
                if deepest < worst_self:
                    worst_self = deepest
                    worst_pair = (
                        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g1])),
                        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g2])),
                    )
        if deepest < -0.002:
            bad_frames += 1
    print(f"  structural overlaps excluded: {len(structural)} geom pairs")
    print(f"  frames self-penetrating over 2 mm: {bad_frames}/{len(frames)}   worst {worst_self * 1000:.1f} mm"
          + (f"   ({worst_pair[0]} into {worst_pair[1]})" if worst_pair else ""))

    # ---- balance -------------------------------------------------------------
    # Nothing in the solve keeps the robot over its own feet. The source human is balanced because
    # it is real motion, and a pure similarity of it inherits that; any rule that moves the hands
    # without moving the feet does not. A reference whose centre of mass leaves the support polygon
    # cannot be tracked by a policy, so it is measured on every run.
    print("\n[balance]")
    outside = []
    for frame in frames:
        data.qpos[:] = qpos[frame]
        mujoco.mj_forward(model, data)
        com = data.subtree_com[0][:2]
        support = []
        for name, world in robot_geom_vertices(model, data, args.robot_stride):
            if any(k in name for k in FOOT_PATTERN):
                support.append(world[world[:, 2] < 0.03])
        support = [a for a in support if len(a)]
        if not support:
            outside.append(0.0)
            continue
        polygon = np.concatenate(support)[:, :2]
        try:
            inside = Delaunay(polygon).find_simplex(com[None, :])[0] >= 0
        except Exception:
            inside = False
        outside.append(0.0 if inside else float(np.linalg.norm(polygon - com, axis=1).min()))
    outside = np.asarray(outside)
    print(f"  centre of mass outside the support polygon: {100 * (outside > 1e-6).mean():.0f}% of frames"
          f"   mean {outside.mean() * 1000:.0f} mm   worst {outside.max() * 1000:.0f} mm")

    # ---- smoothness ----------------------------------------------------------
    print("\n[smoothness]")
    joints = qpos[:, 7:]
    accel = np.abs(np.diff(joints, n=2, axis=0)) * 900.0
    sign = np.sign(np.diff(joints, axis=0))
    chatter = float((sign[1:] != sign[:-1]).mean())
    root_accel = np.linalg.norm(np.diff(qpos[:, :3], n=2, axis=0), axis=1) * 900.0
    quat_step = np.linalg.norm(np.diff(qpos[:, 3:7], axis=0), axis=1)
    print(f"  joint |accel|   mean {accel.mean():7.3f}  p95 {np.percentile(accel, 95):8.3f} rad/s^2")
    print(f"  joint sign-flip fraction (chatter)  {chatter:.4f}")
    print(f"  root  |accel|   mean {root_accel.mean():7.3f}  p95 {np.percentile(root_accel, 95):8.3f} m/s^2")
    print(f"  root quaternion step  p95 {np.percentile(quat_step, 95):.5f}  max {quat_step.max():.5f}")
    print()


if __name__ == "__main__":
    main()
