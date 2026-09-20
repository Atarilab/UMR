#!/usr/bin/env python3
"""Stage 2: drive the Inspire RH56E2 fingertips onto the MANO fingertips with mink.

Stage 1 (humanoid_retarget_pipeline_hsi_hoi.py) places the whole body, but its
correspondence objective only fits the body SURFACE. It has nothing that says
"this fingertip belongs on that fingertip", so the fingers end up wherever the
surface fit leaves them.

This stage re-solves the fingers AND the arm chain with one inverse-kinematics
task per fingertip, targeting the corresponding SMPL-X/MANO fingertip. Legs,
torso and the floating base stay frozen, so the stance is untouched.

The arms have to be free: measured on this sample, about three quarters of the
fingertip error is hand PLACEMENT rather than finger pose, and no amount of
finger articulation moves a wrist. Arms are held toward their stage-1 pose by a
posture term (--arm-posture-cost) rather than frozen, so they deviate only as far
as a fingertip demands.

Targets default to the ORIGINAL, unscaled MANO fingertips (--tip-frame). With
retarget_object_size="object_frame" the object is unscaled too, so those are the
real contact points. Scaling the targets instead drops them ~8 cm and puts the
hands on the wrong part of the object.

--view opens the result with the HSI/HOI object meshes and green MANO fingertip
markers, so contact can actually be judged.

    python scripts/refine_inspire_fingertips_mink.py \
        --result output/unitree_g1_rh56e2_retarget/chair_tucking_hsi_hoi_unitree_g1_rh56e2.npz \
        --sequence sample_data/embody/chair_tucking

Writes <result stem>_fingertips.npz beside the input unless --out is given.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import mujoco  # noqa: E402
import mink  # noqa: E402

# SMPL-X fingertip landmark indices, from smplx.joint_names.JOINT_NAMES.
MANO_TIPS = {
    "left":  {"thumb": 66, "index": 67, "middle": 68, "ring": 69, "pinky": 70},
    "right": {"thumb": 71, "index": 72, "middle": 73, "ring": 74, "pinky": 75},
}
# Matching force-sensor sites on the distal link of each Inspire finger.
INSPIRE_TIP_SITES = {
    "left":  {"thumb": "l_rh_thumb_force_sensor_4", "index": "l_rh_index_force_sensor_3",
              "middle": "l_rh_middle_force_sensor_3", "ring": "l_rh_ring_force_sensor_3",
              "pinky": "l_rh_pinky_force_sensor_3"},
    "right": {"thumb": "r_rh_thumb_force_sensor_4", "index": "r_rh_index_force_sensor_3",
              "middle": "r_rh_middle_force_sensor_3", "ring": "r_rh_ring_force_sensor_3",
              "pinky": "r_rh_pinky_force_sensor_3"},
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--result", type=Path, required=True, help="Stage-1 retarget .npz")
    p.add_argument("--sequence", type=Path, required=True, help="Source sequence dir (poses.npy, betas.npy, ...)")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--smplx-dir", type=Path, default=ROOT / "smpl")
    p.add_argument("--tip-cost", type=float, default=1.0)
    p.add_argument("--posture-cost", type=float, default=1e-2,
                   help="Keeps the fingers near their stage-1 pose where a tip is unreachable.")
    p.add_argument("--iters", type=int, default=12, help="IK iterations per frame.")
    p.add_argument("--dt", type=float, default=1.0, help="Integration step for the IK velocity.")
    p.add_argument("--damping", type=float, default=1e-2)
    p.add_argument("--solver", default="daqp")
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--solve", choices=("fingers", "fingers+wrists", "fingers+arms", "fingers+arms+waist"),
                   default="fingers+arms",
                   help="Which chain the IK may move. Most fingertip error is hand PLACEMENT, which "
                        "finger joints alone cannot fix, so the arms are freed by default. Everything "
                        "not listed stays frozen at its stage-1 value.")
    p.add_argument("--arm-posture-cost", type=float, default=0.3,
                   help="How strongly arms/waist are held to their stage-1 pose. Higher keeps more of "
                        "stage 1's body fit; lower lets the arm reach further for a fingertip.")
    p.add_argument("--tip-frame", choices=("original", "scaled"), default="original",
                   help="Which MANO fingertips to chase. 'original' (default) is where the real hand "
                        "actually was. 'scaled' scales them with the body and lands the hands on the "
                        "wrong part of the object. See to_robot_frame.")
    p.add_argument("--view-only", action="store_true",
                   help="Skip the IK and just display --result as-is, with objects and tip markers. "
                        "Use it to inspect a stage-2 output that already exists.")
    p.add_argument("--view", action="store_true", help="Play the refined motion after solving.")
    return p.parse_args()


def mano_fingertips_world(seq_dir: Path, smplx_dir: Path, n_frames: int) -> np.ndarray:
    """SMPL-X fingertip positions per frame, in the SOURCE world frame. (T, 10, 3)"""
    import torch
    import smplx

    poses = np.load(seq_dir / "poses.npy").astype(np.float32)[:n_frames]
    transl = np.load(seq_dir / "transl.npy").astype(np.float32)[:n_frames]
    betas = np.load(seq_dir / "betas.npy").astype(np.float32).reshape(1, -1)[:, :10]
    gender = str(np.load(seq_dir / "gender.npy", allow_pickle=True))

    # This repo keeps SMPLX_NEUTRAL.pkl directly in smpl/, not in the nested
    # smplx/ layout smplx.create() expects, so hand the file path straight over
    # the way retarget_body_segment_surface_hoi_hsi.py does.
    direct = Path(smplx_dir) / f"SMPLX_{gender.upper()}.pkl"
    model = smplx.SMPLX(
        str(direct if direct.is_file() else smplx_dir),
        gender=gender, use_pca=False, flat_hand_mean=True, ext="pkl",
        batch_size=len(poses), num_betas=betas.shape[1],
    )
    t = lambda a: torch.as_tensor(a, dtype=torch.float32)
    with torch.no_grad():
        out = model(
            betas=t(np.repeat(betas, len(poses), axis=0)),
            global_orient=t(poses[:, 0:3]),
            body_pose=t(poses[:, 3:66]),
            jaw_pose=t(poses[:, 66:69]),
            leye_pose=t(poses[:, 69:72]),
            reye_pose=t(poses[:, 72:75]),
            left_hand_pose=t(poses[:, 75:120]),
            right_hand_pose=t(poses[:, 120:165]),
            transl=t(transl),
        )
    joints = out.joints.detach().cpu().numpy()
    order = [MANO_TIPS[s][f] for s in ("left", "right")
             for f in ("thumb", "index", "middle", "ring", "pinky")]
    return joints[:, order, :].astype(np.float64)


def to_robot_frame(points: np.ndarray, result, tip_frame: str = "original") -> np.ndarray:
    """Put MANO fingertips into the world the robot lives in.

    The up-axis basis and the ground shift are frame corrections both modes need.
    Only the smpl_scale step is a choice:

    "original" (default) keeps the tips where the real hand actually was. Under
    retarget_object_size="object_frame" the object is unscaled too, so these ARE
    the true contact points. Verified visually on chair_tucking: the human grabs
    the chair BACK, and only these targets put the robot's hands there.

    "scaled" scales the tips about the object frame centre like stage 1 scales
    the body. It drags the targets ~8 cm down, so the hands land on the chair
    SEAT instead of the back. Do not trust a "distance to nearest point on the
    object" metric here: resting on the wrong part of the object scores well on
    it. Measured against the real MANO fingertips, which is the honest ground
    truth:

        mean robot fingertip error vs real MANO tips
        stage 1 only                     0.3468 m
        stage 2, scaled targets          0.2935 m
        stage 2, original targets        0.0293 m
    """
    import smpl_surface_retarget_common as common

    up = str(result["noitom_output_up"]) if "noitom_output_up" in result else "z"
    pts = common.source_points_to_retarget_frame(points.astype(np.float32), "smplx", up).astype(np.float64)

    ground_z = float(np.asarray(result["ground_z"]).reshape(-1)[0]) if "ground_z" in result else 0.0
    pts[..., 2] -= ground_z

    if tip_frame == "original":
        return pts

    scale = float(np.asarray(result["smpl_scale"]).reshape(-1)[0])
    if "object_frame_origin" in result:
        c = np.asarray(result["object_frame_origin"], dtype=np.float64).reshape(-1, 3)
        c = c[: len(pts)] if len(c) >= len(pts) else np.repeat(c[:1], len(pts), axis=0)
    else:
        c = np.zeros((len(pts), 3))
    return (pts - c[:, None, :]) * scale + c[:, None, :]


ARM_KEYS = ("shoulder", "elbow", "wrist")


def classify_dofs(model, scope: str) -> tuple[list[int], list[int], list[int]]:
    """(finger dofs, arm/waist dofs the IK may move, frozen dofs).

    The Inspire joints are NOT contiguous in the DOF array: each hand attaches
    mid-tree after its wrist, so selecting by index range silently grabs leg
    DOFs. Everything here is selected by joint name instead.
    """
    fingers, arms, frozen = [], [], []
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
        adr = int(model.jnt_dofadr[j])
        n = {mujoco.mjtJoint.mjJNT_FREE: 6, mujoco.mjtJoint.mjJNT_BALL: 3}.get(
            mujoco.mjtJoint(int(model.jnt_type[j])), 1)
        idx = list(range(adr, adr + n))
        if name.startswith(("l_rh_", "r_rh_")):
            fingers.extend(idx)
        elif "wrist" in name and scope in ("fingers+wrists", "fingers+arms", "fingers+arms+waist"):
            arms.extend(idx)
        elif any(k in name for k in ARM_KEYS) and scope in ("fingers+arms", "fingers+arms+waist"):
            arms.extend(idx)
        elif "waist" in name and scope == "fingers+arms+waist":
            arms.extend(idx)
        else:
            frozen.extend(idx)
    return fingers, arms, frozen



def build_scene_with_objects(robot_xml: Path, seq_dir: Path, n_tips: int):
    """Robot + HSI/HOI object meshes + fingertip target markers, as one model.

    The combined file is written NEXT TO the robot XML on purpose: the Inspire
    model pulls its hands in via <model file="../rh56e2_left.xml"/>, which
    resolves relative to the including file. Writing the scene anywhere else
    silently breaks those includes.
    """
    import xml.etree.ElementTree as ET

    tree = ET.parse(robot_xml)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")

    stems = []
    for obj_xml in sorted(seq_dir.glob("*.xml")):
        if not (seq_dir / f"prop_{obj_xml.stem}.csv").exists():
            continue
        oroot = ET.parse(obj_xml).getroot()
        compiler = oroot.find("compiler")
        meshdir = compiler.get("meshdir") if compiler is not None else None
        base = (obj_xml.parent / meshdir).resolve() if meshdir else obj_xml.parent.resolve()
        for oasset in oroot.findall("asset"):
            for mesh in oasset.findall("mesh"):
                mesh = ET.fromstring(ET.tostring(mesh))
                mesh.set("name", f"obj_{obj_xml.stem}_{mesh.get('name')}")
                if mesh.get("file"):
                    mesh.set("file", str((base / mesh.get("file")).resolve()))
                asset.append(mesh)
        ob = oroot.find("worldbody")
        if ob is None:
            continue
        for body in ob.findall("body"):
            body = ET.fromstring(ET.tostring(body))
            body.set("name", f"obj_{obj_xml.stem}")
            for j in body.findall("freejoint"):
                j.set("name", f"obj_{obj_xml.stem}_free")
            for g in body.iter("geom"):
                if g.get("mesh"):
                    g.set("mesh", f"obj_{obj_xml.stem}_{g.get('mesh')}")
                if g.get("name"):
                    g.set("name", f"obj_{obj_xml.stem}_{g.get('name')}")
                # visual only: the marker scene must not perturb contact
                g.set("contype", "0"); g.set("conaffinity", "0")
            worldbody.append(body)
        stems.append(obj_xml.stem)

    # fingertip targets as mocap markers: green = MANO target
    for i in range(n_tips):
        mb = ET.SubElement(worldbody, "body")
        mb.set("name", f"tip_target_{i}"); mb.set("mocap", "true"); mb.set("pos", "0 0 -5")
        g = ET.SubElement(mb, "geom")
        g.set("type", "sphere"); g.set("size", "0.008"); g.set("rgba", "0.1 0.9 0.2 0.9")
        g.set("contype", "0"); g.set("conaffinity", "0")

    out = robot_xml.with_name(robot_xml.stem + "_fingertip_scene.xml")
    tree.write(out, encoding="unicode")
    return out, stems



def view_scene(args, result, refined, targets, tips, n_frames):
    """Play the refined motion with the objects and the MANO fingertip markers."""
    import csv
    import time
    import mujoco.viewer as mj_viewer
    from scipy.spatial.transform import Rotation as R

    robot_xml = Path(str(result["robot_xml"]))
    seq_dir = Path(args.sequence)
    scene_path, stems = build_scene_with_objects(robot_xml, seq_dir, len(tips))
    scene = mujoco.MjModel.from_xml_path(str(scene_path))
    sdata = mujoco.MjData(scene)
    print(f"[Fingertips] scene: robot + {len(stems)} object(s) {stems} + {len(tips)} tip markers")

    # Object trajectories. Under retarget_object_size="object_frame" the solver
    # leaves them completely unscaled, so replay the prop CSV verbatim; any other
    # mode scaled them by smpl_scale about the world origin.
    mode = str(result["retarget_object_size"]) if "retarget_object_size" in result else "scaled"
    pscale = 1.0 if mode == "object_frame" else float(np.asarray(result["smpl_scale"]).reshape(-1)[0])
    obj = {}
    for stem in stems:
        rows = list(csv.DictReader((seq_dir / f"prop_{stem}.csv").open(newline="")))
        pos = np.array([[float(r["px"]), float(r["py"]), float(r["pz"])] for r in rows]) * pscale
        xyzw = np.array([[float(r["qx"]), float(r["qy"]), float(r["qz"]), float(r["qw"])] for r in rows])
        wxyz = np.concatenate([xyzw[:, 3:4], xyzw[:, :3]], axis=1)
        jid = mujoco.mj_name2id(scene, mujoco.mjtObj.mjOBJ_JOINT, f"obj_{stem}_free")
        if jid < 0:
            continue
        obj[stem] = (int(scene.jnt_qposadr[jid]), pos, wxyz)

    robot_jid = mujoco.mj_name2id(scene, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint")
    robot_adr = 0 if robot_jid < 0 else int(scene.jnt_qposadr[robot_jid])
    nq_robot = refined.shape[1]
    mocap = [scene.body(f"tip_target_{i}").mocapid[0] for i in range(len(tips))]

    print("[Fingertips] green spheres are the MANO targets; close the window to stop.")
    try:
        _play(scene, sdata, viewer_ctx=mj_viewer, refined=refined, targets=targets,
              obj=obj, mocap=mocap, robot_adr=robot_adr, nq_robot=nq_robot, n_frames=n_frames)
    finally:
        scene_path.unlink(missing_ok=True)


def _play(scene, sdata, viewer_ctx, refined, targets, obj, mocap, robot_adr, nq_robot, n_frames):
    import time
    with viewer_ctx.launch_passive(scene, sdata, show_left_ui=False, show_right_ui=False) as viewer:
        mujoco.mjv_defaultFreeCamera(scene, viewer.cam)
        f = 0
        while viewer.is_running():
            k = f % n_frames
            sdata.qpos[robot_adr:robot_adr + nq_robot] = refined[k]
            for adr, pos, wxyz in obj.values():
                i = min(k, len(pos) - 1)
                sdata.qpos[adr:adr + 3] = pos[i]
                sdata.qpos[adr + 3:adr + 7] = wxyz[i]
            for i, mid in enumerate(mocap):
                sdata.mocap_pos[mid] = targets[k, i]
            mujoco.mj_forward(scene, sdata)
            viewer.sync()
            time.sleep(1.0 / 30.0)
            f += 1


def main() -> int:
    args = parse_args()
    result = np.load(args.result, allow_pickle=True)
    qpos = np.asarray(result["qpos"], dtype=np.float64)
    if args.max_frames > 0:
        qpos = qpos[: args.max_frames]
    n = len(qpos)

    model = mujoco.MjModel.from_xml_path(str(result["robot_xml"]))
    if model.nq != qpos.shape[1]:
        raise ValueError(f"qpos width {qpos.shape[1]} does not match model nq {model.nq}")

    site_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i) for i in range(model.nsite)]
    tips = [(s, f, INSPIRE_TIP_SITES[s][f]) for s in ("left", "right")
            for f in ("thumb", "index", "middle", "ring", "pinky")]
    missing = [t[2] for t in tips if t[2] not in site_names]
    if missing:
        raise ValueError(f"Robot model has no Inspire fingertip sites: {missing}. "
                         "This stage needs the rh56e2 model, not the dummy-hand G1.")

    print(f"[Fingertips] frames={n} robot={str(result['robot_name'])}")
    targets = to_robot_frame(mano_fingertips_world(args.sequence, args.smplx_dir, n), result, args.tip_frame)
    print(f"[Fingertips] tip_frame={args.tip_frame}: MANO tips mapped into the robot frame {targets.shape}")

    if args.view_only:
        data = mujoco.MjData(model)
        err = []
        for f in range(n):
            data.qpos[:] = qpos[f]; mujoco.mj_forward(model, data)
            err.append(np.linalg.norm(
                [data.site_xpos[site_names.index(s)] - targets[f, i] for i, (_, _, s) in enumerate(tips)],
                axis=1).mean())
        print(f"[Fingertips] view only, no IK. mean tip error of this result: {float(np.mean(err)):.4f} m")
        view_scene(args, result, qpos, targets, tips, n)
        return 0

    configuration = mink.Configuration(model)
    finger_dofs, arm_dofs, body_dofs = classify_dofs(model, args.solve)
    # Per-DOF posture cost: fingers are cheap to move, the arm chain is held near
    # its stage-1 pose so it only deviates when a fingertip genuinely needs it.
    posture_cost = np.zeros(model.nv)
    posture_cost[finger_dofs] = args.posture_cost
    posture_cost[arm_dofs] = args.arm_posture_cost
    tasks = [mink.PostureTask(model=model, cost=posture_cost)]
    tip_tasks = []
    for _, _, site in tips:
        task = mink.FrameTask(frame_name=site, frame_type="site",
                              position_cost=args.tip_cost, orientation_cost=0.0, lm_damping=1.0)
        tip_tasks.append(task)
    tasks.extend(tip_tasks)

    print(f"[Fingertips] scope={args.solve}: {len(finger_dofs)} finger + {len(arm_dofs)} arm/waist "
          f"DOFs solved, {len(body_dofs)} frozen")
    limits = [mink.ConfigurationLimit(model)]
    # Freezing is an equality CONSTRAINT in mink, passed separately from limits.
    constraints = [mink.DofFreezingTask(model=model, dof_indices=body_dofs)]

    refined = qpos.copy()
    errs_before, errs_after = [], []
    data = mujoco.MjData(model)
    for f in range(n):
        configuration.update(qpos[f].copy())
        tasks[0].set_target_from_configuration(configuration)

        data.qpos[:] = qpos[f]
        mujoco.mj_forward(model, data)
        errs_before.append(np.linalg.norm(
            [data.site_xpos[site_names.index(s)] - targets[f, i] for i, (_, _, s) in enumerate(tips)], axis=1).mean())

        for i, (_, _, _site) in enumerate(tips):
            tip_tasks[i].set_target(mink.SE3.from_rotation_and_translation(
                mink.SO3.identity(), targets[f, i]))
        for _ in range(args.iters):
            vel = mink.solve_ik(configuration, tasks, args.dt, args.solver,
                                damping=args.damping, limits=limits, constraints=constraints)
            configuration.integrate_inplace(vel, args.dt)
        refined[f] = configuration.q.copy()

        data.qpos[:] = refined[f]
        mujoco.mj_forward(model, data)
        errs_after.append(np.linalg.norm(
            [data.site_xpos[site_names.index(s)] - targets[f, i] for i, (_, _, s) in enumerate(tips)], axis=1).mean())
        if f % 100 == 0:
            print(f"  frame {f:4d}/{n}  tip error {errs_before[-1]:.4f} -> {errs_after[-1]:.4f} m")

    def split(qseq):
        """Mean absolute tip error, and the part left after removing the whole-hand offset."""
        a, r = [], []
        for f in range(n):
            data.qpos[:] = qseq[f]; mujoco.mj_forward(model, data)
            P = np.array([data.site_xpos[site_names.index(s)] for _, _, s in tips])
            for sl in (slice(0, 5), slice(5, 10)):
                p, t = P[sl], targets[f, sl]
                a.append(float(np.linalg.norm(p - t, axis=1).mean()))
                r.append(float(np.linalg.norm((p - p.mean(0)) - (t - t.mean(0)), axis=1).mean()))
        return float(np.mean(a)), float(np.mean(r))

    a0, r0 = split(qpos)
    a1, r1 = split(refined)
    print(f"[Fingertips] absolute tip error   {a0:.4f} -> {a1:.4f} m")
    print(f"[Fingertips] finger-pose error    {r0:.4f} -> {r1:.4f} m  (hand offset removed)")
    print(f"[Fingertips] placement share of the remaining error: {100*(1-r1/max(a1,1e-9)):.0f}%")
    eb, ea = float(np.mean(errs_before)), float(np.mean(errs_after))
    moved = float(np.abs(refined[:, :] - qpos[:, :]).max())
    print(f"[Fingertips] mean tip error {eb:.4f} -> {ea:.4f} m  ({100*(1-ea/max(eb,1e-9)):.1f}% reduction)")
    print(f"[Fingertips] max qpos change {moved:.4f} rad")

    out = args.out or args.result.with_name(args.result.stem + "_fingertips.npz")
    payload = {k: result[k] for k in result.files}
    payload["qpos"] = refined.astype(np.float32)
    np.savez_compressed(out, **payload)
    print(f"[Fingertips] saved {out}")

    if args.view:
        view_scene(args, result, refined, targets, tips, n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
