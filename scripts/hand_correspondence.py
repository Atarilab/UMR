"""Dedicated SMPL-X <-> robot hand surface correspondence.

The whole-body correspondence gives each hand about 2% of the slots, too few
for fingers. Each hand is sampled on its own in a wrist-aligned canonical frame
and normalised per part (palm, thumb, index, middle, ring, pinky), because a
robot hand is not a scaled human hand: the Inspire RH56E2 palm is twice as long
as a human palm while its fingers are human-sized. Parts are "exploded" along
the canonical z axis so Chamfer matching never pairs points across parts. The
learned hand slots replace the body slots that fell on the hands.
"""
from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEGMENTATION_PATH = ROOT / "assets" / "smplx_parts_segm.pkl"

HAND_PARTS = ("palm", "thumb", "index", "middle", "ring", "pinky")
FINGER_PARTS = HAND_PARTS[1:]
SIDES = ("left", "right")
SIDE_INDEX = {"left": 0, "right": 1}
SMPLX_HAND_JOINTS = {
    "left": {"wrist": 20, "index": 25, "middle": 28, "pinky": 31, "ring": 34, "thumb": 37,
             "tips": {"thumb": 66, "index": 67, "middle": 68, "ring": 69, "pinky": 70}},
    "right": {"wrist": 21, "index": 40, "middle": 43, "pinky": 46, "ring": 49, "thumb": 52,
              "tips": {"thumb": 71, "index": 72, "middle": 73, "ring": 74, "pinky": 75}},
}
# assets/smplx_parts_segm.pkl labels each face with the SMPL-X joint it follows.
SMPLX_HAND_FACE_LABELS = {
    "left": {"palm": [20], "index": [25, 26, 27], "middle": [28, 29, 30], "pinky": [31, 32, 33],
             "ring": [34, 35, 36], "thumb": [37, 38, 39]},
    "right": {"palm": [21], "index": [40, 41, 42], "middle": [43, 44, 45], "pinky": [46, 47, 48],
              "ring": [49, 50, 51], "thumb": [52, 53, 54]},
}
SMPLX_HAND_LABELS = sorted({label for side in SMPLX_HAND_FACE_LABELS.values() for labels in side.values() for label in labels})


def smplx_face_labels() -> np.ndarray:
    with SEGMENTATION_PATH.open("rb") as handle:
        return np.asarray(pickle.load(handle, encoding="latin1")["segm"], dtype=np.int32)


def hand_frame(origin, middle, index, pinky, side):
    """Rotation(s) with columns x (wrist->middle knuckle), y (pinky->index), z (out of the palm).

    The right-hand frame is a reflection of the left one, so both hands overlay
    in canonical coordinates.
    """
    origin, middle, index, pinky = (np.asarray(a, dtype=np.float64) for a in (origin, middle, index, pinky))
    x = middle - origin
    x = x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)
    y = index - pinky
    y = y - np.sum(y * x, axis=-1, keepdims=True) * x
    y = y / np.maximum(np.linalg.norm(y, axis=-1, keepdims=True), 1e-12)
    z = np.cross(x, y) if side == "left" else np.cross(y, x)
    return np.stack([x, y, z], axis=-1)


def to_canonical(points, origin, rot):
    return (np.asarray(points, dtype=np.float64) - origin) @ rot


def from_canonical(coords, origin, rot):
    return np.asarray(coords, dtype=np.float64) @ np.swapaxes(rot, -1, -2) + origin


def part_parameters(part_vertices, part_origins):
    """Per-part (origin, scale): palm by bounding box, fingers by MCP and isotropic length."""
    origin = np.zeros((len(HAND_PARTS), 3), dtype=np.float64)
    scale = np.ones((len(HAND_PARTS), 3), dtype=np.float64)
    for k, part in enumerate(HAND_PARTS):
        verts = np.asarray(part_vertices[part], dtype=np.float64)
        if part == "palm":
            origin[k] = verts.min(axis=0)
            scale[k] = np.maximum(verts.max(axis=0) - verts.min(axis=0), 1e-6)
        else:
            origin[k] = np.asarray(part_origins[part], dtype=np.float64)
            scale[k] = max(float(np.linalg.norm(verts - origin[k], axis=1).max()), 1e-6)
    return origin, scale


def part_offsets(part_offset):
    offsets = np.zeros((len(HAND_PARTS), 3), dtype=np.float64)
    offsets[:, 2] = float(part_offset) * np.arange(len(HAND_PARTS))
    return offsets


def normalize_parts(coords, part_ids, origin, scale, part_offset):
    part_ids = np.asarray(part_ids, dtype=np.int32)
    return (np.asarray(coords) - origin[part_ids]) / scale[part_ids] + part_offsets(part_offset)[part_ids]


def denormalize_parts(normalized, part_ids, origin, scale, part_offset):
    part_ids = np.asarray(part_ids, dtype=np.int32)
    return (np.asarray(normalized) - part_offsets(part_offset)[part_ids]) * scale[part_ids] + origin[part_ids]


def part_of_normalized(normalized, part_offset):
    return np.clip(np.round(np.asarray(normalized)[..., 2] / float(part_offset)), 0, len(HAND_PARTS) - 1).astype(np.int32)


def map_parts(coords, part_ids, origin_src, scale_src, origin_dst, scale_dst):
    """Canonical coordinates from one hand to the other, part by part."""
    part_ids = np.asarray(part_ids, dtype=np.int32)
    return origin_dst[part_ids] + (scale_dst[part_ids] / scale_src[part_ids]) * (np.asarray(coords) - origin_src[part_ids])


def _part_meshes(vertices, faces, face_parts):
    """Per-part submeshes (metric), concatenated in HAND_PARTS order."""
    meshes = {}
    for k, part in enumerate(HAND_PARTS):
        part_faces = faces[face_parts == k]
        if len(part_faces) == 0:
            raise ValueError(f"hand part {part!r} has no faces")
        used, inverse = np.unique(part_faces.reshape(-1), return_inverse=True)
        meshes[part] = (vertices[used], inverse.reshape(-1, 3).astype(np.int32))
    return meshes


def _sample_parts(meshes, params, num_points, seed, oversample_ratio, part_offset, exterior):
    from build_correspondence_ae_dataset import sample_surface  # noqa: WPS433

    verts, faces, face_part = [], [], []
    offset = 0
    for k, part in enumerate(HAND_PARTS):
        v, f = meshes[part]
        verts.append(v)
        faces.append(f + offset)
        face_part.append(np.full(len(f), k, dtype=np.int32))
        offset += len(v)
    verts = np.concatenate(verts)
    faces = np.concatenate(faces)
    face_part = np.concatenate(face_part)
    kwargs = {"exterior_only": True, "exterior_method": "first_hit"} if exterior else {}
    points, face_ids = sample_surface(verts, faces, num_points, seed, oversample_ratio, **kwargs)
    point_parts = face_part[face_ids]
    origin, scale = params
    vertex_parts = np.zeros(len(verts), dtype=np.int32)
    vertex_parts[faces.reshape(-1)] = np.repeat(face_part, 3)
    return {
        "points": normalize_parts(points, point_parts, origin, scale, part_offset).astype(np.float32),
        "point_parts": point_parts.astype(np.int32),
        "vertices": normalize_parts(verts, vertex_parts, origin, scale, part_offset).astype(np.float32),
        "faces": faces.astype(np.int32),
        "sample_face_ids": face_ids.astype(np.int32),
    }


def build_human_hand_sample(vertices, faces, joints, side, num_points, seed, oversample_ratio, part_offset):
    """SMPL-X hand of ``side`` from the centred template (body dataset frame)."""
    labels = smplx_face_labels()
    if len(labels) != len(faces):
        raise ValueError(f"SMPL-X segmentation has {len(labels)} labels for {len(faces)} faces")
    ids = SMPLX_HAND_JOINTS[side]
    origin = np.asarray(joints[ids["wrist"]], dtype=np.float64)
    rot = hand_frame(origin, joints[ids["middle"]], joints[ids["index"]], joints[ids["pinky"]], side)
    face_parts = np.full(len(faces), -1, dtype=np.int32)
    for k, part in enumerate(HAND_PARTS):
        face_parts[np.isin(labels, SMPLX_HAND_FACE_LABELS[side][part])] = k
    canonical = to_canonical(vertices, origin, rot)
    meshes = _part_meshes(canonical, np.asarray(faces, dtype=np.int32), face_parts)
    landmarks = {part: to_canonical(joints[ids[part]], origin, rot) for part in FINGER_PARTS}
    params = part_parameters({part: mesh[0] for part, mesh in meshes.items()}, landmarks)
    sample = _sample_parts(meshes, params, num_points, seed, oversample_ratio, part_offset, exterior=False)
    sample.update({"frame_origin": origin, "frame_axes": rot, "part_origin": params[0], "part_scale": params[1]})
    return sample


def robot_hand_pose(model, robot, sample_qpos, joint_couplings=None):
    """MjData at the correspondence sample pose (T-pose joints, mimic, couplings)."""
    from build_correspondence_ae_dataset import apply_joint_qpos, apply_mimic_qpos  # noqa: WPS433

    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    apply_joint_qpos(model, data, sample_qpos or {}, required=False)
    mimic = robot.get("mimic_qpos") or {}
    apply_mimic_qpos(model, data, {k: (v["source"], v["multiplier"]) if isinstance(v, dict) else tuple(v) for k, v in mimic.items()})
    if joint_couplings is not None:
        data.qpos[:] = joint_couplings.project(data.qpos)
    mujoco.mj_forward(model, data)
    return data


def _body_id(model, name):
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, str(name))
    if body < 0:
        raise ValueError(f"robot.hands references unknown body {name!r}")
    return body


def robot_sample_frame_points(model, data, points_world, center_name, to_smpl_frame=True):
    """World points in the correspondence sample frame (point-cloud centre, then SMPL axes)."""
    from build_correspondence_ae_dataset import g1_mujoco_to_smpl_frame  # noqa: WPS433
    from mujoco_point_cloud_center import point_cloud_center_frame  # noqa: WPS433

    center_pos, center_rot, _label = point_cloud_center_frame(model, data, center_name)
    root = (np.asarray(points_world, dtype=np.float64) - center_pos) @ center_rot
    return np.asarray(g1_mujoco_to_smpl_frame(root) if to_smpl_frame else root, dtype=np.float64)


def build_robot_hand_sample(model, data, side_cfg, side, center_name, num_points, seed, oversample_ratio, part_offset, to_smpl_frame=True):
    from mujoco_geom_surface import geom_local_mesh, surface_geom_ids  # noqa: WPS433

    wrist = data.xpos[_body_id(model, side_cfg["wrist_body"])]
    frame_pts = {key: data.xpos[_body_id(model, name)] for key, name in side_cfg["frame_bodies"].items()}
    to_frame = lambda pts: robot_sample_frame_points(model, data, np.atleast_2d(pts), center_name, to_smpl_frame)  # noqa: E731
    origin = to_frame(wrist)[0]
    rot = hand_frame(origin, to_frame(frame_pts["middle"])[0], to_frame(frame_pts["index"])[0], to_frame(frame_pts["pinky"])[0], side)
    visual = set(int(g) for g in surface_geom_ids(model))
    meshes = {}
    landmarks = {}
    for part in HAND_PARTS:
        spec = side_cfg["parts"][part]
        bodies = {_body_id(model, name) for name in spec["bodies"]}
        verts, faces, offset = [], [], 0
        for geom in sorted(visual):
            if int(model.geom_bodyid[geom]) not in bodies:
                continue
            local_v, local_f = geom_local_mesh(model, geom)
            world = np.asarray(local_v, dtype=np.float64) @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
            verts.append(to_canonical(to_frame(world), origin, rot))
            faces.append(np.asarray(local_f, dtype=np.int32) + offset)
            offset += len(local_v)
        if not verts:
            raise ValueError(f"robot hand part {side}/{part} has no visual geoms on bodies {spec['bodies']}")
        meshes[part] = (np.concatenate(verts), np.concatenate(faces))
        if part != "palm":
            landmarks[part] = to_canonical(to_frame(data.xpos[_body_id(model, spec["origin_body"])])[0], origin, rot)
    params = part_parameters({part: mesh[0] for part, mesh in meshes.items()}, landmarks)
    sample = _sample_parts(meshes, params, num_points, seed, oversample_ratio, part_offset, exterior=True)
    sample.update({"frame_origin": origin, "frame_axes": rot, "part_origin": params[0], "part_scale": params[1]})
    return sample


def hand_config_hash(payload) -> str:
    return hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def write_hand_dataset(out, side, human, robot, human_name, robot_name, num_points, seed, oversample_ratio, part_offset, config_hash):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save = {
        "names": np.asarray([human_name, robot_name]),
        "points": np.stack([human["points"], robot["points"]]).astype(np.float32),
        "root_offsets": np.zeros((2, 3), dtype=np.float32),
        "center_modes": np.asarray([f"hand_{side}_canonical"] * 2),
        "num_points": np.asarray(num_points, dtype=np.int32),
        "seed": np.asarray(seed, dtype=np.int32),
        "surface_oversample_ratio": np.asarray(oversample_ratio, dtype=np.int32),
        "surface_curvature_weight": np.asarray(0.0, dtype=np.float32),
        "surface_curvature_power": np.asarray(1.0, dtype=np.float32),
        "robot_exterior_surface": np.asarray(True),
        "robot_exterior_occlusion_distance": np.asarray(0.12, dtype=np.float32),
        "robot_exterior_method": np.asarray("first_hit"),
        "robot_exterior_ray_distance": np.asarray(0.0, dtype=np.float32),
        "custom_robot_name": np.asarray(robot_name),
        "hand_side": np.asarray(side),
        "hand_part_names": np.asarray(HAND_PARTS),
        "hand_part_offset": np.asarray(float(part_offset), dtype=np.float64),
        "hand_part_origin": np.stack([human["part_origin"], robot["part_origin"]]),
        "hand_part_scale": np.stack([human["part_scale"], robot["part_scale"]]),
        "hand_frame_origin": np.stack([human["frame_origin"], robot["frame_origin"]]),
        "hand_frame_axes": np.stack([human["frame_axes"], robot["frame_axes"]]),
        "point_part_ids": np.stack([human["point_parts"], robot["point_parts"]]),
        "meta_hand_config_hash": np.asarray(config_hash),
    }
    for idx, sample in enumerate((human, robot)):
        save[f"mesh_vertices_{idx}"] = sample["vertices"]
        save[f"mesh_faces_{idx}"] = sample["faces"]
        save[f"sample_face_ids_{idx}"] = sample["sample_face_ids"]
        save[f"betas_{idx}"] = np.zeros(0, dtype=np.float32)
    np.savez_compressed(out, **save)
    return out


def _robot_body_face_ids(model, data):
    """Body id per face of the robot body-dataset mesh (same geom order as the dataset builder)."""
    from mujoco_geom_surface import geom_local_mesh, surface_geom_ids  # noqa: WPS433

    ids = []
    for geom in surface_geom_ids(model):
        _v, f = geom_local_mesh(model, int(geom))
        ids.append(np.full(len(f), int(model.geom_bodyid[int(geom)]), dtype=np.int32))
    return np.concatenate(ids)


def merge_hand_slots(body_slots_path, body_dataset_path, hand_slots_paths, hand_dataset_paths, out, model, robot, human_name, robot_name, config_hash):
    """Body slots with the hand slots replacing the body slots that fall on either hand."""
    import smpl_surface_retarget_common as common  # noqa: WPS433

    body = dict(np.load(body_slots_path, allow_pickle=True))
    dataset = np.load(body_dataset_path, allow_pickle=True)
    names = [str(n) for n in body["names"]]
    ih, ir = names.index(human_name), names.index(robot_name)
    ds_names = [str(n) for n in dataset["names"]]
    dh, dr = ds_names.index(human_name), ds_names.index(robot_name)

    # Drop body slots on a hand, judged on either the human or the robot surface.
    labels = smplx_face_labels()
    human_bind = common.bind_points_to_mesh(body["reconstructed_slots"][ih], dataset[f"mesh_vertices_{dh}"], dataset[f"mesh_faces_{dh}"])
    human_on_hand = np.isin(labels[human_bind["face_ids"]], SMPLX_HAND_LABELS)
    robot_faces = dataset[f"mesh_faces_{dr}"]
    face_body = _robot_body_face_ids(model, mujoco.MjData(model))
    if len(face_body) != len(robot_faces):
        raise ValueError(f"robot dataset mesh has {len(robot_faces)} faces, expected {len(face_body)}")
    hand_bodies = {
        _body_id(model, name)
        for side in SIDES
        for part in HAND_PARTS
        for name in robot["hands"]["sides"][side]["parts"][part]["bodies"]
    }
    robot_bind = common.bind_points_to_mesh(body["reconstructed_slots"][ir], dataset[f"mesh_vertices_{dr}"], robot_faces)
    robot_on_hand = np.isin(face_body[robot_bind["face_ids"]], sorted(hand_bodies))
    keep = ~(human_on_hand | robot_on_hand)

    hand_points = {ih: [], ir: []}
    slot_side, slot_part, agreement = [], [], {}
    params = {}
    for side in SIDES:
        slots = np.load(hand_slots_paths[side], allow_pickle=True)
        hds = np.load(hand_dataset_paths[side], allow_pickle=True)
        hn = [str(n) for n in slots["names"]]
        sh, sr = hn.index(human_name), hn.index(robot_name)
        offset = float(hds["hand_part_offset"])
        n_h = slots["reconstructed_slots"][sh]
        n_r = slots["reconstructed_slots"][sr]
        parts = part_of_normalized(n_h, offset)
        robot_parts = part_of_normalized(n_r, offset)
        agreement[side] = float(np.mean(parts == robot_parts))
        origin, scale = hds["hand_part_origin"], hds["hand_part_scale"]
        frame_o, frame_r = hds["hand_frame_origin"], hds["hand_frame_axes"]
        for sample, n, dst in ((0, n_h, ih), (1, n_r, ir)):
            coords = denormalize_parts(n, parts, origin[sample], scale[sample], offset)
            hand_points[dst].append(from_canonical(coords, frame_o[sample], frame_r[sample]).astype(np.float32))
        slot_side.append(np.full(len(parts), SIDE_INDEX[side], dtype=np.int8))
        slot_part.append(parts.astype(np.int8))
        params[side] = {"origin": origin, "scale": scale}

    n_body = int(keep.sum())
    n_hand = sum(len(s) for s in slot_side)
    merged = dict(body)
    slots_per_sample = []
    for s in range(len(names)):
        extra = np.concatenate(hand_points[s]) if s in hand_points else np.concatenate(hand_points[ih])
        slots_per_sample.append(np.concatenate([body["reconstructed_slots"][s][keep], extra]))
    merged["reconstructed_slots"] = np.stack(slots_per_sample).astype(np.float32)
    template = merged["reconstructed_slots"][ih]
    merged["residual_slots"] = (merged["reconstructed_slots"] - template[None]).astype(np.float32)
    centers = np.asarray(body["normalization_centers"], dtype=np.float32)[:, None, :]
    scales = np.asarray(body["normalization_scales"], dtype=np.float32)[:, None, None]
    merged["normalized_reconstructed_slots"] = ((merged["reconstructed_slots"] - centers) / scales).astype(np.float32)
    merged["normalized_residual_slots"] = (merged["residual_slots"] / scales).astype(np.float32)
    merged["template_points"] = np.concatenate([body["template_points"][keep], np.concatenate(hand_points[ih])]).astype(np.float32)
    old_to_new = np.full(len(keep), -1, dtype=np.int64)
    old_to_new[keep] = np.arange(n_body)
    edges = old_to_new[np.asarray(body["template_edge_index"], dtype=np.int64)]
    merged["template_edge_index"] = edges[np.all(edges >= 0, axis=1)]
    merged["slot_index"] = np.arange(n_body + n_hand, dtype=np.int32)
    if "template_sort_index" in body:
        merged["template_sort_index"] = np.concatenate([body["template_sort_index"][keep], np.full(n_hand, -1)]).astype(np.int32)
    merged["hand_slot_side"] = np.concatenate([np.full(n_body, -1, dtype=np.int8)] + slot_side)
    merged["hand_slot_part"] = np.concatenate([np.full(n_body, -1, dtype=np.int8)] + slot_part)
    merged["merged_body_slot_index"] = np.concatenate([np.flatnonzero(keep), np.full(n_hand, -1)]).astype(np.int32)
    merged["hand_sample_names"] = np.asarray([human_name, robot_name])
    merged["hand_part_names"] = np.asarray(HAND_PARTS)
    merged["hand_part_origin"] = np.stack([params[side]["origin"] for side in SIDES])  # (side, human/robot, part, 3)
    merged["hand_part_scale"] = np.stack([params[side]["scale"] for side in SIDES])
    merged["hand_robot_config"] = np.asarray(json.dumps(robot["hands"]["sides"]))
    merged["hand_merge_dropped"] = np.asarray([int((~keep).sum()), int(human_on_hand.sum()), int(robot_on_hand.sum())])
    merged["hand_part_agreement"] = np.asarray([agreement[side] for side in SIDES])
    merged["meta_hand_config_hash"] = np.asarray(config_hash)
    merged["meta_body_slots_sha1"] = np.asarray(hashlib.sha1(np.ascontiguousarray(body["reconstructed_slots"]).tobytes()).hexdigest())
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **merged)
    print(
        f"[HandCorrespondence] merged {out}: body slots {len(keep)} -> {n_body} "
        f"(dropped human-hand {int(human_on_hand.sum())}, robot-hand {int(robot_on_hand.sum())}), "
        f"hand slots {n_hand}, part agreement left={agreement['left']:.3f} right={agreement['right']:.3f}"
    )
    return Path(out)


def load_hand_slot_info(slots_path):
    """Hand slot metadata from a merged slots file, or None for body-only slots."""
    with np.load(slots_path, allow_pickle=True) as data:
        if "hand_slot_side" not in data:
            return None
        return {
            "side": np.asarray(data["hand_slot_side"], dtype=np.int32),
            "part": np.asarray(data["hand_slot_part"], dtype=np.int32),
            "origin": np.asarray(data["hand_part_origin"], dtype=np.float64),
            "scale": np.asarray(data["hand_part_scale"], dtype=np.float64),
            "robot": json.loads(str(np.asarray(data["hand_robot_config"]).item())),
        }


def source_hand_frames(joints, side):
    ids = SMPLX_HAND_JOINTS[side]
    wrist = np.asarray(joints[:, ids["wrist"]], dtype=np.float64)
    rot = hand_frame(wrist, joints[:, ids["middle"]], joints[:, ids["index"]], joints[:, ids["pinky"]], side)
    return wrist, rot


def hand_local_targets(source_slots, joints_scaled, body_scale, info):
    """Replace hand slot targets by the robot-proportioned hand, placed at the scaled human wrist.

    The source hand is read in its canonical frame at human size, mapped part by
    part to the robot hand's proportions, and put back with the human hand's
    orientation. Fingers therefore take the robot's real finger lengths instead
    of the body scale.
    """
    out = np.array(source_slots, dtype=np.float32, copy=True)
    for side in SIDES:
        s = SIDE_INDEX[side]
        ids = np.flatnonzero(info["side"] == s)
        if ids.size == 0:
            continue
        parts = info["part"][ids]
        wrist, rot = source_hand_frames(joints_scaled, side)
        coords = np.einsum("tnd,tdk->tnk", out[:, ids] - wrist[:, None], rot) / float(body_scale)
        mapped = map_parts(coords, parts, info["origin"][s, 0], info["scale"][s, 0], info["origin"][s, 1], info["scale"][s, 1])
        out[:, ids] = (wrist[:, None] + np.einsum("tnk,tdk->tnd", mapped, rot)).astype(np.float32)
    return out


def fingertip_metrics(model, qpos_seq, joints_scaled, body_scale, info):
    """Fingertip error of the robot tip sites vs. the robot-proportioned human fingertips."""
    data = mujoco.MjData(model)
    names, world, local = [], [], []
    targets = []
    for side in SIDES:
        cfg = info["robot"][side]
        s = SIDE_INDEX[side]
        wrist, rot = source_hand_frames(joints_scaled, side)
        for finger in FINGER_PARTS:
            k = HAND_PARTS.index(finger)
            tip = np.asarray(joints_scaled[:, SMPLX_HAND_JOINTS[side]["tips"][finger]], dtype=np.float64)
            coords = np.einsum("td,tdk->tk", tip - wrist, rot) / float(body_scale)
            mapped = map_parts(coords, np.full(len(coords), k), info["origin"][s, 0], info["scale"][s, 0], info["origin"][s, 1], info["scale"][s, 1])
            targets.append(wrist + np.einsum("tk,tdk->td", mapped, rot))
            names.append(f"{side}_{finger}")
    targets = np.stack(targets, axis=1)
    site_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, info["robot"][side]["tip_sites"][finger]) for side in SIDES for finger in FINGER_PARTS]
    body = {side: [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in (
        info["robot"][side]["wrist_body"], info["robot"][side]["frame_bodies"]["middle"],
        info["robot"][side]["frame_bodies"]["index"], info["robot"][side]["frame_bodies"]["pinky"])] for side in SIDES}
    for frame, qpos in enumerate(qpos_seq):
        data.qpos[: len(qpos)] = qpos
        mujoco.mj_kinematics(model, data)
        tips = data.site_xpos[site_ids]
        world.append(np.linalg.norm(tips - targets[frame], axis=1))
        err_local = []
        for j, side in enumerate(s for s in SIDES for _ in FINGER_PARTS):
            o, m, i, p = (data.xpos[b] for b in body[side])
            r_robot = hand_frame(o, m, i, p, side)
            wrist, rot = source_hand_frames(joints_scaled[frame : frame + 1], side)
            target_local = (targets[frame, j] - wrist[0]) @ rot[0]
            err_local.append(np.linalg.norm((tips[j] - o) @ r_robot - target_local))
        local.append(err_local)
    return {"names": names, "targets": targets.astype(np.float32), "world": np.asarray(world, dtype=np.float32), "local": np.asarray(local, dtype=np.float32)}
