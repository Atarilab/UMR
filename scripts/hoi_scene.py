from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.spatial import Delaunay, cKDTree


RETARGET_SCENE_MODES = ("scaled", "true_scale")
TRUE_SCALE_ANCHOR_MODES = ("contact_shift", "root_ground", "origin")


def resolve_scene_mode(value) -> str:
    mode = str(value or "scaled").strip().lower()
    if mode not in RETARGET_SCENE_MODES:
        raise ValueError(f"retarget_scene_mode must be one of {RETARGET_SCENE_MODES}, got {value!r}")
    return mode


def scene_scale_anchors(root_positions, mode="root_ground"):
    """Per-frame points the source body is scaled about in true-scale mode.

    ``root_ground`` keeps the human root's horizontal path unscaled, so the
    robot walks to where the true-size objects are, while heights still scale
    about the ground plane (z=0 after source ground preprocessing).
    """
    mode = str(mode or "root_ground").strip().lower()
    if mode not in TRUE_SCALE_ANCHOR_MODES:
        raise ValueError(f"true_scale_anchor must be one of {TRUE_SCALE_ANCHOR_MODES}, got {mode!r}")
    root_positions = np.asarray(root_positions, dtype=np.float64).reshape(-1, 3)
    anchors = np.zeros_like(root_positions)
    if mode in {"root_ground", "contact_shift"}:
        anchors[:, :2] = root_positions[:, :2]
    return anchors


def contact_alignment_shift(scaled_slots, target_points, distances, threshold, fps, smoothing_seconds, max_shift):
    """Horizontal body shift that aligns scaled contact slots with true-scale targets.

    A smaller robot scaled about its own root stands where the human stood and
    has to over-reach for true-size objects. Per frame, the weighted mean xy
    residual of in-contact slots gives the shift that best closes that gap. It
    is interpolated across frames without contact and Gaussian-smoothed so the
    robot steps toward objects instead of jumping.
    """
    scaled_slots = np.asarray(scaled_slots, dtype=np.float64)
    target_points = np.asarray(target_points, dtype=np.float64)
    distances = np.asarray(distances, dtype=np.float64)
    frame_count = len(scaled_slots)
    threshold = max(float(threshold), 1e-8)
    weights = np.clip((threshold - distances) / threshold, 0.0, 1.0)
    weight_sums = weights.sum(axis=1)
    residual = (target_points - scaled_slots)[..., :2]
    raw = np.zeros((frame_count, 2), dtype=np.float64)
    has_contact = weight_sums > 1e-8
    raw[has_contact] = (
        np.einsum("tn,tnk->tk", weights[has_contact], residual[has_contact]) / weight_sums[has_contact, None]
    )
    shift = np.zeros((frame_count, 3), dtype=np.float64)
    if not np.any(has_contact):
        return shift, has_contact
    frames = np.arange(frame_count)
    contact_frames = frames[has_contact]
    for axis in range(2):
        shift[:, axis] = np.interp(frames, contact_frames, raw[has_contact, axis])
    sigma = float(smoothing_seconds) * float(fps)
    if sigma > 0.0 and frame_count > 1:
        from scipy.ndimage import gaussian_filter1d

        shift[:, :2] = gaussian_filter1d(shift[:, :2], sigma=sigma, axis=0, mode="nearest")
    if float(max_shift) > 0.0:
        norms = np.linalg.norm(shift[:, :2], axis=1, keepdims=True)
        shift[:, :2] *= np.minimum(1.0, float(max_shift) / np.maximum(norms, 1e-12))
    return shift, has_contact


def scale_about_anchors(points, anchors, scale):
    points = np.asarray(points, dtype=np.float64)
    anchors = np.asarray(anchors, dtype=np.float64).reshape(len(points), *([1] * (points.ndim - 2)), 3)
    return (anchors + float(scale) * (points - anchors)).astype(np.float32)


def unscale_about_anchors(points, anchors, scale):
    points = np.asarray(points, dtype=np.float64)
    anchors = np.asarray(anchors, dtype=np.float64).reshape(len(points), *([1] * (points.ndim - 2)), 3)
    return (anchors + (points - anchors) / max(float(scale), 1e-12)).astype(np.float32)


def compute_scene_contact_map(source_slots_world, scene_points_world, scale, snap_threshold):
    """Nearest true-scale scene point for every unscaled source slot.

    Distances stay in source metres. Offsets are scaled by the body scale so a
    robot slot targets ``scene_point + scale * offset``: touching slots land on
    the true object surface and hovering slots keep a proportional clearance.
    """
    source_slots_world = np.asarray(source_slots_world, dtype=np.float32)
    scene_points_world = np.asarray(scene_points_world, dtype=np.float32)
    frame_count, slot_count = source_slots_world.shape[:2]
    distances = np.empty((frame_count, slot_count), dtype=np.float32)
    object_ids = np.empty((frame_count, slot_count), dtype=np.int32)
    pair_vectors = np.empty((frame_count, slot_count, 3), dtype=np.float32)
    snap_threshold = float(snap_threshold)
    snapped_count = 0
    for frame_idx in range(frame_count):
        scene = scene_points_world[frame_idx]
        slots = source_slots_world[frame_idx]
        dist, ids = cKDTree(scene).query(slots, k=1)
        ids = np.asarray(ids, dtype=np.int32)
        dist = np.asarray(dist, dtype=np.float32)
        vectors = slots - scene[ids]
        if snap_threshold > 0.0:
            snap_mask = dist < snap_threshold
            snapped_count += int(snap_mask.sum())
            vectors[snap_mask] = 0.0
            dist[snap_mask] = 0.0
        distances[frame_idx] = dist
        object_ids[frame_idx] = ids
        pair_vectors[frame_idx] = vectors * float(scale)
    return distances, object_ids, pair_vectors, snapped_count


def _delaunay_neighbors(points):
    try:
        tri = Delaunay(points, qhull_options="QJ")
    except Exception:
        return None
    return tri.vertex_neighbor_vertices


def build_interaction_mesh_frames(
    source_slots_world,
    slot_ids,
    scene_points_world,
    scale,
    radius,
    max_object_points,
    seed,
    log_prefix="Retarget",
):
    """Per-frame interaction-mesh Laplacian targets.

    Vertices are the selected unscaled human slots plus true-scale scene points
    within ``radius`` of the body. For each human vertex i with Delaunay
    neighbours j (inverse-distance weights w_ij), the robot should satisfy

        x_i - sum_j w_ij y_j = scale * (p_i - sum_j w_ij q_j)

    where x/y are robot slots or fixed scene points and p/q the source points.
    Rearranged per frame as ``L @ X = target``, with L = I - W_human and the
    scene contribution folded into ``target``.
    """
    source_slots_world = np.asarray(source_slots_world, dtype=np.float64)
    scene_points_world = np.asarray(scene_points_world, dtype=np.float64)
    slot_ids = np.asarray(slot_ids, dtype=np.int32).reshape(-1)
    human_count = len(slot_ids)
    rng = np.random.default_rng(int(seed))
    radius = float(radius)
    max_object_points = int(max_object_points)
    identity = sp.identity(human_count, format="csr", dtype=np.float64)

    frames = []
    object_counts = []
    failed = 0
    for frame_idx in range(len(source_slots_world)):
        human = source_slots_world[frame_idx, slot_ids]
        scene = scene_points_world[frame_idx]
        near = np.zeros(0, dtype=np.int64)
        if len(scene) > 0 and radius > 0.0:
            dist, _ids = cKDTree(human).query(scene, k=1, distance_upper_bound=radius)
            near = np.where(np.isfinite(dist))[0]
            if max_object_points > 0 and near.size > max_object_points:
                near = np.sort(rng.choice(near, size=max_object_points, replace=False))
        scene_near = scene[near]
        vertices = np.concatenate([human, scene_near], axis=0)
        neighbors = _delaunay_neighbors(vertices)
        if neighbors is None:
            failed += 1
            frames.append(None)
            object_counts.append(0)
            continue
        indptr, indices = neighbors
        rows = np.repeat(np.arange(human_count), np.diff(indptr[: human_count + 1]))
        cols = np.asarray(indices[: indptr[human_count]], dtype=np.int64)
        lengths = np.linalg.norm(vertices[rows] - vertices[cols], axis=1)
        weights = 1.0 / np.maximum(lengths, 1e-4)
        row_sums = np.bincount(rows, weights=weights, minlength=human_count)
        weights = weights / np.maximum(row_sums[rows], 1e-12)

        human_mask = cols < human_count
        w_human = sp.csr_matrix(
            (weights[human_mask], (rows[human_mask], cols[human_mask])),
            shape=(human_count, human_count),
        )
        w_scene = sp.csr_matrix(
            (weights[~human_mask], (rows[~human_mask], cols[~human_mask] - human_count)),
            shape=(human_count, len(scene_near)),
        )
        scene_term = w_scene @ scene_near if len(scene_near) > 0 else np.zeros_like(human)
        source_delta = human - w_human @ human - scene_term
        frames.append(
            {
                "slot_ids": slot_ids,
                "laplacian": (identity - w_human).tocsr(),
                "target": (scene_term + float(scale) * source_delta).astype(np.float64),
            }
        )
        object_counts.append(int(len(near)))

    object_counts = np.asarray(object_counts, dtype=np.int32)
    print(
        f"[{log_prefix}][InteractionMesh] frames={len(frames)} human_vertices={human_count} "
        f"object_vertices min/mean/max={int(object_counts.min(initial=0))}/"
        f"{float(object_counts.mean()) if object_counts.size else 0.0:.1f}/{int(object_counts.max(initial=0))} "
        f"radius={radius:.3f} max_object_points={max_object_points} failed_frames={failed}"
    )
    return frames


def interaction_mesh_rows(mesh_frame, slot_cache, cost):
    """Linearised interaction-mesh rows ``sqrt(c) * (L @ J)`` and residuals."""
    slot_ids = mesh_frame["slot_ids"]
    laplacian = mesh_frame["laplacian"]
    points = slot_cache.points(slot_ids)
    jac = np.stack([slot_cache.point_jacobian(int(slot_id)) for slot_id in slot_ids], axis=0)
    nv = jac.shape[-1]
    lap_jac = (laplacian @ jac.reshape(len(slot_ids), 3 * nv)).reshape(len(slot_ids) * 3, nv)
    residual = (laplacian @ points - mesh_frame["target"]).reshape(-1)
    sqrt_cost = np.sqrt(float(cost))
    return sqrt_cost * lap_jac, sqrt_cost * residual


def summarize(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return {"count": 0, "mean": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }
