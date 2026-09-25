#!/usr/bin/env python3
"""Export the learned SMPL-X <-> robot hand correspondence to a Three.js viewer.

Shows, per hand, the SMPL-X hand and the robot hand side by side in the
canonical hand frame with their learned slots. Slots can be coloured by
correspondence (matched slots share a colour), by finger, or by mismatch (a
human finger slot whose robot slot lies on another link). Clicking a slot
highlights its partner.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import hand_correspondence as hc  # noqa: E402
from export_adapt_correspondence_viewer import correspondence_colors, write_binary  # noqa: E402

TEMPLATE_DIR = ROOT / "static/hand_correspondence_viewer"
PART_COLORS = np.asarray(
    [[140, 150, 162], [226, 87, 76], [242, 169, 59], [88, 179, 104], [72, 140, 214], [157, 102, 204]],
    dtype=np.uint8,
)
# Display: fingers point up (+y), palm faces the camera (+z), hands laid out along x.
LAYOUT_X = {("left", 0): -0.42, ("left", 1): -0.14, ("right", 0): 0.14, ("right", 1): 0.42}


def canonical_to_display(points, side, sample):
    points = np.asarray(points, dtype=np.float32)
    out = np.empty_like(points)
    mirror = 1.0 if side == "right" else -1.0
    out[:, 0] = mirror * points[:, 1] + LAYOUT_X[(side, sample)]
    out[:, 1] = points[:, 0]
    out[:, 2] = points[:, 2]
    return out


def _side_arrays(dataset_path, slots_path):
    ds = np.load(dataset_path, allow_pickle=True)
    slots = np.load(slots_path, allow_pickle=True)
    offset = float(ds["hand_part_offset"])
    origin, scale = ds["hand_part_origin"], ds["hand_part_scale"]
    ds_names = [str(n) for n in ds["names"]]
    slot_names = [str(n) for n in slots["names"]]
    meshes, clouds = [], []
    for sample, name in enumerate(ds_names):
        vertices = ds[f"mesh_vertices_{sample}"]
        vparts = hc.part_of_normalized(vertices, offset)
        meshes.append((hc.denormalize_parts(vertices, vparts, origin[sample], scale[sample], offset), ds[f"mesh_faces_{sample}"], vparts))
    n_h = slots["reconstructed_slots"][slot_names.index(ds_names[0])]
    n_r = slots["reconstructed_slots"][slot_names.index(ds_names[1])]
    parts = hc.part_of_normalized(n_h, offset)
    clouds.append(hc.denormalize_parts(n_h, parts, origin[0], scale[0], offset))
    clouds.append(hc.denormalize_parts(n_r, parts, origin[1], scale[1], offset))
    return ds_names, meshes, clouds, parts


def export_hand_correspondence_viewer_from_paths(hand_datasets, hand_slots, out_dir, title="Hand correspondence"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(TEMPLATE_DIR / "index.html", out_dir / "index.html")
    shutil.copy2(TEMPLATE_DIR / "app.js", out_dir / "app.js")
    manifest = {"title": title, "parts": list(hc.HAND_PARTS), "part_colors": PART_COLORS.tolist(), "hands": []}
    analysis = {}
    for side in hc.SIDES:
        names, meshes, clouds, parts = _side_arrays(hand_datasets[side], hand_slots[side])
        robot_vertices, _robot_faces, robot_vparts = meshes[1]
        dist, nearest = cKDTree(robot_vertices).query(clouds[1])
        bound_part = robot_vparts[nearest]
        mismatch = bound_part != parts
        corr = correspondence_colors(clouds[0])
        part_rgb = PART_COLORS[parts]
        mismatch_rgb = np.tile(np.asarray([[196, 204, 212]], dtype=np.uint8), (len(parts), 1))
        mismatch_rgb[mismatch] = [255, 170, 35]
        hand = {"side": side, "samples": []}
        for sample, label in enumerate(("SMPL-X", "robot")):
            prefix = f"{side}_{'human' if sample == 0 else 'robot'}"
            vertices, faces, _vparts = meshes[sample]
            hand["samples"].append(
                {
                    "label": f"{side} {label} ({names[sample]})",
                    "positions": write_binary(out_dir / f"{prefix}_slots.f32", canonical_to_display(clouds[sample], side, sample)),
                    "color_correspondence": write_binary(out_dir / f"{prefix}_corr.u8", corr),
                    "color_part": write_binary(out_dir / f"{prefix}_part.u8", part_rgb),
                    "color_mismatch": write_binary(out_dir / f"{prefix}_mismatch.u8", mismatch_rgb),
                    "mesh_vertices": write_binary(out_dir / f"{prefix}_mesh.f32", canonical_to_display(vertices, side, sample)),
                    "mesh_faces": write_binary(out_dir / f"{prefix}_faces.u32", np.asarray(faces, dtype=np.uint32)),
                    "count": int(len(clouds[sample])),
                }
            )
        hand["slot_parts"] = write_binary(out_dir / f"{side}_slot_parts.u8", parts.astype(np.uint8))
        hand["stats"] = {
            part: {
                "slots": int((parts == k).sum()),
                "mapped_to_same_link_group": float(np.mean(~mismatch[parts == k])) if np.any(parts == k) else 0.0,
                "robot_surface_distance_mm": float(1000.0 * dist[parts == k].mean()) if np.any(parts == k) else 0.0,
            }
            for k, part in enumerate(hc.HAND_PARTS)
        }
        manifest["hands"].append(hand)
        analysis[f"{side}_human_slots"] = clouds[0]
        analysis[f"{side}_robot_slots"] = clouds[1]
        analysis[f"{side}_slot_parts"] = parts
        analysis[f"{side}_robot_bound_parts"] = bound_part
        analysis[f"{side}_robot_surface_distance"] = dist
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    np.savez_compressed(out_dir / "hand_correspondence_analysis.npz", **analysis)
    rates = ", ".join(
        f"{hand['side']}: " + " ".join(f"{p}={100 * s['mapped_to_same_link_group']:.0f}%" for p, s in hand["stats"].items())
        for hand in manifest["hands"]
    )
    print(f"[HandCorrVis] exported {out_dir}; same-finger mapping rate {rates}")
    print(f"[HandCorrVis] view with: python -m http.server --directory {out_dir} 8000  (then open http://localhost:8000)")
    return out_dir


def export_hand_correspondence_viewer(config, merged_slots=None, out_dir=None):
    from humanoid_retarget_pipeline import hand_dataset_out, hand_train_out_dir, train_out_dir

    datasets = {side: hand_dataset_out(config, side) for side in hc.SIDES}
    slots = {side: hand_train_out_dir(config, side) / "correspondence_slots_final.npz" for side in hc.SIDES}
    out_dir = Path(out_dir) if out_dir else train_out_dir(config) / "hand_correspondence_viewer"
    return export_hand_correspondence_viewer_from_paths(datasets, slots, out_dir, title=f"Hand correspondence: {train_out_dir(config).name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--name",
        required=True,
        help="Correspondence stem, e.g. correspondence_unitree_g1_rh56e2_bowl_in_microwave_hsi_hoi "
        "(data/<name>_hand_<side>.npz and output/<name>_hand_<side>/).",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    datasets = {side: ROOT / "data" / f"{args.name}_hand_{side}.npz" for side in hc.SIDES}
    slots = {side: ROOT / "output" / f"{args.name}_hand_{side}" / "correspondence_slots_final.npz" for side in hc.SIDES}
    out = args.out or ROOT / "output" / args.name / "hand_correspondence_viewer"
    export_hand_correspondence_viewer_from_paths(datasets, slots, out, title=f"Hand correspondence: {args.name}")


if __name__ == "__main__":
    main()
