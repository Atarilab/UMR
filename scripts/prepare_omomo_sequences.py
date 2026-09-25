#!/usr/bin/env python3
"""Convert OMOMO release sequences into the UMR flat SMPL-X + object layout.

The upstream OMOMO distribution stores each sequence as a single ``.npz`` with
body-only SMPL-X pose parameters and a per-frame object transform, plus a shared
directory of captured object meshes.  ``sample_data/omomo/README.md`` documents
the layout UMR expects instead: one directory per sequence holding the motion
arrays, a sequence-local MJCF, and a ``prop_<object>.csv`` trajectory, with the
object meshes shared under ``object_mjcf/assets/``.

Object meshes are decomposed once per object at their authored size and reused
by every sequence.  OMOMO scales the object per sequence, so the sequence-local
MJCF carries that factor as a ``scale`` attribute on the mesh assets rather than
baking it into the shared geometry.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import xml.etree.ElementTree as ET
from dataclasses import asdict
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from prepare_grail_object_mjcf import (  # noqa: E402
    build_collision_cache,
    geometry_digest,
    validate_mjcf,
)
from obj2mjcf.cli import CoacdArgs  # noqa: E402

ASSET_VERSION = 1
SMPLX_POSE_DIM = 165
OMOMO_POSE_DIM = 66


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="OMOMO release root containing motions/ and captured_objects/.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "sample_data/omomo",
        help="UMR OMOMO data root to populate.",
    )
    parser.add_argument(
        "--split-dir",
        type=str,
        default="train_and_test",
        help="Sequence directory below the output root.",
    )
    parser.add_argument("--seq-key", action="append", default=None, help="Convert this sequence only; repeatable.")
    parser.add_argument(
        "--gender",
        choices=("source", "neutral", "male", "female"),
        default="source",
        help=(
            "Body-model gender recorded per sequence. OMOMO labels every subject male or female, so "
            "'source' requires the matching smpl/SMPLX_<GENDER>.pkl; pass 'neutral' to run with the "
            "neutral model alone at the cost of a less faithful body shape."
        ),
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional sequence limit; 0 converts all discovered.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.03)
    parser.add_argument("--max-convex-hull", type=int, default=32)
    parser.add_argument("--mcts-iterations", type=int, default=200)
    parser.add_argument("--resolution", type=int, default=2000)
    parser.add_argument("--no-validate", dest="validate", action="store_false", default=True)
    args = parser.parse_args()
    args.source = args.source.resolve()
    args.output = args.output.resolve()
    return args


def discover_sequences(source: Path, seq_keys: list[str] | None) -> list[Path]:
    candidates = sorted(source.glob("motions/*/*.npz")) + sorted(source.glob("motions/*.npz"))
    if not candidates:
        candidates = sorted(source.glob("*.npz"))
    if seq_keys:
        wanted = set(seq_keys)
        candidates = [path for path in candidates if path.stem in wanted]
        missing = wanted - {path.stem for path in candidates}
        if missing:
            raise FileNotFoundError(f"Sequences not found below {source}: {sorted(missing)}")
    return candidates


def object_mesh_path(source: Path, object_name: str) -> Path:
    candidates = [
        source / "captured_objects" / f"{object_name}_cleaned_simplified.obj",
        source / "captured_objects" / f"{object_name}.obj",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Object mesh not found for {object_name!r} below {source / 'captured_objects'}")


def expand_pose(poses: np.ndarray) -> np.ndarray:
    """Pad OMOMO's root+body axis-angle block out to the full SMPL-X pose vector."""
    poses = np.asarray(poses, dtype=np.float32).reshape(len(poses), -1)
    if poses.shape[1] == SMPLX_POSE_DIM:
        return poses
    if poses.shape[1] != OMOMO_POSE_DIM:
        raise ValueError(f"Expected {OMOMO_POSE_DIM} or {SMPLX_POSE_DIM} pose channels, got {poses.shape[1]}")
    expanded = np.zeros((len(poses), SMPLX_POSE_DIM), dtype=np.float32)
    expanded[:, :OMOMO_POSE_DIM] = poses
    return expanded


def write_motion_arrays(sequence_dir: Path, payload, frames: int, gender_mode: str = "source") -> dict:
    poses = expand_pose(payload["poses"])
    trans = np.asarray(payload["trans"], dtype=np.float32).reshape(frames, 3)
    betas = np.asarray(payload["betas"], dtype=np.float32).reshape(-1)
    source_gender = str(payload["gender"]).lower()
    gender = source_gender if gender_mode == "source" else gender_mode
    fps = float(np.asarray(payload["mocap_frame_rate"]).reshape(-1)[0])

    np.save(sequence_dir / "poses.npy", poses)
    np.save(sequence_dir / "transl.npy", trans)
    np.save(sequence_dir / "betas.npy", betas)
    np.save(sequence_dir / "gender.npy", np.array(gender))
    np.save(sequence_dir / "model_type.npy", np.array("smplx"))
    np.save(sequence_dir / "mocap_framerate.npy", np.array(fps, dtype=np.float32))
    np.save(sequence_dir / "output_up.npy", np.array("z"))
    return {
        "frames": int(frames),
        "gender": gender,
        "source_gender": source_gender,
        "fps": fps,
        "betas": int(betas.size),
    }


def write_object_motion(prop_path: Path, payload, frames: int) -> None:
    positions = np.asarray(payload["obj_trans"], dtype=np.float64).reshape(frames, 3)
    rotations = np.asarray(payload["obj_rot_mat"], dtype=np.float64).reshape(frames, 3, 3)
    quats = R.from_matrix(rotations).as_quat()
    with prop_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["px", "py", "pz", "qx", "qy", "qz", "qw"])
        for position, quat in zip(positions, quats):
            writer.writerow([*position.tolist(), *quat.tolist()])


def publish_object_assets(
    object_name: str,
    mesh_source: Path,
    assets_dir: Path,
    cache_root: Path,
    coacd_args: CoacdArgs,
    overwrite: bool,
) -> tuple[Path, list[Path], str, str]:
    """Copy the visual mesh and its convex pieces into the shared assets directory."""
    assets_dir.mkdir(parents=True, exist_ok=True)
    visual_target = assets_dir / f"{object_name}.obj"
    if overwrite or not visual_target.exists():
        shutil.copy2(mesh_source, visual_target)

    mesh = trimesh.load(visual_target, force="mesh", process=True)
    digest = geometry_digest(mesh)
    cached_parts, method = build_collision_cache(visual_target, cache_root / digest, coacd_args)

    for stale in assets_dir.glob(f"{object_name}_collision_*.obj"):
        stale.unlink()
    collision_targets = []
    for index, source in enumerate(cached_parts):
        target = assets_dir / f"{object_name}_collision_{index}.obj"
        try:
            os.link(source, target)
        except OSError:
            shutil.copy2(source, target)
        collision_targets.append(target)
    return visual_target, collision_targets, method, digest


def write_mjcf(
    xml_path: Path,
    object_name: str,
    visual_mesh: Path,
    collision_meshes: list[Path],
    mesh_scale: float,
) -> None:
    scale_attr = f"{mesh_scale:.12g} {mesh_scale:.12g} {mesh_scale:.12g}"

    def relative(path: Path) -> str:
        return os.path.relpath(path.resolve(), xml_path.resolve().parent)

    root = ET.Element("mujoco", {"model": object_name})
    asset = ET.SubElement(root, "asset")
    ET.SubElement(
        asset,
        "mesh",
        {"name": f"{object_name}_visual_mesh", "file": relative(visual_mesh), "scale": scale_attr},
    )
    for index, mesh_path in enumerate(collision_meshes):
        ET.SubElement(
            asset,
            "mesh",
            {"name": f"{object_name}_collision_{index}_mesh", "file": relative(mesh_path), "scale": scale_attr},
        )
    worldbody = ET.SubElement(root, "worldbody")
    body = ET.SubElement(worldbody, "body", {"name": object_name})
    ET.SubElement(body, "freejoint", {"name": f"{object_name}_freejoint"})
    ET.SubElement(
        body,
        "geom",
        {
            "name": f"{object_name}_visual",
            "type": "mesh",
            "mesh": f"{object_name}_visual_mesh",
            "group": "2",
            "contype": "0",
            "conaffinity": "0",
            "rgba": "1 1 1 1",
        },
    )
    for index, _mesh_path in enumerate(collision_meshes):
        ET.SubElement(
            body,
            "geom",
            {
                "name": f"{object_name}_collision_{index}",
                "type": "mesh",
                "mesh": f"{object_name}_collision_{index}_mesh",
                "group": "3",
                "contype": "1",
                "conaffinity": "1",
                "rgba": "0.25 0.45 0.8 0.15",
            },
        )
    xml_path.write_text(ET.tostring(root, encoding="unicode") + "\n", encoding="utf-8")


def convert_sequence(npz_path: Path, args, coacd_args: CoacdArgs) -> dict:
    seq_key = npz_path.stem
    sequence_dir = args.output / args.split_dir / seq_key
    metadata_path = sequence_dir / "metadata.json"
    if metadata_path.exists() and not args.overwrite:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if int(metadata.get("asset_version", 0)) == ASSET_VERSION:
            return {**metadata, "status": "reused"}

    payload = np.load(npz_path, allow_pickle=True)
    frames = int(len(payload["poses"]))
    object_name = str(payload["obj_name"])
    scales = np.asarray(payload["obj_scale"], dtype=np.float64).reshape(-1)
    mesh_scale = float(np.median(scales))
    scale_spread = float((scales.max() - scales.min()) / max(abs(mesh_scale), 1e-12))

    sequence_dir.mkdir(parents=True, exist_ok=True)
    motion_meta = write_motion_arrays(sequence_dir, payload, frames, args.gender)
    write_object_motion(sequence_dir / f"prop_{object_name}.csv", payload, frames)

    visual_mesh, collision_meshes, method, digest = publish_object_assets(
        object_name,
        object_mesh_path(args.source, object_name),
        args.output / "object_mjcf" / "assets",
        args.output / "object_mjcf" / "_collision_cache",
        coacd_args,
        args.overwrite,
    )
    xml_path = sequence_dir / f"{object_name}.xml"
    write_mjcf(xml_path, object_name, visual_mesh, collision_meshes, mesh_scale)
    if args.validate:
        validate_mjcf(xml_path, len(collision_meshes))

    metadata = {
        "asset_version": ASSET_VERSION,
        "sequence_key": seq_key,
        "source_npz": str(npz_path),
        "object_name": object_name,
        "object_mesh_scale": mesh_scale,
        "object_scale_spread": scale_spread,
        "geometry_digest": digest,
        "decomposition_method": method,
        "collision_parts": len(collision_meshes),
        "coacd_args": asdict(coacd_args),
        "status": "converted",
        **motion_meta,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def main() -> None:
    args = parse_args()
    coacd_args = CoacdArgs(
        threshold=float(args.threshold),
        max_convex_hull=int(args.max_convex_hull),
        mcts_iterations=int(args.mcts_iterations),
        resolution=int(args.resolution),
    )
    sequences = discover_sequences(args.source, args.seq_key)
    if args.limit > 0:
        sequences = sequences[: args.limit]
    if not sequences:
        raise SystemExit(f"No OMOMO sequences discovered below {args.source}")

    converted = 0
    for index, npz_path in enumerate(sequences, start=1):
        metadata = convert_sequence(npz_path, args, coacd_args)
        converted += int(metadata.get("status") == "converted")
        print(
            f"[OMOMO] {index}/{len(sequences)} {metadata['status']} {metadata['sequence_key']} "
            f"object={metadata['object_name']} frames={metadata['frames']} gender={metadata['gender']} "
            f"scale={metadata['object_mesh_scale']:.5f} parts={metadata['collision_parts']} method={metadata['decomposition_method']}",
            flush=True,
        )
    print(f"[OMOMO] complete sequences={len(sequences)} converted={converted} output={args.output}")


if __name__ == "__main__":
    main()
