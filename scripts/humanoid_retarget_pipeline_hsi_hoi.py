#!/usr/bin/env python3
"""Run humanoid retargeting for fixed-layout HSI-HOI object-interaction sequences."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import re
import subprocess
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from humanoid_retarget_config import load_config, resolve_path, robot_config, section  # noqa: E402
from humanoid_retarget_pipeline import (  # noqa: E402
    RETARGET_VISUALIZATION_FIELDS,
    build_correspondence_dataset,
    correspondence_slots_compatible,
    dataset_out,
    ensure_retarget_slots,
    retarget_motion,
    retarget_out,
    retarget_result_has_final_qpos_only,
    slots_out,
    train_correspondence,
    visualize_result,
)


DEFAULTS_CONFIG = ROOT / "humanoid_retarget_defaults_hsi_hoi_standard.json"
DEFAULT_ROBOT_CONFIG = ROOT / "robot_configs" / "humanoid_retarget_unitree_g1_example.json"
GRAIL_OBJECT_ASSET_VERSION = "4"
GRAIL_CONVEX_MJCF_VERSION = 3
GRAIL_SIZE_DESCRIPTOR_TOLERANCE = 0.02
GRAIL_TABLE_STEM = "grail_table"
OBJECT_COLLISION_VERSION = 1
OBJECT_COLLISION_SCENE_ROOT = ROOT / "output" / "hsi_hoi_objects"
OBJECT_COLLISION_CACHE_ROOT = ROOT / "output" / "object_collision_cache"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_ROBOT_CONFIG, help="Robot config.")
    parser.add_argument("--defaults", type=Path, default=DEFAULTS_CONFIG, help="HSI/HOI data and solver defaults.")
    parser.add_argument(
        "--stage",
        choices=["all", "build", "train", "retarget", "view", "prepare"],
        default="all",
        help="prepare only exports the HSI/HOI sequence and temporary config.",
    )
    parser.add_argument("--data", type=Path, default=None, help="OmniContact/OMOMO root, GRAIL root, or one sequence source.")
    parser.add_argument("--seq-key", type=str, default=None, help="Sequence key, e.g. sofa006, sub9_vacuum_054, or a GRAIL recon stem.")
    parser.add_argument("--out", type=Path, default=None, help="Override retarget output .npz.")
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--scene-mode",
        choices=("scaled", "true_scale"),
        default=None,
        help="Override solver.retarget_scene_mode: scale the whole scene to the robot, or keep objects at true size.",
    )
    parser.add_argument("--force-build", action="store_true")
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--force-retarget", action="store_true")
    parser.add_argument("--skip-view", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def clean_config(value):
    if isinstance(value, dict):
        return {str(k): clean_config(v) for k, v in value.items() if not str(k).startswith("_")}
    if isinstance(value, list):
        return [clean_config(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in {"extends", "_config_path", "_config_dir"}:
            continue
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = deep_merge(current, value)
        else:
            merged[key] = value
    return merged


def load_hsi_hoi_config(
    path: Path, defaults_path: Path = DEFAULTS_CONFIG
) -> dict[str, Any]:
    path = Path(path)
    hsi_defaults = load_config(Path(defaults_path), use_default_extends=False)
    user_config = load_config(path, use_default_extends=False)
    merged = deep_merge(clean_config(hsi_defaults), clean_config(user_config))
    user_robot = section(user_config, "robot")
    merged_robot = section(merged, "robot")
    if user_robot.get("xml"):
        merged_robot["xml"] = str(resolve_path(user_robot.get("xml"), user_config))
        merged["robot"] = merged_robot
    merged["_config_path"] = str(Path(user_config["_config_path"]).resolve())
    merged["_config_dir"] = str(Path(user_config["_config_dir"]).resolve())
    return merged


def safe_name(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "sequence"
    if len(name) <= 180:
        return name
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:12]
    return f"{name[:160]}_{digest}"


def scalar_npy(path: Path, default: Any = None):
    if not path.exists():
        return default
    value = np.load(path, allow_pickle=True)
    return value.item() if value.shape == () else value


def hsi_config(config: dict[str, Any]) -> dict[str, Any]:
    value = section(config, "hsi_hoi")
    return value


def is_hsi_hoi_sequence_dir(path: Path) -> bool:
    path = Path(path)
    has_pose = (path / "poses.npy").exists() or (path / "smpl_pose_axis_angle.npy").exists()
    has_trans = (path / "transl.npy").exists() or (path / "trans.npy").exists()
    return bool(path.is_dir() and has_pose and has_trans and (path / "betas.npy").exists())


def is_grail_root(path: Path) -> bool:
    path = Path(path)
    return bool(path.is_dir() and (path / "recon").is_dir() and (path / "object_usd").is_dir())


def is_grail_sequence(path: Path) -> bool:
    path = Path(path)
    if not path.is_file() or path.suffix.lower() != ".pkl" or path.parent.name != "recon":
        return False
    root = path.parent.parent
    return (root / "object_usd" / f"{path.stem}.usd").exists()


def load_grail_pickle(path: Path) -> dict[str, Any]:
    # NumPy 2 pickles use numpy._core module names; NumPy 1.x only exposes numpy.core.
    try:
        import numpy._core as numpy_core  # type: ignore[attr-defined]  # noqa: WPS433
        import numpy._core.multiarray as numpy_multiarray  # type: ignore[attr-defined]  # noqa: WPS433
        import numpy._core.numeric as numpy_numeric  # type: ignore[attr-defined]  # noqa: WPS433
    except ImportError:
        import numpy.core as numpy_core  # type: ignore[no-redef]  # noqa: WPS433
        import numpy.core.multiarray as numpy_multiarray  # type: ignore[no-redef]  # noqa: WPS433
        import numpy.core.numeric as numpy_numeric  # type: ignore[no-redef]  # noqa: WPS433
    sys.modules.setdefault("numpy._core", numpy_core)
    sys.modules.setdefault("numpy._core.numeric", numpy_numeric)
    sys.modules.setdefault("numpy._core.multiarray", numpy_multiarray)
    return joblib.load(path)


def sequence_dir_candidates(data_path: Path, seq_key: str) -> list[Path]:
    data_path = Path(data_path)
    seq_key = str(seq_key)
    candidates = [
        data_path / seq_key,
        data_path / "train_and_test" / seq_key,
        data_path / "train" / seq_key,
        data_path / "test" / seq_key,
    ]
    manifest_path = data_path / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            records = manifest if isinstance(manifest, list) else manifest.get("sequences", [])
            for record in records:
                if not isinstance(record, dict):
                    continue
                if str(record.get("seq_name", "")) != seq_key:
                    continue
                data_dir = str(record.get("data_dir", ""))
                if data_dir:
                    candidates.append(data_path / data_dir)
                    candidates.append(data_path / "train_and_test" / seq_key)
        except Exception as exc:
            print(f"[HSIHOI][WARN] could not read sequence manifest {manifest_path}: {exc}")
    seen = set()
    out = []
    for path in candidates:
        key = str(path)
        if key not in seen:
            out.append(path)
            seen.add(key)
    return out


def sample_sequence_names(data_path: Path, limit: int = 12) -> str:
    names = []
    try:
        if is_grail_root(data_path):
            return ", ".join(path.stem for path in sorted((Path(data_path) / "recon").glob("*.pkl"))[:limit])
        for path in sorted(Path(data_path).iterdir()):
            if path.is_dir() and is_hsi_hoi_sequence_dir(path):
                names.append(path.name)
            if len(names) >= limit:
                break
        for split in ("train_and_test", "train", "test"):
            split_dir = Path(data_path) / split
            if not split_dir.is_dir():
                continue
            for path in sorted(split_dir.iterdir()):
                if path.is_dir() and is_hsi_hoi_sequence_dir(path):
                    names.append(path.name)
                if len(names) >= limit:
                    break
            if len(names) >= limit:
                break
    except Exception:
        pass
    return ", ".join(names[:limit])


def source_dataset_kind(seq_dir: Path) -> str:
    if is_grail_sequence(seq_dir):
        return "grail"
    parts = {part.lower() for part in Path(seq_dir).resolve().parts}
    if "omomo" in parts:
        return "omomo"
    if "samp" in parts:
        return "samp"
    return "hsi_hoi"


def source_format_for_sequence(seq_dir: Path) -> str:
    kind = source_dataset_kind(seq_dir)
    return f"{kind}_hsi_hoi_smplx" if kind in {"samp", "omomo", "grail"} else "hsi_hoi_smplx"


def resolve_samp_sequence(config: dict[str, Any], data_override: Path | None, seq_key_override: str | None) -> tuple[str, Path]:
    motion = section(config, "motion")
    data_root = data_override if data_override is not None else motion.get("data", "sample_data/omnicontact/soccer/case3_kick_right")
    data_path = resolve_path(data_root, config)
    if data_path is None:
        raise ValueError("motion.data must point to an OmniContact/OMOMO/GRAIL root or sequence source.")
    data_path = Path(data_path)

    seq_key = seq_key_override if seq_key_override is not None else motion.get("seq_key", "")
    if is_grail_sequence(data_path):
        return str(seq_key or data_path.stem), data_path.resolve()
    if is_grail_root(data_path):
        if not seq_key:
            raise ValueError(f"Set motion.seq_key or --seq-key for GRAIL root: {data_path}")
        sequence_path = data_path / "recon" / f"{seq_key}.pkl"
        if is_grail_sequence(sequence_path):
            return str(seq_key), sequence_path.resolve()
        sample = sample_sequence_names(data_path)
        raise FileNotFoundError(f"GRAIL sequence {seq_key!r} not found under {data_path / 'recon'}. First sequences: {sample}")
    if is_hsi_hoi_sequence_dir(data_path):
        return str(seq_key or data_path.name), data_path.resolve()
    if not seq_key:
        raise ValueError(f"Set motion.seq_key or --seq-key when motion.data is a sequence root: {data_path}")

    for seq_dir in sequence_dir_candidates(data_path, str(seq_key)):
        if is_hsi_hoi_sequence_dir(seq_dir):
            return str(seq_key), seq_dir.resolve()
    sample = sample_sequence_names(data_path)
    raise FileNotFoundError(
        f"HSI-HOI sequence {seq_key!r} not found under {data_path}. "
        f"Checked direct/train_and_test/train/test. First sequence dirs: {sample}"
    )


def load_samp_motion(seq_dir: Path) -> dict[str, Any]:
    if is_grail_sequence(seq_dir):
        payload = load_grail_pickle(seq_dir)
        human = payload.get("human_data", {})
        poses = np.asarray(human["poses"], dtype=np.float32).reshape(-1, 165)
        trans = np.asarray(human["trans"], dtype=np.float32).reshape(len(poses), 3)
        betas = np.asarray(human.get("betas", np.zeros((1, 10))), dtype=np.float32).reshape(-1)
        model_type = str(human.get("model", "smplx")).lower()
        if model_type != "smplx":
            raise ValueError(f"GRAIL HSI pipeline expects SMPL-X data, got model={model_type!r}: {seq_dir}")
        return {
            "poses": poses,
            "trans": trans,
            "betas": betas,
            "gender": str(human.get("gender", "neutral")).lower(),
            "model_type": model_type,
            "fps": float(human.get("mocap_frame_rate", 30.0)),
            "output_up": "z",
            "human_scale": float(np.asarray(human.get("scale", 1.0), dtype=np.float32).reshape(-1)[0]),
            "human_scale_mode": "local",
        }
    pose_path = seq_dir / "poses.npy"
    if not pose_path.exists():
        pose_path = seq_dir / "smpl_pose_axis_angle.npy"
    trans_path = seq_dir / "transl.npy"
    if not trans_path.exists():
        trans_path = seq_dir / "trans.npy"

    raw_poses = np.load(pose_path)
    poses = np.asarray(raw_poses, dtype=np.float32).reshape(-1, raw_poses.shape[-1])
    trans = np.asarray(np.load(trans_path), dtype=np.float32).reshape(len(poses), 3)
    betas = np.asarray(np.load(seq_dir / "betas.npy"), dtype=np.float32).reshape(-1)
    gender = str(scalar_npy(seq_dir / "gender.npy", "neutral")).lower()
    model_type = str(scalar_npy(seq_dir / "model_type.npy", "smplx")).lower()
    fps = float(scalar_npy(seq_dir / "mocap_framerate.npy", scalar_npy(seq_dir / "fps.npy", 30.0)))
    output_up = str(scalar_npy(seq_dir / "output_up.npy", "z")).lower()
    if model_type != "smplx":
        raise ValueError(f"HSI-HOI pipeline currently expects SMPL-X motion data, got model_type={model_type!r}.")
    if poses.shape[0] != trans.shape[0]:
        raise ValueError(f"Frame count mismatch in {seq_dir}: poses={poses.shape}, trans={trans.shape}")
    return {
        "poses": poses,
        "trans": trans,
        "betas": betas,
        "gender": gender,
        "model_type": model_type,
        "fps": fps,
        "output_up": output_up,
        "human_scale": 1.0,
        "human_scale_mode": "off",
    }


def discover_samp_objects(seq_dir: Path) -> list[dict[str, str]]:
    objects = []
    stems = sorted({p.stem for p in seq_dir.glob("*.xml")} | {p.stem for p in seq_dir.glob("*.obj")})
    for stem in stems:
        xml_path = seq_dir / f"{stem}.xml"
        obj_path = seq_dir / f"{stem}.obj"
        prop_path = seq_dir / f"prop_{stem}.csv"
        if not prop_path.exists():
            continue
        objects.append(
            {
                "name": stem,
                "xml": str(xml_path if xml_path.exists() else ""),
                "obj": str(obj_path if obj_path.exists() else ""),
                "prop": str(prop_path),
            }
        )
    return objects


def _mjcf_mesh_base(root: ET.Element, xml_path: Path) -> Path:
    compiler = root.find("compiler")
    meshdir = compiler.get("meshdir") if compiler is not None else None
    base = Path(xml_path).resolve().parent
    if meshdir:
        base = Path(meshdir) if Path(meshdir).is_absolute() else (base / meshdir).resolve()
    return base


def _mjcf_collision_mesh_geoms(root: ET.Element) -> list[ET.Element]:
    """Mesh geoms MuJoCo collides with (compiled defaults: contype = conaffinity = 1)."""
    geoms = []
    for geom in root.iter("geom"):
        if geom.get("type") != "mesh" or not geom.get("mesh"):
            continue
        collides = int(geom.get("contype", "1")) != 0 or int(geom.get("conaffinity", "1")) != 0
        if collides or geom.get("group") == "3":
            geoms.append(geom)
    return geoms


def _load_scaled_mesh(mesh_el: ET.Element, mesh_base: Path) -> tuple[Path, trimesh.Trimesh]:
    mesh_path = Path(mesh_el.get("file"))
    mesh_path = mesh_path if mesh_path.is_absolute() else (mesh_base / mesh_path).resolve()
    mesh = trimesh.load(mesh_path, force="mesh", process=True)
    scale = np.fromstring(mesh_el.get("scale", "1 1 1"), sep=" ", dtype=np.float64)
    if scale.size == 3 and not np.allclose(scale, 1.0):
        mesh.apply_scale(scale)
    return mesh_path, mesh


def _concave_ratio(mesh: trimesh.Trimesh) -> float:
    """Convex-hull volume over mesh volume; inf when the mesh has no reliable volume."""
    if not mesh.is_watertight or mesh.volume <= 1e-12:
        return float("inf")
    return float(mesh.convex_hull.volume / mesh.volume)


def _object_collision_sources(seq_dir: Path, objects: list[dict[str, str]]) -> dict[str, int]:
    sources = {}
    for obj in objects:
        for key in ("xml", "obj", "prop"):
            if obj[key]:
                sources[obj[key]] = Path(obj[key]).stat().st_mtime_ns
        if obj["xml"]:
            root = ET.parse(obj["xml"]).getroot()
            base = _mjcf_mesh_base(root, Path(obj["xml"]))
            for mesh_el in root.iter("mesh"):
                if mesh_el.get("file"):
                    path = Path(mesh_el.get("file"))
                    path = path if path.is_absolute() else (base / path).resolve()
                    if path.exists():
                        sources[str(path)] = path.stat().st_mtime_ns
    return sources


def prepare_standard_object_collision(
    seq_key: str,
    seq_dir: Path,
    config: dict[str, Any],
    force: bool,
    dry_run: bool,
) -> Path:
    """Object dir whose concave collision meshes are split into convex pieces.

    MuJoCo collides a mesh geom as its convex hull, which fills cavities (a
    microwave, a bowl, the space under a table) the hand has to enter. Any
    collision mesh noticeably larger as a hull than as itself is decomposed with
    CoACD into cached pieces, and the scene is rewritten under
    ``output/hsi_hoi_objects/<seq>``; the source folder is never modified. With
    nothing to decompose the sequence dir is returned unchanged.
    """
    object_cfg = section(hsi_config(config), "object")
    if not bool(object_cfg.get("auto_convex_decomposition", True)):
        return seq_dir
    objects = discover_samp_objects(seq_dir)
    if not objects:
        return seq_dir
    from object_collision import CoacdArgs, build_collision_cache, geometry_digest  # noqa: WPS433

    coacd_args = CoacdArgs(
        threshold=float(object_cfg.get("coacd_threshold", 0.03)),
        max_convex_hull=int(object_cfg.get("coacd_max_convex_hull", 32)),
        mcts_iterations=int(object_cfg.get("coacd_mcts_iterations", 200)),
        resolution=int(object_cfg.get("coacd_resolution", 2000)),
    )
    tolerance = float(object_cfg.get("coacd_concavity_tolerance", 0.05))
    out_dir = OBJECT_COLLISION_SCENE_ROOT / safe_name(seq_key)
    sources = _object_collision_sources(seq_dir, objects)
    settings = {"version": OBJECT_COLLISION_VERSION, "coacd": vars(coacd_args), "tolerance": tolerance}
    metadata_path = out_dir / "metadata.json"
    if metadata_path.exists() and not force:
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("settings") == settings and metadata.get("sources") == sources:
                return out_dir if metadata.get("decomposed") else seq_dir
        except Exception:
            pass

    plans = []
    for obj in objects:
        if not obj["xml"]:
            plans.append((obj, None, []))
            continue
        root = ET.parse(obj["xml"]).getroot()
        base = _mjcf_mesh_base(root, Path(obj["xml"]))
        meshes = {mesh_el.get("name"): mesh_el for mesh_el in root.iter("mesh")}
        flagged = []
        for geom in _mjcf_collision_mesh_geoms(root):
            mesh_el = meshes.get(geom.get("mesh"))
            if mesh_el is None or not mesh_el.get("file"):
                continue
            mesh_path, mesh = _load_scaled_mesh(mesh_el, base)
            ratio = _concave_ratio(mesh)
            if ratio > 1.0 + tolerance:
                flagged.append((geom, mesh_path, mesh, ratio))
        plans.append((obj, root, flagged))
    decomposed = any(flagged for _obj, _root, flagged in plans)
    if dry_run:
        print(f"[HSIHOI][Collision] would {'decompose into ' + str(out_dir) if decomposed else 'keep'} {seq_dir}")
        return out_dir if decomposed else seq_dir
    if decomposed:
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    report = {}
    for obj, root, flagged in plans:
        stem = obj["name"]
        if root is None:
            if decomposed:
                shutil.copy2(obj["obj"], out_dir / Path(obj["obj"]).name)
                shutil.copy2(obj["prop"], out_dir / Path(obj["prop"]).name)
            continue
        if not decomposed:
            continue
        base = _mjcf_mesh_base(root, Path(obj["xml"]))
        for mesh_el in root.iter("mesh"):
            if mesh_el.get("file") and not Path(mesh_el.get("file")).is_absolute():
                mesh_el.set("file", str((base / mesh_el.get("file")).resolve()))
        compiler = root.find("compiler")
        if compiler is not None and "meshdir" in compiler.attrib:
            del compiler.attrib["meshdir"]
        asset = root.find("asset")
        parents = {child: parent for parent in root.iter() for child in parent}
        for geom, mesh_path, mesh, ratio in flagged:
            digest = geometry_digest(mesh)
            input_obj = OBJECT_COLLISION_CACHE_ROOT / "_inputs" / f"{digest}.obj"
            input_obj.parent.mkdir(parents=True, exist_ok=True)
            if not input_obj.exists():
                mesh.export(input_obj)
            reason = "not watertight" if np.isinf(ratio) else f"hull/mesh volume={ratio:.2f}"
            print(f"[HSIHOI][Collision] {stem}: {geom.get('name')} {reason}; convex pieces for {mesh_path.name}")
            parts, method = build_collision_cache(input_obj, OBJECT_COLLISION_CACHE_ROOT / digest, coacd_args)
            parent = parents[geom]
            index = list(parent).index(geom)
            parent.remove(geom)
            geom_name = geom.get("name") or f"{stem}_collision"
            for part_index, part in enumerate(parts):
                piece_mesh = f"{geom_name}_piece_{part_index}"
                ET.SubElement(asset, "mesh", {"name": piece_mesh, "file": str(part)})
                attrs = {key: value for key, value in geom.attrib.items() if key not in {"name", "mesh"}}
                attrs.update({"name": piece_mesh, "mesh": piece_mesh})
                parent.insert(index + part_index, ET.Element("geom", attrs))
            report[f"{stem}/{geom_name}"] = {
                "hull_over_mesh": None if np.isinf(ratio) else round(ratio, 3),
                "pieces": len(parts),
                "method": method,
            }
        used = {geom.get("mesh") for geom in root.iter("geom")}
        for mesh_el in list(asset):
            if mesh_el.tag == "mesh" and mesh_el.get("name") not in used:
                asset.remove(mesh_el)
        (out_dir / f"{stem}.xml").write_text(ET.tostring(root, encoding="unicode") + "\n", encoding="utf-8")
        shutil.copy2(obj["prop"], out_dir / Path(obj["prop"]).name)
    if not decomposed:
        OBJECT_COLLISION_SCENE_ROOT.mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps({"settings": settings, "sources": sources, "decomposed": decomposed, "report": report}, indent=2) + "\n",
        encoding="utf-8",
    )
    if decomposed:
        print(f"[HSIHOI][Collision] scene with convex collision pieces: {out_dir}")
        return out_dir
    return seq_dir


def grail_root_for_sequence(sequence_path: Path) -> Path:
    if not is_grail_sequence(sequence_path):
        raise ValueError(f"Not a GRAIL sequence: {sequence_path}")
    return Path(sequence_path).parent.parent


def repair_obj_face_winding(obj_path: Path) -> int:
    """Orient closed OBJ components outward without changing vertices or UVs."""
    obj_path = Path(obj_path)
    mesh = trimesh.load(obj_path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        return 0

    # USD assets may duplicate positions at UV/normal seams. Merge only in this
    # temporary topology so trimesh can propagate a consistent orientation;
    # the OBJ itself keeps its original vertices, UVs, and face ordering.
    mesh.merge_vertices(merge_tex=True, merge_norm=True)
    faces_before = np.asarray(mesh.faces, dtype=np.int64).copy()
    trimesh.repair.fix_normals(mesh, multibody=True)
    faces_after = np.asarray(mesh.faces, dtype=np.int64)
    unchanged = np.all(faces_after == faces_before, axis=1)
    reversed_faces = np.all(faces_after == faces_before[:, ::-1], axis=1)
    if not np.all(unchanged | reversed_faces):
        raise RuntimeError(f"OBJ winding repair unexpectedly reordered faces: {obj_path}")
    if not np.any(reversed_faces):
        return 0

    output_lines = []
    face_index = 0
    for line in obj_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("f "):
            tokens = line.split()
            if reversed_faces[face_index]:
                tokens[1:] = tokens[:0:-1]
            line = " ".join(tokens)
            face_index += 1
        output_lines.append(line)
    if face_index != len(reversed_faces):
        raise RuntimeError(
            f"OBJ winding repair face count mismatch: text={face_index}, "
            f"mesh={len(reversed_faces)}: {obj_path}"
        )
    obj_path.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
    flipped = int(reversed_faces.sum())
    print(f"[HSIHOI][GRAIL] repaired outward winding for {flipped} faces: {obj_path}")
    return flipped


def export_grail_usd_obj(
    usd_path: Path,
    obj_path: Path,
    texture_path: Path | None,
    object_scale: np.ndarray,
) -> None:
    try:
        from pxr import Usd, UsdGeom  # noqa: WPS433
    except ImportError as exc:
        raise ImportError("GRAIL USD export requires the pxr Python package (available in the sphere environment).") from exc

    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise ValueError(f"Could not open GRAIL USD: {usd_path}")
    xform_cache = UsdGeom.XformCache()
    vertex_lines = []
    uv_lines = []
    face_lines = []
    vertex_offset = 0
    uv_offset = 0
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64).reshape(-1, 3)
        matrix = np.asarray(xform_cache.GetLocalToWorldTransform(prim), dtype=np.float64)
        points = np.concatenate([points, np.ones((len(points), 1))], axis=1) @ matrix.T
        points = points[:, :3]
        points *= np.asarray(object_scale, dtype=np.float64).reshape(1, 3)
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
        st = UsdGeom.PrimvarsAPI(prim).GetPrimvar("st")
        uv_values = np.asarray(st.Get() or [], dtype=np.float64).reshape(-1, 2)
        uv_indices = np.asarray(st.GetIndices() or [], dtype=np.int64).reshape(-1)
        interpolation = str(st.GetInterpolation()) if st else ""
        vertex_lines.extend(f"v {p[0]:.9g} {p[1]:.9g} {p[2]:.9g}" for p in points)
        uv_lines.extend(f"vt {uv[0]:.9g} {uv[1]:.9g}" for uv in uv_values)
        cursor = 0
        for count in counts:
            count = int(count)
            polygon = indices[cursor : cursor + count]
            if interpolation == "faceVarying":
                polygon_uv = uv_indices[cursor : cursor + count] if len(uv_indices) else np.arange(cursor, cursor + count)
            elif interpolation in {"vertex", "varying"}:
                polygon_uv = polygon
            else:
                polygon_uv = np.zeros(count, dtype=np.int64)
            for corner in range(1, count - 1):
                tri = (0, corner, corner + 1)
                tokens = []
                for local in tri:
                    vi = int(polygon[local]) + vertex_offset + 1
                    if len(uv_values):
                        ti = int(polygon_uv[local]) + uv_offset + 1
                        tokens.append(f"{vi}/{ti}")
                    else:
                        tokens.append(str(vi))
                face_lines.append("f " + " ".join(tokens))
            cursor += count
        vertex_offset += len(points)
        uv_offset += len(uv_values)
    if not vertex_lines or not face_lines:
        raise ValueError(f"No triangle mesh found in GRAIL USD: {usd_path}")

    mtl_path = obj_path.with_suffix(".mtl")
    obj_text = [f"mtllib {mtl_path.name}", "usemtl grail_object", *vertex_lines, *uv_lines, *face_lines]
    obj_path.write_text("\n".join(obj_text) + "\n", encoding="utf-8")
    mtl_lines = ["newmtl grail_object", "Kd 1 1 1", "Ka 0 0 0", "Ks 0.05 0.05 0.05"]
    if texture_path is not None:
        mtl_lines.append(f"map_Kd {texture_path.resolve().as_posix()}")
    mtl_path.write_text("\n".join(mtl_lines) + "\n", encoding="utf-8")
    repair_obj_face_winding(obj_path)


def grail_usd_extent(usd_path: Path) -> np.ndarray:
    from pxr import Usd, UsdGeom  # noqa: WPS433

    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise ValueError(f"Could not open GRAIL USD: {usd_path}")
    xform_cache = UsdGeom.XformCache()
    lower = np.full(3, np.inf)
    upper = np.full(3, -np.inf)
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        points = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=np.float64).reshape(-1, 3)
        matrix = np.asarray(xform_cache.GetLocalToWorldTransform(prim), dtype=np.float64)
        points = (np.concatenate([points, np.ones((len(points), 1))], axis=1) @ matrix.T)[:, :3]
        lower = np.minimum(lower, points.min(axis=0))
        upper = np.maximum(upper, points.max(axis=0))
    return upper - lower


def resolve_grail_object_scale(usd_path: Path, obj_data: dict[str, Any]) -> tuple[np.ndarray, bool]:
    """Mesh scale to apply to a GRAIL USD and whether ``obj_scale`` was applied.

    Pickup objects ship metric USD meshes whose ``obj_scale`` equals their
    largest extent (a size descriptor, not a scale): applying it again shrinks
    them several times and leaves them floating above their support. Scene
    objects such as chairs store a true scale factor. Treat ``obj_scale`` as a
    size descriptor when it matches the USD's largest extent.
    """
    object_scale = np.asarray(obj_data.get("obj_scale", np.ones(3)), dtype=np.float64).reshape(-1)
    if object_scale.size == 1:
        object_scale = np.repeat(object_scale, 3)
    object_scale = object_scale[:3]
    if np.allclose(object_scale, 1.0) or not np.allclose(object_scale, object_scale[0]):
        return object_scale, True
    max_extent = float(grail_usd_extent(usd_path).max())
    if abs(max_extent - float(object_scale[0])) <= GRAIL_SIZE_DESCRIPTOR_TOLERANCE * float(object_scale[0]):
        print(
            f"[HSIHOI][GRAIL] obj_scale={float(object_scale[0]):.4f} equals the metric USD extent "
            f"{max_extent:.4f}; keeping the USD at its original size: {usd_path.name}"
        )
        return np.ones(3, dtype=np.float64), False
    return object_scale, True


def grail_table_box(payload: dict[str, Any]) -> dict[str, np.ndarray] | None:
    scene_data = payload.get("scene_data")
    if not isinstance(scene_data, dict) or not isinstance(scene_data.get("table"), dict):
        return None
    table = scene_data["table"]
    return {
        "pos": np.asarray(table["pos"], dtype=np.float64).reshape(3),
        "size": np.asarray(table["size"], dtype=np.float64).reshape(3),
    }


def write_grail_table_object(out_dir: Path, table: dict[str, np.ndarray], frame_count: int) -> None:
    """Export the static GRAIL table box as a second scene object.

    ``scene_data.table.size`` holds full extents; the sequence object rests on
    ``pos.z + size.z / 2``. The box is written as a mesh so it is sampled for
    contacts and collision-checked like any other object.
    """
    mesh = trimesh.creation.box(extents=table["size"])
    mesh.export(out_dir / f"{GRAIL_TABLE_STEM}.obj")
    (out_dir / f"{GRAIL_TABLE_STEM}.xml").write_text(
        f"<mujoco model=\"{GRAIL_TABLE_STEM}\">\n"
        f"  <asset><mesh name=\"{GRAIL_TABLE_STEM}_mesh\" file=\"{GRAIL_TABLE_STEM}.obj\"/></asset>\n"
        f"  <worldbody><body name=\"{GRAIL_TABLE_STEM}\"><freejoint name=\"{GRAIL_TABLE_STEM}_freejoint\"/>"
        f"<geom name=\"{GRAIL_TABLE_STEM}_visual\" type=\"mesh\" mesh=\"{GRAIL_TABLE_STEM}_mesh\" "
        "group=\"2\" contype=\"0\" conaffinity=\"0\" rgba=\"0.72 0.6 0.45 1\"/>"
        f"<geom name=\"{GRAIL_TABLE_STEM}_collision\" type=\"mesh\" mesh=\"{GRAIL_TABLE_STEM}_mesh\" "
        "group=\"3\" contype=\"1\" conaffinity=\"1\" rgba=\"0.25 0.45 0.8 0.15\"/>"
        "</body></worldbody>\n</mujoco>\n",
        encoding="utf-8",
    )
    with (out_dir / f"prop_{GRAIL_TABLE_STEM}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["px", "py", "pz", "qx", "qy", "qz", "qw"])
        row = [*table["pos"].tolist(), 0.0, 0.0, 0.0, 1.0]
        for _ in range(int(frame_count)):
            writer.writerow(row)


def grail_asset_scale_current(sequence_path: Path, usd_path: Path, metadata: dict[str, Any], asset_dir: Path) -> bool:
    """Whether cached GRAIL assets match the current scale rule and scene export."""
    payload = load_grail_pickle(sequence_path)
    if grail_table_box(payload) is not None and not (asset_dir / f"prop_{GRAIL_TABLE_STEM}.csv").exists():
        return False
    _scale, applied = resolve_grail_object_scale(usd_path, payload.get("obj_data", {}))
    # Assets written before the rule existed always applied obj_scale.
    return bool(metadata.get("object_scale_applied", True)) == bool(applied)


def prepare_grail_object_assets(
    seq_key: str,
    sequence_path: Path,
    config: dict[str, Any],
    force: bool,
    dry_run: bool,
    work_dir: Path,
) -> Path:
    root = grail_root_for_sequence(sequence_path)
    hsi = hsi_config(config)
    object_cfg = section(hsi, "object")
    offline_root_value = object_cfg.get("grail_mjcf_dir")
    if isinstance(offline_root_value, str) and offline_root_value.strip().lower() == "auto":
        offline_root_value = root / "object_mjcf"
    if offline_root_value:
        offline_root = (
            offline_root_value
            if isinstance(offline_root_value, Path)
            else resolve_path(offline_root_value, config)
        )
        assert offline_root is not None
        offline_dir = offline_root / safe_name(seq_key)

        def offline_asset_current() -> bool:
            required = [
                offline_dir / "grail_object.xml",
                offline_dir / "grail_object.obj",
                offline_dir / "prop_grail_object.csv",
                offline_dir / "metadata.json",
            ]
            if not all(path.exists() for path in required):
                return False
            try:
                metadata = json.loads((offline_dir / "metadata.json").read_text(encoding="utf-8"))
                collision_count = int(metadata.get("collision_parts", 0))
                collision_paths = [
                    offline_dir / f"grail_object_collision_{index}.obj"
                    for index in range(collision_count)
                ]
                return bool(
                    int(metadata.get("asset_version", 0)) == GRAIL_CONVEX_MJCF_VERSION
                    and collision_count > 0
                    and all(path.exists() for path in collision_paths)
                    and grail_asset_scale_current(
                        sequence_path,
                        root / "object_usd" / f"{sequence_path.stem}.usd",
                        metadata,
                        offline_dir,
                    )
                )
            except Exception:
                return False

        if offline_asset_current():
            return offline_dir
        if dry_run:
            print(f"[HSIHOI][GRAIL] offline convex MJCF required: {offline_dir}")
            return offline_dir
        if bool(object_cfg.get("grail_auto_prepare_mjcf", True)):
            prepare_script = ROOT / "scripts/prepare_grail_object_mjcf.py"
            prepare_python_value = object_cfg.get("grail_mjcf_python")
            if prepare_python_value:
                prepare_python = Path(prepare_python_value).expanduser()
                if not prepare_python.is_file():
                    # Bare command names such as "python" resolve through PATH.
                    resolved = shutil.which(str(prepare_python_value))
                    if resolved:
                        prepare_python = Path(resolved)
            else:
                prepare_python = Path(sys.executable)
            if not prepare_python.is_file():
                raise FileNotFoundError(f"GRAIL MJCF preparation Python not found: {prepare_python}")
            command = [
                str(prepare_python),
                "-u",
                str(prepare_script),
                "--data-root",
                str(root),
                "--output-dir",
                str(offline_root),
                "--seq-key",
                str(seq_key),
                "--threshold",
                str(float(object_cfg.get("grail_coacd_threshold", 0.03))),
                "--max-convex-hull",
                str(int(object_cfg.get("grail_coacd_max_convex_hull", 32))),
                "--mcts-iterations",
                str(int(object_cfg.get("grail_coacd_mcts_iterations", 200))),
                "--resolution",
                str(int(object_cfg.get("grail_coacd_resolution", 2000))),
                "--progress-every",
                "1",
                "--fail-fast",
            ]
            print(f"[HSIHOI][GRAIL] convex MJCF missing; preparing {seq_key} before retarget")
            print(f"[HSIHOI][GRAIL] {' '.join(command)}")
            subprocess.run(command, check=True, cwd=ROOT)
            if offline_asset_current():
                return offline_dir
            raise RuntimeError(f"GRAIL convex MJCF preparation completed without valid assets: {offline_dir}")
        raise FileNotFoundError(
            f"Offline GRAIL convex MJCF not found for {seq_key!r}: {offline_dir}. "
            "Set hsi_hoi.object.grail_auto_prepare_mjcf=true or run the offline converter."
        )

    out_dir = work_dir / "grail_objects" / safe_name(seq_key)
    stem = "grail_object"
    obj_path = out_dir / f"{stem}.obj"
    xml_path = out_dir / f"{stem}.xml"
    prop_path = out_dir / f"prop_{stem}.csv"
    version_path = out_dir / ".grail_asset_version"
    usd_path = root / "object_usd" / f"{sequence_path.stem}.usd"
    texture_dir = root / "object_usd" / "textures" / sequence_path.stem
    texture_path = next((p for p in sorted(texture_dir.glob("*")) if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}), None)
    asset_current = version_path.exists() and version_path.read_text(encoding="utf-8").strip() == GRAIL_OBJECT_ASSET_VERSION
    if obj_path.exists() and xml_path.exists() and prop_path.exists() and asset_current and not force:
        return out_dir
    print(f"[HSIHOI][GRAIL] prepare object USD={usd_path} texture={texture_path or '<none>'} -> {out_dir}")
    if dry_run:
        return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = load_grail_pickle(sequence_path)
    obj_data = payload.get("obj_data", {})
    object_scale, _applied = resolve_grail_object_scale(usd_path, obj_data)
    export_grail_usd_obj(usd_path, obj_path, texture_path, object_scale)
    xml_path.write_text(
        "<mujoco model=\"grail_object\">\n"
        "  <asset><mesh name=\"grail_object_mesh\" file=\"grail_object.obj\"/></asset>\n"
        "  <worldbody><body name=\"grail_object\"><freejoint name=\"grail_object_freejoint\"/>"
        "<geom name=\"grail_object_visual\" type=\"mesh\" mesh=\"grail_object_mesh\" rgba=\"1 1 1 1\"/>"
        "</body></worldbody>\n</mujoco>\n",
        encoding="utf-8",
    )
    positions = np.asarray(obj_data["obj_t"], dtype=np.float64).reshape(-1, 3)
    rotations = np.asarray(obj_data["obj_R"], dtype=np.float64).reshape(-1, 3, 3)
    quats = R.from_matrix(rotations).as_quat()
    with prop_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["px", "py", "pz", "qx", "qy", "qz", "qw"])
        for pos, quat in zip(positions, quats):
            writer.writerow([*pos.tolist(), *quat.tolist()])
    table = grail_table_box(payload)
    if table is not None:
        write_grail_table_object(out_dir, table, len(positions))
    version_path.write_text(GRAIL_OBJECT_ASSET_VERSION + "\n", encoding="utf-8")
    return out_dir


def export_standard_motion_npz(
    seq_key: str,
    seq_dir: Path,
    work_dir: Path,
    dry_run: bool,
    object_dir: Path | None = None,
) -> Path:
    out = work_dir / "motion" / f"{safe_name(seq_key)}.npz"
    motion = load_samp_motion(seq_dir)
    objects = discover_samp_objects(object_dir or seq_dir)
    source_dataset = source_dataset_kind(seq_dir)
    source_format = source_format_for_sequence(seq_dir)
    print(
        f"[HSIHOI] export {source_dataset} {seq_key} frames={len(motion['poses'])} "
        f"fps={motion['fps']:.3f} gender={motion['gender']} output_up={motion['output_up']} objects={len(objects)}"
    )
    if dry_run:
        print(f"[HSIHOI] would write exported motion: {out}")
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        poses=motion["poses"].astype(np.float32),
        pose_aa=motion["poses"].astype(np.float32),
        trans=motion["trans"].astype(np.float32),
        transl=motion["trans"].astype(np.float32),
        betas=motion["betas"].astype(np.float32),
        beta=motion["betas"].astype(np.float32),
        gender=np.asarray(motion["gender"]),
        fps=np.asarray([motion["fps"]], dtype=np.float32),
        mocap_frame_rate=np.asarray([motion["fps"]], dtype=np.float32),
        source_dataset=np.asarray(source_dataset),
        source_format=np.asarray(source_format),
        source_sequence_key=np.asarray(seq_key),
        source_data_dir=np.asarray(str(seq_dir)),
        samp_output_up=np.asarray(motion["output_up"]),
        human_scale=np.asarray([motion["human_scale"]], dtype=np.float32),
        human_scale_mode=np.asarray(motion["human_scale_mode"]),
        samp_objects=np.asarray(json.dumps(objects, sort_keys=True)),
    )
    return out


def default_result_path(config: dict[str, Any], seq_key: str) -> Path:
    robot = robot_config(config)
    return ROOT / "output" / f"{robot['name']}_retarget" / f"{safe_name(seq_key)}_hsi_hoi_{robot['name']}.npz"


def configure_single_smpl_template(config: dict[str, Any], seq_key: str, seq_dir: Path) -> None:
    motion = load_samp_motion(seq_dir)
    config.setdefault("correspondence", {})
    corr = config["correspondence"]
    corr.setdefault("dataset", {})
    corr.setdefault("train", {})
    dataset = corr["dataset"]
    train = corr["train"]
    model_type = str(motion["model_type"]).lower()
    gender = str(motion["gender"]).lower()
    model_dir = config.get(f"{model_type}_model_dir", config.get("smplx_model_dir", "smpl"))
    dataset["smpl_models"] = [
        {
            "type": model_type,
            "dir": model_dir,
            "genders": [gender],
        }
    ]
    configured_template_name = str(section(config, "smpl_template").get("name", "auto"))
    if configured_template_name not in {"", "auto"}:
        dataset["smpl_models"][0]["name"] = configured_template_name
    robot_name = safe_name(robot_config(config)["name"])
    seq_name = safe_name(seq_key)
    dataset["out"] = f"data/correspondence_{robot_name}_{seq_name}_hsi_hoi.npz"
    train["out_dir"] = f"output/correspondence_{robot_name}_{seq_name}_hsi_hoi"


def normalize_solver_config(config: dict[str, Any]) -> None:
    solver = config.setdefault("solver", {})
    warm_start = str(solver.get("trajectory_warm_start_mode", "sequential"))
    if warm_start == "forward":
        solver["trajectory_warm_start_mode"] = "sequential"


def make_runtime_config(
    base_config: dict[str, Any],
    seq_key: str,
    seq_dir: Path,
    motion_npz: Path,
    work_dir: Path,
    args,
    object_dir: Path | None = None,
) -> tuple[dict[str, Any], Path]:
    config = copy.deepcopy(clean_config(base_config))
    config.setdefault("motion", {})
    config["motion"]["data"] = str(motion_npz)
    # A standalone NPZ is indexed by its filename stem in load_motion_collection.
    # Long source names are shortened by safe_name(), so the original GRAIL key
    # cannot be used to select the cached motion.
    config["motion"]["seq_key"] = Path(motion_npz).stem
    config["motion"]["seq_index"] = 0
    for key in ("start", "end", "stride", "max_frames"):
        value = getattr(args, key)
        if value is not None:
            config["motion"][key] = value

    configure_single_smpl_template(config, seq_key, seq_dir)
    normalize_solver_config(config)
    if getattr(args, "scene_mode", None):
        config.setdefault("solver", {})["retarget_scene_mode"] = str(args.scene_mode)
    source_motion = load_samp_motion(seq_dir)
    object_cfg = config.setdefault("hsi_hoi", {}).setdefault("object", {})
    object_cfg["output_up"] = source_motion["output_up"]

    config.setdefault("retarget", {})
    if args.out is not None:
        config["retarget"]["out"] = str(args.out)
    elif not config["retarget"].get("out"):
        config["retarget"]["out"] = str(default_result_path(config, seq_key))

    temp_path = work_dir / "configs" / f"{safe_name(seq_key)}_runtime_config.json"
    config["_config_path"] = str(temp_path)
    config["_config_dir"] = str(temp_path.parent)
    config.setdefault("hsi_hoi", {})
    config["hsi_hoi"]["source_sequence_dir"] = str(object_dir or seq_dir)
    config["hsi_hoi"]["source_sequence_key"] = str(seq_key)
    if is_grail_sequence(seq_dir):
        object_cfg = config["hsi_hoi"].setdefault("object", {})
        object_cfg.update({"output_up": "z", "convert_y_up": False, "object_scale": 1.0})
        view_args = config.setdefault("view", {}).setdefault("extra_args", {})
        view_args["source_object_scale"] = 1.0

    if not args.dry_run:
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path.write_text(json.dumps(clean_config(config), indent=2, sort_keys=True) + "\n")
    else:
        print(f"[HSIHOI] would write runtime config: {temp_path}")
    return config, temp_path


def patch_result_with_samp_metadata(
    result_path: Path,
    config: dict[str, Any],
    seq_key: str,
    seq_dir: Path,
    motion_npz: Path,
    dry_run: bool,
    object_dir: Path | None = None,
) -> None:
    result_path = Path(result_path)
    if dry_run:
        print(f"[HSIHOI] would patch result metadata for HSI/HOI objects: {result_path}")
        return
    if not result_path.exists():
        raise FileNotFoundError(f"Retarget result does not exist: {result_path}")

    motion = load_samp_motion(seq_dir)
    source_format = source_format_for_sequence(seq_dir)
    hsi = hsi_config(config)
    object_cfg = section(hsi, "object")
    source_object_dir = Path(object_dir or seq_dir)
    objects = discover_samp_objects(source_object_dir)
    with np.load(result_path, allow_pickle=True) as data:
        payload = {name: data[name] for name in data.files if name in RETARGET_VISUALIZATION_FIELDS}

    payload.update(
        {
            "source_data": np.asarray(str(Path(seq_dir).resolve())),
            "source_object_dir": np.asarray(str(source_object_dir)),
            "source_sequence_key": np.asarray(str(seq_key)),
            "source_format": np.asarray(source_format),
            "noitom_output_up": np.asarray(motion["output_up"]),
            "noitom_convert_y_up": np.asarray([bool(object_cfg.get("convert_y_up", True))]),
            "noitom_ground_align": np.asarray([bool(object_cfg.get("ground_align", False))]),
            "noitom_floor_y": np.asarray([float(object_cfg.get("floor_y", 0.0))], dtype=np.float32),
            "noitom_ground_offset": np.asarray([float(object_cfg.get("ground_offset", 0.0))], dtype=np.float32),
        }
    )
    np.savez_compressed(result_path, **payload)
    print(f"[HSIHOI] patched result object metadata: {result_path} objects={len(objects)} source={seq_dir}")


def expected_frame_count(config: dict[str, Any], motion_npz: Path) -> int:
    motion = section(config, "motion")
    with np.load(motion_npz, allow_pickle=True) as data:
        if "poses" in data:
            total = int(np.asarray(data["poses"]).shape[0])
        elif "pose_aa" in data:
            total = int(np.asarray(data["pose_aa"]).shape[0])
        else:
            return -1
    start = int(motion.get("start", 0))
    end_value = int(motion.get("end", -1))
    end = total if end_value < 0 else min(end_value, total)
    stride = max(1, int(motion.get("stride", 1)))
    count = len(range(start, end, stride))
    max_frames = int(motion.get("max_frames", 0))
    if max_frames > 0:
        count = min(count, max_frames)
    return int(count)


def source_input_paths(seq_dir: Path) -> list[Path]:
    seq_dir = Path(seq_dir).resolve()
    if seq_dir.is_file():
        return [seq_dir]
    if not seq_dir.is_dir():
        return []
    return sorted(path for path in seq_dir.iterdir() if path.is_file())


def hsi_result_compatible(
    result_path: Path,
    config: dict[str, Any],
    motion_npz: Path,
    seq_key: str,
    seq_dir: Path,
) -> bool:
    result_path = Path(result_path)
    motion_npz = Path(motion_npz).resolve()
    if not result_path.exists() or not motion_npz.exists():
        return False
    try:
        with np.load(result_path, allow_pickle=True) as data:
            if not retarget_result_has_final_qpos_only(data):
                raise ValueError("result is not limited to playback/visualization fields")
            saved_frames = int(np.asarray(data["qpos"]).shape[0])
            saved_seq_key = str(np.asarray(data["source_sequence_key"]).item())
            saved_format = str(np.asarray(data["source_format"]).item())
            saved_scene_mode = (
                str(np.asarray(data["retarget_scene_mode"]).item()) if "retarget_scene_mode" in data.files else "scaled"
            )
        expected_scene_mode = str(section(config, "solver").get("retarget_scene_mode", "scaled")).lower()
        if saved_scene_mode != expected_scene_mode:
            raise ValueError(f"saved scene mode={saved_scene_mode}, expected scene mode={expected_scene_mode}")
        expected_frames = expected_frame_count(config, motion_npz)
        if expected_frames >= 0 and saved_frames != expected_frames:
            raise ValueError(f"saved_frames={saved_frames}, expected_frames={expected_frames}")
        if saved_seq_key != str(seq_key):
            raise ValueError(f"saved sequence={saved_seq_key}, expected sequence={seq_key}")
        expected_format = source_format_for_sequence(seq_dir)
        if saved_format != expected_format:
            raise ValueError(f"saved format={saved_format}, expected format={expected_format}")
        result_mtime_ns = result_path.stat().st_mtime_ns
        newer_sources = [
            path for path in source_input_paths(seq_dir)
            if path.stat().st_mtime_ns > result_mtime_ns
        ]
        if newer_sources:
            raise ValueError(f"source is newer than result: {newer_sources[0]}")
    except Exception as exc:
        print(f"[HSIHOI] rebuild retarget result: {result_path}: {exc}")
        return False
    return True


def retarget_hsi_motion(
    config: dict[str, Any],
    slots_path: Path,
    motion_npz: Path,
    seq_key: str,
    seq_dir: Path,
    force: bool = False,
    dry_run: bool = False,
) -> Path:
    result_path = retarget_out(config)
    if result_path.exists() and not force:
        if hsi_result_compatible(result_path, config, motion_npz, seq_key, seq_dir):
            print(f"[HSIHOI] reuse retarget result: {result_path}")
            return result_path
        print(f"[HSIHOI] rebuild retarget result: {result_path}")
    return retarget_motion(config, slots_path, force=True if result_path.exists() and not force else force, dry_run=dry_run)


def run_pipeline(args, work_dir: Path):
    base_config = load_hsi_hoi_config(args.config, args.defaults)
    seq_key, seq_dir = resolve_samp_sequence(base_config, args.data, args.seq_key)
    object_dir = (
        prepare_grail_object_assets(
            seq_key,
            seq_dir,
            base_config,
            args.force_retarget,
            args.dry_run,
            work_dir,
        )
        if is_grail_sequence(seq_dir)
        else prepare_standard_object_collision(
            seq_key,
            seq_dir,
            base_config,
            args.force_retarget,
            args.dry_run,
        )
    )
    motion_npz = export_standard_motion_npz(
        seq_key,
        seq_dir,
        work_dir,
        args.dry_run,
        object_dir=object_dir,
    )
    config, temp_config_path = make_runtime_config(
        base_config,
        seq_key,
        seq_dir,
        motion_npz,
        work_dir,
        args,
        object_dir=object_dir,
    )
    print(f"[HSIHOI] runtime config={temp_config_path}")

    if args.stage == "prepare":
        return

    stages = ["build", "train", "retarget"] if args.stage == "all" else [args.stage]
    dataset_path = dataset_out(config)
    slots_path = slots_out(config)
    result_path = retarget_out(config)

    if "build" in stages:
        dataset_path = build_correspondence_dataset(config, force=args.force_build, dry_run=args.dry_run)
    if "train" in stages:
        if not args.dry_run:
            dataset_path = build_correspondence_dataset(config, force=args.force_build, dry_run=args.dry_run)
        slots_path = train_correspondence(config, dataset_path, force=args.force_train, dry_run=args.dry_run)
    if "retarget" in stages:
        if (not slots_path.exists() or not correspondence_slots_compatible(slots_path, config)) and not args.dry_run:
            dataset_path = build_correspondence_dataset(config, force=args.force_build, dry_run=args.dry_run)
            slots_path = train_correspondence(config, dataset_path, force=args.force_train, dry_run=args.dry_run)
        slots_path = ensure_retarget_slots(
            config, slots_path, force_build=args.force_build, force_train=args.force_train, dry_run=args.dry_run
        )
        result_path = retarget_hsi_motion(
            config,
            slots_path,
            motion_npz,
            seq_key,
            seq_dir,
            force=args.force_retarget,
            dry_run=args.dry_run,
        )
        patch_result_with_samp_metadata(result_path, config, seq_key, seq_dir, motion_npz, args.dry_run, object_dir=object_dir)
    if args.stage == "view":
        patch_result_with_samp_metadata(result_path, config, seq_key, seq_dir, motion_npz, args.dry_run, object_dir=object_dir)
        visualize_result(config, result_path, dry_run=args.dry_run)
    elif args.stage == "all" and not args.skip_view:
        visualize_result(config, result_path, dry_run=args.dry_run)


def main():
    args = parse_args()
    with tempfile.TemporaryDirectory(prefix="umr_hsi_hoi_", dir="/tmp") as work_dir:
        run_pipeline(args, Path(work_dir))


if __name__ == "__main__":
    main()
