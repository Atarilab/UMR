"""Convex collision pieces for object meshes, cached by geometry.

MuJoCo collides every mesh geom as its convex hull, so a concave object (a bowl,
a microwave cavity, a chair) needs several convex pieces to be touched and
entered the way it was in the source. Shared by the GRAIL and OMOMO converters
and the HSI/HOI pipeline.
"""
from __future__ import annotations

import fcntl
import hashlib
import sys
from pathlib import Path

import numpy as np
import trimesh
from obj2mjcf.cli import CoacdArgs, decompose_convex

__all__ = [
    "CoacdArgs",
    "build_collision_cache",
    "collision_part_index",
    "exact_convex_components",
    "geometry_digest",
    "sanitize_collision_cache",
    "sorted_collision_parts",
    "valid_collision_part",
]


def geometry_digest(mesh: trimesh.Trimesh) -> str:
    vertices = np.asarray(mesh.vertices, dtype=np.float64).round(8)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    digest = hashlib.sha256()
    digest.update(vertices.tobytes())
    digest.update(faces.tobytes())
    return digest.hexdigest()[:20]


def exact_convex_components(mesh: trimesh.Trimesh) -> list[trimesh.Trimesh] | None:
    processed = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=True)
    components = list(processed.split(only_watertight=False))
    if components and all(component.is_watertight and component.is_convex for component in components):
        return components
    return None


def collision_part_index(path: Path) -> int:
    tail = path.stem.rsplit("_", 1)[-1]
    return int(tail) if tail.isdigit() else sys.maxsize


def sorted_collision_parts(directory: Path) -> list[Path]:
    return sorted(directory.glob("collision_*.obj"), key=collision_part_index)


def valid_collision_part(path: Path) -> bool:
    try:
        mesh = trimesh.load(path, force="mesh", process=False)
        vertices = np.asarray(mesh.vertices, dtype=np.float64).reshape(-1, 3)
        unique = np.unique(vertices.round(10), axis=0)
        return bool(len(unique) >= 4 and np.linalg.matrix_rank(unique - unique.mean(axis=0)) >= 3)
    except Exception:
        return False


def sanitize_collision_cache(cache_dir: Path) -> list[Path]:
    parts = sorted_collision_parts(cache_dir)
    valid_parts = []
    for part in parts:
        if valid_collision_part(part):
            valid_parts.append(part)
        else:
            print(f"[ObjectCollision][WARN] dropping degenerate collision hull: {part}", flush=True)
            part.unlink()
    if not valid_parts:
        return []
    temporary = []
    for index, part in enumerate(valid_parts):
        target = cache_dir / f".collision_{index}.obj.tmp"
        part.replace(target)
        temporary.append(target)
    outputs = []
    for index, temporary_path in enumerate(temporary):
        target = cache_dir / f"collision_{index}.obj"
        temporary_path.replace(target)
        outputs.append(target)
    return outputs


def build_collision_cache(visual_obj: Path, cache_dir: Path, coacd_args: CoacdArgs) -> tuple[list[Path], str]:
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_path = cache_dir.parent / f".{cache_dir.name}.lock"
    with lock_path.open("w", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        cached_parts = sorted_collision_parts(cache_dir)
        method_path = cache_dir / "method.txt"
        if cached_parts and method_path.exists():
            cached_parts = sanitize_collision_cache(cache_dir)
            if cached_parts:
                return cached_parts, method_path.read_text(encoding="utf-8").strip()

        cache_dir.mkdir(parents=True, exist_ok=True)
        for partial in cache_dir.glob("*.obj"):
            partial.unlink()
        mesh = trimesh.load(visual_obj, force="mesh", process=True)
        if not isinstance(mesh, trimesh.Trimesh):
            raise TypeError(f"Expected one Trimesh from {visual_obj}, got {type(mesh)}")
        components = exact_convex_components(mesh)
        if components is not None:
            method = "exact_convex_components"
            for index, component in enumerate(components):
                component.export(cache_dir / f"collision_{index}.obj")
        else:
            method = "obj2mjcf_coacd"
            decompose_convex(visual_obj, cache_dir, coacd_args)
            generated = sorted(
                cache_dir.glob(f"{visual_obj.stem}_collision_*.obj"),
                key=collision_part_index,
            )
            for index, path in enumerate(generated):
                path.replace(cache_dir / f"collision_{index}.obj")
        parts = sanitize_collision_cache(cache_dir)
        if not parts:
            raise RuntimeError(f"Convex decomposition produced no collision parts: {visual_obj}")
        method_path.write_text(method + "\n", encoding="utf-8")
        return parts, method
