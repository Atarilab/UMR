#!/usr/bin/env python3
"""Flatten an MJCF that uses <model>/<attach> into one self-contained XML.

UMR writes temporary robot XMLs (floating base, merged objects, viewer scenes)
in other directories, where the relative <model file=...> includes and the
per-model mesh directories of an attached hand no longer resolve. MuJoCo
compiles the original assembly, saves the flattened spec, and every mesh file
is rewritten relative to the output XML.
"""
from __future__ import annotations

import argparse
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np


def mesh_search_dirs(xml_path: Path) -> list[Path]:
    """Mesh directories of the main model and of every model it attaches."""
    dirs = []
    pending = [Path(xml_path).resolve()]
    seen = set()
    while pending:
        path = pending.pop(0)
        if path in seen:
            continue
        seen.add(path)
        # MuJoCo tolerates '--' inside comments; ElementTree does not.
        root = ET.fromstring(re.sub(r"<!--.*?-->", "", path.read_text(encoding="utf-8"), flags=re.S))
        compiler = root.find("compiler")
        meshdir = compiler.get("meshdir") if compiler is not None else None
        dirs.append((path.parent / meshdir).resolve() if meshdir else path.parent)
        for asset in root.findall("asset"):
            for model in asset.findall("model"):
                if model.get("file"):
                    pending.append((path.parent / model.get("file")).resolve())
    return dirs


def flatten_mjcf(src: Path, out: Path) -> Path:
    src = Path(src).resolve()
    out = Path(out).resolve()
    model = mujoco.MjModel.from_xml_path(str(src))
    tmp = out.with_suffix(".unresolved.xml")
    mujoco.mj_saveLastXML(str(tmp), model)
    text = tmp.read_text(encoding="utf-8")
    tmp.unlink()

    root = ET.fromstring(text)
    compiler = root.find("compiler")
    if compiler is not None:
        for key in ("meshdir", "texturedir", "assetdir"):
            compiler.attrib.pop(key, None)
    search = mesh_search_dirs(src)
    for element in root.iter():
        if element.tag not in {"mesh", "texture", "hfield", "skin"} or not element.get("file"):
            continue
        name = element.get("file")
        matches = [directory / name for directory in search if (directory / name).is_file()]
        if not matches:
            raise FileNotFoundError(f"{element.tag} file {name!r} not found under {search}")
        if len({m.resolve() for m in matches}) > 1:
            raise ValueError(f"{element.tag} file {name!r} is ambiguous: {matches}")
        rel = Path(matches[0]).resolve().relative_to(out.parent) if matches[0].resolve().is_relative_to(out.parent) else None
        element.set("file", str(rel) if rel is not None else str(matches[0].resolve()))
    out.write_text(ET.tostring(root, encoding="unicode") + "\n", encoding="utf-8")
    return out


def check_equivalent(src: Path, out: Path, seed: int = 0) -> None:
    """Same sizes, and body/geom/site poses within 1 um at random configurations (XML float rounding)."""
    a = mujoco.MjModel.from_xml_path(str(src))
    b = mujoco.MjModel.from_xml_path(str(out))
    for field in ("nq", "nv", "nbody", "njnt", "ngeom", "nsite", "nmesh", "neq", "nu"):
        if getattr(a, field) != getattr(b, field):
            raise AssertionError(f"{field}: {getattr(a, field)} != {getattr(b, field)}")
    rng = np.random.default_rng(seed)
    max_diff = 0.0
    da, db = mujoco.MjData(a), mujoco.MjData(b)
    for _ in range(3):
        qpos = a.qpos0.copy()
        for joint in range(a.njnt):
            if a.jnt_type[joint] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE) and a.jnt_limited[joint]:
                low, high = a.jnt_range[joint]
                qpos[a.jnt_qposadr[joint]] = rng.uniform(low, high)
        da.qpos[:] = qpos
        db.qpos[:] = qpos
        mujoco.mj_forward(a, da)
        mujoco.mj_forward(b, db)
        for field in ("xpos", "geom_xpos", "site_xpos"):
            diff = float(np.abs(getattr(da, field) - getattr(db, field)).max(initial=0.0))
            max_diff = max(max_diff, diff)
            if diff > 1e-6:
                raise AssertionError(f"{field} differs by {diff}")
    print(
        f"[FlattenMJCF] {out.name}: nq={b.nq} nv={b.nv} ngeom={b.ngeom} nsite={b.nsite} neq={b.neq}; "
        f"max pose difference to the source {max_diff:.1e} m"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    out = flatten_mjcf(args.src, args.out)
    check_equivalent(args.src, out)


if __name__ == "__main__":
    main()
