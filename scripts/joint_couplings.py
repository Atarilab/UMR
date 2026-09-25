"""Exact joint couplings (MJCF ``<equality><joint>``) for the per-frame QP.

Dexterous hands such as the Inspire RH56E2 drive a distal joint from a proximal
one (``dip = 1.0843 * joint``). The QP solves in reduced coordinates
``dq = T dq_r`` over independent dofs, so the couplings hold exactly instead of
being fought by the surface objective, and every qpos is projected back onto
them after integration.
"""
from __future__ import annotations

import mujoco
import numpy as np


SCALAR_JOINTS = (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))


def _poly(coef, x):
    return coef[0] + coef[1] * x + coef[2] * x**2 + coef[3] * x**3 + coef[4] * x**4


def _dpoly(coef, x):
    return coef[1] + 2.0 * coef[2] * x + 3.0 * coef[3] * x**2 + 4.0 * coef[4] * x**3


class JointCouplings:
    """``q_dep - q0_dep = poly(q_drv - q0_drv)`` for every coupled scalar joint."""

    def __init__(self, model, couplings):
        self.nv = int(model.nv)
        by_dep = {}
        for item in couplings:
            if item["dep_dof"] in by_dep:
                raise ValueError(f"joint {item['name']!r} has more than one coupling")
            by_dep[item["dep_dof"]] = item
        order = []
        done = set()

        def visit(dof, stack=()):
            if dof in done or dof not in by_dep:
                return
            if dof in stack:
                raise ValueError(f"cyclic joint couplings through {by_dep[dof]['name']!r}")
            driver = by_dep[dof]["drv_dof"]
            if driver is not None:
                visit(driver, stack + (dof,))
            done.add(dof)
            order.append(by_dep[dof])

        for dof in sorted(by_dep):
            visit(dof)
        self.couplings = order
        self.dependent_dofs = np.asarray(sorted(by_dep), dtype=np.int32)
        self.independent_dofs = np.asarray(
            [dof for dof in range(self.nv) if dof not in by_dep], dtype=np.int32
        )
        self.column_of_dof = {int(dof): col for col, dof in enumerate(self.independent_dofs)}
        # Every dependent resolves to one independent root driver (or None for a locked joint).
        self.root_of = {}
        for item in self.couplings:
            driver = item["drv_dof"]
            self.root_of[item["dep_dof"]] = None if driver is None else self.root_of.get(driver, driver)
        self.linear = all(np.allclose(item["poly"][2:], 0.0) for item in self.couplings)
        self.names = [item["name"] for item in self.couplings]

    @classmethod
    def from_model(cls, model, mode="off"):
        """Couplings from the MJCF equality block, an explicit list, or ``None``."""
        if mode is None or (isinstance(mode, str) and mode.strip().lower() in {"off", "none", "false", ""}):
            return None
        if isinstance(mode, str) and mode.strip().lower() not in {"mjcf", "auto"}:
            raise ValueError(f"joint_couplings must be 'mjcf', 'auto', 'off', or a list, got {mode!r}")
        items = []
        if isinstance(mode, (list, tuple)):
            for spec in mode:
                dep = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, str(spec["joint"]))
                drv = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, str(spec["source"])) if spec.get("source") else -1
                if dep < 0 or (spec.get("source") and drv < 0):
                    raise ValueError(f"joint coupling references an unknown joint: {spec}")
                items.append(cls._item(model, dep, drv, np.pad(np.asarray(spec["polycoef"], dtype=np.float64), (0, 5))[:5]))
        else:
            for eq in range(model.neq):
                if int(model.eq_type[eq]) != int(mujoco.mjtEq.mjEQ_JOINT) or not bool(model.eq_active0[eq]):
                    continue
                items.append(cls._item(model, int(model.eq_obj1id[eq]), int(model.eq_obj2id[eq]), model.eq_data[eq][:5]))
        if not items:
            return None
        return cls(model, items)

    @staticmethod
    def _item(model, dep_joint, drv_joint, poly):
        for joint in (dep_joint, drv_joint):
            if joint >= 0 and int(model.jnt_type[joint]) not in SCALAR_JOINTS:
                raise ValueError("joint couplings support hinge and slide joints only")
        return {
            "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, dep_joint) or f"joint_{dep_joint}",
            "dep_q": int(model.jnt_qposadr[dep_joint]),
            "dep_dof": int(model.jnt_dofadr[dep_joint]),
            "drv_q": None if drv_joint < 0 else int(model.jnt_qposadr[drv_joint]),
            "drv_dof": None if drv_joint < 0 else int(model.jnt_dofadr[drv_joint]),
            "q0_dep": float(model.qpos0[model.jnt_qposadr[dep_joint]]),
            "q0_drv": 0.0 if drv_joint < 0 else float(model.qpos0[model.jnt_qposadr[drv_joint]]),
            "poly": np.asarray(poly, dtype=np.float64).copy(),
        }

    def _dependent_value(self, item, qpos):
        x = 0.0 if item["drv_q"] is None else float(qpos[item["drv_q"]]) - item["q0_drv"]
        return item["q0_dep"] + _poly(item["poly"], x)

    def project(self, qpos):
        qpos = np.array(qpos, dtype=np.float64, copy=True)
        for item in self.couplings:
            qpos[item["dep_q"]] = self._dependent_value(item, qpos)
        return qpos

    def residual(self, qpos_seq):
        """Largest coupling violation per frame (radians or metres)."""
        qpos_seq = np.atleast_2d(np.asarray(qpos_seq, dtype=np.float64))
        out = np.zeros(len(qpos_seq), dtype=np.float64)
        for frame, qpos in enumerate(qpos_seq):
            out[frame] = max(abs(float(qpos[item["dep_q"]]) - self._dependent_value(item, qpos)) for item in self.couplings)
        return out

    def basis(self, qpos):
        """T (nv x nr): full dq from reduced dq over the independent dofs."""
        T = np.zeros((self.nv, len(self.independent_dofs)), dtype=np.float64)
        T[self.independent_dofs, np.arange(len(self.independent_dofs))] = 1.0
        slope = {}
        for item in self.couplings:
            if item["drv_dof"] is None:
                slope[item["dep_dof"]] = 0.0
                continue
            x = float(qpos[item["drv_q"]]) - item["q0_drv"]
            slope[item["dep_dof"]] = _dpoly(item["poly"], x) * slope.get(item["drv_dof"], 1.0)
            root = self.root_of[item["dep_dof"]]
            T[item["dep_dof"], self.column_of_dof[root]] = slope[item["dep_dof"]]
        return T

    def reduce_step_bounds(self, T, lower, upper):
        """Box bounds on dq_r: independent bounds intersected with dependents' bounds via their slope."""
        lower = np.asarray(lower, dtype=np.float64)
        upper = np.asarray(upper, dtype=np.float64)
        lower_r = lower[self.independent_dofs].copy()
        upper_r = upper[self.independent_dofs].copy()
        for dof in self.dependent_dofs:
            root = self.root_of[int(dof)]
            if root is None:
                continue
            col = self.column_of_dof[root]
            k = T[dof, col]
            if abs(k) < 1e-12:
                continue
            lo, hi = lower[dof] / k, upper[dof] / k
            if k < 0.0:
                lo, hi = hi, lo
            lower_r[col] = max(lower_r[col], lo)
            upper_r[col] = min(upper_r[col], hi)
        return lower_r, upper_r

    def reduce_l2_groups(self, T, groups):
        """(dof_ids, radius, label) groups as (cols, radius, label, weights) with exact norms."""
        reduced = []
        for group in groups:
            dof_ids, radius = group[0], group[1]
            label = group[2] if len(group) > 2 else "l2"
            sub = T[np.asarray(dof_ids, dtype=np.int32)]
            cols = np.flatnonzero(np.any(np.abs(sub) > 0.0, axis=0)).astype(np.int32)
            weights = np.sqrt((sub[:, cols] ** 2).sum(axis=0))
            reduced.append((cols, radius, label, weights))
        return reduced

    def effective_joint_limits(self, joint_limits_by_qpos):
        """Narrow each root driver's limits so no dependent can leave its own range (linear couplings)."""
        if not self.linear or joint_limits_by_qpos is None:
            return joint_limits_by_qpos
        limits = {key: dict(value) for key, value in joint_limits_by_qpos.items()}
        dof_to_qadr = {int(info["dof_id"]): qadr for qadr, info in limits.items()}
        affine = {}  # dep dof -> (c, K) with q_dep = c + K * q_root
        for item in self.couplings:
            a0, a1 = float(item["poly"][0]), float(item["poly"][1])
            if item["drv_dof"] is None:
                continue
            c_drv, k_drv = affine.get(item["drv_dof"], (0.0, 1.0))
            c = item["q0_dep"] + a0 + a1 * (c_drv - item["q0_drv"])
            affine[item["dep_dof"]] = (c, a1 * k_drv)
        for dof, (c, k) in affine.items():
            root = self.root_of[dof]
            if abs(k) < 1e-12 or dof not in dof_to_qadr or root not in dof_to_qadr:
                continue
            dep_info = limits[dof_to_qadr[dof]]
            root_info = limits[dof_to_qadr[root]]
            lo, hi = (dep_info["lower"] - c) / k, (dep_info["upper"] - c) / k
            if k < 0.0:
                lo, hi = hi, lo
            new_lo, new_hi = max(root_info["lower"], lo), min(root_info["upper"], hi)
            if new_lo <= new_hi:
                root_info["lower"], root_info["upper"] = float(new_lo), float(new_hi)
        return limits
