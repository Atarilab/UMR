"""Per-finger surface segments for dedicated hand correspondences.

Installed on top of the loaded body-segment module after it is configured:
``leftHand``/``rightHand`` then mean the palm, and each finger becomes its own
segment with its own sample count and costs, so a clip's selected solver slots
cover every finger instead of a random 15 per hand.
"""
from __future__ import annotations

from hand_correspondence import FINGER_PARTS, HAND_PARTS

FINGER_PART_ID_START = 21
DEFAULT_FINGER_COST = {"sample_slots": 16, "point_cost": 10.0, "normal_cost": 1.0}
DEFAULT_PALM_COST = {"sample_slots": 24, "point_cost": 10.0, "normal_cost": 1.0}


def segment_name(side: str, part: str) -> str:
    return f"{side}Hand" if part == "palm" else f"{side}{part.capitalize()}"


def hand_part_ids(module) -> dict[tuple[str, str], int]:
    """(side, part) -> body-segment part id, after ``install``."""
    return {(side, part): int(module.SMPLX_PART_IDS[segment_name(side, part)]) for side in ("left", "right") for part in HAND_PARTS}


def install(module, hands_cfg) -> dict[tuple[str, str], int]:
    costs = dict(hands_cfg.get("segment_costs") or {})
    part_id = FINGER_PART_ID_START
    for side in ("left", "right"):
        for part in FINGER_PARTS:
            name = segment_name(side, part)
            if name not in module.SMPLX_PART_IDS:
                module.SMPLX_PART_IDS[name] = part_id
                module.BODY_SEGMENT_PART_NAMES[part_id] = name
            part_id += 1
        for part in HAND_PARTS:
            name = segment_name(side, part)
            default = DEFAULT_PALM_COST if part == "palm" else DEFAULT_FINGER_COST
            module.BODY_SEGMENT_SURFACE_COST_CONFIG[name] = {**default, **costs.get(name, {})}
        # Fingers, palm and forearm of one hand are "adjacent": their proximity is
        # anatomy, not contact, so they must not fill the self-contact budget.
        chain = [f"{side}ForeArm"] + [segment_name(side, part) for part in HAND_PARTS]
        for i, a in enumerate(chain):
            for b in chain[i + 1:]:
                module.SMPLX_ADJACENT_SEGMENT_PAIRS.add((a, b))
    return hand_part_ids(module)
