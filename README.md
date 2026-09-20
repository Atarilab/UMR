<h1 align="center">
  Unified Motion Retargeting for Humanoids with Learned Point Cloud Correspondence
</h1>

<p align="center">
  Hanyang Cao<sup>1,2,*</sup>,
  Yuetong Fang<sup>1,2,*</sup>,
  Taesoo Kwon<sup>3,*</sup>,
  Runyi Yu<sup>2,4</sup>,
  Ji Ma<sup>5</sup>,
  Jing Tan<sup>1,2</sup>,<br>
  Yangchen Zhou<sup>1</sup>,
  Baoze Du<sup>2</sup>,
  Yi Gu<sup>1</sup>,
  Yukang Gao<sup>1,2</sup>,
  Ruoli Dai<sup>2</sup>,
  Lei Han<sup>2,†</sup>,
  Renjing Xu<sup>1,†</sup>
</p>

<p align="center">
  <sup>1</sup>HKUST (Guangzhou)&nbsp;&nbsp;
  <sup>2</sup>Noitom Robotics&nbsp;&nbsp;
  <sup>3</sup>Hanyang University&nbsp;&nbsp;
  <sup>4</sup>HKUST&nbsp;&nbsp;
  <sup>5</sup>HKU
</p>

<p align="center">
  <sup>*</sup>Equal contribution&nbsp;&nbsp;&nbsp;
  <sup>†</sup>Corresponding authors
</p>

<p align="center">
  <a href="https://hanyang9.github.io/UMR/"><img src="https://img.shields.io/badge/Project-Page-2ea44f" alt="Project Page"></a>
  <a href="https://arxiv.org/abs/2609.02134"><img src="https://img.shields.io/badge/arXiv-2609.02134-b31b1b" alt="arXiv"></a>
</p>

---

<p align="center">
  <img src="teaser.png" alt="UMR teaser" width="100%">
</p>

UMR treats the moving exterior body surface as a shared interface between human
motion and humanoid robots. It has two main stages:

- **Point Cloud Correspondence Learning** learns ordered source-robot surface
  correspondence in aligned canonical poses.
- **Correspondence-Guided Retargeting** optimizes robot motion using matched
  surface positions, orientations, contacts, and kinematic constraints.

A learned correspondence is reused by motions with the same source template
and target robot.

## Supported Motion Sources

UMR samples the moving exterior surface, so any source with surface-level motion
information can be integrated through the same formulation.

| Motion source | Dataset | Adapter guide |
| --- | --- | --- |
| BONES-SEED / SOMA | [BONES-SEED](https://huggingface.co/datasets/bones-studio/seed) | [`sample_data/bones-seed/README.md`](sample_data/bones-seed/README.md) |
| GRAIL | [NVIDIA GRAIL](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Locomanipulation-GRAIL) | [`sample_data/grail/README.md`](sample_data/grail/README.md) |
| OmniContact | [Paper and dataset](https://huggingface.co/papers/2606.26201) | [`sample_data/omnicontact/README.md`](sample_data/omnicontact/README.md) |
| LAFAN1 / SMPL-X | [LAFAN1](https://github.com/ubisoft/ubisoft-laforge-animation-dataset) | [`sample_data/lafan1_smplx/README.md`](sample_data/lafan1_smplx/README.md) |
| OMOMO | [OMOMO](https://github.com/lijiaman/omomo_release) | [`sample_data/omomo/README.md`](sample_data/omomo/README.md) |
| Humanoid Character | [MimicKit](https://github.com/xbpeng/MimicKit) | [`sample_data/humanoid_character/README.md`](sample_data/humanoid_character/README.md) |
| AdaPT body+racket | [AdaPT](https://humanoidtennis.github.io/AdaPT/) | [`sample_data/adapt/README.md`](sample_data/adapt/README.md) |
| NR FBX/BVH | FBX/BVH motion | [`sample_data/nr/README.md`](sample_data/nr/README.md) |

> **OmniContact support.** An internal development version of UMR was used to produce the Unitree G1 retargeting data released by [OmniContact](https://omnicontact.github.io/). OmniContact provides the source motions as BVH, while UMR uses SMPL-X inputs. The internal BVH-to-SMPL-X converter is not included in this repository, so the current release does not directly support these BVH files.

For LAFAN1, use [`lafan_to_smplx`](https://github.com/jaraujo98/lafan_to_smplx)
to convert BVH motion to SMPL-X before retargeting. Each adapter guide documents
the expected local layout.

## Installation

```bash
conda create -n umr python=3.12 pip -y
conda activate umr
python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.4.1
python -m pip install -r requirements-umr.txt
```

### SMPL-X Body Models

SMPL-X body-model files are not distributed with this repository. Download
them from the official SMPL-X provider after accepting its terms. Both `.pkl`
and `.npz` models are supported; place at least the neutral model at:

```text
smpl/SMPLX_NEUTRAL.pkl
# or
smpl/SMPLX_NEUTRAL.npz
```

Add `SMPLX_MALE` and `SMPLX_FEMALE` in either format when a sequence requires
those genders. The NPZ path has been tested with both neutral SMPL-X motion and
female OMOMO motion.

The GRAIL example applies its bundled G1-SMPL-X template and pose-corrective
overlay to the user-provided neutral SMPL-X model at runtime; the derived baked
SMPL-X weights are not distributed.

## Quick Start

Retarget the included LAFAN1-derived SMPL-X motion:

```bash
python scripts/humanoid_retarget_pipeline.py \
  --config robot_configs/humanoid_retarget_unitree_g1_example.json
```

The default configuration uses the included LAFAN1-derived SMPL-X sequence
`sample_data/lafan1_smplx/dance1_subject2.npz`. It builds or reuses the learned
point-cloud correspondence, runs correspondence-guided retargeting, and opens
the MuJoCo viewer.

To use another robot, copy the example config in `robot_configs/` and update
its name and MJCF path. Prepare the robot T-pose in
[UMR Studio](https://hanyang9.github.io/UMR/umr_studio.html): load the robot
asset folder, select its MJCF, adjust it into a T-pose, and click **Copy T-pose
Config**. Paste the copied `tpose_qpos` into the new robot config, then run the
pipeline with that config. No manual human-robot mapping is required.

The same surface-based formulation is exposed for other motion representations
and interaction settings:

```bash
# BONES-SEED SOMA motion
python scripts/humanoid_retarget_pipeline.py \
  --config robot_configs/humanoid_retarget_unitree_g1_example.json \
  --defaults humanoid_retarget_defaults_bones_seed.json

# Humanoid Character spin-kick
python scripts/humanoid_retarget_pipeline_character.py \
  --config robot_configs/humanoid_retarget_unitree_g1_example.json

# GRAIL human-scene interaction
python scripts/humanoid_retarget_pipeline_hsi_hoi.py \
  --config robot_configs/humanoid_retarget_unitree_g1_example.json \
  --defaults humanoid_retarget_defaults_hsi_hoi_grail.json

# OmniContact human-object interaction (pre-converted SMPL-X input)
python scripts/humanoid_retarget_pipeline_hsi_hoi.py \
  --config robot_configs/humanoid_retarget_unitree_g1_example.json \
  --defaults humanoid_retarget_defaults_hsi_hoi_standard.json

# OMOMO human-object interaction
python scripts/humanoid_retarget_pipeline_hsi_hoi.py \
  --config robot_configs/humanoid_retarget_unitree_g1_example.json \
  --defaults humanoid_retarget_defaults_hsi_hoi_standard.json \
  --data sample_data/omomo \
  --seq-key sub1_plasticbox_015

# NR FBX/BVH human motion
python scripts/humanoid_retarget_pipeline_nr.py \
  --config robot_configs/humanoid_retarget_unitree_g1_example.json

# AdaPT body+racket correspondence and retargeting
python scripts/humanoid_retarget_pipeline_adapt.py
```

## Visualize a Result

Open a result with the default GLFW viewer:

```bash
python scripts/visualize_robot_retarget_result.py \
  --result output/unitree_g1_retarget/dance1_subject2_smplx_unitree_g1.npz \
  --play
```

For browser or headless batch visualization, point the Viser backend at a
result folder:

```bash
python scripts/visualize_robot_retarget_result.py \
  --result-dir output/unitree_g1_retarget \
  --viewer-backend viser \
  --viser-host 0.0.0.0 \
  --viser-port 8080 \
  --play
```

Open the clickable `Network` URL printed in the terminal.

Use **Refresh** to load new results while batch retargeting is running.

## Batch Retargeting

Run the SMPL-X/LAFAN batch pipeline with:

```bash
python scripts/humanoid_retarget_pipeline_batch.py \
  --config robot_configs/humanoid_retarget_unitree_g1_example.json \
  --batch-config humanoid_retarget_defaults_batch.json
```

BONES-SEED uses its own batch defaults:

```bash
python scripts/humanoid_retarget_pipeline_batch.py \
  --config robot_configs/humanoid_retarget_unitree_g1_example.json \
  --batch-config humanoid_retarget_defaults_batch_bones_seed.json
```

Add `--motion-folder sample_data/bones-seed/motions_proportional/bvh` for the
actor-proportional subset. BONES-SEED associates each `Axxx` motion with its
matching shape and reuses one correspondence per source-template/robot pair.

Batch defaults use **bidirectional warm start with dynamic programming** to
reduce sensitivity to occasional singularities. This mode is recommended for
large-scale retargeting.

| Option | Meaning |
| --- | --- |
| `--motion-folder PATH` | Select the input directory. |
| `--recursive` / `--pattern GLOB` | Control motion discovery. |
| `--workers N` | Set parallel retargeting jobs. |
| `--correspondence-workers N` | Set parallel correspondence preparation jobs. |
| `--retarget-gpus` | Control GPU assignment. |
| `--force-retarget` | Rebuild existing results. |

Results are saved under `output/batch_retarget/<robot-name>/`;
`batch_summary.json` records each clip status.

## Configuration

`--config` selects the target robot. Robot-specific `tpose_qpos`, joint limits,
and model paths belong in this file. `--defaults` selects source- and
task-specific settings.

| Defaults | Source |
| --- | --- |
| `humanoid_retarget_defaults.json` | SMPL/SMPL-X |
| `humanoid_retarget_defaults_bones_seed.json` | BONES-SEED / SOMA |
| `humanoid_retarget_defaults_humanoid_character.json` | Humanoid Character |
| `humanoid_retarget_defaults_hsi_hoi_grail.json` | GRAIL |
| `humanoid_retarget_defaults_hsi_hoi_standard.json` | OmniContact / OMOMO |
| `humanoid_retarget_defaults_hsi_hoi_real.json` | OmniContact / OMOMO, objects kept at real size |
| `humanoid_retarget_defaults_hsi_hoi_real_xy.json` | OmniContact / OMOMO, real-size objects on a scaled floor plan |
| `humanoid_retarget_defaults_nr.json` | NR FBX/BVH |
| `robot_configs/humanoid_retarget_defaults_adapt.json` | AdaPT SMPL-X+racket |

### Surface Objective Weights

Surface weights are defined on the motion source, not per robot:

| Source/task | Parameter file |
| --- | --- |
| SMPL/SMPL-X and SOMA | [`retarget_body_segment_surface.py`](scripts/retarget_body_segment_surface.py) |
| Humanoid Character | [`retarget_body_segment_surface_character.py`](scripts/retarget_body_segment_surface_character.py) |
| HSI/HOI and NR | [`retarget_body_segment_surface_hoi_hsi.py`](scripts/retarget_body_segment_surface_hoi_hsi.py) |
| AdaPT body+racket | [`retarget_body_segment_surface_adapt.py`](scripts/retarget_body_segment_surface_adapt.py) |

Each segment specifies `sample_slots`, `point_cost`, and `normal_cost`. Robots
sharing the same source/task use the same values; only the robot config changes.
Interaction defaults give more weight to end-effector preservation. These
settings work well for G1 and generally transfer to other robots, but may not be
optimal for every embodiment.

### Real-Size Objects (`retarget_object_size: real`)

`scaled` shrinks the whole world by `smpl_scale = robot_height / human_height`, so the object the
robot manipulates is not the object that exists. `original` keeps the mesh real but still scales
its trajectory, which decouples the two and drops objects through the floor. A third value, `real`,
keeps **both** the mesh and the recorded trajectory, and scales the human about the ground point it
is touching instead of about the world origin:

```text
X'(t) = smpl_scale * X(t) + (1 - smpl_scale) * a(t)      a(t) = contact centroid, projected to z = 0
```

Objects are never scaled and never moved, so the scene's internal geometry is exactly the source's.
Because `a_z = 0` the floor is a fixed point and every ground term keeps its meaning, and because
the offset is one translation shared by the whole body, the human-object relative motion is
preserved. The anchor is recomputed per frame, so it follows an object that is lifted and carried;
between contact episodes it interpolates, and outside them it is held.

This mode also loads **every** object in the sequence rather than the first one alphabetically,
for both the contact map and non-penetration.

```bash
python scripts/humanoid_retarget_pipeline_hsi_hoi.py \
  --config robot_configs/humanoid_retarget_unitree_g1_example.json \
  --defaults humanoid_retarget_defaults_hsi_hoi_real.json \
  --data sample_data/embody --seq-key chair_tucking
```

Measured on a 798-frame two-object sequence where a person lifts a chair, carries it 1.5 m and
tucks it under a table (`scripts/evaluate_hsi_hoi_retarget.py`, all frames, G1 with Inspire hands):

| | `scaled` | `real` |
|---|---|---|
| object size | 0.727x recorded | **1.000x** |
| object-to-object distance | 0.727x | **1.000x** |
| robot hand to the object the human touched, mean / p95 | 22.5 / 49.3 mm | **12.8 / 36.5 mm** |
| worst robot-object penetration | -61.5 mm | **-2.0 mm** |
| frames penetrating more than 5 mm | 66 | **0** |
| lowest robot vertex, min / mean | -1.6 / -0.3 mm | -1.8 / 0.0 mm |
| joint acceleration, mean / p95 | 2.19 / 9.87 rad/s^2 | **1.94 / 8.34** |
| joint sign-flip fraction | 0.137 | **0.126** |
| root acceleration, mean / p95 | 0.86 / 2.32 m/s^2 | **0.65 / 1.67** |

What it costs, and what no warp of this kind can fix: a real-size object is
`(1 - smpl_scale) * height` higher relative to the robot than it was for the human -- 18 cm on
average here -- so the arm has to make that up, and a single translation cannot put two hands 0.4 m
apart on their exact contact points at once (2.5 cm mean residual on this sequence).

### Real-Size Objects, Second Rule (`retarget_object_size: real_xy`)

`real` above keeps an object's recorded trajectory as well as its size, and warps the human about
a moving contact anchor to meet it. That is the right choice when the object trajectory is itself
the product. It has a cost: the anchor moves whenever a touched object moves, each step of the
robot falls slightly short, and the lag accumulates. On the chair sequence the centre of mass ends
up outside the robot's own support polygon on 60 % of frames, which no tracking policy can follow.

`real_xy` keeps the object's size and its **height**, and scales only the scene's floor plan:

```text
human:   X'(t) = smpl_scale * X(t)                  exactly as `scaled`, no warp at all
objects: mesh scale 1.0,  p'(t) = (s*px, s*py, pz)
```

Horizontal alignment is then exact for every object on every frame, by construction, because both
sides are multiplied by the same factor. A carried object needs no anchor to chase: the chair's
1.5 m path becomes 1.09 m and the shrunk human walks 1.09 m. Heights are untouched, so a table top
stays at 0.792 m and a chair still rests on the floor.

**Where the scaling is centred matters, and it is a knob.** A scaling is inherently
distance-dependent: an object is displaced from where it was recorded by `(1 - s)` times its
distance from the centre of the scaling, so objects do **not** all move by the same vector.
`scene_scale_anchor` chooses that centre:

| value | centre | displacement from the recorded position, this sequence |
| --- | --- | --- |
| `origin` | the capture frame's origin | chair 0.03-0.46 m, table 0.55 m |
| `scene_centroid` (default) | each object's mean position, averaged over objects | chair 0.01-0.37 m, table 0.16 m |

`scene_centroid` more than halves it, mean 0.393 m to 0.182 m, and removes the dependence on where
the capture frame happened to put its origin. The two differ by a single rigid translation applied
to the human and to every object together, so the retargeted motion and every metric below are
equivalent; only the scene's absolute placement changes. If you need absolute inter-object
distances preserved rather than merely their ratios, use `real` instead: it does not move objects
at all.

```bash
python scripts/humanoid_retarget_pipeline_hsi_hoi.py \
  --config robot_configs/humanoid_retarget_unitree_g1_example.json \
  --defaults humanoid_retarget_defaults_hsi_hoi_real_xy.json \
  --data sample_data/embody --seq-key chair_tucking
```

Measured on the same 798-frame sequence, all frames, G1 with Inspire hands:

| | `scaled` | `real` | `real_xy` |
|---|---|---|---|
| object size | 0.727x | **1.000x** | **1.000x** |
| object height above the floor | 0.727x | **recorded** | **recorded** |
| object-to-object distance | 0.727x | **recorded** | 0.727x horizontally |
| **object-object intersection not in the source** | none | **none** | **46 of 160 frames, up to 16 cm** |
| CoM outside the support polygon | **4 %** | 21 % | 5 % |
| CoM overshoot, mean | **1 mm** | 10 mm | 2 mm |
| feet to the chair during the tuck | n/a | **0.274 m** | 0.187 m (the human stood at 0.437 m) |
| hand to the object the human touched | 22.5 mm | **12.5 mm** | 16.3 mm |
| worst robot-object penetration | -61.5 mm | **-1.9 mm** | -2.0 mm |
| frames with both feet off the floor | 0 | **0** | **0** |
| joint sign-flip fraction | 0.137 | **0.117** | 0.136 |
| root acceleration, mean | 0.855 m/s^2 | **0.655** | 0.653 |

**Which to use: `real`.** It is the only rule that keeps object sizes AND introduces no contact
between objects that was not in the source, which `real_xy` cannot do (see below). It also has the
best contact and the smoothest trajectory. Its cost is balance: the centre of mass leaves the
support polygon on 21 % of frames against 4 % for the pure similarity, by 10 mm on average.

**A warning about `real_xy`, and why it is kept anyway.** Scaling object positions while keeping
mesh sizes fixed is not a similarity transform, so it does not preserve the distance between two
objects: here the chair-to-table distance shrinks from 1.199 m to 0.871 m while both meshes stay
full size, and the chair ends up passing through the table on 46 of 160 frames. Sweeping the
compression shows this scene tolerates about 4 % before the two first touch, and the robot needs
27 %. No single factor can do better. `real_xy` remains the right choice for a sparse scene, where
nothing is close enough to crowd, because its horizontal alignment is exact and its balance is
almost that of the similarity. Check `object-to-object surface clearance` in the evaluation, and
note that a sampled point-to-point clearance is NOT sufficient to detect two thin surfaces passing
through each other: the evaluation reports it, but a voxel occupancy test is what settles it.

**What `real_xy` costs.** Object-to-object distances shrink horizontally by `smpl_scale` while the
meshes stay full size, so the floor plan is a horizontal similarity rather than a rigid copy, and a
dense scene crowds. On this take the chair-to-table clearance drops from 27.7 mm to
1.3 mm and never goes negative, and the evaluation reports that margin on every run. The vertical
reach demand is the same in both rules: the contact point stays at its real height while the hand
is at `smpl_scale` times that height, so the arm makes up about 18 cm here.

### Self Collision

The HSI/HOI defaults ship `robot_self_penetration_cost: 0.0` and
`robot_self_penetration_hard_constraint: false`, so nothing stops the robot passing through itself.
Measured on the chair sequence, UMR's own `scaled` default puts an arm **112.7 mm** inside the
torso, on 190 of 266 sampled frames. Both real-size defaults now enable the hard constraint with
`robot_self_penetration_hard_slack_cost: 5000`, which takes the worst overlap to **24.7 mm** and the
overlapping frames from 217 to 70 of 266.

Two things make it usable. Some robot descriptions overlap themselves while standing still -- on
the G1 each ankle link sits 20 mm inside its own knee -- so those pairs are detected once at the
neutral pose and excluded by geom id, with no knowledge of the robot's naming. And the post-filter
re-projection pass enforces the constraint too, so the trajectory filter cannot undo it.

What remains is a shoulder against the torso at about 25 mm. The robot's shoulder is thicker than
the human's, so matching a pose where the human's arm rests against their body forces some overlap;
the surface term and the constraint settle there. `robot_self_penetration_margin` trades it against
smoothness: at -0.015 the trajectory is smoother (joint acceleration 3.3 against 4.4 rad/s^2) and
shallow overlaps are simply permitted.

### Smoothness Settings

These are independent of object scale and are enabled in
`humanoid_retarget_defaults_hsi_hoi_real.json`; copy them into any defaults file to get the same
effect. Every one of them defaults to a no-op, so existing configurations are unchanged.

| Setting | Default | What it does |
| --- | --- | --- |
| `trajectory_filter_acceleration_cost` / `_jerk_cost` | 0.1 / 0.01 | Shipped so low the LQR filter barely filters. 8 / 2 cuts joint acceleration and chatter substantially at no measured cost. |
| `trajectory_filter_root_rotation` | `false` | The filter never touched the root quaternion; this filters it too, in a sign-continuous way. |
| `root_smooth_cost` / `root_temporal_smooth_cost` | 0 / 0 | The floating base is excluded from `smooth_cost`/`temporal_smooth_cost`, which cover scalar joints only. These are its only in-solve regularisers. |
| `root_step_limit_mode` | `off` | The root has no per-frame bound by default. `box` applies one from the second frame on. |
| `trajectory_filter_reproject` | `false` | The filter runs after the constrained solve and can push a foot through the floor or a hand into an object. This re-solves a minimum-norm step against the same hard constraints, so frames that violate nothing keep the filter's result exactly. `trajectory_filter_reproject_root_weight` keeps it from lurching the body to resolve a hand contact. |
| `trajectory_filter_reproject_smooth_cost` | 0 | Each frame is repaired on its own, so the CORRECTION is high-frequency over time and applied raw it undoes the filter: measured at 6x the arm acceleration and 9x the wrist's, visible as vibration. This smooths the correction rather than the trajectory, so the repair survives and its jitter does not. At 1.0 arm acceleration p99 drops from 94.8 to 31.9 rad/s^2 and the worst wrist joint from 168.8 to 37.6, for 2 mm of object penetration. |
| `object_contact_pair_latch`, `_release_threshold_scale`, `_fade_frames` | `false`, 1.0, 0 | A contact row currently appears at half weight the instant a slot crosses the threshold, and its object-local target hops between surface samples. These latch the target for the episode and ramp the weight. |
| `ground_contact_anchor_release_threshold` | 0.0 | A foot anchor is deleted and re-latched at a new position whenever source noise moves a slot off exactly zero. This keeps it through the noise. |

## Data Preparation Notes

- For HSI/HOI, convex-decompose concave objects with
  [CoACD](https://github.com/SarahWeiii/CoACD) before retargeting. MuJoCo treats
  a single mesh collision geom as its convex hull.
- BONES-SEED motion and SOMA-X asset placement is documented in the
  [BONES-SEED guide](sample_data/bones-seed/README.md); `py-soma-x` is included
  in the requirements.
- NR FBX/BVH input requires Node.js 18 or newer. The bundled minimal Three.js
  code is used only for FBX mesh parsing, not visualization.

## Citation

If you find UMR useful in your research, please cite the paper:

```bibtex
@misc{cao2026unifiedmotionretargetinghumanoids,
  title={Unified Motion Retargeting for Humanoids with Learned Point Cloud Correspondence},
  author={Hanyang Cao and Yuetong Fang and Taesoo Kwon and Runyi Yu and Ji Ma and Jing Tan and Yangchen Zhou and Baoze Du and Yi Gu and Yukang Gao and Ruoli Dai and Lei Han and Renjing Xu},
  year={2026},
  eprint={2609.02134},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2609.02134},
}
```
