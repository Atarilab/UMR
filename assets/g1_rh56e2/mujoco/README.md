# MuJoCo Assets

This directory contains the Unitree G1 29-DoF body, the standalone RH56E2
hand assets, and the combined G1 + RH56E2 assembly. The hand XMLs can also be
used by `scripts/xml2usda.py` to regenerate the standalone Isaac Sim hand
assets.

- `g1_29dof.xml`: full G1 29-DoF body with the stock rubber hands.
- `g1_29dof_rh56e2.xml`: full G1 29-DoF body with RH56E2 hands attached at
  the wrists (via `<attach>`, prefixed `l_rh_`/`r_rh_`).
- `rh56e2_right.xml`: standalone right-hand MuJoCo model.
- `rh56e2_left.xml`: standalone left-hand MuJoCo model.
- `rh56e2_gains.json`: per-finger position-servo PD gains (`kp`, `kd`,
  joint `damping`, `vmax`) for the RH56E2 hands.
- `meshes/g1/`: visual and collision meshes for the G1 body.
- `meshes/inspire_rh56e2/`: visual and collision meshes referenced by the
  hand XMLs.

## Viewing

`scripts/view_g1_mujoco.py` and `scripts/view_hands_mujoco.py` load these
models on a simple floor scene with position-servo actuators (rather than
the raw torque motors defined in the XMLs), so the G1 stands still and the
hand fingers hold their commanded pose instead of going limp under gravity.
