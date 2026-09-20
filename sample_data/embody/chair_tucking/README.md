# `chair_tucking` as a UMR HSI/HOI sample

The ORIGINAL motion: SMPL-X body from s3 + s2 hands, floor at z = 0, nothing scaled, nothing
shifted. UMR applies its own `smpl_scale` and `retarget_object_size` when it runs.

Objects (each `<stem>.xml` + `<stem>.obj` + `prop_<stem>.csv`, one row per frame, real trajectory):

- `office_chair_0` -- s9 mesh on the s5 trajectory, s5 motion `moving`, 733/798 frames tracked (untracked frames hold the last pose)
- `table_2` -- s9 mesh on the s5 trajectory, s5 motion `fixed`, 704/798 frames tracked (untracked frames hold the last pose)

UMR's released HOI solver keeps the FIRST object it discovers (alphabetical stem) for its contact
term and warns "multiple objects found". Rename or remove the other `prop_*.csv` files to choose.

From the embody repo root:

    envs/umr/.venv/bin/python third_party/UMR/scripts/humanoid_retarget_pipeline_hsi_hoi.py \
        --config configs/umr/robot_g1_rh56e2.json --defaults configs/umr/defaults_hoi.json \
        --data /home/ilyass/workspace/embody/third_party/UMR/sample_data/embody --seq-key chair_tucking
