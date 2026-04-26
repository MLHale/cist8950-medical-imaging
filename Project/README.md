# Liver Atlas Capstone — Tristan Jones, Spring 2026

## Project Structure


```
capstone/
  Registration/              # CT-to-CT registration pipeline (refactored)
    Config.py                # All tunable parameters
    Run.py                   # CLI entry point: python -m Registration.Run --ref 0004 --src 0010
    stages/
      load.py                # Stage 1: load NIfTI segmentations
      landmarks.py           # Stage 2: extract anatomical landmark clusters
      align.py               # Stage 3: Procrustes rigid alignment in mm space
      tps.py                 # Stage 4: Thin Plate Spline (kept, not used by Atlas)
      evaluate.py            # Stage 5: apply transform + Dice
    utils/
      Checkpoint.py          # Save/load intermediate stage results
      Nifti.py               # Shared NIfTI helpers (voxels_to_mm, mm_to_voxels, etc.)

  Atlas/                     # Probabilistic liver atlas
    utils.py                 # Shared geometry helpers + Plotly building blocks
    registration.py          # Rigid alignment wrapper + cache (no TPS)
    liver_atlas.py           # Average liver + common basis diagnostic
    vascular_distance.py     # Distance-to-vasculature electron cloud

  Segmentation/
    run_totalsegmentator.py  # Wrapper to run TotalSegmentator on CT volumes

  Validation/
    validate_dataset.py      # Scan dataset zip for bad/missing segmentations
    visualize_registration.py # Before/after 3D mesh viewer

  Data/                      # Patient folders: Data/{id}/liver.nii.gz etc.
  outputs/
    atlas/                   # Atlas density volumes + visualization HTML
    reg_cache/               # Cached rigid alignment .npz files (one per patient)
```

## Run Order

Just ignore this for right now, I created a quick full pipeline wrapper that does a small test set just for "quick" illustraton sakes it should still take some time to do the segmentations its about 3-5 mins per so probably close to 2 hours or so for 20 people 




### 1. Extract CTs and run TotalSegmentator
```bash
python Segmentation/run_totalsegmentator.py
```

### 2. Validate the dataset
```bash
python Validation/validate_dataset.py --source zip --zip-path Data/Totalsegmentator_dataset_v201.zip
# Outputs: validation_report.csv, usable_patient_ids.txt
```

### 3. Build the liver atlas (rigid alignment only)
```bash
python -m Atlas.liver_atlas
# Outputs: outputs/atlas/atlas_liver_density.nii.gz
#          outputs/atlas/01_common_basis.html  
#          outputs/atlas/02_average_liver.html
#          outputs/atlas/03_density_slices.html
#          outputs/reg_cache/rigid_0004_*.npz  ← cached alignments
```

### 4. Build the vascular distance cloud
```bash
python -m Atlas.vascular_distance
# Uses cached alignments from step 3 — no re-registration
# Outputs: outputs/atlas/vdc_*.html
```

### 5. (Optional) Pairwise TPS registration for evaluation
```bash
python -m Registration.Run --ref 0004 --src 0010 --skip-if-done
```

## Key Design Decisions

**Rigid-only for the atlas** — distances are computed in each patient's native
space then rigidly warped. This avoids introducing TPS registration error into
the distance values. TPS is kept in Registration/stages/tps.py for evaluation
and comparison but is not called by the atlas pipeline.

**Shared cache** — `outputs/reg_cache/rigid_0004_{id}.npz` stores the Procrustes
alignment for each patient. Both `liver_atlas.py` and `vascular_distance.py` read
from the same cache so registration runs only once per patient total.

**LOAD_EXISTING flag** — set to True in each Atlas script after the first run
to reload saved outputs without re-running anything.
