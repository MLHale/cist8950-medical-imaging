# Liver Atlas Capstone — Tristan Jones, Spring 2026

See other readme in project folder this is the same thing 

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


## Install Instructions

### Requirements
- Python 3.9+
- ~25GB disk space for the full dataset
- CPU hardware (no GPU required, however GPU will be faster for segmentations)
- nibabel
- numpy
- scipy
- plotly
- matplotlib
- totalsegmentator

### Installation Instructions
1. Clone the repository from GitHub
2. Run `pip install nibabel numpy totalsegmentator matplotlib scipy plotly`
3. Download `Totalsegmentator_dataset_v201.zip` from https://zenodo.org/records/10047292
4. Extract into a `Data/` folder in the project directory (rename folder)


## Run Order

Run from the project root (`Project/`) in this order.

### 1. Run TotalSegmentator on CT scans
```bash
# CPU (default — slow but works anywhere)
python -m Segmentation.run_totalsegmentator

# GPU (much faster if you have CUDA)
python -m Segmentation.run_totalsegmentator --gpu

# CPU fast mode (lower resolution, faster than default CPU)
python -m Segmentation.run_totalsegmentator --fast
```
Produces `Data/segmentations.zip`. Note: the raw dataset (`Totalsegmentator_dataset_v201.zip`)
was manually reviewed to keep only patients with a complete liver CT. TotalSegmentator
can struggle with vessel segmentation on some scans, so the validation step below is
important for filtering out bad results before atlas construction.

### 2. Validate the dataset
```bash
python -m Validation.validate_dataset --source zip --zip-path Data/segmentations.zip
# Outputs: Validation/validation_report.csv
#          Validation/usable_patient_ids.txt
```

### 3. Build the liver atlas
```bash
python -m Atlas.liver_atlas
# Outputs: outputs/atlas_male/  and  outputs/atlas_female/
#            atlas_liver_density.nii.gz
#            01_common_basis.html
#            02_average_liver.html
#            03_density_slices.html
#          outputs/reg_cache/rigid_<atlas>_<patient>.npz  (cached alignments)
```

### 4. Build the vascular distance cloud
```bash
python -m Atlas.vascular_distance
# Uses cached alignments from step 3 — no re-registration
# Outputs: outputs/atlas_male/vdc_*.html  and  outputs/atlas_female/vdc_*.html
```

### 5. Compute alignment statistics (for research paper)
```bash
python -m Validation.alignment_statistics
# Outputs: outputs/reg_cache/alignment_stats_summary.json
#          outputs/reg_cache/alignment_stats_rows.csv
```

### Equivalent direct .py commands (no -m)
```bash
python ./Atlas/vascular_distance.py
python ./research_paper/scripts/alignment_statistics.py
```

### 6. (Optional) Pairwise TPS registration
This is old/legacy and may not work lots of changes since it was completed previously 
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


## Detailed Documentation Pass

For file-by-file technical documentation, use:

- Documentation.md

This is the working checklist and deep-dive notes document for each Python file in the pipeline.
