# run_pipeline_test.py
# Tristan Jones — Spring 2026 Capstone
#
# End-to-end pipeline test on a small subset of patients.
# Essentially I just asked claude if it could give me a wrapper for the whole pipeline 
# just to kind of do a quick test on the dataset. 

# prior to runnning just add the Totalsegmentator_dataset_v201.zip folder to the data folder
#

# This should take somewhere around 2ish hours to run it does the segmentations from scratch 
# but will run for 10 patients for both the male and female cohort.


# Run from Project/ root:
#   python run_pipeline_test.py
#
# What this does:
#   1. Checks which test patients already have segmentations in segmentations.zip
#   2. Runs TotalSegmentator only for any missing patients
#   3. Builds male liver atlas (10 patients)
#   4. Builds female liver atlas (10 patients)
#   5. Builds vascular distance cloud for both atlases
#   6. Saves all visualizations to outputs/pipeline_test_male/ and outputs/pipeline_test_female/
#
# After the first run, set LOAD_EXISTING = True to skip recomputation (of atlases and vasculature)
# and just reload + revisualize.

import sys
import zipfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
DATA_DIR          = Path("Data")
CACHE_DIR         = Path("outputs/reg_cache")
SEG_ZIP           = DATA_DIR / "segmentations.zip"
CT_ZIP            = DATA_DIR / "Totalsegmentator_dataset_v201.zip"
N_PATIENTS        = 10        # number of male + female patients each

MALE_ATLAS_ID     = "0004"    # reference patient for male atlas
FEMALE_ATLAS_ID   = "0012"    # ← update to a confirmed female patient ID

MALE_OUT_DIR      = Path("outputs/pipeline_test_male")
FEMALE_OUT_DIR    = Path("outputs/pipeline_test_female")

LOAD_EXISTING     = False     # set True after first run to skip recomputation
DENSITY_THRESHOLD = 0.5       # fraction of patients that must have liver at a voxel
K_NEIGHBORS       = 5         # KD-tree neighbors for distance computation
# ------------------------------------------------------------------

MALE_OUT_DIR.mkdir(parents=True, exist_ok=True)
FEMALE_OUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# Step 1 — Check which patients need segmentation
# ------------------------------------------------------------------
from Validation.dataset_loader import load_cohort

print(f"\n{'='*60}")
print(f"  Step 1 — Loading patient lists")
print(f"{'='*60}")

male_all   = load_cohort(DATA_DIR, cohort="male",   exclude_ids=[MALE_ATLAS_ID])
female_all = load_cohort(DATA_DIR, cohort="female", exclude_ids=[FEMALE_ATLAS_ID])

male_source   = male_all[:N_PATIENTS]
female_source = female_all[:N_PATIENTS]

# Always include the reference patients
all_test_ids = list({MALE_ATLAS_ID, FEMALE_ATLAS_ID} | set(male_source) | set(female_source))

print(f"  Male reference   : {MALE_ATLAS_ID}")
print(f"  Female reference : {FEMALE_ATLAS_ID}")
print(f"  Male sources     : {male_source}")
print(f"  Female sources   : {female_source}")

# ------------------------------------------------------------------
# Step 2 — Run TotalSegmentator for any missing patients
# ------------------------------------------------------------------
print(f"\n{'='*60}")
print(f"  Step 2 — Checking segmentations")
print(f"{'='*60}")

def get_segmented_ids(seg_zip: Path) -> set:
    if not seg_zip.exists():
        return set()
    with zipfile.ZipFile(seg_zip, "r") as zf:
        return {
            name.split("/")[0]
            for name in zf.namelist()
            if name.endswith("/liver.nii.gz") and "/" in name
        }

segmented = get_segmented_ids(SEG_ZIP)

# Normalize IDs for comparison (strip 's' prefix, zero-pad)
def normalize_id(pid):
    return pid.lstrip("s").zfill(4)

missing = [
    pid for pid in all_test_ids
    if normalize_id(pid) not in segmented
]

if missing:
    print(f"  Missing segmentations for: {missing}")
    print(f"  Running TotalSegmentator...")

    import csv
    import shutil
    import subprocess
    import tempfile
    import importlib.util

    def find_ts():
        for c in ["TotalSegmentator", "totalsegmentator"]:
            found = shutil.which(c)
            if found:
                return [found]
        if importlib.util.find_spec("totalsegmentator.bin.TotalSegmentator"):
            return [sys.executable, "-m", "totalsegmentator.bin.TotalSegmentator"]
        raise FileNotFoundError("TotalSegmentator not found — pip install TotalSegmentator")

    _CMD = find_ts()

    _ROI_SUBSET = {
        "liver":                        "liver",
        "portal_vein_and_splenic_vein": "portal_vein",
    }
    _VESSELS   = ["liver_vessels", "liver_tumor"]
    _N_SEGS    = 8

    def _collect(tmp_dir):
        found = {}
        for ts_name, key in _ROI_SUBSET.items():
            for s in [".nii.gz", ".nii"]:
                p = tmp_dir / f"{ts_name}{s}"
                if p.exists():
                    found[f"{key}.nii.gz"] = p
                    break
        for name in _VESSELS:
            for s in [".nii.gz", ".nii"]:
                p = tmp_dir / f"{name}{s}"
                if p.exists():
                    found[f"{name}.nii.gz"] = p
                    break
        for i in range(1, _N_SEGS + 1):
            for s in [".nii.gz", ".nii"]:
                p = tmp_dir / f"liver_segment_{i}{s}"
                if p.exists():
                    found[f"liver_segment_{i}.nii.gz"] = p
                    break
        return found

    for pid in missing:
        pid_clean = normalize_id(pid)
        ct_entry  = f"s{pid_clean}/ct.nii.gz"

        with zipfile.ZipFile(CT_ZIP, "r") as zf:
            if ct_entry not in set(zf.namelist()):
                print(f"  WARNING: {ct_entry} not in CT zip — skipping {pid}")
                continue

        print(f"\n  Segmenting patient {pid_clean}...")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            ct_path = tmp_dir / f"ct{pid_clean}.nii.gz"

            with zipfile.ZipFile(CT_ZIP, "r") as zf:
                with zf.open(ct_entry) as src, open(ct_path, "wb") as dst:
                    shutil.copyfileobj(src, dst)

            for subdir, task_args in [
                ("roi",      ["--roi_subset", *list(_ROI_SUBSET.keys())]),
                ("segments", ["-ta", "liver_segments"]),
                ("vessels",  ["-ta", "liver_vessels"]),
            ]:
                d = tmp_dir / subdir
                d.mkdir()
                cmd = [*_CMD, "-i", str(ct_path), "-o", str(d), *task_args]
                print(f"    Running: {' '.join(cmd)}")
                subprocess.run(cmd, check=True)

            all_outputs = {}
            for d in [tmp_dir / "roi", tmp_dir / "segments", tmp_dir / "vessels"]:
                all_outputs.update(_collect(d))

            mode = "a" if SEG_ZIP.exists() else "w"
            with zipfile.ZipFile(SEG_ZIP, mode, compression=zipfile.ZIP_DEFLATED) as zf:
                for zip_name, local_path in all_outputs.items():
                    zf.write(local_path, f"{pid_clean}/{zip_name}")
                    print(f"    → {pid_clean}/{zip_name}")

            ct_path.unlink()
            print(f"  Patient {pid_clean} segmented and saved.")
else:
    print(f"  All {len(all_test_ids)} test patients already segmented ✓")

# ------------------------------------------------------------------
# Step 3 — Build atlases + vascular distance clouds
# ------------------------------------------------------------------
from Atlas.liver_atlas import LiverAtlas
from Atlas.vascular_distance import VascularDistanceCloud

for label, atlas_id, source_ids, out_dir in [
    ("MALE",   MALE_ATLAS_ID,   male_source,   MALE_OUT_DIR),
    ("FEMALE", FEMALE_ATLAS_ID, female_source, FEMALE_OUT_DIR),
]:
    print(f"\n{'#'*60}")
    print(f"  {label} ATLAS — {N_PATIENTS} patients  (reference={atlas_id})")
    print(f"{'#'*60}")

    # -- Liver atlas --
    atlas = LiverAtlas(
        atlas_id  = atlas_id,
        data_dir  = DATA_DIR,
        cache_dir = CACHE_DIR,
    )

    if LOAD_EXISTING:
        atlas.load(out_dir)
    else:
        atlas.build(source_ids)
        atlas.save(out_dir)

    atlas.print_registration_summary()
    atlas.visualize_common_basis(
        point_cap   = 3000,
        output_html = str(out_dir / "01_common_basis.html"))
    atlas.visualize_average_liver(
        thresholds  = [0.5, 0.75, 0.9],
        output_html = str(out_dir / "02_average_liver.html"))
    atlas.visualize_density_slices(
        output_html = str(out_dir / "03_density_slices.html"))

    # -- Vascular distance cloud --
    vdc = VascularDistanceCloud(
        atlas_id          = atlas_id,
        data_dir          = DATA_DIR,
        cache_dir         = CACHE_DIR,
        atlas_dir         = out_dir,
        density_threshold = DENSITY_THRESHOLD,
        k_neighbors       = K_NEIGHBORS,
    )

    if LOAD_EXISTING:
        vdc.load(out_dir)
    else:
        vdc.build(source_ids)
        vdc.save(out_dir)

    vdc.visualize(output_html=str(out_dir / "vdc_combined.html"))
    vdc.visualize_all_modes(output_html=str(out_dir / "vdc_all_modes.html"))
    vdc.visualize_distance_slices(output_html=str(out_dir / "vdc_slices.html"))
    vdc.visualize_distance_histogram(output_html=str(out_dir / "vdc_histogram.html"))

    print(f"\n  ✓ {label} complete — outputs in {out_dir}")

print(f"\n{'='*60}")
print(f"  PIPELINE TEST COMPLETE")
print(f"  Male outputs   : {MALE_OUT_DIR}")
print(f"  Female outputs : {FEMALE_OUT_DIR}")
print(f"  Set LOAD_EXISTING = True to reload without recomputing.")
print(f"{'='*60}")




