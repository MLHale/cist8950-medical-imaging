# This was made by Claude I asked Claude if it was possible to create a way to automaticall parse through different files and then 
# give me the segmentations as a wrapper for total segmentator I was really initally just messing around with it wasnt intending on this
# to become a main staple of whats happening here. Prompt: can you help me to create a way to just grab these certains
# segmentations [list of ones in question] output them to files from data that I have currently? Something along these lines
# I made some alterations but definetly a lot made by Claude. 





"""
Run TotalSegmentator on CT volumes found anywhere under Data/ and export
per-patient output folders:

  Data/{id}/liver.nii.gz
  Data/{id}/aorta.nii.gz
  Data/{id}/portal_vein.nii.gz
  Data/{id}/liver_segment_{1-8}.nii.gz

CT files are discovered recursively (any depth) and must follow the naming
convention ct{id}.nii.gz, e.g.:
  Data/ct0001.nii.gz
  Data/new_patients/ct0042.nii.gz

All three vascular structures (liver, aorta, portal vein) are extracted in a
single TotalSegmentator call using --roi_subset for efficiency.
The 8 liver segments are extracted in a second call with -ta liver_segments.
"""

from __future__ import annotations

import importlib.util
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


#I feel like this is relatively self exlplanatory but this just makes sure that total segmentator is on the PATH
# then is used to help grab total segmentator to be used to call it as an executable 
def find_totalsegmentator_command(custom_path: str | None = None) -> list[str]:
    """Return a runnable TotalSegmentator command prefix."""
    if custom_path:
        return [custom_path]

    for candidate in ["TotalSegmentator", "totalsegmentator"]:
        found = shutil.which(candidate)
        if found:
            return [found]

    if importlib.util.find_spec("totalsegmentator.bin.TotalSegmentator") is not None:
        return [sys.executable, "-m", "totalsegmentator.bin.TotalSegmentator"]

    raise FileNotFoundError(
        "Could not find TotalSegmentator CLI on PATH and module fallback was unavailable. "
        "Install it with: python -m pip install TotalSegmentator"
    )

# this is a regex function basically that will raise an error if the file name isnt formated correctly. 
# this grabs the case ID from the file name
def case_id_from_ct_filename(ct_path: Path) -> str:
    """Extract numeric case id from names like ct0001.nii.gz."""
    m = re.match(r"^ct(\d+)\.nii(?:\.gz)?$", ct_path.name, flags=re.IGNORECASE)
    if not m:
        raise ValueError(f"Unexpected CT filename format: {ct_path.name}")
    return m.group(1)


# ---------------------------------------------------------------------------
# ROI subset task: liver + aorta + portal vein in one call
# ---------------------------------------------------------------------------

# Maps the TotalSegmentator output filename to the output key and suffix used
# in this script's named outputs.
_ROI_SUBSET_STRUCTURES = {
    "liver": "liver",
    "aorta": "aorta",
    "portal_vein_and_splenic_vein": "portal_vein",
}

# this is the function that takes in the list of desired substructures and creates the command for total segmentator
# Note: If you run on CPU use the option --fast or --roi_subset to greatly improve runtime. from the total segmentator 
# github so this is runnning on CPU which is the point of adding the --roi_subset command here. 
def run_totalsegmentator_roi_subset(
    cmd_prefix: list[str],
    input_ct: Path,
    output_dir: Path,
    structures: list[str],
) -> None:
    """Run TotalSegmentator with an explicit --roi_subset list."""
    cmd = [
        *cmd_prefix,
        "-i", str(input_ct),
        "-o", str(output_dir),
        "--roi_subset", *structures,
    ]
    print("\nRunning:", " ".join(cmd))
    subprocess.run(cmd, check=True)

# this command essentially takes all the above and combines them to effectively create a dictionary whihc is used
# to dump the files into data/caseid/segments  
def process_roi_structures(
    cmd_prefix: list[str],
    ct_file: Path,
    case_id: str,
    out_dir: Path,
    overwrite: bool,
) -> dict[str, Path]:
    """
    Run the combined ROI subset (liver + aorta + portal vein) and return
    a dict mapping structure key → output Path inside out_dir.
    """
    outputs: dict[str, Path] = {
        key: out_dir / f"{key}.nii.gz"
        for key in _ROI_SUBSET_STRUCTURES.values()
    }

    if all(p.exists() for p in outputs.values()) and not overwrite:
        print(f"  Skipping ROI structures: all files exist for case {case_id}")
        return outputs

    with tempfile.TemporaryDirectory(dir=out_dir) as tmp:
        tmp_dir = Path(tmp)
        run_totalsegmentator_roi_subset(
            cmd_prefix, ct_file, tmp_dir,
            list(_ROI_SUBSET_STRUCTURES.keys()),
        )

        for ts_name, key in _ROI_SUBSET_STRUCTURES.items():
            candidates = [tmp_dir / f"{ts_name}.nii.gz", tmp_dir / f"{ts_name}.nii"]
            raw = next((p for p in candidates if p.exists()), None)
            if raw is None:
                raise FileNotFoundError(
                    f"Expected {ts_name}.nii.gz not found in {tmp_dir}. "
                    f"Contents: {sorted(tmp_dir.iterdir())}"
                )
            shutil.move(str(raw), str(outputs[key]))
            print(f"  Saved: {outputs[key].name}")

    return outputs


# ---------------------------------------------------------------------------
# 8 Couinaud liver segment masks (liver_segments subtask)
# ---------------------------------------------------------------------------

NUM_SEGMENTS = 8

#same things as above the issue is that for these segments these come from a different model 
# https://link.springer.com/chapter/10.1007/978-3-030-32692-0_32
# this is the link to the paper as a citation for this specific TS task 

def run_totalsegmentator_segments(
    cmd_prefix: list[str],
    input_ct: Path,
    output_dir: Path,
) -> None:
    """Run TotalSegmentator with the liver_segments subtask."""
    cmd = [
        *cmd_prefix,
        "-i", str(input_ct),
        "-o", str(output_dir),
        "-ta", "liver_segments",
    ]
    print("\nRunning:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def process_liver_segments(
    cmd_prefix: list[str],
    ct_file: Path,
    case_id: str,
    out_dir: Path,
    overwrite: bool,
) -> list[Path]:
    """Run liver_segments subtask and return list of 8 output paths (1-indexed)."""
    final_paths = [
        out_dir / f"liver_segment_{seg}.nii.gz"
        for seg in range(1, NUM_SEGMENTS + 1)
    ]

    if all(p.exists() for p in final_paths) and not overwrite:
        print(f"  Skipping liver segments: all 8 files exist for case {case_id}")
        return final_paths

    with tempfile.TemporaryDirectory(dir=out_dir) as tmp:
        tmp_dir = Path(tmp)
        run_totalsegmentator_segments(cmd_prefix, ct_file, tmp_dir)

        for seg in range(1, NUM_SEGMENTS + 1):
            candidates = [
                tmp_dir / f"liver_segment_{seg}.nii.gz",
                tmp_dir / f"liver_segment_{seg}.nii",
            ]
            raw = next((p for p in candidates if p.exists()), None)
            if raw is None:
                raise FileNotFoundError(
                    f"Expected liver_segment_{seg}.nii.gz not found in {tmp_dir}. "
                    f"Contents: {sorted(tmp_dir.iterdir())}"
                )
            shutil.move(str(raw), str(final_paths[seg - 1]))
            print(f"  Saved: {final_paths[seg - 1].name}")

    return final_paths


# ---------------------------------------------------------------------------
# liver_vessels subtask → liver_vessels.nii.gz + liver_tumor.nii.gz
# ---------------------------------------------------------------------------

_LIVER_VESSELS_OUTPUTS = ["liver_vessels", "liver_tumor"]


#similarly this is not a part of the main TS model 
# https://arxiv.org/abs/1902.09063
# comes from the work here essentailly same as other functions tho

def process_liver_vessels(
    cmd_prefix: list[str],
    ct_file: Path,
    case_id: str,
    out_dir: Path,
    overwrite: bool,
) -> list[Path]:
    """Run liver_vessels subtask and return output paths."""
    final_paths = [out_dir / f"{name}.nii.gz" for name in _LIVER_VESSELS_OUTPUTS]

    if all(p.exists() for p in final_paths) and not overwrite:
        print(f"  Skipping liver vessels/tumor: all files exist for case {case_id}")
        return final_paths

    with tempfile.TemporaryDirectory(dir=out_dir) as tmp:
        tmp_dir = Path(tmp)
        cmd = [*cmd_prefix, "-i", str(ct_file), "-o", str(tmp_dir), "-ta", "liver_vessels"]
        print("\nRunning:", " ".join(cmd))
        subprocess.run(cmd, check=True)

        for name, dest in zip(_LIVER_VESSELS_OUTPUTS, final_paths):
            candidates = [tmp_dir / f"{name}.nii.gz", tmp_dir / f"{name}.nii"]
            raw = next((p for p in candidates if p.exists()), None)
            if raw is None:
                raise FileNotFoundError(
                    f"Expected {name}.nii.gz not found in {tmp_dir}. "
                    f"Contents: {sorted(tmp_dir.iterdir())}"
                )
            shutil.move(str(raw), str(dest))
            print(f"  Saved: {dest.name}")

    return final_paths



# ---------------------------------------------------------------------------
# Extract CTs from TotalSegmentator dataset zip
# ---------------------------------------------------------------------------



# Had claude write this way after the fact just so that you can download the zip from the website and then 
# just run this from the jump Prompt: Write me a script that will take the zip file and parse through it to find 
# the specific desired IDs then put those into the data folder with the label of ct{ID}.nii.gz

def extract_ct_from_dataset_zip(
    zip_path: Path,
    patient_ids: list[int],
    data_dir: Path,
    overwrite: bool = False,
) -> None:
    """
    Extract ct.nii.gz for each patient_id from the TotalSegmentator dataset zip
    and save as Data/ct{id:04d}.nii.gz.

    The zip is expected to contain entries like:
        s0004/ct.nii.gz
        s0010/ct.nii.gz
        ...
    """
    data_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        all_entries = set(zf.namelist())

        for pid in patient_ids:
            entry = f"s{pid:04d}/ct.nii.gz"
            dest  = data_dir / f"ct{pid:04d}.nii.gz"

            if dest.exists() and not overwrite:
                print(f"  Skipping s{pid:04d}: {dest.name} already exists")
                continue

            if entry not in all_entries:
                print(f"  WARNING: {entry} not found in zip — skipping")
                continue

            print(f"  Extracting {entry} → {dest.name}")
            with zf.open(entry) as src, open(dest, "wb") as out:
                shutil.copyfileobj(src, out)

    print("Done extracting CTs.")


# ---------------------------------------------------------------------------
# Per-case orchestration
# ---------------------------------------------------------------------------


# this in line withthe other ones takes the dict from before to then push the files into their new destination 
# thats the short overview essentially 

def process_one_ct(
    cmd_prefix: list[str],
    ct_file: Path,
    data_dir: Path,
) -> dict:
    """Run all TotalSegmentator tasks on one CT and return output paths."""
    case_id = case_id_from_ct_filename(ct_file)

    out_dir = data_dir / case_id
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Output folder: {out_dir}")

    return {
        "case_id": case_id,
        "roi":      process_roi_structures(cmd_prefix, ct_file, case_id, out_dir, overwrite=False),
        "segments": process_liver_segments(cmd_prefix, ct_file, case_id, out_dir, overwrite=False),
        "vessels":  process_liver_vessels(cmd_prefix, ct_file, case_id, out_dir, overwrite=False),
    }



#This essentially ties all the thigns together its largely a lot of just printing just to showcase whats
# happening during the function. But it works to take all the Total segmentator tasks and implements them. 

def main(data_dir: Path) -> int:
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        return 1

    ct_files = sorted(data_dir.rglob("ct*.nii.gz")) + sorted(data_dir.rglob("ct*.nii"))
    ct_files = sorted({p.resolve() for p in ct_files})
    if not ct_files:
        print(f"No CT files found under {data_dir}. Expected files like ct0001.nii.gz")
        return 1

    try:
        cmd_prefix = find_totalsegmentator_command()
    except FileNotFoundError as e:
        print(str(e))
        return 1

    print(f"Using TotalSegmentator: {' '.join(cmd_prefix)}")
    print(f"Found {len(ct_files)} CT file(s) in: {data_dir}")

    failures = []
    outputs = []

    for ct_file in ct_files:
        print(f"\n--- Processing {ct_file.name} ---")
        try:
            result = process_one_ct(cmd_prefix, ct_file, data_dir)
            outputs.append(result)
        except Exception as e:
            failures.append((ct_file, e))
            print(f"ERROR processing {ct_file.name}: {e}")

    print("\n=== Summary ===")
    print(f"Successful: {len(outputs)}")
    for r in outputs:
        print(f"  Case {r['case_id']}:")
        for key, path in r["roi"].items():
            print(f"    {key}: {path.name}")
        for seg_path in r["segments"]:
            print(f"    {seg_path.name}")
        for p in r["vessels"]:
            print(f"    {p.name}")

    print(f"Failed: {len(failures)}")
    for ct_file, err in failures:
        print(f"  {ct_file.name}: {err}")

    return 1 if failures else 0



#this bascially just extracts the 5 ones that I've manually checked so far and from the zip, adds them to data folder
# and then runs main. 
if __name__ == "__main__":
    _DATASET_ZIP = Path(__file__).resolve().parent / "Totalsegmentator_dataset_v201.zip"
    _DATA_DIR    = Path(__file__).resolve().parent / "Data"
    _EXTRACT_IDS = [4, 10, 11, 12, 13]

    print("=== Extracting CTs from dataset zip ===")
    extract_ct_from_dataset_zip(_DATASET_ZIP, _EXTRACT_IDS, _DATA_DIR)
    print()

    sys.exit(main(_DATA_DIR))




# will be changing a lot of this stuff in the comming weeks just the structure of the pipeline and what not
