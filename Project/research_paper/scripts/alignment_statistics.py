# AI Use Disclosure
#   Student estimate: 70% student-designed, 30% AI-assisted implementation
#   Claude assisted with: .npz cache scanning, Matplotlib figure layout, GIF assembly, compact HTML figures
#   See: "Documentation/AI Use Disclosure.md" for full details

"""
Validation/alignment_statistics.py

Compute Section-4 evaluation statistics from cached atlas artifacts:
1) Dice similarity (before/after) vs reference subject
2) Centroid-anchor displacement before alignment
3) Effective post-canonical rotation angles
4) Qualitative artifact inventory (HTML outputs) for figure references

Outputs:
- JSON summary: outputs/reg_cache/alignment_stats_summary.json
- CSV rows   : outputs/reg_cache/alignment_stats_rows.csv

Usage:
    python -m Validation.alignment_statistics
    python -m Validation.alignment_statistics --cache-dir outputs/reg_cache --data-dir Data
    python -m Validation.alignment_statistics --skip-dice
"""

from __future__ import annotations

import sys
import argparse
import csv
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import zoom as nd_zoom

# Allow direct script execution (python research_paper/scripts/alignment_statistics.py)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Atlas.registration import align_patient, forward_warp_mask
from Atlas.utils import dice
from Registration.stages.load import load_patient

log = logging.getLogger(__name__)


def _rotation_angle_deg(R: np.ndarray) -> float:
    """Return the principal rotation angle (degrees) from a 3x3 rotation matrix."""
    c = (np.trace(R) - 1.0) / 2.0
    c = float(np.clip(c, -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def _orthonormalize(R: np.ndarray) -> np.ndarray:
    """Project a near-rotation matrix onto SO(3) via SVD."""
    U, _, Vt = np.linalg.svd(R)
    Rc = U @ Vt
    if np.linalg.det(Rc) < 0:
        U[:, -1] *= -1
        Rc = U @ Vt
    return Rc


def _describe(values: np.ndarray) -> dict[str, float]:
    """Return common summary stats for a 1D numeric array."""
    if values.size == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "p90": float("nan"),
            "p95": float("nan"),
            "p99": float("nan"),
        }

    std = float(values.std(ddof=1)) if values.size > 1 else 0.0
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std": std,
        "min": float(values.min()),
        "max": float(values.max()),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
    }


def _load_usable_ids(path: Path) -> set[str]:
    """Load usable patient IDs from a txt file (one ID per line)."""
    if not path.exists():
        raise FileNotFoundError(f"Usable ID file not found: {path}")
    ids = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            pid = line.strip()
            if not pid or pid.startswith("#"):
                continue
            ids.add(pid.lstrip("s"))
    return ids


def compute_centroid_before(cache_dir: Path, usable_ids: set[str]) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """
    Compute pre-alignment centroid distance ||t_src - t_ref|| from rigid cache files.

    Returns:
        values_dedup: 1D float array after ID normalization/deduplication
        rows  : per-file rows for CSV output
    """
    dedup: dict[tuple[str, str], float] = {}
    rows: list[dict[str, Any]] = []

    files = sorted(cache_dir.glob("rigid_*.npz"))
    total = len(files)
    kept = 0
    print(f"[centroid] scanning {total} rigid cache files...")

    for i, path in enumerate(files, start=1):
        d = np.load(path)
        t_src = np.asarray(d["t_src"], dtype=np.float64)
        t_ref = np.asarray(d["t_ref"], dtype=np.float64)
        dist = float(np.linalg.norm(t_src - t_ref))

        name = path.stem  # rigid_<atlas>_<patient>
        parts = name.split("_")
        atlas_id = parts[1] if len(parts) > 1 else ""
        patient_id = parts[2] if len(parts) > 2 else ""
        patient_id_norm = patient_id.lstrip("s")

        # Strictly limit stats to IDs in usable_patient_ids.txt
        if atlas_id not in usable_ids or patient_id_norm not in usable_ids:
            continue

        # Keep first seen value per normalized (atlas, patient) pair.
        key = (atlas_id, patient_id_norm)
        if key not in dedup:
            dedup[key] = dist
            kept += 1

        rows.append(
            {
                "metric": "centroid_before_mm",
                "atlas_id": atlas_id,
                "patient_id": patient_id,
                "patient_id_normalized": patient_id_norm,
                "value": dist,
                "source_file": str(path),
            }
        )

        if i % 50 == 0 or i == total:
            print(f"[centroid] {i}/{total} files scanned, {kept} usable pairs kept")

    values_dedup = np.asarray(list(dedup.values()), dtype=np.float64)
    print(f"[centroid] done: n={len(values_dedup)} usable pairs")
    return values_dedup, rows


def compute_effective_rotation(cache_dir: Path, usable_ids: set[str]) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """
    Compute effective post-canonical rotation angle for each patient.

    For each extents_<atlas>.json:
    - compute canonical direction = orthonormalized mean(direction matrices)
    - for each patient record, compute R = canonical @ src_dir.T
    - orthonormalize R and convert to principal angle (degrees)

    Returns:
        values: 1D float array of effective rotation angles (degrees)
        rows  : per-patient rows for CSV output
    """
    values: list[float] = []
    rows: list[dict[str, Any]] = []

    for ext_path in sorted(cache_dir.glob("extents_*.json")):
        with open(ext_path, "r", encoding="utf-8") as f:
            ext = json.load(f)

        atlas_id = ext_path.stem.split("_")[1]
        if atlas_id not in usable_ids:
            print(f"[rotation] skip atlas {atlas_id} (not in usable IDs)")
            continue
        print(f"[rotation] atlas {atlas_id}: loading {ext_path.name}")

        dirs = []
        for rec in ext.values():
            d = np.asarray(rec.get("direction", []), dtype=np.float64)
            if d.shape == (3, 3):
                dirs.append(d)
        if not dirs:
            continue

        canonical = _orthonormalize(np.mean(np.stack(dirs, axis=0), axis=0))

        per_atlas = 0
        for pid, rec in ext.items():
            pid_clean = pid.lstrip("s")
            if pid_clean == atlas_id:
                continue
            if pid_clean not in usable_ids:
                continue

            src_dir = np.asarray(rec.get("direction", []), dtype=np.float64)
            if src_dir.shape != (3, 3):
                continue

            R_affine = canonical @ src_dir.T
            R_clean = _orthonormalize(R_affine)
            ang = _rotation_angle_deg(R_clean)

            values.append(ang)
            per_atlas += 1
            rows.append(
                {
                    "metric": "rotation_effective_deg",
                    "atlas_id": atlas_id,
                    "patient_id": pid,
                    "patient_id_normalized": pid.lstrip("s"),
                    "value": ang,
                    "source_file": str(ext_path),
                }
            )

        print(f"[rotation] atlas {atlas_id}: kept {per_atlas} usable patients")

    return np.asarray(values, dtype=np.float64), rows


def _load_extents(cache_dir: Path, atlas_id: str) -> dict[str, Any]:
    """Load extents_<atlas_id>.json if present, else empty dict."""
    p = cache_dir / f"extents_{atlas_id}.json"
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _canonical_and_grid(all_extents: dict[str, Any]) -> tuple[np.ndarray, tuple[int, int, int], np.ndarray, float]:
    """Recompute canonical direction, global grid shape, global offset, and median volume."""
    all_mins = np.array([e["min"] for e in all_extents.values()], dtype=np.int64)
    all_maxs = np.array([e["max"] for e in all_extents.values()], dtype=np.int64)
    global_min = all_mins.min(axis=0)
    global_max = all_maxs.max(axis=0)

    global_offset = global_min.clip(max=0)
    global_shape = tuple(int(global_max[i] - global_offset[i]) + 1 for i in range(3))

    all_voxel_counts = [e["voxel_count"] for e in all_extents.values() if "voxel_count" in e]
    median_volume = float(np.median(all_voxel_counts))

    all_directions = np.array([e["direction"] for e in all_extents.values() if "direction" in e], dtype=np.float64)
    mean_dir = all_directions.mean(axis=0)
    canonical = _orthonormalize(mean_dir)

    return canonical, global_shape, global_offset, median_volume


def compute_dice_statistics(cache_dir: Path, data_dir: Path, usable_ids: set[str]) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """
    Compute Dice before/after rigid alignment for all patients in extents files.

    Returns:
        before_vals: 1D float array
        after_vals: 1D float array
        rows: per-patient CSV rows
    """
    before_vals: list[float] = []
    after_vals: list[float] = []
    rows: list[dict[str, Any]] = []

    ext_paths = sorted(cache_dir.glob("extents_*.json"))
    t0 = time.time()
    for ext_i, ext_path in enumerate(ext_paths, start=1):
        atlas_id = ext_path.stem.split("_")[1]
        if atlas_id not in usable_ids:
            print(f"[dice] [{ext_i}/{len(ext_paths)}] skip atlas {atlas_id} (not in usable IDs)")
            continue

        all_extents = _load_extents(cache_dir, atlas_id)
        if not all_extents:
            continue

        usable_patients = [pid for pid in all_extents.keys() if pid.lstrip("s") in usable_ids and pid.lstrip("s") != atlas_id]
        print(
            f"[dice] [{ext_i}/{len(ext_paths)}] atlas {atlas_id}: "
            f"{len(usable_patients)} usable patients"
        )

        canonical_direction, global_shape, global_offset, median_volume = _canonical_and_grid(all_extents)

        ref_data = load_patient(data_dir, atlas_id)
        ref_liver = ref_data["liver"]

        ref_shifted = np.zeros(global_shape, dtype=ref_liver.dtype)
        ox, oy, oz = [-int(global_offset[i]) for i in range(3)]
        rx, ry, rz = ref_liver.shape
        ref_shifted[ox:ox + rx, oy:oy + ry, oz:oz + rz] = ref_liver

        for p_i, pid in enumerate(usable_patients, start=1):
            pid_clean = pid.lstrip("s")

            try:
                if p_i % 20 == 1 or p_i == len(usable_patients):
                    elapsed = time.time() - t0
                    print(
                        f"[dice] atlas {atlas_id} patient {p_i}/{len(usable_patients)} "
                        f"(overall elapsed {elapsed/60:.1f} min)"
                    )

                src_data = load_patient(data_dir, pid_clean)
                src_liver = src_data["liver"]

                zf = tuple(r / s for r, s in zip(ref_liver.shape, src_liver.shape))
                src_rs = (nd_zoom(src_liver.astype(np.float32), zf, order=0) > 0.5).astype(src_liver.dtype)
                d_before = float(dice(ref_liver, src_rs, label=1))

                patient_voxels = all_extents.get(pid, {}).get("voxel_count", None)
                alignment = align_patient(
                    patient_id=pid_clean,
                    atlas_id=atlas_id,
                    data_dir=data_dir,
                    cache_dir=cache_dir,
                    median_volume=median_volume,
                    patient_volume=patient_voxels,
                    canonical_direction=canonical_direction,
                )
                if alignment is None:
                    continue

                xfm_liver = forward_warp_mask(
                    src_liver,
                    alignment,
                    global_shape,
                    global_offset=global_offset,
                )
                d_after = float(dice(ref_shifted, xfm_liver, label=1))

                before_vals.append(d_before)
                after_vals.append(d_after)

                rows.append(
                    {
                        "metric": "dice_before",
                        "atlas_id": atlas_id,
                        "patient_id": pid,
                        "patient_id_normalized": pid_clean,
                        "value": d_before,
                        "source_file": str(ext_path),
                    }
                )
                rows.append(
                    {
                        "metric": "dice_after",
                        "atlas_id": atlas_id,
                        "patient_id": pid,
                        "patient_id_normalized": pid_clean,
                        "value": d_after,
                        "source_file": str(ext_path),
                    }
                )

            except Exception as exc:
                log.warning(f"Dice failed for atlas={atlas_id} patient={pid_clean}: {exc}")

        print(f"[dice] atlas {atlas_id}: completed")

    return np.asarray(before_vals, dtype=np.float64), np.asarray(after_vals, dtype=np.float64), rows


def collect_qualitative_artifacts(project_root: Path) -> dict[str, list[str]]:
    """Collect existing qualitative visualization outputs for Section 4.4 references."""
    outputs = project_root / "outputs"
    artifacts = {
        "atlas_html": [str(p) for p in sorted(outputs.glob("atlas_*/**/*.html"))],
        "vessel_comparison_html": [str(p) for p in sorted(outputs.glob("vessel_comparison_*.html"))],
    }
    return artifacts


def write_outputs(cache_dir: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    """Write summary JSON and per-sample CSV rows."""
    summary_path = cache_dir / "alignment_stats_summary.json"
    rows_path = cache_dir / "alignment_stats_rows.csv"

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(rows_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "metric",
                "atlas_id",
                "patient_id",
                "patient_id_normalized",
                "value",
                "source_file",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved summary JSON: {summary_path}")
    print(f"Saved rows CSV   : {rows_path}")


def main() -> None:
    """Parse CLI options, compute Section-4 metrics, and write JSON/CSV outputs."""
    parser = argparse.ArgumentParser(description="Compute alignment statistics from cached artifacts.")
    parser.add_argument("--cache-dir", default="outputs/reg_cache", help="Directory with rigid_*.npz and extents_*.json files")
    parser.add_argument("--data-dir", default="Data", help="Directory containing patient segmentation data")
    parser.add_argument("--usable-ids", default="Validation/usable_patient_ids.txt", help="Path to usable patient IDs file")
    parser.add_argument("--skip-dice", action="store_true", help="Skip Dice computation (faster)")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    data_dir = Path(args.data_dir)
    usable_ids_path = Path(args.usable_ids)
    if not cache_dir.exists():
        raise FileNotFoundError(f"Cache directory not found: {cache_dir}")

    usable_ids = _load_usable_ids(usable_ids_path)
    print(f"Loaded {len(usable_ids)} usable patient IDs from {usable_ids_path}")

    centroid_before, rows_before = compute_centroid_before(cache_dir, usable_ids)
    rotation_eff, rows_rot = compute_effective_rotation(cache_dir, usable_ids)
    rows_dice: list[dict[str, Any]] = []
    dice_before = np.array([], dtype=np.float64)
    dice_after = np.array([], dtype=np.float64)
    if not args.skip_dice:
        dice_before, dice_after, rows_dice = compute_dice_statistics(cache_dir, data_dir, usable_ids)

    project_root = Path(__file__).resolve().parents[1]
    qual_artifacts = collect_qualitative_artifacts(project_root)

    dice_summary = {
        "before": _describe(dice_before),
        "after": _describe(dice_after),
    }
    if dice_before.size > 0 and dice_after.size > 0 and dice_before.size == dice_after.size:
        dice_summary["improvement"] = _describe(dice_after - dice_before)

    summary = {
        "cache_dir": str(cache_dir),
        "data_dir": str(data_dir),
        "usable_ids_file": str(usable_ids_path),
        "usable_ids_count": len(usable_ids),
        "dice_similarity": dice_summary,
        "centroid_before_mm": _describe(centroid_before),
        "centroid_after_mm": {
            "note": "For this rigid transform form, centroid-anchor distance after alignment is 0 mm by construction.",
            "value": 0.0,
        },
        "rotation_effective_deg": _describe(rotation_eff),
        "qualitative_artifacts": qual_artifacts,
        "section4_report": {
            "4_1_dice": "Use dice_similarity.before/after and improvement for quantitative overlap summary.",
            "4_2_centroid": "Use centroid_before_mm and centroid_after_mm for translation spread summary.",
            "4_3_rotation": "Use rotation_effective_deg for canonical orientation verification.",
            "4_4_qualitative": "Qualitative validation should cite artifact files listed in qualitative_artifacts.",
        },
    }

    write_outputs(cache_dir, summary, rows_before + rows_rot + rows_dice)

    print("\nSection-4 ready summaries")
    if not args.skip_dice and dice_before.size > 0:
        db = summary["dice_similarity"]["before"]
        da = summary["dice_similarity"]["after"]
        print(
            f"Dice before: n={db['n']}, mean={db['mean']:.3f}, median={db['median']:.3f}, "
            f"range=[{db['min']:.3f}, {db['max']:.3f}]"
        )
        print(
            f"Dice after : n={da['n']}, mean={da['mean']:.3f}, median={da['median']:.3f}, "
            f"range=[{da['min']:.3f}, {da['max']:.3f}]"
        )
    cb = summary["centroid_before_mm"]
    re = summary["rotation_effective_deg"]
    print(
        f"Centroid before (mm): n={cb['n']}, mean={cb['mean']:.3f}, median={cb['median']:.3f}, "
        f"std={cb['std']:.3f}, range=[{cb['min']:.3f}, {cb['max']:.3f}]"
    )
    print(
        f"Effective rotation (deg): n={re['n']}, mean={re['mean']:.3f}, median={re['median']:.3f}, "
        f"std={re['std']:.3f}, range=[{re['min']:.3f}, {re['max']:.3f}], p95={re['p95']:.3f}"
    )


if __name__ == "__main__":
    main()
