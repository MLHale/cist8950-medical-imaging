# AI Use Disclosure
#   Student estimate: 70% student-designed, 30% AI-assisted implementation
#   Claude assisted with: .npz cache scanning, Matplotlib figure layout, GIF assembly, compact HTML figures
#   See: "Documentation/AI Use Disclosure.md" for full details

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Atlas.utils import density_to_mesh, make_slider_layout


CSV_PATH = PROJECT_ROOT / "outputs" / "reg_cache" / "alignment_stats_rows.csv"
FIGURE_OUT_DIR = PROJECT_ROOT / "research_paper" / "outputs" / "figures"


def compact_stage2_average_liver(
    atlas_dir: Path,
    cohort_label: str,
    out_name: str = "02_average_liver_compact.html",
) -> Path | None:
    """Render compact nested liver-density isosurfaces and save an HTML figure."""
    img = nib.load(str(atlas_dir / "atlas_liver_density.nii.gz"))
    density = np.asarray(img.get_fdata(), dtype=np.float32)
    smooth = gaussian_filter(density, sigma=1.5)

    shell_styles = {
        0.25: ("#a8e6a3", 0.12),
        0.50: ("#3cb44b", 0.30),
        0.75: ("#1a5e26", 0.60),
    }

    traces = []
    for level in [0.25, 0.50, 0.75]:
        color, opacity = shell_styles[level]
        mesh = density_to_mesh(
            smooth,
            level,
            color,
            opacity,
            f"Liver in >= {int(level * 100)}% of subjects",
            img.affine,
        )
        if mesh:
            traces.append(mesh)

    if not traces:
        print(f"[skip] no meshes for {cohort_label} stage 2")
        return None

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(
            text=f"Stage 2 - Average Liver - {cohort_label}<br><sup>Nested shells show cross-subject consistency</sup>",
            x=0.5,
            y=0.96,
            xanchor="center",
            yanchor="top",
        ),
        scene=dict(
            xaxis_title="x (mm)",
            yaxis_title="y (mm)",
            zaxis_title="z (mm)",
            aspectmode="data",
            domain=dict(x=[0.0, 0.80], y=[0.0, 1.0]),
        ),
        legend=dict(
            x=0.81,
            y=0.94,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.75)",
            bordercolor="rgba(0,0,0,0.15)",
            borderwidth=1,
        ),
        margin=dict(l=18, r=18, t=66, b=18),
        width=980,
        height=680,
    )

    out_path = atlas_dir / out_name
    fig.write_html(str(out_path))
    print(f"Saved {out_path}")
    return out_path


def compact_stage3_density_slices(
    atlas_dir: Path,
    cohort_label: str,
    out_name: str = "03_density_slices_compact.html",
) -> Path:
    """Create a compact axial slice browser from atlas liver-density volume."""
    img = nib.load(str(atlas_dir / "atlas_liver_density.nii.gz"))
    vol_full = np.asarray(img.get_fdata(), dtype=np.float32)

    present = vol_full > 0
    if np.any(present):
        vox_idx = np.argwhere(present)
        x_min, y_min, z_min = vox_idx.min(axis=0)
        x_max, y_max, z_max = vox_idx.max(axis=0)
        vol = vol_full[x_min : x_max + 1, y_min : y_max + 1, z_min : z_max + 1]
    else:
        z_min = 0
        vol = vol_full

    n_z = vol.shape[2]
    mid = n_z // 2

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=vol[:, :, mid].T,
            colorscale="Greens",
            zmin=0,
            zmax=1,
            colorbar=dict(title="Probability"),
        )
    )

    fig.frames = [
        go.Frame(
            data=[go.Heatmap(z=vol[:, :, z].T, colorscale="Greens", zmin=0, zmax=1)],
            name=str(z),
        )
        for z in range(n_z)
    ]

    slider = make_slider_layout(n_z, mid, prefix="Axial slice z=")
    fig.update_layout(
        title=dict(
            text=f"Stage 3 - Liver Density Slices - {cohort_label}<br><sup>Cropped to liver extent; global z start={z_min}</sup>",
            x=0.5,
        ),
        xaxis_title="x (voxels)",
        yaxis_title="y (voxels)",
        width=980,
        height=500,
        margin=dict(l=42, r=42, t=66, b=62),
        **slider,
    )

    out_path = atlas_dir / out_name
    fig.write_html(str(out_path))
    print(f"Saved {out_path}")
    return out_path


def compact_vdc_combined(
    atlas_dir: Path,
    cohort_label: str,
    out_name: str = "vdc_combined_compact.html",
    point_cap: int = 50000,
) -> Path:
    """Render a compact 3D VDC scatter plot colored by combined vessel distance."""
    pts = np.load(atlas_dir / "vdc_surface_pts_mm.npy")
    dists = np.load(atlas_dir / "vdc_dist_combined.npy")

    valid = ~np.isnan(dists)
    pts = pts[valid]
    dists = dists[valid]

    if len(pts) > point_cap:
        idx = np.random.choice(len(pts), point_cap, replace=False)
        pts = pts[idx]
        dists = dists[idx]

    fig = go.Figure(
        data=go.Scatter3d(
            x=pts[:, 0],
            y=pts[:, 1],
            z=pts[:, 2],
            mode="markers",
            marker=dict(
                size=2,
                color=dists,
                colorscale="Jet",
                reversescale=True,
                opacity=0.75,
                colorbar=dict(title="Mean dist to vessel (mm)"),
                showscale=True,
            ),
            text=[f"{d:.1f} mm" for d in dists],
            hovertemplate="x=%{x:.1f} y=%{y:.1f} z=%{z:.1f}<br>Mean dist: %{text}<extra></extra>",
            name="Liver volume",
        )
    )

    fig.update_layout(
        title=dict(
            text=f"Distance-to-Vasculature - {cohort_label}<br><sup>combined (k=5), red=close blue=far</sup>",
            x=0.5,
        ),
        scene=dict(
            xaxis_title="x (mm)",
            yaxis_title="y (mm)",
            zaxis_title="z (mm)",
            aspectmode="data",
        ),
        margin=dict(l=18, r=18, t=66, b=18),
        width=980,
        height=680,
    )

    out_path = atlas_dir / out_name
    fig.write_html(str(out_path))
    print(f"Saved {out_path}")
    return out_path


def compact_vdc_histogram(
    atlas_dir: Path,
    cohort_label: str,
    out_name: str = "vdc_histogram_compact.html",
) -> Path | None:
    """Plot the combined VDC distribution with summary lines and save HTML output."""
    dists = np.load(atlas_dir / "vdc_full_dist_combined.npy")
    valid = dists[~np.isnan(dists)]

    if valid.size == 0:
        print(f"[skip] no valid distance values for {cohort_label} histogram")
        return None

    mean_v = float(np.mean(valid))
    med_v = float(np.median(valid))
    min_v = float(np.min(valid))
    max_v = float(np.max(valid))

    print(
        f"Histogram stats [{cohort_label}] "
        f"n={valid.size} mean={mean_v:.3f}mm median={med_v:.3f}mm "
        f"min={min_v:.3f}mm max={max_v:.3f}mm"
    )

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=valid,
            nbinsx=50,
            marker_color="#4363d8",
            opacity=0.82,
            name="Combined",
        )
    )
    fig.add_vline(
        x=mean_v,
        line_dash="dash",
        line_color="red",
        annotation_text=f"mean {mean_v:.1f} mm",
        annotation_position="top left",
        annotation=dict(yshift=10),
    )
    fig.add_vline(
        x=med_v,
        line_dash="dot",
        line_color="orange",
        annotation_text=f"median {med_v:.1f} mm",
        annotation_position="top right",
        annotation=dict(yshift=-8),
    )

    fig.update_layout(
        title=dict(
            text=f"Distance-to-Vasculature Distribution - {cohort_label}",
            x=0.5,
        ),
        xaxis_title="Mean distance to nearest vessel (mm)",
        yaxis_title="Number of liver voxels",
        bargap=0.05,
        margin=dict(l=48, r=22, t=60, b=50),
        width=860,
        height=430,
    )

    out_path = atlas_dir / out_name
    fig.write_html(str(out_path))
    print(f"Saved {out_path}")
    return out_path


def _load_metric(metric: str, dedupe_by_patient: bool = False) -> np.ndarray:
    """Load metric values from the alignment CSV, optionally deduplicated by patient ID."""
    vals: list[float] = []
    seen: set[str] = set()

    with CSV_PATH.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("metric") != metric:
                continue
            pid = (row.get("patient_id_normalized") or "").strip()
            if dedupe_by_patient and pid:
                if pid in seen:
                    continue
                seen.add(pid)
            try:
                vals.append(float(row["value"]))
            except (TypeError, ValueError, KeyError):
                continue

    return np.asarray(vals, dtype=float)


def _style_metrics() -> None:
    """Apply consistent Matplotlib styling for publication metrics figures."""
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "font.size": 11,
            "axes.titlesize": 15,
            "axes.labelsize": 14,
            "axes.facecolor": "#fbfbfd",
            "figure.facecolor": "white",
            "axes.grid": True,
            "grid.color": "#d9d9df",
            "grid.alpha": 0.45,
            "grid.linestyle": "-",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def make_dice_boxplot() -> Path:
    """Generate a before-vs-after Dice boxplot and save it as a compact PNG."""
    dice_before = _load_metric("dice_before", dedupe_by_patient=True)
    dice_after = _load_metric("dice_after", dedupe_by_patient=True)

    fig, ax = plt.subplots(figsize=(6.3, 4.2))
    bp = ax.boxplot(
        [dice_before, dice_after],
        tick_labels=["Before", "After"],
        patch_artist=True,
        widths=0.5,
        medianprops={"color": "#111111", "linewidth": 2.0},
        boxprops={"linewidth": 1.2},
        whiskerprops={"linewidth": 1.2},
        capprops={"linewidth": 1.2},
    )

    for patch, color in zip(bp["boxes"], ["#e49d44", "#4f8fbd"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)

    ax.set_title(
        "Liver Dice Similarity\n"
        "Combined analytic cohort (154 male + 107 female)"
    )
    ax.set_ylabel("Dice score")
    ax.set_ylim(0, 1)

    ax.text(
        1,
        float(np.median(dice_before)) + 0.03,
        f"median={np.median(dice_before):.3f}",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#5a3b16",
    )
    ax.text(
        2,
        float(np.median(dice_after)) + 0.03,
        f"median={np.median(dice_after):.3f}",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#18354a",
    )

    fig.tight_layout()
    out = FIGURE_OUT_DIR / "fig_metrics_dice_boxplot_compact.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def make_alignment_histograms() -> Path:
    """Generate centroid and rotation histograms for the alignment statistics section."""
    centroid = _load_metric("centroid_before_mm", dedupe_by_patient=True)
    rotation = _load_metric("rotation_effective_deg", dedupe_by_patient=True)

    c99 = float(np.percentile(centroid, 99))
    tail_count = int(np.sum(centroid > c99))

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.1))

    axes[0].hist(centroid[centroid <= c99], bins=24, color="#eb8b2d", edgecolor="white")
    axes[0].set_title("Centroid Displacement (Pre-alignment)")
    axes[0].set_xlabel("Distance (mm)")
    axes[0].set_ylabel("Patients")
    axes[0].text(
        0.98,
        0.95,
        f"x-axis clipped at p99={c99:.0f} mm\n{tail_count} outlier(s) > p99",
        transform=axes[0].transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "#dddddd"},
    )

    axes[1].hist(rotation, bins=24, color="#4a78a9", edgecolor="white")
    axes[1].set_title("Effective Rotation Angles")
    axes[1].set_xlabel("Angle (degrees)")
    axes[1].set_ylabel("Patients")

    fig.tight_layout()
    out = FIGURE_OUT_DIR / "fig_metrics_alignment_hist_compact.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def generate_metrics_figures() -> list[Path]:
    """Create all compact metrics figures derived from alignment_stats_rows.csv."""
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Missing metrics CSV: {CSV_PATH}")
    FIGURE_OUT_DIR.mkdir(parents=True, exist_ok=True)
    _style_metrics()

    out1 = make_dice_boxplot()
    out2 = make_alignment_histograms()
    return [out1, out2]


def generate_compact_atlas_and_vdc_figures(args: argparse.Namespace) -> list[Path]:
    """Build compact atlas and VDC HTML figures for the selected cohort(s)."""
    outputs: list[Path] = []
    cohorts: list[tuple[Path, str, str]] = []
    if args.cohort in ("male", "both"):
        cohorts.append((args.male_dir, args.male_label, "male"))
    if args.cohort in ("female", "both"):
        cohorts.append((args.female_dir, args.female_label, "female"))

    for atlas_dir, label, tag in cohorts:
        stage2_out = atlas_dir / "02_average_liver_compact.html"
        stage3_out = atlas_dir / "03_density_slices_compact.html"
        vdc_out = atlas_dir / "vdc_combined_compact.html"
        hist_out = atlas_dir / "vdc_histogram_compact.html"

        if args.skip_existing and stage2_out.exists():
            print(f"[skip-existing] {tag}: {stage2_out}")
        else:
            out = compact_stage2_average_liver(atlas_dir, label)
            if out is not None:
                outputs.append(out)

        if args.skip_existing and stage3_out.exists():
            print(f"[skip-existing] {tag}: {stage3_out}")
        else:
            outputs.append(compact_stage3_density_slices(atlas_dir, label))

        if args.skip_existing and vdc_out.exists():
            print(f"[skip-existing] {tag}: {vdc_out}")
        else:
            outputs.append(compact_vdc_combined(atlas_dir, label))

        if args.skip_existing and hist_out.exists():
            print(f"[skip-existing] {tag}: {hist_out}")
        else:
            out = compact_vdc_histogram(atlas_dir, label)
            if out is not None:
                outputs.append(out)

    return outputs


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser for paper figure generation."""
    parser = argparse.ArgumentParser(
        description="Generate research paper atlas, VDC, and metrics figures."
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        choices=["all", "compact", "metrics"],
        default=["all"],
        help="Choose which figure groups to generate.",
    )
    parser.add_argument("--male-dir", type=Path, default=Path("outputs/atlas_male"))
    parser.add_argument("--female-dir", type=Path, default=Path("outputs/atlas_female"))
    parser.add_argument("--male-label", default="154 male subjects")
    parser.add_argument("--female-label", default="107 female subjects")
    parser.add_argument("--cohort", choices=["male", "female", "both"], default="both")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip regenerating a compact HTML figure if the target already exists.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Parse CLI flags and generate the requested paper figures."""
    parser = build_parser()
    args = parser.parse_args(argv)

    targets = set(args.targets)
    if "all" in targets:
        targets = {"compact", "metrics"}

    outputs: list[Path] = []
    if "compact" in targets:
        outputs.extend(generate_compact_atlas_and_vdc_figures(args))
    if "metrics" in targets:
        outputs.extend(generate_metrics_figures())

    for out in outputs:
        print(f"Generated {out}")


if __name__ == "__main__":
    main()