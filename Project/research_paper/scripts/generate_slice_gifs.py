# AI Use Disclosure
#   Student estimate: 70% student-designed, 30% AI-assisted implementation
#   Claude assisted with: .npz cache scanning, Matplotlib figure layout, GIF assembly, compact HTML figures
#   See: "Documentation/AI Use Disclosure.md" for full details

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGURE_OUT_DIR = PROJECT_ROOT / "research_paper" / "outputs" / "figures"


def _crop_to_nonzero_bounds(vol: np.ndarray) -> tuple[np.ndarray, int]:
    """Crop 3D volume to non-zero region; return cropped volume and z offset."""
    present = vol > 0
    if not np.any(present):
        return vol, 0

    xyz = np.argwhere(present)
    x_min, y_min, z_min = xyz.min(axis=0)
    x_max, y_max, z_max = xyz.max(axis=0)
    cropped = vol[x_min : x_max + 1, y_min : y_max + 1, z_min : z_max + 1]
    return cropped, int(z_min)


def _make_density_volume(atlas_dir: Path) -> tuple[np.ndarray, int]:
    """Load and crop atlas liver-density volume to the nonzero bounding box."""
    img = nib.load(str(atlas_dir / "atlas_liver_density.nii.gz"))
    full = np.asarray(img.get_fdata(), dtype=np.float32)
    return _crop_to_nonzero_bounds(full)


def _make_vdc_volume(atlas_dir: Path) -> tuple[np.ndarray, int]:
    """Reconstruct a dense cropped VDC volume from sparse voxel/value arrays."""
    vox_idx = np.load(atlas_dir / "vdc_all_voxel_idx.npy")
    dists = np.load(atlas_dir / "vdc_full_dist_combined.npy")

    valid = ~np.isnan(dists)
    vox = vox_idx[valid]
    vals = dists[valid]

    if len(vox) == 0:
        raise RuntimeError(f"No valid VDC values found in {atlas_dir}")

    x_min, y_min, z_min = vox.min(axis=0)
    x_max, y_max, z_max = vox.max(axis=0)

    vol = np.full(
        (x_max - x_min + 1, y_max - y_min + 1, z_max - z_min + 1),
        np.nan,
        dtype=np.float32,
    )

    xi = vox[:, 0] - x_min
    yi = vox[:, 1] - y_min
    zi = vox[:, 2] - z_min
    vol[xi, yi, zi] = vals
    return vol, int(z_min)


def _render_slice_frame(
    arr2d: np.ndarray,
    cmap_name: str,
    vmin: float,
    vmax: float,
    colorbar_label: str,
    title: str,
) -> Image.Image:
    """Render one slice as an image with a fixed colorbar using Matplotlib."""
    cmap = mpl.colormaps[cmap_name].copy()
    cmap.set_bad(color=(1.0, 1.0, 1.0, 0.0))

    fig, ax = plt.subplots(figsize=(7.2, 5.6), dpi=140, constrained_layout=True)
    fig.patch.set_facecolor("white")

    im = ax.imshow(
        arr2d,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        origin="lower",
        interpolation="nearest",
        aspect="equal",
    )
    ax.set_title(title)
    ax.set_xlabel("x (voxels)")
    ax.set_ylabel("y (voxels)")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(colorbar_label)

    fig.canvas.draw()
    frame = Image.fromarray(np.asarray(fig.canvas.buffer_rgba()), mode="RGBA")
    plt.close(fig)
    return frame


def _write_gif(
    vol: np.ndarray,
    out_path: Path,
    cmap_name: str,
    colorbar_label: str,
    title_prefix: str,
    fps: int,
    reverse_z: bool = False,
) -> None:
    """Write an axial-slice GIF from a 3D volume (X, Y, Z)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    finite = vol[np.isfinite(vol)]
    if finite.size == 0:
        raise RuntimeError(f"No finite values to render for {out_path.name}")

    vmin = float(np.min(finite))
    vmax = float(np.max(finite))
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-6

    z_indices = list(range(vol.shape[2]))
    if reverse_z:
        z_indices = z_indices[::-1]

    frames: list[Image.Image] = []
    for z in z_indices:
        sl = vol[:, :, z].T  # match existing browser orientation
        frames.append(
            _render_slice_frame(
                sl,
                cmap_name=cmap_name,
                vmin=vmin,
                vmax=vmax,
                colorbar_label=colorbar_label,
                title=f"{title_prefix} - axial slice {z}",
            )
        )

    duration_ms = int(round(1000 / max(fps, 1)))
    frames[0].save(
        str(out_path),
        save_all=True,
        append_images=frames[1:],
        optimize=False,
        duration=duration_ms,
        loop=0,
        disposal=2,
    )


def generate_all_gifs(
    male_dir: Path,
    female_dir: Path,
    out_dir: Path,
    fps: int,
) -> list[Path]:
    """Generate male/female atlas and VDC axial GIFs and return output paths."""
    outputs: list[Path] = []

    # Atlas density (greens)
    m_density, _ = _make_density_volume(male_dir)
    f_density, _ = _make_density_volume(female_dir)

    m_density_out = out_dir / "MAtlas_slices.gif"
    f_density_out = out_dir / "FAtlas_slices.gif"
    _write_gif(
        m_density,
        m_density_out,
        cmap_name="Greens",
        colorbar_label="Probability",
        title_prefix="Male atlas density",
        fps=fps,
    )
    _write_gif(
        f_density,
        f_density_out,
        cmap_name="Greens",
        colorbar_label="Probability",
        title_prefix="Female atlas density",
        fps=fps,
    )
    outputs.extend([m_density_out, f_density_out])

    # Vascular distance combined (jet reversed: red=close, blue=far)
    m_vdc, _ = _make_vdc_volume(male_dir)
    f_vdc, _ = _make_vdc_volume(female_dir)

    m_vdc_out = out_dir / "MVDC_slices.gif"
    f_vdc_out = out_dir / "FVDC_slices.gif"
    _write_gif(
        m_vdc,
        m_vdc_out,
        cmap_name="jet_r",
        colorbar_label="Mean dist to vessel (mm)",
        title_prefix="Male VDC",
        fps=fps,
    )
    _write_gif(
        f_vdc,
        f_vdc_out,
        cmap_name="jet_r",
        colorbar_label="Mean dist to vessel (mm)",
        title_prefix="Female VDC",
        fps=fps,
    )
    outputs.extend([m_vdc_out, f_vdc_out])

    return outputs


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for atlas and VDC GIF generation."""
    p = argparse.ArgumentParser(
        description="Generate axial-slice GIFs for male/female atlas and VDC outputs."
    )
    p.add_argument("--male-dir", type=Path, default=Path("outputs/atlas_male"))
    p.add_argument("--female-dir", type=Path, default=Path("outputs/atlas_female"))
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("research_paper/outputs/figures"),
    )
    p.add_argument("--fps", type=int, default=10, help="Frames per second")
    return p


def main(argv: list[str] | None = None) -> None:
    """Parse CLI args, render slice animations, and print generated GIF paths."""
    args = build_parser().parse_args(argv)

    male_dir = (PROJECT_ROOT / args.male_dir).resolve() if not args.male_dir.is_absolute() else args.male_dir
    female_dir = (PROJECT_ROOT / args.female_dir).resolve() if not args.female_dir.is_absolute() else args.female_dir
    out_dir = (PROJECT_ROOT / args.out_dir).resolve() if not args.out_dir.is_absolute() else args.out_dir

    outputs = generate_all_gifs(male_dir=male_dir, female_dir=female_dir, out_dir=out_dir, fps=args.fps)
    for out in outputs:
        print(f"Generated {out}")


if __name__ == "__main__":
    main()
