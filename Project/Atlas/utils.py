# Atlas/utils.py
# Tristan Jones — Spring 2026 Capstone
#
# Shared geometry helpers and Plotly building blocks used by both
# liver_atlas.py and vascular_distance.py.
#
# Importing from Registration.utils.Nifti where possible so there is one
# canonical implementation of voxel ↔ mm conversion.

from __future__ import annotations

from typing import Optional

import numpy as np
import plotly.graph_objects as go
from scipy.ndimage import sobel
from scipy.spatial import cKDTree

from Registration.utils.Nifti import voxels_to_mm, mm_to_voxels   # canonical helpers


# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------

PATIENT_COLORS = [
    "#e6194b",  # red
    "#4363d8",  # blue
    "#f58231",  # orange
    "#911eb4",  # purple
    "#42d4f4",  # cyan
    "#f032e6",  # magenta
    "#bfef45",  # lime
    "#fabed4",  # pink
]


# ---------------------------------------------------------------------------
# Array padding
# ---------------------------------------------------------------------------

def pad_to_common(a: np.ndarray, b: np.ndarray):
    """Zero-pad two arrays so they share the same shape along every axis."""
    out = tuple(max(a.shape[i], b.shape[i]) for i in range(a.ndim))
    def _pad(arr):
        return np.pad(arr, [(0, out[i] - arr.shape[i]) for i in range(arr.ndim)])
    return _pad(a), _pad(b)


def pad_vol_to(vol: np.ndarray, shape: tuple) -> np.ndarray:
    return np.pad(vol, [(0, shape[i] - vol.shape[i]) for i in range(vol.ndim)])


# ---------------------------------------------------------------------------
# Dice coefficient
# ---------------------------------------------------------------------------

def dice(seg_a: np.ndarray, seg_b: np.ndarray, label: int = 1) -> float:
    """
    Dice Similarity Coefficient between two segmentation volumes.
    Pads both to a common bounding box so grid-size differences don't matter.
    """
    a, b = pad_to_common(seg_a, seg_b)
    a = a == label
    b = b == label
    intersection = np.logical_and(a, b).sum()
    denom = a.sum() + b.sum()
    return float(2.0 * intersection / denom) if denom > 0 else 0.0


# ---------------------------------------------------------------------------
# Point cloud extraction
# ---------------------------------------------------------------------------

def extract_surface_mm(liver_vol: np.ndarray,
                        affine: np.ndarray,
                        downsample: int = 8) -> np.ndarray:
    """
    Sobel-edge surface voxels → mm world coords.
    Used by LiverAtlas for the common-basis diagnostic scatter plot.

    Returns (N, 3) float32 array in mm.
    """
    binary = (liver_vol > 0).astype(np.float32)
    edges  = sobel(binary)
    vox    = np.argwhere(edges > 0)[::downsample].astype(np.float32)
    return voxels_to_mm(vox, affine).astype(np.float32)


def extract_liver_voxels(vol: np.ndarray) -> np.ndarray:
    """
    All nonzero liver voxel indices — full resolution, no downsampling.
    Used by VascularDistanceCloud for accumulation and the slice browser.

    Returns (N, 3) int32 array.
    """
    return np.argwhere(vol > 0).astype(np.int32)


def downsample_voxels_to_mm(voxels: np.ndarray,
                              affine: np.ndarray,
                              target: int = 50_000) -> tuple[np.ndarray, int]:
    """
    Stride-downsample voxels to ~target points and convert to mm.
    Used for 3-D scatter visualization only — not for accumulation.

    Returns (mm_pts, stride).
    """
    n      = len(voxels)
    stride = max(1, n // target)
    subset = voxels[::stride].astype(np.float32)
    print(f"    Viz downsample: {n:,} → {len(subset):,} pts (stride={stride})")
    return voxels_to_mm(subset, affine).astype(np.float32), stride


def extract_vessel_mm(vol: np.ndarray,
                       affine: np.ndarray,
                       downsample: int = 4) -> np.ndarray:
    """
    All nonzero vessel voxels → mm coords, with a fixed stride for speed.
    Returns (N, 3) float32. Returns empty array if mask is empty.
    """
    vox = np.argwhere(vol > 0)[::downsample].astype(np.float32)
    if len(vox) == 0:
        return np.empty((0, 3), dtype=np.float32)
    return voxels_to_mm(vox, affine).astype(np.float32)


# ---------------------------------------------------------------------------
# KNN distance
# ---------------------------------------------------------------------------

def knn_mean_distance(query_pts: np.ndarray,
                       vessel_pts: np.ndarray,
                       k: int = 5) -> np.ndarray:
    """
    For every point in query_pts return the mean distance to the k nearest
    points in vessel_pts.

    Using k>1 makes the estimate robust to single misregistered vessel voxels.
    Returns NaN for all query points if vessel_pts is empty.

    Returns (N,) float32 array.
    """
    if len(vessel_pts) == 0:
        return np.full(len(query_pts), np.nan, dtype=np.float32)

    k_actual = min(k, len(vessel_pts))
    tree     = cKDTree(vessel_pts)
    dists, _ = tree.query(query_pts, k=k_actual, workers=-1)

    if dists.ndim == 1:
        return dists.astype(np.float32)
    return dists.mean(axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# Plotly mesh from density volume (marching cubes)
# ---------------------------------------------------------------------------

def density_to_mesh(vol: np.ndarray,
                     level: float,
                     color: str,
                     opacity: float,
                     name: str,
                     affine: np.ndarray) -> Optional[go.Mesh3d]:
    """
    Run marching cubes on a density volume and return a Plotly Mesh3d trace.
    Vertex coordinates are converted to mm using the atlas affine.

    Returns None if the volume max is below `level` or marching cubes fails.
    """
    try:
        from skimage.measure import marching_cubes
    except ImportError:
        raise ImportError("pip install scikit-image")

    if vol.max() < level:
        print(f"  [skip] {name}: max={vol.max():.3f} < level={level}")
        return None

    zooms  = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
    padded = np.pad(vol, 1, mode="constant", constant_values=0)
    try:
        verts, faces, _, _ = marching_cubes(padded, level=level)
    except Exception as e:
        print(f"  [skip] {name}: marching cubes failed — {e}")
        return None

    verts -= 1       # undo padding offset
    # Apply full affine (rotation + scaling + translation) so the mesh sits
    # in the same mm coordinate space as the surface point clouds.
    # Previously only zooms (voxel spacing) was applied, which ignored the
    # affine origin and caused the mesh to float away from the point clouds.
    ones  = np.ones((len(verts), 1))
    verts = (affine @ np.hstack([verts, ones]).T).T[:, :3]

    return go.Mesh3d(
        x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        color=color, opacity=opacity, name=name, showlegend=True,
    )


# ---------------------------------------------------------------------------
# Plotly slider layout helper
# ---------------------------------------------------------------------------

def make_slider_layout(n_frames: int,
                        mid: int,
                        prefix: str = "Axial slice z=") -> dict:
    """Return a Plotly sliders + updatemenus layout block for an axial browser."""
    return dict(
        sliders=[dict(
            active=mid,
            currentvalue=dict(prefix=prefix, visible=True),
            pad=dict(t=50),
            steps=[dict(
                method="animate",
                args=[[str(z)], dict(
                    mode="immediate",
                    frame=dict(duration=0, redraw=True),
                    transition=dict(duration=0),
                )],
                label=str(z),
            ) for z in range(n_frames)],
        )],
        updatemenus=[dict(
            type="buttons", showactive=False,
            y=0, x=0.5, xanchor="center", yanchor="top",
            buttons=[dict(
                label="▶ Play",
                method="animate",
                args=[None, dict(
                    frame=dict(duration=80, redraw=True),
                    fromcurrent=True, transition=dict(duration=0),
                )],
            )],
        )],
    )