# Validation/visualize_vessel_comparison.py
# Tristan Jones — Spring 2026 Capstone
#
# AI Use Disclosure
#   Student estimate: 70% student-designed, 30% AI-assisted implementation
#   Claude assisted with: marching cubes mesh generation, two-panel Plotly figure assembly
#   See: "Documentation/AI Use Disclosure.md" for full details
#

# I literally just made this file to check something it was made by AI with the prompt:
# script that shows the portal and hepatic segmetnations side by side using the 
# the marching cubes method (shows as shell) to showcase the differnence between them 

#
# Side-by-side 3D comparison of the two vessel segmentation masks for a
# single patient using marching cubes:
#
#   LEFT  — portal_vein_and_splenic_vein.nii.gz  (main TS model, --roi_subset)
#   RIGHT — liver_vessels.nii.gz                 (dedicated liver_vessels sub-task)
#
# Renders both alongside the liver surface so anatomical context is clear.
# Usage:
#   python -m Validation.visualize_vessel_comparison --patient 0004
#   python -m Validation.visualize_vessel_comparison --patient 0004 --data Data

import argparse
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from skimage.measure import marching_cubes

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Registration.stages.load import load_patient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mesh_from_mask(mask: np.ndarray,
                    affine: np.ndarray,
                    color: str,
                    opacity: float,
                    name: str) -> go.Mesh3d | None:
    """
    Run marching cubes on a binary mask and return a Plotly Mesh3d trace.
    Vertex coords are mapped to mm via the patient affine.
    Returns None if the mask is empty or marching cubes fails.
    """
    if mask is None or mask.max() == 0:
        print(f"  [skip] {name}: mask is empty or None")
        return None

    padded = np.pad(mask > 0, 1, mode="constant", constant_values=0).astype(np.float32)
    try:
        verts, faces, _, _ = marching_cubes(padded, level=0.5)
    except Exception as e:
        print(f"  [skip] {name}: marching cubes failed — {e}")
        return None

    verts -= 1  # undo padding offset
    ones   = np.ones((len(verts), 1))
    verts  = (affine @ np.hstack([verts, ones]).T).T[:, :3]

    return go.Mesh3d(
        x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        color=color, opacity=opacity, name=name,
        showlegend=True,
    )


def _camera() -> dict:
    """Return a shared 3D camera preset for both comparison scenes."""
    return dict(eye=dict(x=1.6, y=1.6, z=0.8))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def visualize_vessel_comparison(patient_id: str, data_dir: Path) -> None:
    """Render portal vs hepatic vessel segmentations side by side for one patient."""
    print(f"Loading patient {patient_id} …")
    data = load_patient(data_dir, patient_id)

    affine       = data["affine"]
    liver        = data["liver"]
    portal_mask  = data.get("portal_vein")   # portal_vein_and_splenic_vein
    hepatic_mask = data.get("hepatic_vein")  # liver_vessels

    # ------------------------------------------------------------------
    # Build meshes
    # ------------------------------------------------------------------
    liver_mesh_L = _mesh_from_mask(liver,        affine, "#c8a97e", 0.15, "Liver")
    liver_mesh_R = _mesh_from_mask(liver,        affine, "#c8a97e", 0.15, "Liver")
    portal_mesh  = _mesh_from_mask(portal_mask,  affine, "#3a86ff", 0.85, "Portal + Splenic vein (roi_subset)")
    hepatic_mesh = _mesh_from_mask(hepatic_mask, affine, "#ff006e", 0.85, "Hepatic vessels (liver_vessels task)")

    # ------------------------------------------------------------------
    # Figure — two subplots side by side
    # ------------------------------------------------------------------
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=[
            f"Patient {patient_id} — Portal + Splenic Vein<br><sup>portal_vein_and_splenic_vein.nii.gz (main model, --roi_subset)</sup>",
            f"Patient {patient_id} — Hepatic Vessels<br><sup>liver_vessels.nii.gz (dedicated liver_vessels sub-task)</sup>",
        ],
    )

    # Left subplot — portal/splenic
    for trace in [liver_mesh_L, portal_mesh]:
        if trace is not None:
            fig.add_trace(trace, row=1, col=1)

    # Right subplot — hepatic vessels
    for trace in [liver_mesh_R, hepatic_mesh]:
        if trace is not None:
            # duplicate liver mesh so it appears independently on the right scene
            if trace.name == "Liver":
                dup = go.Mesh3d(
                    x=trace.x, y=trace.y, z=trace.z,
                    i=trace.i, j=trace.j, k=trace.k,
                    color=trace.color, opacity=trace.opacity,
                    name=trace.name, showlegend=False,
                )
                fig.add_trace(dup, row=1, col=2)
            else:
                fig.add_trace(trace, row=1, col=2)

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    shared_scene = dict(
        aspectmode="data",
        camera=_camera(),
        xaxis_title="X (mm)",
        yaxis_title="Y (mm)",
        zaxis_title="Z (mm)",
    )

    fig.update_layout(
        title=dict(
            text=f"Vessel Segmentation Comparison — Patient {patient_id}",
            x=0.5,
        ),
        scene =shared_scene,
        scene2=shared_scene,
        legend=dict(x=0.01, y=0.99),
        paper_bgcolor="#1a1a2e",
        plot_bgcolor ="#1a1a2e",
        font=dict(color="white"),
        width=1600,
        height=800,
    )

    out_path = Path("outputs") / f"vessel_comparison_{patient_id}.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path))
    print(f"Saved → {out_path}")
    fig.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Side-by-side 3D vessel segmentation comparison for one patient."
    )
    parser.add_argument("--patient", default="0004",
                        help="Patient ID (default: 0004)")
    parser.add_argument("--data",    default="Data",
                        help="Root data directory (default: Data)")
    args = parser.parse_args()

    visualize_vessel_comparison(args.patient, Path(args.data))
