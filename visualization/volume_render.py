from __future__ import annotations
import numpy as np, plotly.graph_objects as go
from skimage import measure


def mask_mesh_figure(masks: dict, spacing_zyx=(1, 1, 1), step=2):
    """Marching-cubes surfaces (physical mm axes). masks: {name: (3D bool array, color)}"""
    fig = go.Figure()
    for name, (m, color) in masks.items():
        if m is None or not np.asarray(m).any(): continue
        v, f, _, _ = measure.marching_cubes(np.asarray(m)[::step, ::step, ::step].astype(np.uint8), 0.5, spacing=tuple(s * step for s in spacing_zyx))
        fig.add_trace(go.Mesh3d(x=v[:, 2], y=v[:, 1], z=v[:, 0], i=f[:, 0], j=f[:, 1], k=f[:, 2], color=color, opacity=0.8, name=name, showlegend=True))
    fig.update_layout(scene=dict(aspectmode="data"), height=500, margin=dict(l=0, r=0, t=0, b=0))
    return fig
