"""Plotly 3-plane viewer with physically correct aspect ratio, window/level, mask overlays, boxes and centroids."""
from __future__ import annotations
import numpy as np, plotly.graph_objects as go
from .overlays import window, get_slice, plane_aspect

COL = {"gt": "rgba(0,200,0,{a})", "pred": "rgba(255,40,40,{a})"}


def slice_figure(vol, plane, idx, spacing_zyx=(1, 1, 1), level=-600, width=1500, gt=None, pred=None, candidates=None, alpha=0.45, title=None):
    img = window(get_slice(vol, plane, idx), level, width)
    # coronal/sagittal: flip so superior is up (volume is LPS => z increases superiorly)
    flip = plane != "axial"; ex = (lambda a: a[::-1]) if flip else (lambda a: a)
    fig = go.Figure(go.Heatmap(z=ex(img), colorscale="gray", showscale=False, zmin=0, zmax=1, hoverinfo="skip"))
    for m, name in ((gt, "gt"), (pred, "pred")):
        if m is None: continue
        s = ex(get_slice(m, plane, idx) > 0).astype(float)
        if s.any():
            fig.add_trace(go.Heatmap(z=np.where(s > 0, 1, np.nan), colorscale=[[0, COL[name].format(a=alpha)], [1, COL[name].format(a=alpha)]],
                                     showscale=False, hoverinfo="skip", name=name))
    for c in candidates or []:
        (z0, z1), (y0, y1), (x0, x1) = c["bbox_zyx"]; cz, cy, cx = c["centroid_zyx"]
        pl = {"axial": (z0 <= idx < z1, x0, x1, y0, y1, cx, cy), "coronal": (y0 <= idx < y1, x0, x1, z0, z1, cx, cz), "sagittal": (x0 <= idx < x1, y0, y1, z0, z1, cy, cz)}[plane]
        if pl[0]:
            r0, r1, cy_ = (pl[3], pl[4], pl[6]) if not flip else (vol.shape[0] - pl[4], vol.shape[0] - pl[3], vol.shape[0] - pl[6])
            fig.add_shape(type="rect", x0=pl[1], x1=pl[2], y0=r0, y1=r1, line=dict(color="yellow", width=2))
            fig.add_trace(go.Scatter(x=[pl[5]], y=[cy_], mode="markers", marker=dict(color="yellow", size=6, symbol="x"),
                                     hovertext=f"d≈{c['diameter_mm']:.1f} mm conf {c['confidence']:.2f}", showlegend=False))
    asp = plane_aspect(plane, spacing_zyx)
    fig.update_yaxes(scaleanchor="x", scaleratio=1 / asp if asp else 1, autorange=True if flip else "reversed", showticklabels=False)
    fig.update_xaxes(showticklabels=False)
    fig.update_layout(margin=dict(l=0, r=0, t=28, b=0), title=title or f"{plane} {idx}", height=420, paper_bgcolor="#111", font_color="#ddd", plot_bgcolor="#000")
    return fig
