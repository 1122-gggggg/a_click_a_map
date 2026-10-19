"""Directional HTML, explainable zone reports, and colored risk-sphere PLYs."""

from __future__ import annotations

import html
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from sfm_diagnosis.io import write_json
from sfm_diagnosis.models import MapData
from sfm_diagnosis.risk_ply import (
    robust_spatial_clip,
    sphere_points,
    write_binary_ply,
)

from .schema import RiskClass, SpatialDiagnostic


@dataclass(frozen=True)
class RiskPalette:
    """Visualization palette; classification never reads these values."""

    good: tuple[int, int, int] = (40, 180, 80)
    view_direction_sensitive: tuple[int, int, int] = (245, 210, 40)
    weak: tuple[int, int, int] = (245, 135, 35)
    dead_zone: tuple[int, int, int] = (220, 45, 45)
    unknown: tuple[int, int, int] = (145, 145, 145)

    def color(self, risk_class: RiskClass) -> tuple[int, int, int]:
        return {
            RiskClass.GOOD: self.good,
            RiskClass.VIEW_DIRECTION_SENSITIVE: self.view_direction_sensitive,
            RiskClass.WEAK: self.weak,
            RiskClass.DEAD_ZONE: self.dead_zone,
            RiskClass.UNKNOWN: self.unknown,
        }[risk_class]


def write_diagnosis_artifacts(
    map_data: MapData,
    rows: Sequence[SpatialDiagnostic],
    output_dir: str | Path,
    *,
    voxel_size: float = 1.0,
    palette: RiskPalette | None = None,
    sphere_samples: int = 48,
    display_radius: float | None = None,
) -> dict[str, Path]:
    """Write all human-facing diagnosis artifacts from normalized rows."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    colors = palette or RiskPalette()
    groups = _position_groups(rows)
    zones = _zone_records(groups, voxel_size=voxel_size)
    weak_zones = [zone for zone in zones if zone["risk_class"] == RiskClass.WEAK.value]
    dead_zones = [zone for zone in zones if zone["risk_class"] == RiskClass.DEAD_ZONE.value]

    weak_path = output / "weak_zones.json"
    dead_path = output / "dead_zones.json"
    write_json(weak_path, _zone_payload("WEAK_ZONES", weak_zones))
    write_json(dead_path, _zone_payload("DEAD_ZONES", dead_zones))

    sphere_xyz, sphere_rgb = _risk_spheres(
        groups,
        palette=colors,
        radius=max(float(voxel_size) * 0.25, 1e-3),
        samples=sphere_samples,
    )
    spheres_path = write_binary_ply(
        output / "risk_spheres.ply",
        sphere_xyz,
        sphere_rgb,
        comments=("one colored shell per spatial waypoint",),
    )
    clip = robust_spatial_clip(
        map_data.points_xyz,
        camera_xyz=map_data.image_centers,
    )
    robust_keep = np.asarray(clip["keep"], dtype=bool)
    anchor_xyz = np.vstack(
        (
            map_data.image_centers,
            np.asarray([position for position, _ in groups], dtype=float).reshape(-1, 3),
        )
    )
    if display_radius is None:
        camera_diagonal = (
            float(
                np.linalg.norm(
                    np.quantile(map_data.image_centers, 0.99, axis=0)
                    - np.quantile(map_data.image_centers, 0.01, axis=0)
                )
            )
            if len(map_data.image_centers)
            else 0.0
        )
        display_radius = max(float(voxel_size) * 8.0, camera_diagonal * 1.5, 1e-3)
    if display_radius <= 0.0:
        raise ValueError("display_radius must be positive")
    if len(anchor_xyz):
        from scipy.spatial import cKDTree

        nearest_distance = cKDTree(anchor_xyz).query(map_data.points_xyz, k=1)[0]
        anchor_keep = np.isfinite(nearest_distance) & (
            nearest_distance <= float(display_radius)
        )
    else:
        anchor_keep = np.ones(map_data.num_points, dtype=bool)
    keep = robust_keep & anchor_keep
    clipped_xyz = map_data.points_xyz[keep]
    clipped_rgb = map_data.point_rgb[keep]
    base_path = write_binary_ply(
        output / "risk_map_base.ply",
        clipped_xyz,
        clipped_rgb,
        comments=("viewer-safe robust-clipped GlueMap landmarks with original RGB",),
    )
    combined_path = write_binary_ply(
        output / "risk_map.ply",
        np.vstack((clipped_xyz, sphere_xyz)),
        np.vstack((clipped_rgb, sphere_rgb)),
        comments=("viewer-safe robust-clipped GlueMap RGB plus risk spheres",),
    )
    full_path = write_binary_ply(
        output / "risk_map_full.ply",
        np.vstack((map_data.points_xyz, sphere_xyz)),
        np.vstack((map_data.point_rgb, sphere_rgb)),
        comments=("full-extent archival GlueMap RGB plus risk spheres; do not auto-fit",),
    )
    clipping_path = output / "risk_ply_clipping.json"
    write_json(
        clipping_path,
        {
            "schema_version": 1,
            "artifact_type": "EDM_RISK_PLY_DISPLAY_CLIP",
            **{
                key: value
                for key, value in clip.items()
                if key not in {"keep", "retained_count", "excluded_count"}
            },
            "robust_retained_count": int(np.sum(robust_keep)),
            "display_anchor_count": int(len(anchor_xyz)),
            "display_radius": float(display_radius),
            "retained_count": int(np.sum(keep)),
            "excluded_count": int(len(keep) - np.sum(keep)),
            "display_min": (
                np.min(clipped_xyz, axis=0).tolist() if len(clipped_xyz) else None
            ),
            "display_max": (
                np.max(clipped_xyz, axis=0).tolist() if len(clipped_xyz) else None
            ),
            "display_diagonal": (
                float(
                    np.linalg.norm(
                        np.max(clipped_xyz, axis=0) - np.min(clipped_xyz, axis=0)
                    )
                )
                if len(clipped_xyz)
                else 0.0
            ),
            "viewer_safe_ply": str(combined_path),
            "full_extent_archive_ply": str(full_path),
            "original_rgb_preserved": True,
        },
    )

    risk_html = output / "risk_map.html"
    report_html = output / "diagnosis_report.html"
    risk_html.write_text(
        _directional_html(groups, colors), encoding="utf-8"
    )
    report_html.write_text(
        _report_html(groups, zones, colors), encoding="utf-8"
    )
    return {
        "risk_map_html": risk_html,
        "diagnosis_report_html": report_html,
        "weak_zones_json": weak_path,
        "dead_zones_json": dead_path,
        "risk_spheres_ply": spheres_path,
        "risk_map_base_ply": base_path,
        "risk_map_ply": combined_path,
        "risk_map_full_ply": full_path,
        "risk_ply_clipping_json": clipping_path,
    }


def _position_groups(
    rows: Sequence[SpatialDiagnostic],
) -> list[tuple[tuple[float, float, float], list[SpatialDiagnostic]]]:
    grouped: dict[tuple[float, float, float], list[SpatialDiagnostic]] = {}
    for row in rows:
        grouped.setdefault(tuple(float(value) for value in row.position), []).append(row)
    return [(position, grouped[position]) for position in sorted(grouped)]


def _zone_payload(kind: str, zones: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_type": kind,
        "zone_count": len(zones),
        "zones": zones,
    }


def _zone_records(
    groups: list[tuple[tuple[float, float, float], list[SpatialDiagnostic]]],
    *,
    voxel_size: float,
) -> list[dict[str, Any]]:
    counters: Counter[str] = Counter()
    records = []
    for position, group in groups:
        risk_class = _group_class(group)
        if risk_class not in {
            RiskClass.VIEW_DIRECTION_SENSITIVE,
            RiskClass.WEAK,
            RiskClass.DEAD_ZONE,
        }:
            continue
        counters[risk_class.value] += 1
        available = [row for row in group if row.failure_probability is not None]
        best = min(available, key=lambda row: float(row.failure_probability)) if available else group[0]
        worst = max(available, key=lambda row: float(row.failure_probability)) if available else group[0]
        causes = Counter(cause for row in group for cause in row.primary_failure_causes)
        records.append(
            {
                "zone_id": f"{risk_class.value}_{counters[risk_class.value]:03d}",
                "risk_class": risk_class.value,
                "position": list(position),
                "failure_probability_best": best.failure_probability,
                "failure_probability_worst": worst.failure_probability,
                "primary_causes": [name for name, _ in causes.most_common()],
                "evidence": _evidence(worst),
                "recommended_action": best.recommended_action,
                "recommended_capture_position": best.recommended_capture_position,
                "recommended_yaw_deg": best.yaw_deg,
                "recommended_pitch_deg": best.pitch_deg,
                "recommended_distance": float(voxel_size) * 1.5,
            }
        )
    return records


def _evidence(row: SpatialDiagnostic) -> dict[str, Any]:
    return {
        "visible_landmarks": row.visible_landmarks,
        "effective_landmarks": row.effective_landmarks,
        "parallax_p10_deg": row.parallax_p10_deg,
        "fim_lambda_min": row.fim_lambda_min,
        "fim_condition": row.fim_condition,
        "independent_observers": row.independent_observers,
        "edm_loo_success_rate": row.edm_loo_success_rate,
        "edm_inliers_median": row.edm_inliers_median,
        "num_pose_modes": row.num_pose_modes,
        "mode_support_margin": row.mode_support_margin,
        "consecutive_failure_probability": row.consecutive_failure_probability,
    }


def _group_class(group: Sequence[SpatialDiagnostic]) -> RiskClass:
    classes = Counter(row.risk_class for row in group)
    return classes.most_common(1)[0][0] if classes else RiskClass.UNKNOWN


def _risk_spheres(
    groups: list[tuple[tuple[float, float, float], list[SpatialDiagnostic]]],
    *,
    palette: RiskPalette,
    radius: float,
    samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    xyz = []
    rgb = []
    for position, group in groups:
        shell = sphere_points(position, radius, samples)
        color = np.asarray(palette.color(_group_class(group)), dtype=np.uint8)
        xyz.append(shell)
        rgb.append(np.repeat(color[None], len(shell), axis=0))
    if not xyz:
        return np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=np.uint8)
    return np.vstack(xyz), np.vstack(rgb)


def _directional_html(
    groups: list[tuple[tuple[float, float, float], list[SpatialDiagnostic]]],
    palette: RiskPalette,
) -> str:
    payload = []
    for position, group in groups:
        payload.append(
            {
                "position": list(position),
                "risk_class": _group_class(group).value,
                "rows": [row.to_dict() for row in group],
            }
        )
    palette_payload = {
        risk.value: f"rgb{palette.color(risk)}"
        for risk in RiskClass
    }
    data_json = _safe_script_json(payload)
    colors_json = _safe_script_json(palette_payload)
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>EDM Localization Risk Map</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>body{{font-family:system-ui;margin:0;background:#111827;color:#e5e7eb}}
.layout{{display:grid;grid-template-columns:60% 40%;height:100vh}} .panel{{padding:12px}}
#map,#heatmap{{height:82vh}} select{{padding:6px;background:#1f2937;color:white}}</style></head>
<body><div class="layout"><div id="map" class="panel"></div><div class="panel">
<h2>yaw-pitch directional diagnosis</h2><label>Metric <select id="metric">
<option value="actloc_score">ActLoc score</option><option value="fim_logdet">FIM score</option>
<option value="edm_loo_success_rate">empirical EDM success</option>
<option value="failure_probability" selected>final failure probability</option>
</select></label><div id="selection"></div><div id="heatmap"></div></div></div>
<script>const groups={data_json}; const colors={colors_json}; let selected=0;
const traces=Object.keys(colors).map(cls=>{{const rows=groups.map((g,i)=>[g,i]).filter(x=>x[0].risk_class===cls);
return {{type:'scatter3d',mode:'markers',name:cls,x:rows.map(x=>x[0].position[0]),y:rows.map(x=>x[0].position[1]),z:rows.map(x=>x[0].position[2]),
customdata:rows.map(x=>x[1]),marker:{{size:5,color:colors[cls]}},text:rows.map(x=>x[0].risk_class),hovertemplate:'%{{text}}<br>(%{{x:.2f}}, %{{y:.2f}}, %{{z:.2f}})<extra></extra>'}};}});
Plotly.newPlot('map',traces,{{title:'3D Localization Risk Map',paper_bgcolor:'#111827',plot_bgcolor:'#111827',font:{{color:'#e5e7eb'}}}});
document.getElementById('map').on('plotly_click',e=>{{selected=e.points[0].customdata;drawHeatmap();}});
document.getElementById('metric').addEventListener('change',drawHeatmap);
function drawHeatmap(){{if(!groups.length)return;const g=groups[selected],metric=document.getElementById('metric').value;
const yaws=[...new Set(g.rows.map(r=>r.yaw_deg))].sort((a,b)=>a-b), pitches=[...new Set(g.rows.map(r=>r.pitch_deg))].sort((a,b)=>a-b);
const z=pitches.map(p=>yaws.map(y=>{{const r=g.rows.find(v=>v.yaw_deg===y&&v.pitch_deg===p);return r?r[metric]:null;}}));
document.getElementById('selection').textContent=`position=${{g.position.join(', ')}} class=${{g.risk_class}}`;
Plotly.react('heatmap',[{{type:'heatmap',x:yaws,y:pitches,z:z,colorscale:'RdYlGn',reversescale:metric==='failure_probability'}}],
{{title:document.getElementById('metric').selectedOptions[0].text,xaxis:{{title:'yaw (deg)'}},yaxis:{{title:'pitch (deg)'}},paper_bgcolor:'#111827',plot_bgcolor:'#111827',font:{{color:'#e5e7eb'}}}});}} drawHeatmap();</script></body></html>"""


def _report_html(
    groups: list[tuple[tuple[float, float, float], list[SpatialDiagnostic]]],
    zones: list[dict[str, Any]],
    palette: RiskPalette,
) -> str:
    counts = Counter(_group_class(group).value for _, group in groups)
    cards = "".join(
        f'<div class="card"><b>{html.escape(risk.value)}</b><br>{counts[risk.value]}</div>'
        for risk in RiskClass
    )
    table_rows = "".join(
        "<tr>"
        f"<td>{html.escape(zone['zone_id'])}</td>"
        f"<td>{html.escape(zone['risk_class'])}</td>"
        f"<td>{html.escape(str(zone['position']))}</td>"
        f"<td>{html.escape(str(zone['failure_probability_worst']))}</td>"
        f"<td>{html.escape(', '.join(zone['primary_causes']))}</td>"
        f"<td>{html.escape(str(zone['recommended_action']))}</td>"
        "</tr>"
        for zone in zones
    )
    legend = " ".join(
        f'<span style="color:rgb{palette.color(risk)}">● {risk.value}</span>'
        for risk in RiskClass
    )
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>EDM Diagnosis Report</title>
<style>body{{font-family:system-ui;margin:2rem;color:#172033}}.cards{{display:flex;gap:1rem}}.card{{padding:1rem;border:1px solid #ddd}}
table{{border-collapse:collapse;width:100%;margin-top:1rem}}td,th{{border:1px solid #ddd;padding:.5rem;text-align:left}}</style></head>
<body><h1>EDM Localization Risk Diagnosis</h1><p>{legend}</p><div class="cards">{cards}</div>
<h2>Weak and dead zones</h2><table><thead><tr><th>ID</th><th>Class</th><th>Position</th><th>Worst P(fail)</th><th>Causes</th><th>Action</th></tr></thead>
<tbody>{table_rows}</tbody></table></body></html>"""


def _safe_script_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")


__all__ = ["RiskPalette", "write_diagnosis_artifacts"]
