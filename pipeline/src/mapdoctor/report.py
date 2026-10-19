from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from mapdoctor.benchmark import BenchmarkSummary, QueryLocalizationResult
from mapdoctor.comparison import ComparisonResult
from mapdoctor.metrics import HealthMetrics
from mapdoctor.recapture.audit import MetricAuditReport
from mapdoctor.recapture.types import RecaptureDecision
from mapdoctor.scoring import ReadinessResult


def _page(title: str, body: str) -> str:
    return f"""<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>{html.escape(title)}</title><style>body{{font-family:system-ui,sans-serif;max-width:1100px;margin:40px auto;padding:0 20px;color:#1f2937}}table{{border-collapse:collapse;width:100%;margin:12px 0 28px}}th,td{{text-align:left;padding:8px;border-bottom:1px solid #ddd;vertical-align:top}}.hero{{padding:20px;border:1px solid #ddd;border-radius:12px;background:#f8fafc}}.warning{{padding:12px 16px;border-left:4px solid #d97706;background:#fffbeb;margin:12px 0}}code{{word-break:break-all}}</style></head><body><h1>{html.escape(title)}</h1>{body}</body></html>"""


def _display_value(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value) if value else "none"
    return str(value)


def _provenance_html(metrics: HealthMetrics) -> str:
    provenance = metrics.producer_provenance
    if not provenance:
        return ""
    rows: list[tuple[str, Any]] = []
    if "producer" in provenance:
        rows.append(("Producer", provenance["producer"]))
    if "adapter" in provenance:
        rows.append(("Adapter", provenance["adapter"]))
    gluemap = provenance.get("gluemap")
    if isinstance(gluemap, dict):
        rows.extend(
            [
                ("GLUEMAP provenance mode", gluemap.get("mode")),
                ("GLUEMAP workspace", gluemap.get("workspace") or "not detected"),
                ("Detected GLUEMAP artifacts", gluemap.get("detected_artifacts", [])),
                ("Detected GLUEMAP stages", gluemap.get("detected_stages", [])),
                ("Coarse reconstructions", gluemap.get("coarse_reconstructions", [])),
            ]
        )
    table_rows = "".join(
        f"<tr><th>{html.escape(label)}</th><td>{html.escape(_display_value(value))}</td></tr>"
        for label, value in rows
        if value is not None
    )
    return f"<h2>Producer provenance</h2><table>{table_rows}</table>" if table_rows else ""


def write_analysis_bundle(
    metrics: HealthMetrics,
    readiness: ReadinessResult,
    output_dir: str | Path,
) -> dict[str, Path]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "report.json"
    html_path = root / "report.html"
    csv_path = root / "weak_images.csv"
    json_path.write_text(
        json.dumps({"metrics": metrics.to_dict(), "readiness": readiness.to_dict()}, indent=2),
        encoding="utf-8",
    )
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fields = ["image_id", "name", "observations", "hull_coverage", "grid4_occupancy", "reasons"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in metrics.weak_images:
            writer.writerow({**item, "reasons": ";".join(item["reasons"])})

    rows = "".join(
        f"<tr><th>{html.escape(key)}</th><td>{html.escape(str(value))}</td></tr>"
        for key, value in [
            ("Map adapter", metrics.source),
            ("Format", metrics.format),
            ("Registered images", metrics.registered_images),
            ("3D points", metrics.points3d),
            ("P10 observations/image", metrics.observations_per_image_p10),
            ("Median track length", metrics.track_length_median),
            ("Reprojection p90 (px)", metrics.reprojection_error_p90_px),
            ("Hull coverage p10", metrics.hull_coverage_p10),
            ("4x4 coverage p10", metrics.grid4_coverage_p10),
            ("Largest covisibility component", metrics.largest_covisibility_component_ratio),
        ]
    )
    weak = "".join(
        f"<tr><td>{item['image_id']}</td><td>{html.escape(item['name'])}</td>"
        f"<td>{item['observations']}</td><td>{item['grid4_occupancy']}</td>"
        f"<td>{html.escape(', '.join(item['reasons']))}</td></tr>"
        for item in metrics.weak_images
    ) or "<tr><td colspan='5'>No weak images.</td></tr>"
    suggestions = "".join(
        f"<li><code>{html.escape(item['reference_image'])}</code>: "
        f"{html.escape(item['suggested_action'])}</li>"
        for item in metrics.recapture_suggestions
    ) or "<li>No static capture hints.</li>"
    body = (
        f"<div class='hero'><strong>Readiness {readiness.score}/100 ({readiness.grade})</strong></div>"
        f"<h2>Map health</h2><table>{rows}</table>"
        f"{_provenance_html(metrics)}"
        f"<h2>Weak references</h2><table><tr><th>ID</th><th>Name</th><th>Obs.</th>"
        f"<th>Grid</th><th>Reasons</th></tr>{weak}</table>"
        f"<h2>Static capture hints (screening only)</h2>"
        f"<div class='warning'>These hints come only from reference-map geometry. They do not authorize recapture. "
        f"Run the metric-audited recapture planner with held-out localization, FIM, directional coverage, and "
        f"existing-data counterfactual evidence before deciding to collect new mapping data.</div>"
        f"<ul>{suggestions}</ul>"
    )
    html_path.write_text(_page("MapDoctor map-health report", body), encoding="utf-8")
    return {"json": json_path, "html": html_path, "weak_images_csv": csv_path}


def write_benchmark_bundle(
    summary: BenchmarkSummary,
    results: list[QueryLocalizationResult],
    output_dir: str | Path,
) -> dict[str, Path]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "benchmark.json"
    html_path = root / "benchmark.html"
    csv_path = root / "queries.csv"
    json_path.write_text(
        json.dumps({"summary": summary.to_dict(), "queries": [result.to_dict() for result in results]}, indent=2),
        encoding="utf-8",
    )
    fields = list(results[0].to_dict()) if results else []
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(result.to_dict() for result in results)
    failure_rows = "".join(
        f"<tr><td>{html.escape(item['query'])}</td>"
        f"<td>{html.escape(', '.join(item['reasons']))}</td>"
        f"<td>{item['inliers']}</td><td>{item['reproj_p90_px']}</td></tr>"
        for item in summary.failures
    ) or "<tr><td colspan='4'>No strict failures.</td></tr>"
    body = (
        f"<div class='hero'><strong>Strict success {summary.strict_success_rate:.1%}</strong> · "
        f"raw success {summary.raw_success_rate:.1%}</div>"
        f"<h2>Failed/weak queries</h2><table><tr><th>Query</th><th>Reasons</th>"
        f"<th>Inliers</th><th>Reproj p90</th></tr>{failure_rows}</table>"
        f"<p>Weak spatial regions: {len(summary.weak_regions)}</p>"
    )
    html_path.write_text(_page("MapDoctor localization benchmark", body), encoding="utf-8")
    return {"json": json_path, "html": html_path, "queries_csv": csv_path}


def write_comparison_bundle(result: ComparisonResult, output_dir: str | Path) -> dict[str, Path]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "comparison.json"
    html_path = root / "comparison.html"
    csv_path = root / "query_deltas.csv"
    json_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    fields = list(result.query_deltas[0]) if result.query_deltas else []
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(result.query_deltas)
    gates = "".join(f"<li>{html.escape(reason)}</li>" for reason in result.gate_failures) or "<li>All regression gates passed.</li>"
    body = (
        f"<div class='hero'><strong>Regression gate: {result.status}</strong></div>"
        f"<p>Base strict success: {result.base_strict_success_rate:.1%}<br>"
        f"Candidate strict success: {result.candidate_strict_success_rate:.1%}</p>"
        f"<p>New failures: {html.escape(', '.join(result.newly_failed) or 'none')}</p>"
        f"<h2>Gate findings</h2><ul>{gates}</ul>"
    )
    html_path.write_text(_page("MapDoctor map regression comparison", body), encoding="utf-8")
    return {"json": json_path, "html": html_path, "query_deltas_csv": csv_path}


def write_recapture_bundle(
    decisions: Sequence[RecaptureDecision],
    audits: Mapping[str, MetricAuditReport],
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write the canonical audited repair/recapture report bundle."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "recapture_plan.json"
    html_path = root / "recapture_plan.html"
    csv_path = root / "recapture_decisions.csv"
    audit_path = root / "metric_audit_by_region.json"

    payload = {
        "schema_version": 2,
        "decisions": [decision.as_dict() for decision in decisions],
        "metric_audit_by_region": {region: audit.as_dict() for region, audit in audits.items()},
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    audit_path.write_text(
        json.dumps({region: audit.as_dict() for region, audit in audits.items()}, indent=2),
        encoding="utf-8",
    )

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "region_id",
            "status",
            "recapture_required",
            "confidence",
            "existing_data_repairability",
            "structural_health",
            "directional_sensitivity",
            "blocked_by",
            "capture_pass_count",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for decision in decisions:
            writer.writerow(
                {
                    "region_id": decision.region_id,
                    "status": decision.status.value,
                    "recapture_required": decision.recapture_required,
                    "confidence": f"{decision.confidence:.4f}",
                    "existing_data_repairability": decision.existing_data_repairability,
                    "structural_health": decision.structural_health,
                    "directional_sensitivity": decision.directional_sensitivity,
                    "blocked_by": ";".join(decision.blocked_by),
                    "capture_pass_count": len(decision.capture_passes),
                }
            )

    decision_rows = "".join(
        "<tr>"
        f"<td>{html.escape(decision.region_id)}</td>"
        f"<td>{html.escape(decision.status.value)}</td>"
        f"<td>{'yes' if decision.recapture_required else 'no'}</td>"
        f"<td>{decision.existing_data_repairability if decision.existing_data_repairability is not None else 'unknown'}</td>"
        f"<td>{decision.structural_health if decision.structural_health is not None else 'unknown'}</td>"
        f"<td>{html.escape(', '.join(decision.blocked_by) or 'none')}</td>"
        f"<td>{len(decision.capture_passes)}</td>"
        "</tr>"
        for decision in decisions
    ) or "<tr><td colspan='7'>No regions supplied.</td></tr>"

    pass_sections: list[str] = []
    for decision in decisions:
        if not decision.capture_passes:
            continue
        items = "".join(
            f"<li><strong>{html.escape(capture_pass.mode.value)}</strong> — "
            f"{len(capture_pass.poses)} pose(s), safety={html.escape(capture_pass.safety_status.value)}, "
            f"units={html.escape(capture_pass.map_units)}</li>"
            for capture_pass in decision.capture_passes
        )
        pass_sections.append(f"<h3>{html.escape(decision.region_id)}</h3><ul>{items}</ul>")

    body = (
        "<div class='hero'><strong>Metric-audited weak-region repair and recapture plan</strong></div>"
        "<div class='warning'>Concrete capture poses are planning hypotheses only. Collision, geofence, dynamics, "
        "communications, wind, battery, and vehicle safety are not checked by MapDoctor.</div>"
        "<table><tr><th>Region</th><th>Decision</th><th>Recapture</th><th>Existing-data repairability</th>"
        f"<th>Structural health</th><th>Blocked by</th><th>Passes</th></tr>{decision_rows}</table>"
        + ("<h2>Targeted capture passes</h2>" + "".join(pass_sections) if pass_sections else "")
    )
    html_path.write_text(_page("MapDoctor targeted recapture plan", body), encoding="utf-8")
    return {
        "json": json_path,
        "html": html_path,
        "decisions_csv": csv_path,
        "metric_audit_json": audit_path,
    }


def write_json(metrics: HealthMetrics, readiness: ReadinessResult, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"metrics": metrics.to_dict(), "readiness": readiness.to_dict()}, indent=2),
        encoding="utf-8",
    )
    return out


def write_html(metrics: HealthMetrics, readiness: ReadinessResult, path: str | Path) -> Path:
    bundle = write_analysis_bundle(metrics, readiness, Path(path).parent)
    generated = bundle["html"]
    out = Path(path)
    if generated != out:
        out.write_text(generated.read_text(encoding="utf-8"), encoding="utf-8")
    return out
