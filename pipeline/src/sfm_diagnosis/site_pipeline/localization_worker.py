"""Stage-14 strict EDM LOO orchestration over a deployment-owned provider."""

from __future__ import annotations

import importlib
import csv
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Callable, Mapping

from sfm_diagnosis.edm_risk.edm_loo import EDMLOOMode, EDMLOORunner, EDMQuery


def load_provider(
    specification: str,
    kwargs: Mapping[str, Any],
    *,
    allowed_modules: tuple[str, ...],
):
    if ":" not in specification:
        raise ValueError("provider_factory must be package.module:callable")
    module_name, attribute = specification.split(":", 1)
    if not allowed_modules or not any(
        module_name == prefix or module_name.startswith(f"{prefix}.") for prefix in allowed_modules
    ):
        raise ValueError(f"provider module is not allowlisted: {module_name}")
    # Provider modules are constrained by provider_allowlist above.
    module = importlib.import_module(  # nosemgrep: python.lang.security.audit.non-literal-import.non-literal-import
        module_name
    )
    factory = getattr(module, attribute)
    if not callable(factory):
        raise TypeError("provider factory is not callable")
    return factory(**dict(kwargs))


def run_adapter_request(
    payload: Mapping[str, Any],
    *,
    provider_factory: Callable[[Mapping[str, Any]], Any] | None = None,
) -> dict[str, Any]:
    config = dict(payload.get("config") or {})
    request = dict(payload.get("payload") or {})
    query_manifest = Path(str(config.get("query_manifest") or ""))
    output_validation = Path(str(request.get("output_validation") or ""))
    output_references = Path(str(request.get("output_references") or ""))
    roles_path = Path(str(request.get("roles") or ""))
    if not query_manifest.is_file() or not roles_path.is_file():
        raise ValueError("localizer requires query_manifest and roles")
    if not str(output_validation) or not str(output_references):
        raise ValueError("localizer output paths are missing")
    queries: dict[str, list[EDMQuery]] = {}
    for row in _jsonl(query_manifest):
        query = EDMQuery(
            query_id=str(row["query_id"]),
            session_id=str(row["session_id"]),
            timestamp=float(row["timestamp"]),
            image_path=None if row.get("image_path") is None else str(row["image_path"]),
            pose_provenance=row.get("pose_provenance"),
        )
        queries.setdefault(query.session_id, []).append(query)
    corpus_path = Path(str(request.get("corpus") or ""))
    metadata_path = Path(str(request.get("metadata") or ""))
    holdout_provenance = None
    if corpus_path.is_file() and metadata_path.is_file():
        corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
        metadata_rows = {
            str(row["source_id"]): row
            for row in csv.DictReader(metadata_path.open(encoding="utf-8"))
        }
        allowed_sessions = set()
        provenance = set()
        for source in corpus.get("sources") or ():
            if source.get("evaluation_role") != "HOLDOUT":
                continue
            row = metadata_rows.get(str(source["source_id"]), {})
            allowed_sessions.update(
                value
                for value in (
                    row.get("session_id"),
                    source.get("source_id"),
                    source.get("video_id"),
                    Path(str(source.get("path") or "")).stem,
                    Path(str(source.get("path") or "")).name,
                )
                if value
            )
            if source.get("holdout_provenance"):
                provenance.add(str(source["holdout_provenance"]))
        leaked = sorted(set(queries) - allowed_sessions)
        if leaked:
            raise RuntimeError(f"query sessions are not declared hold-outs: {leaked}")
        holdout_provenance = sorted(provenance)
    if provider_factory is None:
        specification = str(config.get("provider_factory") or "")
        provider = load_provider(
            specification,
            config.get("provider_kwargs") or {},
            allowed_modules=tuple(str(value) for value in config.get("provider_allowlist") or ()),
        )
    else:
        provider = provider_factory(config)
    runner = EDMLOORunner(
        provider=provider,
        cache_dir=output_validation.parent / "cache",
    )
    mode = EDMLOOMode(str(config.get("loo_mode") or "strict"))
    result = runner.run(queries, mode=mode)
    output_validation.parent.mkdir(parents=True, exist_ok=True)
    result_payload = result.to_dict()
    result_payload["heldout_provenance"] = holdout_provenance
    overlap = bool(config.get("map_query_identity_overlap", False))
    evidence_class = str(config.get("evidence_class") or "mapping_disjoint_holdout")
    result_payload["evidence_class"] = evidence_class
    result_payload["map_query_identity_overlap"] = overlap
    result_payload["mapping_disjoint"] = not overlap
    result_payload["excluded_query_session_ids"] = sorted(
        str(session_id) for session_id in queries.keys()
    )
    if overlap:
        result_payload["status"] = "SHADOW_GROUP_REFERENCE_EXCLUSION"
        result_payload["deployment_authorized"] = False
        result_payload["authority"] = "SHADOW_ONLY_NOT_MAPPING_DISJOINT_NOT_DEPLOYABLE"
    output_validation.write_text(json.dumps(result_payload, indent=2) + "\n", encoding="utf-8")
    references = []
    for role in _jsonl(roles_path):
        if role.get("localization") == "INCLUDE":
            references.append(
                {
                    "segment_id": role.get("segment_id") or role.get("segment"),
                    "post_sfm_role": role.get("post_sfm_role"),
                    "base_map": role.get("base_map"),
                    "localization": "INCLUDE",
                }
            )
    output_references.parent.mkdir(parents=True, exist_ok=True)
    output_references.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in references),
        encoding="utf-8",
    )
    return {
        "status": "completed",
        "outputs": [str(output_validation), str(output_references)],
        "queries": len(result.results),
        "strict_loo": result.strict_loo,
        "evidence_class": evidence_class,
        "map_query_identity_overlap": overlap,
    }


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def main() -> int:
    output = sys.stdout
    try:
        with redirect_stdout(sys.stderr):
            result = run_adapter_request(json.load(sys.stdin))
    except Exception as error:
        output.write(json.dumps({"status": "error", "error": str(error)}) + "\n")
        return 1
    output.write(json.dumps(result) + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["load_provider", "run_adapter_request"]
