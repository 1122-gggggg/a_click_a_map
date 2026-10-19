from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from sfm_diagnosis.io import write_json

from .config import CANONICAL_STAGES_V2, PipelineConfig
from .inventory import (
    IMAGE_SUFFIXES,
    discover_corpus,
    image_sequence_fingerprint,
    write_corpus_manifest,
    write_metadata_template,
)
from .stages import STAGE_BY_NAME, STAGE_SPECS
from .store import PipelineStore


class ApprovalRequired(RuntimeError):
    """Raised when Final GLUEMAP is requested without a matching human approval."""


class StageBlocked(RuntimeError):
    """Raised by a production stage when required evidence or an adapter is absent."""


@dataclass(frozen=True)
class StageContext:
    run_dir: Path
    stage_name: str
    config: PipelineConfig
    expected_outputs: tuple[Path, ...]
    fingerprint: str
    inputs: tuple[Path, ...] = ()


@dataclass(frozen=True)
class StageOutcome:
    outputs: tuple[Path, ...]
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StageResult:
    name: str
    status: str
    receipt: Path | None
    outputs: tuple[Path, ...] = ()
    reason: str | None = None


@dataclass(frozen=True)
class PipelineResult:
    run_dir: Path
    status: str
    stages: tuple[StageResult, ...]
    receipt: Path


StageHandler = Callable[[StageContext], StageOutcome]


class SitePipeline:
    """Deep module for the graph-aware 16-stage site mapping lifecycle.

    The module owns ordering, immutable inventory, metadata/approval gates,
    fingerprints, cache invalidation and output contracts. Expensive tools sit
    behind injected stage handlers or command adapters.
    """

    def __init__(
        self,
        config: PipelineConfig,
        *,
        handlers: Mapping[str, StageHandler] | None = None,
    ) -> None:
        self.config = config
        from .handlers import build_default_handlers

        self.handlers: dict[str, StageHandler] = build_default_handlers(config)
        self.handlers.update(handlers or {})

    @classmethod
    def open(
        cls,
        run_dir: str | Path,
        *,
        handlers: Mapping[str, StageHandler] | None = None,
    ) -> "SitePipeline":
        root = Path(run_dir).expanduser().resolve()
        config_path = root / "inputs/pipeline_config.json"
        if not config_path.is_file():
            raise FileNotFoundError(config_path)
        return cls(PipelineConfig.from_json(config_path), handlers=handlers)

    def initialize(
        self,
        corpus_root: str | Path,
        run_dir: str | Path,
        *,
        metadata: str | Path | None = None,
        force: bool = False,
    ) -> PipelineResult:
        root = Path(run_dir).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        inputs = root / "inputs"
        inputs.mkdir(parents=True, exist_ok=True)
        config_path = inputs / "pipeline_config.json"
        if config_path.exists() and not force:
            existing = PipelineConfig.from_json(config_path)
            if existing != self.config:
                raise RuntimeError("run already exists with a different pipeline configuration")
        write_json(config_path, self.config.as_dict())

        manifest = discover_corpus(corpus_root, self.config)
        manifest_path = write_corpus_manifest(inputs / "corpus_manifest.json", manifest)
        metadata_path = inputs / "metadata.csv"
        if metadata is None:
            if not metadata_path.exists():
                write_metadata_template(metadata_path, manifest["sources"])
        else:
            source_metadata = Path(metadata).expanduser().resolve()
            if not source_metadata.is_file():
                raise FileNotFoundError(source_metadata)
            shutil.copyfile(source_metadata, metadata_path)

        state_path = inputs / "run_state.json"
        write_json(
            state_path,
            {
                "schema_version": 2,
                "site_name": self.config.site_name,
                "corpus_root": str(Path(corpus_root).expanduser().resolve()),
                "metadata": str(metadata_path),
                "run_id": root.name,
            },
        )
        store = PipelineStore(root / "ledger.sqlite").initialize()
        for source in manifest["sources"]:
            self._register_source(store, source)

        outputs = (manifest_path, metadata_path)
        fingerprint = _json_hash(
            {
                "config": self.config.as_dict(),
                "manifest": _file_fingerprint(manifest_path),
                "metadata": _file_fingerprint(metadata_path),
                "implementation": _implementation_fingerprint(),
            }
        )
        receipt = self._write_stage_receipt(
            root,
            "stage00_inventory",
            fingerprint,
            outputs,
            details={"source_count": len(manifest["sources"])},
        )
        store.record_stage(root.name, "stage00_inventory", {"fingerprint": fingerprint})
        status = "READY" if self._metadata_complete(root) else "WAITING_FOR_METADATA"
        return self._finish(
            root,
            status,
            (StageResult("stage00_inventory", "COMPLETED", receipt, outputs),),
        )

    def run(
        self,
        run_dir: str | Path,
        *,
        from_stage: str | None = None,
        to_stage: str | None = None,
        force: bool = False,
    ) -> PipelineResult:
        root = Path(run_dir).expanduser().resolve()
        self._validate_run(root)
        corpus_fingerprint = self._validate_corpus_integrity(root)
        selected = _stage_slice(from_stage, to_stage)
        results: list[StageResult] = []
        metadata_complete = self._metadata_complete(root)

        for stage_name in selected:
            if stage_name == "stage00_inventory":
                state = json.loads((root / "inputs/run_state.json").read_text(encoding="utf-8"))
                refreshed = self.initialize(state["corpus_root"], root, force=force)
                results.extend(refreshed.stages)
                metadata_complete = self._metadata_complete(root)
                continue
            spec = STAGE_BY_NAME[stage_name]
            if spec.requires_metadata and not metadata_complete:
                return self._finish(root, "WAITING_FOR_METADATA", tuple(results))
            if spec.approval_gate:
                self._assert_final_approval(root)
            if CANONICAL_STAGES_V2.index(stage_name) >= 3:
                try:
                    self._assert_holdout_isolation(root)
                except StageBlocked as error:
                    results.append(StageResult(stage_name, "BLOCKED", None, (), str(error)))
                    return self._finish(root, "BLOCKED", tuple(results))

            outputs = tuple(root / value for value in spec.outputs)
            inputs = self._stage_inputs(root, stage_name)
            fingerprint = self._stage_fingerprint(
                root,
                stage_name,
                inputs,
                corpus_fingerprint=corpus_fingerprint,
            )
            receipt = root / "receipts" / f"{stage_name}.json"
            if not force and _receipt_matches(
                receipt,
                fingerprint,
                outputs=outputs,
                stage_name=stage_name,
            ):
                results.append(StageResult(stage_name, "CACHED", receipt, outputs))
                continue
            handler = self.handlers.get(stage_name) or self._default_handler(stage_name)
            if handler is None:
                reason = f"no production adapter/handler configured for {stage_name}"
                results.append(StageResult(stage_name, "BLOCKED", None, (), reason))
                return self._finish(root, "BLOCKED", tuple(results))
            context = StageContext(root, stage_name, self.config, outputs, fingerprint, inputs)
            started = time.time()
            try:
                outcome = handler(context)
            except StageBlocked as error:
                failure = self._write_failure_receipt(
                    root, stage_name, fingerprint, error, blocked=True
                )
                results.append(StageResult(stage_name, "BLOCKED", failure, (), str(error)))
                return self._finish(root, "BLOCKED", tuple(results))
            except (OSError, RuntimeError, ValueError) as error:
                failure = self._write_failure_receipt(
                    root, stage_name, fingerprint, error, blocked=False
                )
                results.append(StageResult(stage_name, "FAILED", failure, (), str(error)))
                return self._finish(root, "BLOCKED", tuple(results))
            absent = [str(path) for path in outputs if not path.exists()]
            if absent:
                error = RuntimeError(
                    f"stage {stage_name} did not produce contracted outputs: {absent}"
                )
                failure = self._write_failure_receipt(
                    root, stage_name, fingerprint, error, blocked=False
                )
                results.append(StageResult(stage_name, "FAILED", failure, (), str(error)))
                return self._finish(root, "BLOCKED", tuple(results))
            details = dict(outcome.details)
            details["runtime_seconds"] = time.time() - started
            receipt = self._write_stage_receipt(
                root, stage_name, fingerprint, outputs, details=details
            )
            PipelineStore(root / "ledger.sqlite").record_stage(
                root.name, stage_name, {"fingerprint": fingerprint, **details}
            )
            results.append(StageResult(stage_name, "COMPLETED", receipt, outputs))

            if stage_name == "stage11_reinforcement":
                decision = json.loads(
                    (root / "decisions/final_build_decision.json").read_text(encoding="utf-8")
                )
                status = (
                    "APPROVAL_REQUIRED" if decision.get("approval_allowed", True) else "BLOCKED"
                )
                return self._finish(root, status, tuple(results))

        if not metadata_complete:
            status = "WAITING_FOR_METADATA"
        elif selected and selected[-1] == "stage11_reinforcement":
            status = "APPROVAL_REQUIRED"
        else:
            status = "COMPLETED"
        return self._finish(root, status, tuple(results))

    def approve_final(
        self,
        run_dir: str | Path,
        decision_sha: str,
        *,
        approver: str,
    ) -> Path:
        root = Path(run_dir).expanduser().resolve()
        decision = root / "decisions/final_build_decision.json"
        if not decision.is_file():
            raise FileNotFoundError(decision)
        actual = _sha256_file(decision)
        if decision_sha != actual:
            raise ValueError("decision_sha does not match final_build_decision.json")
        if not approver.strip():
            raise ValueError("approver must not be empty")
        decision_payload = json.loads(decision.read_text(encoding="utf-8"))
        if not decision_payload.get("approval_allowed", False) or decision_payload.get("issues"):
            raise ApprovalRequired("Final decision is blocked and cannot be approved")
        self._assert_decision_inputs(root, decision_payload)
        approval = root / "approvals/final.json"
        write_json(
            approval,
            {
                "schema_version": 2,
                "artifact_type": "FINAL_MAPPING_APPROVAL",
                "decision_sha256": actual,
                "config_sha256": _sha256_file(root / "inputs/pipeline_config.json"),
                "approver": approver.strip(),
                "approved_at_unix": time.time(),
            },
        )
        store = PipelineStore(root / "ledger.sqlite")
        store.add_approval(
            root.name,
            actual,
            approver.strip(),
            json.loads(approval.read_text(encoding="utf-8")),
        )
        return approval

    def status(self, run_dir: str | Path) -> dict[str, Any]:
        root = Path(run_dir).expanduser().resolve()
        self._validate_run(root)
        stages = []
        for name in CANONICAL_STAGES_V2:
            receipt = root / "receipts" / f"{name}.json"
            stages.append(
                {
                    "name": name,
                    "status": "COMPLETED" if receipt.is_file() else "PENDING",
                    "receipt": str(receipt) if receipt.is_file() else None,
                }
            )
        approval = root / "approvals/final.json"
        decision = root / "decisions/final_build_decision.json"
        approval_valid = False
        decision_payload: dict[str, Any] = {}
        if decision.is_file():
            try:
                decision_payload = json.loads(decision.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                decision_payload = {}
        if approval.is_file() and decision.is_file():
            try:
                approval_valid = json.loads(approval.read_text(encoding="utf-8")).get(
                    "decision_sha256"
                ) == _sha256_file(decision)
            except json.JSONDecodeError:
                approval_valid = False
        return {
            "run_dir": str(root),
            "metadata_complete": self._metadata_complete(root),
            "approval_valid": approval_valid,
            "approval_allowed": decision_payload.get("approval_allowed"),
            "decision_issues": decision_payload.get("issues") or [],
            "stages": stages,
        }

    def export(self, run_dir: str | Path) -> PipelineResult:
        return self.run(
            run_dir,
            from_stage="stage15_publish",
            to_stage="stage15_publish",
            force=True,
        )

    def _register_source(self, store: PipelineStore, source: Mapping[str, Any]) -> None:
        source_id = str(source["source_id"])
        sha = str(source.get("sha256") or source.get("content_fingerprint") or source_id)
        try:
            store.add_raw_source(
                source_id,
                str(source["path"]),
                sha,
                dict(source),
                source_kind=str(source.get("source_kind") or "video"),
                evaluation_role=(
                    None
                    if source.get("evaluation_role") is None
                    else str(source["evaluation_role"])
                ),
            )
        except sqlite3.IntegrityError:
            existing = store.connection.execute(
                "SELECT source_uri, sha256, payload FROM raw_sources WHERE video_id=?",
                (source_id,),
            ).fetchone()
            canonical = json.dumps(dict(source), sort_keys=True, separators=(",", ":"), default=str)
            if (
                existing is None
                or existing["source_uri"] != str(source["path"])
                or existing["sha256"] != sha
                or existing["payload"] != canonical
            ):
                raise

    def _default_handler(self, stage_name: str) -> StageHandler | None:
        if stage_name == "stage15_publish":
            return self._publish
        return None

    def _publish(self, context: StageContext) -> StageOutcome:
        base_geometry, rejection, weak_markdown, selection, final_receipt = context.expected_outputs
        dense_model = context.run_dir / "artifacts/mapping/final/model"
        robust_model = context.run_dir / "artifacts/mapping/robust/model"
        robust_receipt = context.run_dir / "receipts/robust_filter.json"
        if robust_model.exists() and not robust_receipt.is_file():
            raise RuntimeError("robust model exists without its filter receipt")
        release_model = robust_model if robust_receipt.is_file() else dense_model
        if not release_model.exists():
            raise RuntimeError("Final mapping product is absent")
        for required in (
            context.run_dir / "products/candidate_pool/manifest.jsonl",
            context.run_dir / "products/localization_reference/manifest.jsonl",
            context.run_dir / "products/weak_region_reshoot_plan.json",
        ):
            if not required.is_file():
                raise RuntimeError(f"required product input is absent: {required}")
        base_geometry.parent.mkdir(parents=True, exist_ok=True)
        if not base_geometry.exists() and not base_geometry.is_symlink():
            base_geometry.symlink_to(release_model.resolve(), target_is_directory=True)
        elif base_geometry.is_symlink() and base_geometry.resolve() != release_model.resolve():
            raise RuntimeError("base-geometry product points to the wrong map layer")
        rejection.parent.mkdir(parents=True, exist_ok=True)
        roles = context.run_dir / "artifacts/selection/roles.jsonl"
        rows: list[dict[str, Any]] = []
        if roles.is_file():
            for line in roles.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rows.append(json.loads(line))
        rejection.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                for row in rows
                if row.get("post_sfm_role") == "REJECT"
            ),
            encoding="utf-8",
        )
        weak_plan_path = context.run_dir / "products/weak_region_reshoot_plan.json"
        weak_plan = (
            json.loads(weak_plan_path.read_text(encoding="utf-8"))
            if weak_plan_path.is_file()
            else {"required": False, "issues": [], "recommendations": []}
        )
        weak_markdown.write_text(
            "# Weak-region / Reshoot Plan\n\n"
            f"Required: **{str(bool(weak_plan.get('required'))).lower()}**\n\n"
            "## Issues\n\n"
            + "".join(f"- {item}\n" for item in weak_plan.get("issues") or ["None"])
            + "\n## Recommendations\n\n"
            + "".join(f"- {item}\n" for item in weak_plan.get("recommendations") or ["None"]),
            encoding="utf-8",
        )
        fields = [
            "Video",
            "Segment",
            "Pre-SfM role",
            "Post-SfM role",
            "Base map",
            "Localization",
            "Risk",
            "Reason",
        ]
        selection.parent.mkdir(parents=True, exist_ok=True)
        with selection.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(
                {
                    "Video": row.get("video", ""),
                    "Segment": row.get("segment_id") or row.get("segment", ""),
                    "Pre-SfM role": row.get("pre_sfm_role", ""),
                    "Post-SfM role": row.get("post_sfm_role", ""),
                    "Base map": row.get("base_map", ""),
                    "Localization": row.get("localization", ""),
                    "Risk": row.get("risk", ""),
                    "Reason": row.get("reason", ""),
                }
                for row in rows
            )
        localization_path = context.run_dir / "artifacts/localization/validation.json"
        localization = (
            json.loads(localization_path.read_text(encoding="utf-8"))
            if localization_path.is_file()
            else {}
        )
        success_rates = [
            float(row.get("success", False)) for row in localization.get("results") or ()
        ]
        localization_rate = None if not success_rates else sum(success_rates) / len(success_rates)
        final_diagnosis_path = context.run_dir / "artifacts/diagnosis/final_map.json"
        final_diagnosis = (
            json.loads(final_diagnosis_path.read_text(encoding="utf-8"))
            if final_diagnosis_path.is_file()
            else {}
        )
        diagnosis_warnings = set(final_diagnosis.get("warnings") or ())
        geometry_blocked = bool(
            diagnosis_warnings
            & {
                "LOO_STABILITY_WARNING",
                "CYCLE_INCONSISTENT",
                "MULTIMODAL_ALIGNMENT",
                "STRUCTURAL_INVALID",
            }
        )
        if geometry_blocked:
            status = "GEOMETRY_REPAIR_REQUIRED"
            write_json(
                context.run_dir / "approvals/INVALIDATED_BY_FINAL_DIAGNOSIS.json",
                {
                    "schema_version": 2,
                    "reason": "final_geometry_diagnosis_failed",
                    "warnings": sorted(diagnosis_warnings),
                    "prior_approval": str(context.run_dir / "approvals/final.json"),
                },
            )
        elif localization_rate is None:
            status = "MAP_SCREENED_LOCALIZATION_UNCHECKED"
        elif localization_rate >= 0.95:
            status = "READY"
        else:
            status = "MAP_READY_LOCALIZATION_BELOW_TARGET"
        write_json(
            final_receipt,
            {
                "schema_version": 2,
                "artifact_type": "GRAPH_AWARE_SITE_PRODUCTS",
                "config_sha256": _sha256_file(context.run_dir / "inputs/pipeline_config.json"),
                "decision_sha256": _sha256_file(
                    context.run_dir / "decisions/final_build_decision.json"
                ),
                "status": status,
                "localization_success_rate": localization_rate,
                "geometry_warnings": sorted(diagnosis_warnings),
                "base_geometry_layer": (
                    "robust" if robust_receipt.is_file() else "dense_unfiltered"
                ),
                "products": {
                    "base_geometry": str(base_geometry),
                    "localization_reference": str(
                        context.run_dir / "products/localization_reference"
                    ),
                    "candidate_pool": str(context.run_dir / "products/candidate_pool"),
                    "dense_localization_layer": str(
                        context.run_dir / "artifacts/mapping/final/model"
                    ),
                    "localization_ensemble": str(
                        context.run_dir / "products/localization_ensemble"
                    ),
                    "rejection_manifest": str(rejection),
                    "weak_region_reshoot_plan": str(weak_plan_path),
                    "selection_manifest": str(selection),
                },
            },
        )
        return StageOutcome(context.expected_outputs, {"selection_rows": len(rows)})

    def _metadata_complete(self, root: Path) -> bool:
        path = root / "inputs/metadata.csv"
        if not path.is_file():
            return False
        try:
            rows = list(csv.DictReader(path.open(encoding="utf-8")))
        except (OSError, csv.Error):
            return False
        material = [row for row in rows if row.get("source_kind") != "archive"]
        if not material:
            return False
        fields_complete = all(
            all(str(row.get(field) or "").strip() for field in self.config.required_metadata)
            for row in material
        )
        if not fields_complete:
            return False
        from .intrinsics import validate_intrinsics_group

        groups: dict[str, list[dict[str, str]]] = {}
        for row in material:
            groups.setdefault(str(row["intrinsics_group"]), []).append(row)
        try:
            for group in groups.values():
                validate_intrinsics_group(
                    group,
                    allow_unknown_metadata=self.config.allow_unknown_metadata,
                )
        except (KeyError, TypeError, ValueError):
            return False
        return True

    def _assert_final_approval(self, root: Path) -> None:
        decision = root / "decisions/final_build_decision.json"
        approval = root / "approvals/final.json"
        if not decision.is_file() or not approval.is_file():
            raise ApprovalRequired("Final mapping requires a human approval receipt")
        try:
            payload = json.loads(approval.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ApprovalRequired("Final approval receipt is invalid JSON") from error
        if payload.get("decision_sha256") != _sha256_file(decision):
            raise ApprovalRequired("Final approval does not match the current decision")
        if payload.get("config_sha256") != _sha256_file(root / "inputs/pipeline_config.json"):
            raise ApprovalRequired("Final approval does not match the current configuration")
        decision_payload = json.loads(decision.read_text(encoding="utf-8"))
        if not decision_payload.get("approval_allowed", False) or decision_payload.get("issues"):
            raise ApprovalRequired("Final decision is blocked")
        self._assert_decision_inputs(root, decision_payload)

    def _assert_holdout_isolation(self, root: Path) -> None:
        manifest = json.loads((root / "inputs/corpus_manifest.json").read_text(encoding="utf-8"))
        holdouts = [
            source
            for source in manifest.get("sources") or ()
            if source.get("evaluation_role") == "HOLDOUT"
        ]
        holdout_ids = {
            str(value)
            for source in holdouts
            for value in (source.get("source_id"), source.get("video_id"))
            if value
        }
        holdout_paths = [Path(str(source["path"])).resolve() for source in holdouts]
        keyframe_path = root / "artifacts/keyframes/keyframes.jsonl"
        if not keyframe_path.is_file():
            raise StageBlocked("mapping keyframe manifest is missing")
        keyframes = [
            json.loads(line)
            for line in keyframe_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        known_ids = {str(row.get("keyframe_id")) for row in keyframes}
        segment_ids = {str(row.get("segment_id")) for row in keyframes}
        leaked = []
        for row in keyframes:
            if row.get("evaluation_role") == "HOLDOUT" or str(row.get("video_id")) in holdout_ids:
                leaked.append(str(row.get("keyframe_id")))
                continue
            source = row.get("source_uri") or row.get("image_uri")
            if source:
                candidate = Path(str(source)).resolve()
                if any(candidate == path or path in candidate.parents for path in holdout_paths):
                    leaked.append(str(row.get("keyframe_id")))
        if leaked:
            raise StageBlocked(f"hold-out keyframes leaked into mapping: {sorted(leaked)[:10]}")
        for relative in (
            "artifacts/retrieval/candidates.jsonl",
            "artifacts/pairs/geometry.jsonl",
        ):
            path = root / relative
            if not path.is_file():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if (
                    str(row.get("image_i")) not in known_ids
                    or str(row.get("image_j")) not in known_ids
                ):
                    raise StageBlocked(
                        f"pair artifact references a non-mapping keyframe: {relative}"
                    )
        roles_path = root / "artifacts/selection/roles.jsonl"
        if roles_path.is_file():
            for line in roles_path.read_text(encoding="utf-8").splitlines():
                if line.strip() and str(json.loads(line).get("segment_id")) not in segment_ids:
                    raise StageBlocked("role artifact references a non-mapping segment")
        for relative in (
            "artifacts/selection/diagnostic_selection.json",
            "decisions/final_selection.json",
        ):
            path = root / relative
            if not path.is_file():
                continue
            selection = json.loads(path.read_text(encoding="utf-8"))
            unknown_keyframes = set(map(str, selection.get("selected_keyframes") or ())) - known_ids
            unknown_segments = set(map(str, selection.get("active_segments") or ())) - segment_ids
            if unknown_keyframes or unknown_segments:
                raise StageBlocked(f"selection references non-mapping data: {relative}")

    def _assert_decision_inputs(self, root: Path, decision: Mapping[str, Any]) -> None:
        known = {
            "final_selection": root / "decisions/final_selection.json",
            "roles": root / "artifacts/selection/roles.jsonl",
            "keyframes": root / "artifacts/keyframes/keyframes.jsonl",
            "pair_geometry": root / "artifacts/pairs/geometry.jsonl",
            "diagnostic_selection": root / "artifacts/selection/diagnostic_selection.json",
            "metadata": root / "inputs/metadata.csv",
            "corpus_manifest": root / "inputs/corpus_manifest.json",
            "post_sfm_diagnosis": root / "artifacts/diagnosis/post_sfm.json",
        }
        expected = dict(decision.get("input_hashes") or {})
        required_names = set(known)
        if set(expected) != required_names:
            missing = sorted(required_names - set(expected))
            extra = sorted(set(expected) - required_names)
            raise ApprovalRequired(
                f"Final decision input hash set is incomplete: missing={missing}, extra={extra}"
            )
        for name, digest in expected.items():
            path = known.get(str(name))
            if path is None or not path.is_file() or _sha256_file(path) != str(digest):
                raise ApprovalRequired(f"Final decision input changed: {name}")

    def _stage_inputs(self, root: Path, stage_name: str) -> tuple[Path, ...]:
        index = CANONICAL_STAGES_V2.index(stage_name)
        paths = [root / "inputs/pipeline_config.json", root / "inputs/corpus_manifest.json"]
        metadata = root / "inputs/metadata.csv"
        if metadata.exists():
            paths.append(metadata)
        for previous in STAGE_SPECS[:index]:
            for output in previous.outputs:
                candidate = root / output
                if candidate.exists():
                    paths.append(candidate)
        return tuple(paths)

    def _stage_fingerprint(
        self,
        root: Path,
        stage_name: str,
        inputs: Sequence[Path],
        *,
        corpus_fingerprint: str,
    ) -> str:
        stage_index = CANONICAL_STAGES_V2.index(stage_name)
        return _json_hash(
            {
                "stage": stage_name,
                "config": self.config.as_dict(),
                "inputs": [_path_fingerprint(path) for path in inputs],
                "corpus_fingerprint": corpus_fingerprint,
                "implementation": _implementation_fingerprint(),
                "keyframe_artifacts": (
                    _keyframe_references_fingerprint(root / "artifacts/keyframes/keyframes.jsonl")
                    if stage_index >= 3
                    else None
                ),
                "approval": (
                    _path_fingerprint(root / "approvals/final.json") if stage_index >= 12 else None
                ),
            }
        )

    def _write_stage_receipt(
        self,
        root: Path,
        stage_name: str,
        fingerprint: str,
        outputs: Sequence[Path],
        *,
        details: Mapping[str, Any],
    ) -> Path:
        receipt = root / "receipts" / f"{stage_name}.json"
        write_json(
            receipt,
            {
                "schema_version": 2,
                "artifact_type": "SITE_PIPELINE_STAGE_RECEIPT",
                "stage": stage_name,
                "fingerprint": fingerprint,
                "outputs": [str(path) for path in outputs],
                "implementation_fingerprint": _implementation_fingerprint(),
                "outputs_fingerprint": _outputs_fingerprint(outputs),
                "referenced_artifacts": _referenced_artifacts_fingerprint(stage_name, outputs),
                "details": dict(details),
            },
        )
        return receipt

    def _write_failure_receipt(
        self,
        root: Path,
        stage_name: str,
        fingerprint: str,
        error: Exception,
        *,
        blocked: bool,
    ) -> Path:
        receipt = root / "receipts" / f"{stage_name}.failed.json"
        payload = {
            "schema_version": 2,
            "artifact_type": "SITE_PIPELINE_STAGE_FAILURE",
            "stage": stage_name,
            "fingerprint": fingerprint,
            "implementation_fingerprint": _implementation_fingerprint(),
            "status": "BLOCKED" if blocked else "FAILED",
            "error_type": type(error).__name__,
            "reason": str(error),
        }
        write_json(receipt, payload)
        PipelineStore(root / "ledger.sqlite").record_stage(root.name, stage_name, payload)
        return receipt

    def _validate_run(self, root: Path) -> None:
        if not (root / "inputs/pipeline_config.json").is_file():
            raise FileNotFoundError(root / "inputs/pipeline_config.json")
        stored = PipelineConfig.from_json(root / "inputs/pipeline_config.json")
        if stored != self.config:
            raise RuntimeError("pipeline configuration differs from the initialized run")

    def _validate_corpus_integrity(self, root: Path) -> str:
        manifest_path = root / "inputs/corpus_manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"cannot validate immutable corpus: {manifest_path}") from error
        sources = manifest.get("sources")
        if not isinstance(sources, list) or not sources:
            raise RuntimeError("cannot validate immutable corpus: source manifest is empty")
        verified: list[dict[str, Any]] = []
        for row in sources:
            if not isinstance(row, dict) or not row.get("path"):
                raise RuntimeError("cannot validate immutable corpus: invalid source row")
            path = Path(str(row["path"]))
            if path.is_symlink():
                raise RuntimeError(f"immutable corpus source changed: {path} became a symlink")
            source_kind = str(row.get("source_kind") or "")
            expected_sha = str(row.get("sha256") or "")
            expected_size = int(row.get("size") or 0)
            if source_kind == "image_sequence":
                if not path.is_dir():
                    raise RuntimeError(f"immutable corpus source changed: missing {path}")
                actual_sha, actual_count = image_sequence_fingerprint(
                    path, hash_contents=self.config.hash_contents
                )
                expected_count = int(row.get("file_count") or 0)
                if expected_count and actual_count != expected_count:
                    raise RuntimeError(
                        f"immutable corpus source changed: file count differs for {path}"
                    )
                actual_size = sum(
                    child.stat().st_size
                    for child in path.iterdir()
                    if child.is_file() and child.suffix.lower() in IMAGE_SUFFIXES
                )
            else:
                if not path.is_file():
                    raise RuntimeError(f"immutable corpus source changed: missing {path}")
                actual_size = path.stat().st_size
                actual_sha = (
                    _sha256_file(path)
                    if self.config.hash_contents
                    else _json_hash({"size": actual_size, "mtime_ns": path.stat().st_mtime_ns})
                )
            if expected_size != actual_size:
                raise RuntimeError(f"immutable corpus source changed: size differs for {path}")
            if expected_sha and actual_sha != expected_sha:
                raise RuntimeError(f"immutable corpus source changed: SHA256 differs for {path}")
            verified.append(
                {
                    "source_id": row.get("source_id"),
                    "path": str(path),
                    "size": actual_size,
                    "sha256": actual_sha,
                }
            )
        return _json_hash(verified)

    def _finish(
        self,
        root: Path,
        status: str,
        stages: tuple[StageResult, ...],
    ) -> PipelineResult:
        receipt = root / "pipeline_receipt.json"
        write_json(
            receipt,
            {
                "schema_version": 2,
                "artifact_type": "GRAPH_AWARE_SITE_PIPELINE",
                "site_name": self.config.site_name,
                "status": status,
                "stages": [
                    {
                        "name": stage.name,
                        "status": stage.status,
                        "receipt": str(stage.receipt) if stage.receipt else None,
                        "outputs": [str(path) for path in stage.outputs],
                        "reason": stage.reason,
                    }
                    for stage in stages
                ],
            },
        )
        return PipelineResult(root, status, stages, receipt)


def _stage_slice(from_stage: str | None, to_stage: str | None) -> tuple[str, ...]:
    names = list(CANONICAL_STAGES_V2)
    if from_stage is not None and from_stage not in names:
        raise ValueError(f"unknown from_stage {from_stage!r}")
    if to_stage is not None and to_stage not in names:
        raise ValueError(f"unknown to_stage {to_stage!r}")
    start = names.index(from_stage) if from_stage is not None else 1
    stop = names.index(to_stage) + 1 if to_stage is not None else len(names)
    if start >= stop:
        raise ValueError("from_stage must not come after to_stage")
    return tuple(names[start:stop])


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "sha256": _sha256_file(path)}


def _path_fingerprint(path: Path) -> dict[str, Any]:
    if path.is_file():
        return _file_fingerprint(path)
    if path.is_dir():
        rows = [_file_fingerprint(child) for child in sorted(path.rglob("*")) if child.is_file()]
        return {"path": str(path), "type": "directory", "files": rows}
    return {"path": str(path), "type": "missing"}


def _implementation_fingerprint() -> str:
    package_root = Path(__file__).resolve().parent
    sources = [
        {
            "path": path.relative_to(package_root).as_posix(),
            "sha256": _sha256_file(path),
        }
        for path in sorted(package_root.rglob("*.py"))
        if path.is_file()
    ]
    return _json_hash(sources)


def _outputs_fingerprint(outputs: Sequence[Path]) -> str:
    return _json_hash([_path_fingerprint(path) for path in outputs])


def _keyframe_references_fingerprint(manifest: Path) -> dict[str, Any]:
    if not manifest.is_file():
        return {"count": 0, "fingerprint": _json_hash([]), "artifacts": []}
    entries: dict[str, dict[str, Any]] = {}
    with manifest.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{manifest}:{line_number} is not a JSON object")
            value = row.get("image_uri")
            if not value:
                continue
            image = Path(str(value)).expanduser()
            if not image.is_absolute():
                image = manifest.parent / image
            image = image.resolve()
            actual = _file_fingerprint(image)
            declared = str(row.get("image_sha256") or "")
            if declared and actual["sha256"] != declared:
                raise ValueError(f"keyframe image hash mismatch: {image}")
            previous = entries.get(str(image))
            if previous is not None and previous != actual:
                raise ValueError(f"conflicting keyframe image reference: {image}")
            entries[str(image)] = actual
    ordered = [entries[key] for key in sorted(entries)]
    return {
        "count": len(ordered),
        "fingerprint": _json_hash(ordered),
        "artifacts": ordered,
    }


def _referenced_artifacts_fingerprint(stage_name: str, outputs: Sequence[Path]) -> dict[str, Any]:
    if stage_name != "stage02_segment_keyframes":
        return {"count": 0, "fingerprint": _json_hash([]), "artifacts": []}
    manifest = next((path for path in outputs if path.name == "keyframes.jsonl"), None)
    if manifest is None:
        raise ValueError("Stage 2 receipt requires keyframes.jsonl")
    return _keyframe_references_fingerprint(manifest)


def _json_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def _receipt_matches(
    path: Path,
    fingerprint: str,
    *,
    outputs: Sequence[Path],
    stage_name: str,
) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return bool(
            payload.get("fingerprint") == fingerprint
            and payload.get("implementation_fingerprint") == _implementation_fingerprint()
            and payload.get("outputs_fingerprint") == _outputs_fingerprint(outputs)
            and payload.get("referenced_artifacts")
            == _referenced_artifacts_fingerprint(stage_name, outputs)
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


__all__ = [
    "ApprovalRequired",
    "PipelineResult",
    "SitePipeline",
    "StageBlocked",
    "StageContext",
    "StageOutcome",
    "StageResult",
]
