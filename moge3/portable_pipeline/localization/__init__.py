"""Portable EDM + MegaLoc localization stage helpers (stdlib + numpy only)."""

from localization.bundle import (
    BUNDLE_SCHEMA,
    PROFILE_SCHEMA,
    build_bundle,
    render_readme,
)
from localization.io_utils import (
    read_json,
    read_jsonl,
    sha256_file,
    write_json,
    write_jsonl,
)
from localization.query_calibration import (
    QueryCalibration,
    resolve_query_calibration,
    scale_pinhole,
)
from localization.reference_bank import (
    MEGALOC_DESCRIPTOR_DIM,
    build_reference_bank,
    validate_reference_bank,
)

__all__ = [
    "BUNDLE_SCHEMA",
    "MEGALOC_DESCRIPTOR_DIM",
    "PROFILE_SCHEMA",
    "QueryCalibration",
    "build_bundle",
    "build_reference_bank",
    "read_json",
    "read_jsonl",
    "render_readme",
    "resolve_query_calibration",
    "scale_pinhole",
    "sha256_file",
    "validate_reference_bank",
    "write_json",
    "write_jsonl",
]
