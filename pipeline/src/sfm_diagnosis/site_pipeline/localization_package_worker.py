"""Fresh-process MegaLoc reference packaging for a completed direct map."""

from __future__ import annotations

import json
import sys
from contextlib import redirect_stdout
from typing import Any, Mapping

from .direct_mapping import DirectMappingRuntime, build_localization_package


def run_request(payload: Mapping[str, Any]) -> dict[str, Any]:
    runtime = DirectMappingRuntime(**dict(payload["runtime"]))
    result = build_localization_package(
        model_dir=str(payload["model_dir"]),
        keyframes_path=str(payload["keyframes_path"]),
        output_dir=str(payload["output_dir"]),
        runtime=runtime,
        intrinsics=dict(payload["intrinsics"]),
    )
    return {"status": "completed", **result}


def main() -> int:
    output = sys.stdout
    try:
        with redirect_stdout(sys.stderr):
            result = run_request(json.load(sys.stdin))
    except Exception as error:
        output.write(json.dumps({"status": "error", "error": str(error)}) + "\n")
        return 1
    output.write(json.dumps(result) + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["run_request"]
