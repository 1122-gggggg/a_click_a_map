"""Desktop route previews must not invent continuity through lost poses."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gui"))

import jobs


def test_trajectory_sampling_keeps_gaps_and_last_pose(tmp_path: Path) -> None:
    count = 4097
    (tmp_path / "summary.json").write_text(json.dumps({"dense_frames": count, "pose_coverage": 0.999}))
    with (tmp_path / "frames.jsonl").open("w", encoding="utf-8") as stream:
        for index in range(count):
            state = "NO_POSE" if index == 1 else "FAST_TRACK"
            position = None if index in (2050, 4093) else [index, 0, 0]
            stream.write(json.dumps({"position": position, "status": state}) + "\n")

    result = jobs.cruise_result({"result_dir": str(tmp_path)})
    assert result is not None
    points = result["points"]
    segments = {pos[0]: segment for pos, _, segment in points}
    assert len(points) <= 2001
    assert 1 not in segments and 2050 not in segments and 4093 not in segments
    assert segments[0] != segments[3] == segments[2049]
    assert segments[2049] != segments[2052] == segments[4092]
    assert segments[4092] != segments[4095] == segments[4096]
    assert points[-1][0] == [4096, 0, 0]
    assert result["coverage"] == 0.999
