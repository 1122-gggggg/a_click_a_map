"""Post-mapping report (runs in the engine environment: needs pycolmap).

    python -m acam.report RUN_DIR [--min-shared 50]

Checks what "the map built" does not prove: every video registered, and the videos are
connected to each other through shared 3D points (otherwise they may sit in unrelated
frames or only appear aligned by luck).
"""
from __future__ import annotations

import argparse
import collections
import itertools
import json
from pathlib import Path


def video_components(shared: dict[tuple[str, str], int], videos: list[str], min_shared: int) -> list[list[str]]:
    parent = {v: v for v in videos}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for (a, b), count in shared.items():
        if count >= min_shared:
            parent[find(a)] = find(b)
    groups: dict[str, list[str]] = collections.defaultdict(list)
    for video in videos:
        groups[find(video)].append(video)
    return sorted((sorted(g) for g in groups.values()), key=lambda g: (-len(g), g))


def build_report(run: Path, min_shared: int = 50) -> dict:
    import pycolmap

    keyframes = [json.loads(line) for line in (run / "artifacts/keyframes/keyframes.jsonl").read_text().splitlines() if line.strip()]
    per_video_total = collections.Counter(row["video_id"] for row in keyframes)
    source_of = {row["video_id"]: Path(row["source_uri"]).name for row in keyframes}
    pose_only = collections.Counter(row["video_id"] for row in keyframes if row.get("mapping_mode") == "POSE_ONLY")
    model = pycolmap.Reconstruction(str(run / "artifacts/mapping/edm/model"))
    video_of_image = {image_id: image.name.split("/")[0] for image_id, image in model.images.items()}
    registered = collections.Counter(video_of_image.values())
    shared: dict[tuple[str, str], int] = collections.Counter()
    for point in model.points3D.values():
        videos = sorted({video_of_image[element.image_id] for element in point.track.elements})
        for pair in itertools.combinations(videos, 2):
            shared[pair] += 1
    videos = sorted(per_video_total)
    components = video_components(shared, videos, min_shared)
    receipt = json.loads((run / "products/FINAL_RECEIPT.json").read_text())
    return {
        "videos": {
            video: {
                "source": source_of[video],
                "keyframes": per_video_total[video],
                "registered": registered.get(video, 0),
                "pose_only": pose_only.get(video, 0),
            }
            for video in videos
        },
        "images_registered": model.num_images(),
        "images_total": len(keyframes),
        "points3D": model.num_points3D(),
        "megaloc_references": receipt["localization"]["references"],
        "receipt_status": receipt["status"],
        "video_pairs_min_shared": min_shared,
        "video_components": components,
        "connected": len(components) == 1,
        "shared_points_top": [
            {"a": a, "b": b, "points": count}
            for (a, b), count in sorted(shared.items(), key=lambda kv: -kv[1])[:20]
        ],
    }


def to_markdown(report: dict, name: str, mapping: dict) -> str:
    lines = [f"# Map report: {name}", "",
             f"- status: `{report['receipt_status']}` (no external ground truth; native map units, not metres)",
             f"- registered images: {report['images_registered']} / {report['images_total']}",
             f"- EDM 3D points: {report['points3D']:,}; MegaLoc references: {report['megaloc_references']}",
             f"- num_track_per_img used: {mapping['num_track_per_img']}; attempts: {len(mapping['attempts'])}",
             f"- videos connected through shared points (>= {report['video_pairs_min_shared']}): "
             f"**{'yes' if report['connected'] else 'NO'}**", ""]
    if not report["connected"]:
        lines += ["> WARNING: the videos form separate components; they may not share one coordinate frame.", ""]
        lines += [f"> - component: {', '.join(c)}" for c in report["video_components"]] + [""]
    lines += ["| video | source | keyframes | registered | pose-only |", "|---|---|---:|---:|---:|"]
    for video, row in report["videos"].items():
        lines.append(f"| {video} | {row['source']} | {row['keyframes']} | {row['registered']} | {row['pose_only']} |")
    lines += ["", "Attempts:", ""]
    lines += [f"- #{a['attempt']}: K={a['num_track_per_img']} rc={a['returncode']} peak={a['peak_gib']} GiB" for a in mapping["attempts"]]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--min-shared", type=int, default=50)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = build_report(args.run, args.min_shared)
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
