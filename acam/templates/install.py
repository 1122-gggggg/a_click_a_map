#!/usr/bin/env python3
"""Render the path templates of this package for the machine it was extracted on.

    python3 install.py --edm-checkpoint /path/to/edm_outdoor.ckpt [--verify]

Writes every `*.in` template (which holds `@PKG@` / `@EDM_CHECKPOINT@`) next to
itself without the `.in` suffix, then checks that every referenced file exists.
`--verify` also re-hashes all keyframe JPEGs against `image_sha256` (a few minutes).
Safe to run again after moving the package: re-run it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--edm-checkpoint', type=Path, required=True, help='existing EDM weight file on this machine (edm_outdoor.ckpt)')
    parser.add_argument('--verify', action='store_true', help='re-hash every keyframe JPEG')
    args = parser.parse_args()
    checkpoint = args.edm_checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        print(f'EDM checkpoint not found: {checkpoint}', file=sys.stderr)
        return 2
    rendered = []
    for template in sorted(PKG.rglob('*.in')):
        text = template.read_text(encoding='utf-8')
        text = text.replace('@PKG@', PKG.as_posix()).replace('@EDM_CHECKPOINT@', checkpoint.as_posix())
        if '@PKG@' in text or '@EDM_CHECKPOINT@' in text:
            raise RuntimeError(f'unrendered placeholder in {template}')
        target = template.with_suffix('')
        target.write_text(text, encoding='utf-8')
        rendered.append(target.relative_to(PKG).as_posix())

    loc = PKG / 'localization'
    missing = [p for p in (
        *[loc / 'edm_model' / f'{n}.bin' for n in ('cameras', 'images', 'points3D', 'rigs', 'frames')],
        loc / 'bank/megaloc_references.npy', loc / 'bank/megaloc_references.names.json',
        loc / 'bank/reference_manifest.jsonl', loc / 'bank/localizer_config.json',
        loc / 'keyframes/keyframes.jsonl', PKG / 'megaloc/model.safetensors', PKG / 'megaloc/source/megaloc_model.py',
        PKG / 'code/EDM/src', PKG / 'code/normal_map_pipeline_src/sfm_diagnosis',
    ) if not p.exists()]
    rows = [json.loads(line) for line in (loc / 'keyframes/keyframes.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    absent = [r['image_uri'] for r in rows if not Path(r['image_uri']).is_file()]
    bad = []
    if args.verify:
        for row in rows:
            digest = hashlib.sha256(Path(row['image_uri']).read_bytes()).hexdigest()
            if digest != row['image_sha256']:
                bad.append(row['image_uri'])
    receipt = {'package': PKG.as_posix(), 'edm_checkpoint': checkpoint.as_posix(), 'rendered': rendered,
               'keyframes': len(rows), 'missing_files': [p.as_posix() for p in missing],
               'missing_keyframe_images': absent[:20], 'missing_keyframe_images_count': len(absent),
               'sha256_verified': bool(args.verify), 'sha256_mismatch_count': len(bad)}
    (PKG / 'install_receipt.json').write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in receipt.items() if k != 'rendered'}, indent=2, ensure_ascii=False))
    return 0 if not (missing or absent or bad) else 1


if __name__ == '__main__':
    raise SystemExit(main())
