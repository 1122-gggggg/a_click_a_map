# $name: localization + densification bundle

Built by a_click_a_map from $n_videos video(s): GlueMap poses (frozen), then EDM fixed-pose
retriangulation. Receipt status `$status`.

**The map is in native map units: no metric scale, no external ground-truth validation.**

## Contents

| Path | What |
|---|---|
| `localization/edm_model/` | **Map used for localization** (COLMAP .bin; $n_points points, $n_images images) |
| `localization/keyframes/` | `keyframes.jsonl.in` and the $n_images keyframe JPEGs (needed for localization and for densification) |
| `localization/bank/` | MegaLoc retrieval bank ($n_refs references), `reference_manifest.jsonl.in`, `localizer_config.json.in` |
| `localization/intrinsics.json` | calibration the map was built with (undistorted PINHOLE, rescaled per resolution) |
| `megaloc/` | MegaLoc code$megaloc_weights |
| `code/` | `normal_map_pipeline_src`, `river_map_quality_src`, EDM code. **EDM weights are not included** |
| `dense/` | GlueMap pose model (BA points) and the original RGB point cloud |
| `receipts/` | final receipt, preprocessing receipt, mapping/memory report |
| `install.py`, `run_localize.py` | the two adapter scripts |

## Use

```bash
tar --zstd -xf $archive && cd $name
sha256sum -c MANIFEST.sha256                                  # optional
python3 install.py --edm-checkpoint /path/to/edm_outdoor.ckpt # renders the *.in path templates; add --verify to re-hash JPEGs
python3 run_localize.py --video flight.MP4 --route test --output out/
#   or: --frames DIR_OF_960x540_JPEGS --fps 29.97
```

Re-run `install.py` after moving the folder. Needs Python 3.11, a CUDA GPU, `ffmpeg`, and the
packages in `requirements_freeze.txt` (key ones: torch 2.7.1+cu128, pycolmap-cuda12 4.0.4).
The query camera must be the camera/calibration the map was built with.

Localization settings are the validated ones and should not be tuned casually: KLT fast loop
(`track_cap=500`), MegaLoc top-2 + EDM + PnP every 0.3 s, lift radius 4 px, VO + dead reckoning on.

## Densification

- Inputs: `localization/keyframes/images/` + `keyframes.jsonl` (`output_name` = relative image path)
  + a COLMAP model. `edm_model` and `dense/glue_pose_model` share camera and image poses (EDM
  retriangulation keeps poses fixed); they differ only in 3D points.
- Keyframe JPEGs are the source videos' native-resolution frames, treated as undistorted.
- Keyframes with `mapping_mode = POSE_ONLY` (pure rotation) have no triangulated points.
- Not included: source videos, EDM `matches/` and `database.db`. They stay in the mapping run directory.
- Dense output is also in map units.

## Mapping record

$mapping_summary
