# End-to-end smoke test (2026-10-19, author's machine)

Existing engine environment (RTX 5090, 30.5 GiB RAM, torch 2.7.1+cu128, pycolmap 4.0.4);
`./run.sh doctor` all green. Not a from-scratch environment build.

Corpus: three 40 s clips cut from one 2688x1512 video (`P1240124`, 60-100 s, 90-130 s, 120-160 s;
10 s overlaps), run as

```
./run.sh videos/ --out out --name smoke --query query_d.MP4
```

| step | result |
|---|---|
| preprocessing | 139 keyframes -> `num_track_per_img=1024` (2.2 M points, budget 15.4 M) |
| mapping | 1 attempt, rc 0, 14 min wall clock, cgroup peak 18.4 GiB (includes page cache) |
| report | 139/139 registered, 372,469 EDM points, 139 MegaLoc references, 3 videos connected |
| bundle | `smoke_bundle.tar.zst`, 1.13 GiB; `sha256sum -c`, `install.py --verify` (139 JPEG hashes) pass |
| localize, query 200-225 s (outside the mapped 60-160 s) | **0 / 599 poses, all NO_POSE** (the drone is off the map; the loop abstained, no false lock) |
| localize, query 65-85 s (inside the map) | 98.54% pose coverage (431 FAST_TRACK, 41 RELOC_SEED, 7 NO_POSE) |
| same query via the extracted bundle (`install.py` + `run_localize.py`, bundle's own code and MegaLoc) | 98.54% (433 / 39 / 7) |

What this does and does not show:

- It shows that every stage is wired: preprocess -> GlueMap -> EDM -> MegaLoc bank -> report -> bundle -> relocated
  localization. The in-map query re-uses mapped frames, so 98.5% is an integration check, **not** an accuracy or
  generalization number.
- The OOM retry path (`runner.run_map` lowering `num_track_per_img` and moving caches) is covered by unit tests of its
  parts (`transplant_caches`, `reduce_tracks`, `is_oom_exit`) but was **not** triggered in this smoke test.
- Run-to-run counts differ by a few frames (GPU nondeterminism).


## Second smoke test: bad videos and a small memory cap (2026-10-19)

Corpus of 9 files: three good 40 s clips (2688x1512) plus six deliberately bad ones made with ffmpeg
(`blur` sigma 15, `dark`, `static` single frame, `tiny` 320x180, `truncated` 2 MB, `mixed30` = 14 s good + 6 s blur),
run with `--max-keyframes 60` to force the memory-budget path on a machine that could hold far more.

| step | result |
|---|---|
| screening (49 s with 4 workers) | REJECT blur, dark, tiny, truncated; WARN mixed30 (29% poor), static (static camera); OK the 3 clips |
| budget | cap 60 keyframes -> sampling x0.50, 2 more videos dropped for memory (`clip_c`, `static`), 3 of 9 used |
| preprocessing | 56 keyframes (estimate 58) |
| mapping | 1 attempt, rc 0, 6 min; 56/56 registered, 149,235 points, the 3 videos connected |
| report | lists every excluded file with its reason; per-video screening verdict in the table |
| localize (query 65-85 s, inside the map) | 98.54% (427 FAST_TRACK, 45 RELOC_SEED, 7 NO_POSE) from only 56 keyframes |

Also verified separately: the 11 real Wulai videos all screen OK; the engine refuses symlinked sources, so the
staged corpus uses hard links (copies only if `--out` is on another filesystem). A first version used symlinks and
failed on that rule; fixed. Not verified: the post-preprocess re-extraction loop (`rescale_after_preprocess`,
`weak_sources` exclusion) end to end; both have unit tests only.
