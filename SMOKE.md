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
