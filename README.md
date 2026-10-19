# a_click_a_map

一個指令，從一個資料夾的影片，產生可用 **EDM + KLT** 定位的完整地圖，並打包成可搬到另一台電腦的 bundle。

```bash
./scripts/setup.sh              # 一次：clone 並釘住第三方程式、套 patch、下載權重（EDM 權重需自備）
./run.sh doctor                 # 檢查環境
./run.sh /path/to/videos --out acam_out --query /path/to/query.MP4
```

輸出（`acam_out/<name>/`）：

| 檔案 | 內容 |
|---|---|
| `run/artifacts/mapping/edm/model/` | **定位用地圖**（COLMAP .bin） |
| `run/products/` | MegaLoc 檢索 bank、定位設定、RGB 點雲、`FINAL_RECEIPT.json` |
| `MAP_REPORT.md`、`map_report.json` | 每支影片註冊張數與篩選結果、被排除的影片與原因，**影片之間是否由共享 3D 點連通** |
| `<name>_bundle.tar.zst` | 轉接包：地圖、關鍵影格、檢索 bank、程式、`install.py`、`run_localize.py`、稠密化輸入（EDM 權重不含） |
| `localize_test/` | 若給了 `--query`：EDM+KLT 定位結果 |
| `acam/` | `screening.json`、`excluded.json`、建圖日誌與記憶體軌跡（`logs/memory.csv`，cgroup 記憶體含 page cache） |

## 做了什麼

```
videos ─► 品質篩選 ─► 依本機 RAM 規劃 ─► 前處理（SHA、自適應關鍵影格、純旋轉標為 pose-only）
       ─► GlueMap 全序列一次建圖，姿態凍結（Pi3 + SALAD + VGGSfM）
       ─► EDM 重配對、固定姿態三角化（2 px cell，每格一個觀測）
       ─► MegaLoc 檢索 bank
       ─► 報告（註冊、跨影片連通）─► 打包 ─► 選用：EDM+KLT 定位測試
```

定位迴圈（`localize` 與 bundle 內的 `run_localize.py`）：CPU KLT+PnP 每幀；GPU MegaLoc top-2 + EDM + PnP
每 0.3 秒；lift 半徑 4 px；VO 與 dead reckoning 開。這是已驗證的設定，不要隨意調。

## 輸入影片品質把關（建圖前）

使用者放進來的影片不一定能用，所以先分三層擋掉，再花數小時建圖：

1. **影片層篩選**（`acam/screen.py`，約 1 分鐘）：每支影片均勻取 48 張（不解碼整支），量銳利度（Laplacian 變異數）、
   曝光（過暗／過曝）、紋理（FAST 角點，擋天空／水面／空白牆）、可讀性（截斷／損毀）、解析度、幀率、鏡頭是否靜止。
   - `REJECT`（自動排除）：打不開或 >25% 取樣點讀不出來、短於 5 秒、短邊 < 480、≥60% 取樣影格不可用。
   - `WARN`（保留並寫進報告）：25–60% 影格不佳、幀率 < 8、鏡頭靜止、10–25% 取樣點讀不出來。
   - `OK`。分數（0–1）之後在記憶體不夠、必須挑子集時用來排序。`--keep-bad` 關掉自動排除。
2. **引擎逐影格清理**（原本就有）：損毀、嚴重模糊、全黑／全白、重複影格，純旋轉標為 pose-only。
3. **前處理後的把關**：引擎統計某支影片可用關鍵影格 < 8，或超過一半的分析影格被丟掉 → 排除該影片、重新抽幀。
   另外長寬比與標定不同的影片會被拒絕（一次只支援一組標定相機）。

篩選閾值是保守的（針對「根本無法建圖」的影片，不是「品質普通」的影片），在 11 支真實影片（Wulai／River，
最差的 Laplacian 變異數 1084、角點 2364，遠高於閾值 20／150）與 10 支合成壞片（模糊、過暗、過曝、全灰、靜止、
截斷、320×180、3 秒、30%／60% 混合）上驗證過。**沒有在「品質普通但邊緣」的真實影片上校準過**，這類影片會被放行，
由第 2、3 層與報告處理。

## 影片太多：依這台電腦做記憶體保護

建圖前就依本機 RAM 規劃，而不是跑到一半才 OOM（`acam/plan.py`）：

1. **算上限**：本機能放的關鍵影格數 = 預算點數 ÷ 每張點數（K=512，已驗證設定）。預算點數是 30 GiB 上成功跑完的 15.04 M 點，
   乘上 (RAM−6)/24。30 GiB → 約 2043 張；16 GiB → 約 850 張；64 GiB → 約 4900 張。
2. **全部放得下**：照常，`num_track_per_img` 由預算決定（不低於 512）。
3. **放不下**：先把關鍵影格取樣變稀（下限 0.5 倍），K 固定 512；
4. **還是放不下**：依篩選分數由低到高丟影片，直到放得下；被丟的影片與原因寫進報告。
5. 預估的關鍵影格數（1.15 張/秒）與實際不同時，前處理後會用真實張數再檢查一次，必要時重新抽幀（最多兩次）。
6. 啟動檢查：目前可用 RAM < 10 GiB 會拒絕並列出占記憶體最多的程序（`--force` 可略過）；VRAM < 16 GiB 警告；
   磁碟空間不足拒絕；列出預估時間（每個關鍵影格約 6.5 秒，RTX 5090 的實測值，其他 GPU 不同）。
7. **執行中**：建圖放在 `systemd-run --user --scope` 的記憶體 cgroup（`MemoryMax = RAM − 1.5 GiB`），OOM 只殺這個 scope；
   被殺時自動把 `num_track_per_img` 降 25% 並搬移與它無關的快取後續跑（最多 `--max-attempts` 次）。
   **CUDA（顯存）OOM 不重試**——降低主機記憶體設定沒有幫助，會直接說明原因。
8. 手動覆寫：`--max-keyframes N`、`--sampling-scale S`、`--tracks K`。

預算是由三次實測得到的經驗規則（30 GiB、RTX 5090：1389 張 @1024 完成；2043 張 @1024 在追蹤建立被 OOM 殺掉；
2043 張 @512 完成），**不是模型，也只在 ≤2043 張驗證過**；超過這個規模、或稀疏取樣下建圖品質，沒有驗證。
丟影片可能讓影片之間不再連通，`MAP_REPORT.md` 會檢查並警告。

另外修了兩個 2000+ 張規模才出現的問題（已含在 `pipeline/`）：EDM 配對上限 25,000 → 40,000
（每張約 13.4 對，上限必須隨規模增加）；RGB 預覽 PLY 容許極少數（≤0.01%）觀測全在影像外的點，
並把被略過的點 ID 寫進 receipt，超過比例仍會失敗。

## 限制（請先讀）

- **一次執行只支援同一組標定相機。** 預設內參是河濱相機（1280×720 基準，依解析度等比縮放）。影片長寬比與標定不同會
  被拒絕；換相機請給 `--intrinsics` 或分開建圖。
- 地圖是 **map 原生單位，沒有公尺尺度，沒有真實 GT 驗證**（receipt：`MAP_BUILT_UNVALIDATED_ALL_INPUTS`）。
  定位報告的是位姿覆蓋率，不是精度。
- 需要 NVIDIA GPU（來源為 RTX 5090，sm_120）與約 30 GiB RAM 才能跑 2000 張級的語料。
- 只在作者的機器上、用既有環境端到端驗證過。**沒有從零建立環境的驗證**：`setup.sh` 的 clone、
  釘 commit、套 patch 已實測（patch 可乾淨套用到釘住的上游 commit，結果與作者的 runtime 逐檔相同）；
  `pygluemap`（C++ 擴充，需要 Ceres/Eigen/METIS/Boost）的從零編譯（`setup.sh --build`）沒有測。
  環境版本見 `environment/`。
- `pipeline/` 內非預設的光流追蹤（NeuFlow／SEA-RAFT／LocoTrack／XFeat）程式碼含作者機器的絕對路徑，採用的 KLT 路徑不會用到它們。
- EDM 權重（`edm_outdoor.ckpt`）不會自動下載，請放到 `third_party/EDM/weights/` 或設 `ACAM_EDM_CHECKPOINT`。
- 建圖是長時間作業：2043 張約 3.5 小時 + 重試（見下）。

## 驗證紀錄

- 單元測試（55 個）：`PYTHONPATH=pipeline/src:pipeline/compat_src:. pytest tests pipeline/tests`（預算規劃、品質篩選判定、OOM 判斷、快取搬移、路徑範本、影片連通、引擎 direct mapping／EDM／CLI）。
- 規模：Wulai 9 支影片、2043 關鍵影格，一次 GlueMap + EDM（`num_track_per_img=512`）→ 4,270,546 點、
  2019 個 MegaLoc 參考；holdout 位姿覆蓋率 96.35%（P0170017，獨立 session）/ 99.90%（P0230023，非獨立）。
- 端到端冒煙：見 `SMOKE.md`。

## 設定

路徑解析順序：環境變數 > `acam.toml` > `third_party/`。

| 環境變數 | 預設 |
|---|---|
| `ACAM_PYTHON` | 目前的 `python`（需有 torch+CUDA、pycolmap、gluemap） |
| `ACAM_GLUEMAP` / `ACAM_GLUEMAP_CHECKPOINTS` | `third_party/gluemap` / `…/checkpoints` |
| `ACAM_EDM` / `ACAM_EDM_CHECKPOINT` | `third_party/EDM` / `…/weights/edm_outdoor.ckpt` |
| `ACAM_MEGALOC_SOURCE` / `ACAM_MEGALOC_CHECKPOINT` | `third_party/MegaLoc` / `third_party/checkpoints/megaloc/model.safetensors` |

## 指令

```
./run.sh VIDEOS [--out DIR] [--name N] [--intrinsics K.json] [--tracks auto|256..1024]
               [--max-keyframes N] [--sampling-scale S] [--keep-bad] [--force]
               [--query VIDEO ...] [--no-pack] [--no-megaloc-weights] [--dry-run] [--max-attempts 3]
./run.sh doctor
./run.sh pack RUN_DIR --out DIR --name NAME
./run.sh localize RUN_DIR VIDEO --out DIR
./run.sh setup [--build] [--no-checkpoints]
```

重跑同一個 `--out/--name` 會續跑（階段快取），已完成的建圖直接跳到報告與打包。
結束碼：0 成功且全部註冊並連通；5 建圖完成但有影像未註冊或影片不連通（看 `MAP_REPORT.md`）。

## 結構

```
acam/        orchestrator：config、hw、plan（預算）、screen（品質篩選，需 OpenCV）、runner、report、pack、cli
pipeline/    normal_map_pipeline 原始碼（MIT）與 river_map_quality 相容模組
patches/     gluemap.patch（對 colmap/gluemap adc9e4b）、edm.patch（對 chicleee/EDM 8a31d81）
configs/     GlueMap 基底設定、河濱內參、第三方 commit 與 SHA256
scripts/     setup.sh
environment/ 作者機器的 pip freeze
tests/
```

授權：本倉庫 MIT（見 `LICENSE`）；第三方見 `NOTICE`。
