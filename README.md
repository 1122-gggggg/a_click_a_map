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
| `MAP_REPORT.md`、`map_report.json` | 每支影片註冊張數，**影片之間是否由共享 3D 點連通** |
| `<name>_bundle.tar.zst` | 轉接包：地圖、關鍵影格、檢索 bank、程式、`install.py`、`run_localize.py`、稠密化輸入（EDM 權重不含） |
| `localize_test/` | 若給了 `--query`：EDM+KLT 定位結果 |
| `acam/logs/` | 建圖日誌與記憶體軌跡（`memory.csv`） |

## 做了什麼

```
videos ─► 前處理（SHA、自適應關鍵影格、純旋轉標為 pose-only）
       ─► GlueMap 全序列一次建圖，姿態凍結（Pi3 + SALAD + VGGSfM）
       ─► EDM 重配對、固定姿態三角化（2 px cell，每格一個觀測）
       ─► MegaLoc 檢索 bank
       ─► 報告（註冊、跨影片連通）─► 打包 ─► 選用：EDM+KLT 定位測試
```

定位迴圈（`localize` 與 bundle 內的 `run_localize.py`）：CPU KLT+PnP 每幀；GPU MegaLoc top-2 + EDM + PnP
每 0.3 秒；lift 半徑 4 px；VO 與 dead reckoning 開。這是已驗證的設定，不要隨意調。

## 記憶體（這是重點）

GlueMap 的追蹤階段與 BA 吃很多 RAM，會隨影格數增長。`acam` 這樣處理：

1. 前處理後依關鍵影格數與 RAM 選 `num_track_per_img`（256–1024）。預算來自三次實測
   （30 GiB RAM、RTX 5090）：1389 張 @1024 完成；2043 張 @1024 在追蹤建立階段被 OOM 殺掉（21.3 GiB）；
   2043 張 @512 完成（cgroup 峰值 25.2 GiB）。公式見 `acam/plan.py`。**這是由三個資料點得到的經驗規則，不是模型。**
2. 建圖放在 `systemd-run --user --scope` 的記憶體 cgroup 內（`MemoryMax = RAM − 1.5 GiB`），OOM 只殺這個
   scope，不拖垮主機。沒有 systemd 時會直接執行並提示。
3. 被 SIGKILL／OOM 時自動把 `num_track_per_img` 降 25%，把與它無關的兩視圖／檢索快取搬進新 workspace
   後接著跑（最多 `--max-attempts` 次）。
4. 另外修了兩個在 2000+ 張規模才出現的問題（已含在 `pipeline/`）：EDM 配對上限 25,000 → 40,000
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

- 單元測試：`pytest tests/test_acam.py`（K 選擇、OOM 判斷、快取搬移、路徑範本、影片連通）。
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
acam/        orchestrator（標準函式庫）：config、hw、plan、runner、report、pack、cli
pipeline/    normal_map_pipeline 原始碼（MIT）與 river_map_quality 相容模組
patches/     gluemap.patch（對 colmap/gluemap adc9e4b）、edm.patch（對 chicleee/EDM 8a31d81）
configs/     GlueMap 基底設定、河濱內參、第三方 commit 與 SHA256
scripts/     setup.sh
environment/ 作者機器的 pip freeze
tests/
```

授權：本倉庫 MIT（見 `LICENSE`）；第三方見 `NOTICE`。
