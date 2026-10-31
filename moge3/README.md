# MoGe-3 版入口（`./run_moge3.sh`）

多支影片 → **稀疏原始 RGB 點雲** + **MVSFormer++ 稠密點雲** + **可在另一台電腦用 EDM + MegaLoc 定位的 bundle**。
三者在同一個 COLMAP 世界座標系（恆等轉換）。與原始版（`./run.sh`：Pi3 + SALAD 建圖）的差異：

| | 原始版 `./run.sh` | MoGe-3 版 `./run_moge3.sh` |
|---|---|---|
| 建圖引擎 | GlueMap（Pi3 深度 + SALAD 檢索） | GlueMap fork：**MoGe-3 深度** + **MegaLoc 檢索**（`patches/gluemap-moge3-megaloc.patch`） |
| 稠密化 | 只提供稠密化輸入 | **MVSFormer++** 官方 TNT 權重，直接輸出 `dense.ply` |
| 輸入 | 一個影片資料夾，一組標定相機 | 多個影片／資料夾；`--intrinsics` 或 `--auto_calibrate` |
| 定位包 | `<name>_bundle.tar.zst` | `OUT/localization/bundle/`（自含，含 EDM／MegaLoc 權重） |

## 安裝（每台機器一次）

需求：Linux x86_64、NVIDIA GPU + CUDA 12.8+ 驅動（RTX 5090 / sm_120 測過）、`git`、`micromamba` 或 `conda`、`uv`、`wget`、`ffmpeg`。

```bash
# 必須自備的權重（其餘自動下載並驗 sha256）
export ACAM_EDM_CHECKPOINT=/path/edm_outdoor.ckpt
export ACAM_MVS_CHECKPOINT=/path/tnt_ckpt/model_best.pth   # MVSFormer++ 官方 TNT 權重
export ACAM_MVS_CONFIG=/path/tnt_ckpt/config.json
moge3/scripts/setup.sh                 # 產生 third_party/moge3/{repo,resources,weights,envs}
moge3/scripts/setup.sh --no-envs       # 只抓程式與 patch（不建環境）
moge3/scripts/setup.sh --no-sources --no-weights   # third_party/moge3 已備妥（例如結案包）時，只建環境
```

釘住的來源與權重 sha256 在 `moge3/configs/versions.json`；`setup.sh` 會 clone 上游 `colmap/gluemap@adc9e4b` 並套用 patch，
EDM 沿用 `patches/edm.patch`，MVSFormer++ 套用 `patches/mvsformerpp-sdpa.patch`（Blackwell 上以 PyTorch SDPA 取代 FlashAttention2）。

## 使用

```bash
./run_moge3.sh a.mp4 b.mp4 c.mp4 --output OUT                 # 影片可為檔案或資料夾
./run_moge3.sh videos/ --output OUT --intrinsics K.json        # 已知相機（建議）
```

`K.json`：`{"image_width":W,"image_height":H,"K":[[fx,0,cx],[0,fy,cy],[0,0,1]],"images_are_undistorted":true}`。
未提供時用 `--auto_calibrate`（初始焦距 1.2×max(W,H)，BA 精修，收據標為 `APPROXIMATE_AUTO_CALIBRATE`）。
影片必須已去畸變（或相機鏡頭畸變可忽略）。

| 產出 | 路徑 |
|---|---|
| 稀疏原始 RGB 點雲 | `OUT/map/sparse_rgb.ply`（COLMAP 地圖在 `OUT/map/model/`） |
| 稠密點雲 | `OUT/dense/dense.ply` |
| 定位 bundle | `OUT/localization/bundle/` |
| 流程收據 | `OUT/pipeline_receipt.json` |

各階段可單獨執行（`moge3/portable_pipeline/{map,dense,localization}_stage.py --help`）；階段收據存在時重跑會跳過該階段。

## 在另一台電腦定位

把 `OUT/localization/bundle/` 整個複製過去。該機器需要 `ffmpeg`，以及 torch 2.7.1+cu128、pycolmap-cuda12 4.0.4 與
`moge3/envs/gluemap.pip.txt` 套件的環境（`setup.sh` 建的 `envs/gluemap` 即可）。

```bash
cd bundle
python install.py                              # 轉成這台機器的路徑；搬動資料夾後要再跑
python run_localize.py --video query.mp4 --route r1 --output result/
```

必須用「自己的 `sys.prefix` 擁有 torch」的直譯器（conda 環境或 `uv venv` 建的環境）；僅繼承系統 torch 的 venv 會被審計檢查拒絕。
`LOC_PYTHON=/path/python` 可覆寫 `run_moge3.sh` 建 bundle 時使用的定位直譯器。

## 限制

- **尺度不是公尺**：單目地圖只有任意尺度（Sim3）。MoGe-3 深度在建圖時被對齊到 Pi3 的尺度，公尺資訊沒有保留；要公尺需已知距離、GCP 或另外的尺度估計。
- 定位成功率取決於地圖與查詢影片的相似度；不同日期／光線的查詢可能無法重定位，目前沒有外部真值的精度驗證。
- 純旋轉影格會標為 pose-only（不三角化）；這段遮蔽邏輯在短片測試中沒有被觸發過。
- 記憶體：數百張以上的多影片建圖需限制主機 RAM、且 GPU 不能同時跑其他任務（GlueMap 約 20 GB VRAM，MVSFormer++ 約 8 GB）。
- `setup.sh` 的環境建置（`gluemap` 的 pygluemap 編譯、`moge` 的 flex-gemm 編譯）未在乾淨機器上完整跑過；`mvs` 環境、conda 規格解算、
  來源 clone + patch 已實測。
