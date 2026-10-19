from configs.data.base import cfg

TEST_BASE_PATH = "assets/megadepth_test_1500_scene_info"

cfg.DATASET.TEST_DATA_SOURCE = "MegaDepth"
cfg.DATASET.TEST_DATA_ROOT = "data/megadepth/test"
cfg.DATASET.TEST_NPZ_ROOT = f"{TEST_BASE_PATH}"
cfg.DATASET.TEST_LIST_PATH = f"{TEST_BASE_PATH}/megadepth_test_1500.txt"

cfg.DATASET.MIN_OVERLAP_SCORE_TEST = 0.0

cfg.EDM.TRAIN_RES_H = 832
cfg.EDM.TRAIN_RES_W = 832
import os as _os
_HR = int(_os.environ.get("P172_TEST_RES", "640"))
cfg.EDM.TEST_RES_H = _HR
cfg.EDM.TEST_RES_W = _HR

cfg.EDM.NECK.NPE = [
    cfg.EDM.TRAIN_RES_H,
    cfg.EDM.TRAIN_RES_W,
    cfg.EDM.TEST_RES_H,
    cfg.EDM.TEST_RES_W,
]
