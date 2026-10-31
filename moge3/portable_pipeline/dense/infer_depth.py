"""Run official MVSFormer++ weights in the original EDM world frame."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

REPO = Path(os.environ['MVSFORMERPP_ROOT'])
sys.path.insert(0, str(REPO))
from datasets.general_eval import MVSDataset
from utils import init_model


class SceneDataset(MVSDataset):
    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        # COLMAP coordinates have centres at x+0.5; the network uses integer grids.
        # The official loader already resizes K; convert the full-res centre by
        # -0.5, then propagate that same full-res correction to each pyramid.
        for stage, scale in [('stage1',1/8), ('stage2',1/4), ('stage3',1/2), ('stage4',1)]:
            sample['proj_matrices'][stage][:,1,0,2] -= 0.5*scale
            sample['proj_matrices'][stage][:,1,1,2] -= 0.5*scale
        return sample


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--width', type=int, default=1536)
    p.add_argument('--height', type=int, default=832)
    p.add_argument('--views', type=int, default=5)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--ids', type=str)
    a=p.parse_args()
    if a.width%64 or a.height%64:raise ValueError('Dimensions must be divisible by 64')
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark=True
    config=json.loads(a.config.read_text())
    output=a.run/'depths'
    output.mkdir(parents=True,exist_ok=True)
    settings={'width':a.width,'height':a.height,'views':a.views,'numdepth':192,'interval_scale':1.06,
              'checkpoint':str(a.checkpoint),'config':str(a.config),'pixel_grid':'OpenCV integer; scaled COLMAP K minus 0.5',
              'world_frame':'unchanged source EDM; no centering, scaling, or pose optimization'}
    settings_path=output/'settings.json'
    if settings_path.exists() and json.loads(settings_path.read_text())!=settings:
        raise RuntimeError('Existing depths have different settings')
    settings_path.write_text(json.dumps(settings,indent=2)+'\n')
    scene=(a.run/'scenes.txt').read_text().split()[0]
    ds=SceneDataset(str(a.run/'data'),[scene],'test',a.views,ndepths=192,interval_scale=1.06,
                    max_h=a.height,max_w=a.width,dataset='mydataset',fix_res=True)
    wanted=set(map(int,a.ids.split(','))) if a.ids else None
    indexes=[i for i,meta in enumerate(ds.metas) if (wanted is None or meta[1] in wanted)
             and not (output/f'{meta[1]:08d}.npz').exists()]
    if not indexes:
        print('All requested depths already exist',flush=True)
        return
    model=init_model(config)
    checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    state={k.removeprefix('module.'):v for k,v in checkpoint['state_dict'].items() if 'pe_dict' not in k}
    print('CHECKPOINT',model.load_state_dict(state,strict=True),flush=True)
    del checkpoint,state
    model=model.cuda().eval()
    loader=DataLoader(Subset(ds,indexes),batch_size=1,num_workers=a.workers,pin_memory=True)
    start=time.time()
    with torch.inference_mode(), (a.run/'inference_progress.jsonl').open('a',buffering=1) as log:
        for i,s in enumerate(loader):
            t=time.time()
            identifier=int(s['filename'][0].split('/')[-1].replace('{}',''))
            imgs=s['imgs'].cuda(non_blocking=True)
            cams={k:v.cuda(non_blocking=True) for k,v in s['proj_matrices'].items()}
            ranges=s['depth_values'].cuda(non_blocking=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                result=model(imgs,cams,ranges,tmp=[5.,5.,5.,1.])
            depth=result['refined_depth'][0].float().cpu().numpy()
            conf=(result['photometric_confidence'][0].float().clamp(0,1).cpu().numpy()*255).astype(np.uint8)
            camera=s['proj_matrices']['stage4'][0,0].numpy()
            rgb=s['imgs'][0,0].permute(1,2,0).numpy()
            rgb=np.clip((rgb*np.array([.229,.224,.225])+np.array([.485,.456,.406]))*255,0,255).astype(np.uint8)
            # Official inverse-depth refinement is unbounded and may cross
            # zero for individual pixels. Preserve predictions for audit;
            # fusion excludes nonpositive physical depths instead of clipping.
            if not np.isfinite(depth).all():
                raise RuntimeError(f'Non-finite depth for {identifier}')
            dest=output/f'{identifier:08d}.npz'
            tmpfile=dest.with_suffix('.tmp')
            with tmpfile.open('wb') as f:
                np.savez(f,depth=depth,confidence=conf,K=camera[1,:3,:3],world_to_camera=camera[0],rgb=rgb)
            tmpfile.replace(dest)
            row={'id':identifier,'done_this_run':i+1,'remaining_this_run':len(indexes)-i-1,'seconds':time.time()-t,
                 'elapsed_seconds':time.time()-start,'median_depth':float(np.median(depth)),
                 'confidence_above_0_5':float((conf>127).mean()),
                 'nonpositive_depth_pixels':int((depth<=0).sum()),
                 'peak_gpu_bytes':torch.cuda.max_memory_allocated()}
            log.write(json.dumps(row)+'\n')
            print(json.dumps(row),flush=True)
            del result,imgs,cams,ranges
    print('INFERENCE_COMPLETE',len(indexes),flush=True)


if __name__=='__main__':main()
