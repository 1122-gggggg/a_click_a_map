"""Official MVSFormer++ dynamic consistency, integer-grid reprojection, bounded RAM fusion."""
from __future__ import annotations
import argparse
from functools import lru_cache
import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np
import torch
import os
sys.path.insert(0,os.environ['MVSFORMERPP_ROOT'])
from misc import fusion

DTYPE=np.dtype([('x','<f8'),('y','<f8'),('z','<f8'),('red','u1'),('green','u1'),('blue','u1')])


def grids(h,w):
    # OpenCV integer pixel centres, consistent with inference K and grid_sample.
    y,x=torch.meshgrid(torch.arange(h,device='cuda',dtype=torch.float32),torch.arange(w,device='cuda',dtype=torch.float32),indexing='ij')
    return torch.stack((x,y,torch.ones_like(x)),dim=-1).unsqueeze(-1)

fusion.get_pixel_grids=grids


def pairs(path):
    out=[]
    with path.open() as f:
        for _ in range(int(f.readline())):
            ref=int(f.readline());src=list(map(int,f.readline().split()[1::2]));out.append((ref,src))
    return out


def ply_header(n):
    return ('ply\nformat binary_little_endian 1.0\nelement vertex '+str(n)+'\nproperty double x\nproperty double y\nproperty double z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n').encode()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--voxel',type=float,default=None,help='voxel size in map units (default: 2.2e-4 * scene extent from conversion_receipt.json)')
    p.add_argument('--ids')
    a=p.parse_args()
    if a.voxel is None:
        a.voxel=2.2e-4*json.loads((a.run/'conversion_receipt.json').read_text())['conversion']['scene_extent_m']
    torch.set_num_threads(4)
    out=a.run/'fusion';out.mkdir(exist_ok=True)
    shard_dir=out/'shards';shard_dir.mkdir(exist_ok=True)
    with np.load(a.run/'geometry_cameras.npz') as geometry:
        exact_E=geometry['world_to_camera']
        exact_K=geometry['K']
    @lru_cache(maxsize=16)
    def load(i):
        with np.load(a.run/'depths'/f'{i:08d}.npz') as d:
            return {k:d[k] for k in d.files}
    def tensor_cam(d):
        c=np.zeros((2,4,4),np.float32);c[0]=d['world_to_camera'];c[1,:3,:3]=d['K'];c[1,3,3]=1
        return torch.from_numpy(c).cuda()[None]
    scene=(a.run/'scenes.txt').read_text().split()[0]
    selection=pairs(a.run/'data'/scene/'pair.txt')
    wanted=set(map(int,a.ids.split(','))) if a.ids else None
    if wanted is not None:selection=[r for r in selection if r[0] in wanted]
    start=time.time()
    with torch.inference_mode(), (out/'per_view.jsonl').open('a',buffering=1) as log:
        for index,(ref,src_ids) in enumerate(selection):
            dest=shard_dir/f'{ref:08d}.npy'
            if dest.exists():continue
            t=time.time();r=load(ref)
            sources=[s for s in src_ids[:10] if (a.run/'depths'/f'{s:08d}.npz').exists()]
            if len(sources)!=min(10,len(src_ids)):raise RuntimeError(f'Missing source depths for {ref}')
            ref_depth=torch.from_numpy(r['depth']).cuda()[None,None]
            ref_cam=tensor_cam(r);h,w=r['depth'].shape
            nsrc=len(sources)
            if nsrc<2:raise RuntimeError(f'Insufficient independent sources for {ref}')
            thresholds=torch.arange(2,nsrc+1,device='cuda',dtype=torch.float32).reshape(-1,1,1)
            strict_counts=torch.zeros((nsrc-1,h,w),device='cuda',dtype=torch.int16)
            loose_counts=torch.zeros((h,w),device='cuda',dtype=torch.int16)
            sum_depth=torch.zeros((h,w),device='cuda')
            xy=grids(h,w)[...,:2,0].permute(2,0,1)
            for source in sources:
                s=load(source)
                d=torch.from_numpy(s['depth']).cuda()[None,None,None]
                reproj=fusion.get_reproj_dynamic(ref_depth,d,ref_cam,tensor_cam(s)[:,None])[0,0]
                distance=(reproj[:2]-xy).norm(dim=0)
                relative=(ref_depth[0,0]-reproj[2]).abs()/ref_depth[0,0]
                valid=torch.isfinite(reproj).all(dim=0)&(reproj[2]>0)
                strict=(distance[None]<thresholds/4)&(relative[None]<thresholds/1300)&valid[None]
                strict_counts+=strict.to(torch.int16)
                loose=(distance<nsrc/4)&(relative<nsrc/1300)&valid
                loose_counts+=loose.to(torch.int16)
                sum_depth+=torch.where(loose,reproj[2],0)
            geo=(strict_counts>=thresholds).any(dim=0)
            photo=torch.from_numpy(r['confidence']).cuda()>127
            keep=geo&photo&torch.isfinite(ref_depth[0,0])&(ref_depth[0,0]>0)
            depth=(sum_depth+ref_depth[0,0])/(loose_counts+1)
            y,x=torch.where(keep)
            # Network geometry uses float32, but the final world coordinates
            # must use the original float64 EDM pose, including very near rays.
            xn=x.cpu().numpy();yn=y.cpu().numpy()
            pixel=np.stack((xn,yn,np.ones(len(xn))),axis=0)
            cam_xyz=(np.linalg.inv(exact_K[ref])@pixel)*depth[y,x].cpu().numpy()[None]
            E=exact_E[ref];xyz=(cam_xyz-E[:3,3,None]).T@E[:3,:3]
            mask=keep.cpu().numpy();rgb=r['rgb'][mask]
            record=np.empty(len(xyz),DTYPE)
            for j,key in enumerate(('x','y','z')):record[key]=xyz[:,j]
            for j,key in enumerate(('red','green','blue')):record[key]=rgb[:,j]
            if not np.isfinite(xyz).all():raise RuntimeError('Non-finite fused points')
            temp=dest.with_suffix('.tmp')
            with temp.open('wb') as f:np.save(f,record)
            temp.replace(dest)
            row={'id':ref,'index':index+1,'total':len(selection),'points':len(record),'sources':sources,
                 'photo_fraction':float(photo.float().mean()),'geometry_fraction':float(geo.float().mean()),
                 'kept_fraction':float(keep.float().mean()),'seconds':time.time()-t,'elapsed_seconds':time.time()-start}
            log.write(json.dumps(row)+'\n');print(json.dumps(row),flush=True)
    if wanted is not None:
        print('SMOKE_FUSION_COMPLETE',flush=True);return
    load.cache_clear()
    # Hash-partition vertices on disk, so final voxel aggregation never loads
    # the entire site into memory. Every bucket includes complete voxel groups.
    buckets=out/'voxel_buckets';buckets.mkdir(exist_ok=True)
    handles=[(buckets/f'{i:02d}.bin').open('wb') for i in range(64)]
    raw_count=0
    for ref,_ in selection:
        rows=np.load(shard_dir/f'{ref:08d}.npy',mmap_mode='r');raw_count+=len(rows)
        for first in range(0,len(rows),500000):
            r=rows[first:first+500000]
            v=np.floor(np.column_stack([r[k] for k in ('x','y','z')])/a.voxel).astype(np.int64)
            hashed=((v[:,0]*73856093)^(v[:,1]*19349663)^(v[:,2]*83492791))%64
            for b in range(64):r[hashed==b].tofile(handles[b])
    for f in handles:f.close()
    payload=out/'voxel_vertices.bin';count=0
    with payload.open('wb') as f:
        for b in range(64):
            path=buckets/f'{b:02d}.bin';rows=np.fromfile(path,dtype=DTYPE)
            if not len(rows):continue
            xyz=np.column_stack([rows[k] for k in ('x','y','z')])
            vox=np.floor(xyz/a.voxel).astype(np.int64)
            _,inv,n=np.unique(vox,axis=0,return_inverse=True,return_counts=True)
            result=np.empty(len(n),DTYPE)
            for key in DTYPE.names:
                value=np.bincount(inv,weights=rows[key],minlength=len(n))/n
                result[key]=value if key in ('x','y','z') else np.rint(value).astype(np.uint8)
            result.tofile(f);count+=len(result)
            print('VOXEL_BUCKET',b,'input',len(rows),'output',len(result),flush=True)
            path.unlink()
    products=a.run/'products';products.mkdir(exist_ok=True)
    ply=products/f'{scene}_mvsformerpp.ply'
    with ply.open('wb') as f,payload.open('rb') as source:
        f.write(ply_header(count));shutil.copyfileobj(source,f,8<<20)
    payload.unlink();buckets.rmdir()
    stats={'method':'MVSFormer++ official TNT checkpoint; dynamic geometric consistency',
           'frames':len(selection),'raw_geometrically_fused_points':raw_count,'points_written':count,
           'voxel_size_original_map_units':a.voxel,'confidence_threshold':.5,'dynamic_distance_base':4,
           'dynamic_relative_depth_base':1300,'world_transform':np.eye(4).tolist(),
           'pixel_grid':'integer OpenCV consistent with scaled COLMAP K minus0.5',
           'output':str(ply),'seconds':time.time()-start}
    (out/'fusion_receipt.json').write_text(json.dumps(stats,indent=2)+'\n')
    print('FUSION_COMPLETE',json.dumps(stats),flush=True)


if __name__=='__main__':main()
