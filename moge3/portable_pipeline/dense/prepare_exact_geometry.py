"""Retain original float64 EDM cameras for output geometry; network inputs stay float32."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser()
p.add_argument('--run',type=Path,required=True)
a=p.parse_args()
settings=json.loads((a.run/'depths/settings.json').read_text())
with (a.run/'mapping_manifest.tsv').open() as f:rows=list(csv.DictReader(f,delimiter='\t'))
E=np.zeros((len(rows),4,4),np.float64);K=np.zeros((len(rows),3,3),np.float64)
for row in rows:
    i=int(row['mvs_index'])
    lines=(a.run/'data'/(a.run/'scenes.txt').read_text().split()[0]/'cams'/f'{i:08d}_cam.txt').read_text().splitlines()
    E[i]=np.fromstring(' '.join(lines[1:5]),sep=' ').reshape(4,4)
    K[i]=np.fromstring(' '.join(lines[7:10]),sep=' ').reshape(3,3)
    K[i,0]*=settings['width']/int(row['width'])
    K[i,1]*=settings['height']/int(row['height'])
    K[i,:2,2]-=.5
np.savez(a.run/'geometry_cameras.npz',world_to_camera=E,K=K)
print('EXACT_GEOMETRY_CAMERAS',len(rows),'float64; no world transform')
