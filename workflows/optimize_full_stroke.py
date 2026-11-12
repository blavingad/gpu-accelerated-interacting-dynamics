"""Joint no-guard full-stroke charge fitting and integer waveform replay.

Uses the paper's archived structural-search cohort (76.8 kHz), not the separate
75-kHz low/high experimental assessment. No font-wide claim from one full stroke.
"""
from pathlib import Path
import sys,argparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from vdcr import structure_search_support as model
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--order',default=','.join(str(i) for i in range(1,17)));p.add_argument('--device',choices=['cpu','gpu'],default='cpu');a=p.parse_args()
 order=list(map(int,a.order.split(',')))
 if sorted(order)!=list(range(1,17)):raise SystemExit('Order must be a permutation of 1..16')
 model.DEVICE=a.device;model.OUT=ROOT/'results/generated/full_stroke';model.OUT.mkdir(parents=True,exist_ok=True)
 print(model.Pilot().fit('chosen_structure',order))
