#!/usr/bin/env python3
"""
make_synthetic.py -- build a FAKE all_variants.csv with the sweep schema so the
R and Python analyses can be exercised before real sweep tables exist.

Structure encodes known truths so the criteria logic can be checked:
  * no method beats baseline under m or c (null)
  * under e, 'bm3d' is given a +0.012 gain in B05/severe only (a planted
    capacity-dependent exception) so criterion 1 and 3 fire for e
  * clean reference sits ~0.05 above baseline (headroom exists)
  * seeds 1,2 are jittered copies of seed 0
THIS IS NOT DATA. Delete after testing.
"""
import numpy as np, pandas as pd, argparse
ap=argparse.ArgumentParser(); ap.add_argument('--out',default='synthetic_all_variants.csv')
ap.add_argument('--full',action='store_true',help='54-condition grid instead of 36 (the real tables are the full grid)')
a=ap.parse_args()
rng=np.random.default_rng(7)
scenes=['panama','rome','brest1','rotterdam1']; bands=['B04','B05','B11']
sevs=['nominal','moderate','severe'] if a.full else ['nominal','severe']
reals=[0,1,2]; methods=['none','bm3d','dncnn','restormer','noise2void','blind2unblind']
variants={'m':0.0,'c':0.010,'e':-0.020}            # offsets on absolute level
band_lvl={'B04':0.250,'B05':0.219,'B11':0.188}       # clean-reference detectability
sev_drop={'nominal':0.03,'moderate':0.06,'severe':0.10}
rows=[]
for v,voff in variants.items():
    for seed in (0,1,2):
        srng=np.random.default_rng(100+seed)
        for sc in scenes:
            sc_off=srng.normal(0,0.01)
            for b in bands:
                cl=band_lvl[b]+voff+sc_off
                rows.append(dict(variant=v,seed=seed,scene=sc,band=b,severity='clean',realization=0,
                                 method='clean',n_tiles=0,n_gt=0,map50=cl,map50_95=cl*0.45,psnr=float('nan'),ssim=float('nan')))
                for sv in sevs:
                    for r in reals:
                        clean=band_lvl[b]+voff+sc_off+srng.normal(0,0.004)
                        base=clean-sev_drop[sv]+srng.normal(0,0.003)
                        for m in methods:
                            eff=0.0
                            if m!='none': eff=srng.normal(-0.002,0.004)     # methods slightly hurt, null-ish
                            if v=='e' and m=='bm3d' and b=='B05' and sv=='severe': eff=0.012+srng.normal(0,0.002)
                            val=base+eff
                            rows.append(dict(variant=v,seed=seed,scene=sc,band=b,severity=sv,
                                             realization=r,method=m,n_tiles=0,n_gt=0,map50=val,map50_95=val*0.45,
                                             psnr=30.0,ssim=0.8))
df=pd.DataFrame(rows); df.to_csv(a.out,index=False)
print(f'wrote {a.out}: {len(df)} rows'); print(df.groupby(['variant','seed']).size().unstack())
