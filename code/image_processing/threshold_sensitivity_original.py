#!/usr/bin/env python3
"""Unsupervised Otsu threshold perturbation stress test; NOT an accuracy assessment."""
from pathlib import Path
import pandas as pd,numpy as np,cv2 as cv,sys,gc
import importlib.util
spec=importlib.util.spec_from_file_location('v8',str(Path(__file__).resolve().parent/'Step1_cauliflower-structure-v8.py'));o=importlib.util.module_from_spec(spec);spec.loader.exec_module(o)
cv.setNumThreads(2)
base=Path(str(Path(__file__).resolve().parent/'input/Select_30'))
summary=pd.read_csv(str(Path(__file__).resolve().parent/'results/image_inventory_and_feature_comparison.csv'));df=summary[summary.status=='ok'].copy()
rows=[]
for k,rec in enumerate(df.itertuples(index=False),1):
 f=base/rec.archive_group/rec.archive_name
 img=cv.imdecode(np.fromfile(str(f),dtype=np.uint8),cv.IMREAD_COLOR);gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY);blur=cv.GaussianBlur(gray,(5,5),0)
 t0=rec.Otsu_threshold
 for shift in [-10,-5,0,5,10]:
  _,bi=cv.threshold(blur,t0+shift,255,cv.THRESH_BINARY)
  contours,_=cv.findContours(bi,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
  if not contours:
   rows.append({'image_id':rec.image_id,'grade':rec.OCI_grade,'shift':shift,'status':'no_contours'});continue
  cnt=max(contours,key=cv.contourArea);m=np.zeros_like(gray);cv.drawContours(m,[cnt],-1,255,-1)
  g,_=o.analyze_gaps_internal(m,bi)
  rows.append({'image_id':rec.image_id,'grade':rec.OCI_grade,'shift':shift,'status':'ok','threshold':t0+shift,'num_gaps':g['num_gaps'],'gap_ratio':g['gap_ratio'],'mask_area':int(np.count_nonzero(m))})
 print(f'[{k:02d}/{len(df)}] {rec.image_id} done',flush=True)
 del img,gray,blur,bi,m;gc.collect()
out=Path(str(Path(__file__).resolve().parent/'results'))
z=pd.DataFrame(rows)
base0=z[z['shift']==0][['image_id','num_gaps','gap_ratio','mask_area']].rename(columns=lambda s:'original_'+s if s!='image_id' else s)
z=z.merge(base0,on='image_id',how='left')
for v in ['num_gaps','gap_ratio','mask_area']:
 z[f'delta_{v}']=z[v]-z['original_'+v]
 z[f'abs_relative_{v}']=abs(z[f'delta_{v}'])/z['original_'+v].replace(0,np.nan)
z.to_csv(out/'otsu_threshold_perturbation_30.csv',index=False,encoding='utf-8-sig')
agg=z[z['shift']!=0].groupby('shift').agg(n=('image_id','count'),mean_abs_gap_count_delta=('delta_num_gaps',lambda v:float(np.mean(abs(v)))),median_abs_rel_gapcount=('abs_relative_num_gaps','median'),max_abs_rel_gapcount=('abs_relative_num_gaps','max'),median_abs_rel_gapratio=('abs_relative_gap_ratio','median'),max_abs_rel_gapratio=('abs_relative_gap_ratio','max'),median_abs_rel_mask_area=('abs_relative_mask_area','median')).reset_index()
agg.to_csv(out/'otsu_threshold_perturbation_summary.csv',index=False,encoding='utf-8-sig')
print(agg.to_string(index=False),flush=True)
print('All done')
