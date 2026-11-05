"""Evaluate fixed native ROI masks; never create references or change segmentation.

Pillow/NumPy/pandas only. Missing references produce a completion report, not accuracy.
Independent references require real completion metadata and all three masks.
"""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from PIL import Image

def read_binary(path,shape=None):
 with Image.open(path) as im:
  a=np.asarray(im)
 if a.ndim!=2 or not set(np.unique(a).tolist())<={0,255}:
  raise ValueError('Expected single-channel binary PNG, values 0/255: '+str(path))
 if shape is not None and a.shape!=shape:raise ValueError('Native dimensions differ: '+str(path))
 return a.astype(bool)

def overlap(pred,ref,valid):
 p=pred&valid;r=ref&valid
 tp=int((p&r).sum());fp=int((p&~r).sum());fn=int((~p&r).sum())
 npred=tp+fp;nref=tp+fn;union=tp+fp+fn
 both_empty=npred==nref==0
 return dict(tp=tp,fp=fp,fn=fn,pred_pixels=npred,reference_pixels=nref,both_empty=both_empty,
  dice=None if both_empty else 2*tp/(npred+nref),iou=None if both_empty else tp/union,
  precision=tp/npred if npred else None,recall=tp/nref if nref else None)

def check_reference(curd,gaps,uncertain,empty_confirmed):
 if not(curd.shape==gaps.shape==uncertain.shape):raise ValueError('Mask dimensions differ')
 if np.any(gaps&~curd):raise ValueError('Reference gaps outside reference curd; do not auto-clip')
 if not gaps.any() and empty_confirmed!='YES':raise ValueError('Empty gap reference requires explicit EMPTY confirmation')
 if uncertain.all():raise ValueError('No independently judgeable pixels remain')
 return dict(curd_pixels=int(curd.sum()),gap_pixels=int(gaps.sum()),uncertain_pixels=int(uncertain.sum()),uncertain_fraction=float(uncertain.mean()))

def compare(pred,ref,uncertain):
 valid=~uncertain
 if not valid.any():raise ValueError('No independently judgeable pixels remain')
 result={'valid_pixels':int(valid.sum()),'excluded_uncertain_pixels':int(uncertain.sum())}
 for kind in ['curd','gaps']:
  for k,v in overlap(pred[kind],ref[kind],valid).items():result[kind+'_'+k]=v
 domain=ref['curd']&valid;den=int(domain.sum())
 result['reference_curd_area_in_valid_domain']=den
 result['pred_gap_pixels_outside_reference_curd']=int((pred['gaps']&~ref['curd']&valid).sum())
 result['pred_gap_fraction_in_reference_curd']=float((pred['gaps']&domain).sum()/den) if den else None
 result['reference_gap_fraction_in_reference_curd']=float((ref['gaps']&domain).sum()/den) if den else None
 result['gap_fraction_difference_pp']=100*(result['pred_gap_fraction_in_reference_curd']-result['reference_gap_fraction_in_reference_curd']) if den else None
 result['gap_fraction_absolute_error_pp']=abs(result['gap_fraction_difference_pp']) if den else None
 return result

def load_reference(folder,roi,shape,metadata):
 files={k:folder/'masks'/f'{roi}_{k}.png' for k in ['curd','gaps','uncertain']}
 missing=[p.name for p in files.values() if not p.exists()]
 if missing:return None,dict(status='MISSING_MASKS',missing=';'.join(missing))
 row=metadata.get(roi,{})
 required=['annotator_id','annotation_started_at','annotation_finished_at','prior_auto_exposure','reference_origin']
 if row.get('status')!='COMPLETE' or any(not row.get(k) for k in required):
  return None,dict(status='PENDING_COMPLETION_METADATA',missing='Completion metadata not supplied')
 if row['reference_origin']!='RAW_IMAGE_DRAWN':return None,dict(status='NOT_INDEPENDENT_REFERENCE',missing='Origin is not raw-image drawing')
 if row['prior_auto_exposure'] not in ['NO','YES','UNKNOWN']:return None,dict(status='BAD_EXPOSURE_METADATA',missing='Expected NO/YES/UNKNOWN')
 try:
  masks={k:read_binary(p,shape) for k,p in files.items()}
  qc=check_reference(masks['curd'],masks['gaps'],masks['uncertain'],row.get('empty_gap_confirmed',''))
 except ValueError as e:return None,dict(status='QC_FAILED',missing=str(e))
 return masks,dict(status='READY',prior_auto_exposure=row['prior_auto_exposure'],reference_origin=row['reference_origin'],**qc)

def metadata(path):
 if not path.exists():return {}
 d=pd.read_csv(path,keep_default_na=False)
 if d.blind_ROI.duplicated().any():raise ValueError('Duplicate completion ROI IDs')
 return {r['blind_ROI']:r for r in d.to_dict('records')}

def aggregate(results,mapping,bootstrap=10000,seed=20261104):
 if not results:return pd.DataFrame(),dict(status='NO_ELIGIBLE_REFERENCE_PAIRS',accuracy_computed=False)
 d=pd.DataFrame(results).merge(mapping[['blind_ROI','blind_parent','group','historical_pilot','additional_329C_exposure']],on='blind_ROI',validate='many_to_one')
 metriccols=[c for c in d if c.endswith(('_dice','_iou','_precision','_recall')) or c in ['gap_fraction_difference_pp','gap_fraction_absolute_error_pp']]
 parent=d.groupby(['comparison','blind_parent','group'],as_index=False)[metriccols].mean()
 counts=d.groupby(['comparison','blind_parent']).size().rename('ROI_count').reset_index()
 parent=parent.merge(counts,on=['comparison','blind_parent'])
 summary={'status':'PARTIAL_OR_COMPLETE_AS_REPORTED','accuracy_computed':True,'unit':'parent curd; three ROI per curd','bootstrap_seed':seed,'bootstrap_repetitions':bootstrap,'comparisons':{}}
 rng=np.random.default_rng(seed)
 for label,g in d.groupby('comparison'):
  par=parent[parent.comparison==label]
  one=dict(roi_pairs=len(g),parent_curds=len(par),both_empty_gap_ROIs=int(g.gaps_both_empty.sum()),metrics={})
  complete=len(g)==45 and len(par)==15 and (par.ROI_count==3).all()
  for col in metriccols:
   vals=par[col];r=dict(equal_curd_mean=None if vals.isna().all() else float(vals.mean()),curds_with_defined_metric=int(vals.notna().sum()),interval_status='DEFERRED_INCOMPLETE_OR_UNDEFINED_METRIC')
   if complete and vals.notna().all():
    draws=[]
    for _,stratum in par.groupby('group'):
     x=stratum[col].to_numpy();draws.append(rng.choice(x,(bootstrap,len(x)),replace=True))
    dist=np.concatenate(draws,axis=1).mean(axis=1)
    r.update(interval_status='STRATIFIED_PARENT_BOOTSTRAP',conditional95_low=float(np.quantile(dist,.025)),conditional95_high=float(np.quantile(dist,.975)))
   one['metrics'][col]=r
  one['claim_boundary']='Fixed sampled ROI agreement only. No all-curd count, 3D, repeat-photo reliability or OCI accuracy inference.'
  one['exposure_note']='Main available set includes declared exposure histories. Strict exposure/development-pilot sensitivity must be predeclared and reviewed before final publication.'
  summary['comparisons'][label]=one
 return parent,summary

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--workspace',type=Path,default=Path(__file__).resolve().parent);ap.add_argument('--out',type=Path);args=ap.parse_args()
 root=args.workspace;out=args.out or root/'evaluation';out.mkdir(parents=True,exist_ok=True)
 tasks=pd.read_csv(root/'Rater_A/tasks_45ROI.csv',keep_default_na=False)
 mapping=pd.read_csv(root/'COORDINATOR_ONLY/mapping.csv',keep_default_na=False)
 assert len(tasks)==45 and tasks.blind_ROI.nunique()==45 and mapping.blind_parent.nunique()==15
 folders={'A':root/'Rater_A','B':root/'Rater_B','ADJ':root/'Adjudicator'}
 meta={k:metadata(f/'completion.csv') for k,f in folders.items()}
 completion=[];results=[];comparison_qc=[]
 def emit(roi,label,pred,ref,uncertain):
  try:
   value=compare(pred,ref,uncertain)
  except ValueError as e:
   comparison_qc.append(dict(blind_ROI=roi,comparison=label,status='NO_COMPARABLE_PIXELS',reason=str(e)));return
  results.append(dict(blind_ROI=roi,comparison=label,**value))
  comparison_qc.append(dict(blind_ROI=roi,comparison=label,status='EVALUATED',reason=''))
 for task in tasks.to_dict('records'):
  roi=task['blind_ROI'];shape=(int(task['native_height']),int(task['native_width']))
  references={}
  for who,folder in folders.items():
   ref,state=load_reference(folder,roi,shape,meta[who]);completion.append(dict(blind_ROI=roi,role=who,**state))
   if ref is not None:references[who]=ref
  auto={k:read_binary(root/'COORDINATOR_ONLY/auto_roi'/f'{roi}_{k}.png',shape) for k in ['curd','gaps']}
  for who,ref in references.items():
   if who=='ADJ' and not all(k in references for k in ['A','B']):continue
   emit(roi,'auto_vs_'+who,auto,ref,ref['uncertain'])
  if 'A' in references and 'B' in references:
   a,b=references['A'],references['B'];emit(roi,'A_vs_B',a,b,a['uncertain']|b['uncertain'])
 pd.DataFrame(completion).to_csv(out/'completion_and_QC.csv',index=False,encoding='utf-8-sig')
 pd.DataFrame(comparison_qc,columns=['blind_ROI','comparison','status','reason']).to_csv(out/'comparison_QC.csv',index=False,encoding='utf-8-sig')
 pd.DataFrame(results,columns=None if results else ['blind_ROI','comparison','curd_dice','gaps_dice']).to_csv(out/'ROI_metrics.csv',index=False,encoding='utf-8-sig')
 parents,summary=aggregate(results,mapping)
 parents.to_csv(out/'parent_curd_metrics.csv',index=False,encoding='utf-8-sig')
 summary['completion_by_role']={who:sum(r['role']==who and r['status']=='READY' for r in completion) for who in folders}
 summary['reference_masks_created']=False;summary['segmentation_algorithm_changed']=False
 summary['both_empty_rule']='Dice/IoU undefined for both-empty foreground; report counts separately. One-empty gives Dice/IoU zero. Precision/recall with zero denominators are undefined.'
 summary['domain_rule']='Fixed native ROI frame excluding independently supplied uncertainty, never clipped to automatic curd.'
 (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
 print(json.dumps(summary,ensure_ascii=False,allow_nan=False))

if __name__=='__main__':main()
