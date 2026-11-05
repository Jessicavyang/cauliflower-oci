"""Recompute fixed scores and saved-prediction summaries, without fitting.

Python standard library only. Run from any working directory. Optional image
archives are verified if extracted beside this package's data directory.
"""
from pathlib import Path
import argparse,bisect,collections,csv,hashlib,json,math,platform,sys

BASE=Path(__file__).resolve().parents[1]
def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def truth(s):return str(s).lower() in ('true','1','yes')
def grade(s):return {'Tight (Type A)':'Compact','Medium (Type B)':'Medium','Loose (Type C)':'Loose'}.get(s,s)
def checksum(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def quantization_bound(value):
    return .5*10**(math.floor(math.log10(max(abs(value),1e-300)))-11)+1e-12
def metrics(rr):
    y=[float(r['y_true']) for r in rr];p=[float(r['y_pred']) for r in rr];n=len(y);mean=math.fsum(y)/n
    sse=math.fsum((a-b)**2 for a,b in zip(y,p));sst=math.fsum((a-mean)**2 for a in y)
    return dict(R2=1-sse/sst,RMSE=math.sqrt(sse/n),MAE=math.fsum(abs(a-b) for a,b in zip(y,p))/n)
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=BASE/'outputs/reproduction')
    parser.add_argument('--skip-checksums',action='store_true',help='Skip core-file SHA checks; scientific comparisons still run')
    parser.add_argument('--verify-images',action='store_true',help='Require the images designated for public sharing in config/release_scope.json')
    args=parser.parse_args();result=dict(status='RUNNING',python=platform.python_version(),model_fitting=False,selection_rerun=False,scope='Fixed-parameter scores, categories, sample partitions, saved prediction metrics and archive integrity only.')
    if not args.skip_checksums:
        manifest=read(BASE/'manifests/files_sha256.csv')
        for r in manifest:
            p=(BASE/r['path']).resolve();p.relative_to(BASE.resolve())
            assert p.is_file() and checksum(p)==r['sha256'],f"Checksum mismatch: {r['path']}"
        result['core_files_verified']=len(manifest)
    config=json.loads((BASE/'config/OCI_frozen_scaling_and_grades.json').read_text(encoding='utf-8'))
    features=config['model_features'];mean=config['x_mean'];sd=config['x_sd_ddof1'];beta=config['coefficients_standardized'];intercept=config['intercept_standardized']
    low=config['OCI_raw_train_min'];high=config['OCI_raw_train_max'];bounds=config['OCI_grade_score_boundaries_train_only'];cats=['Compact','Medium','Loose']
    def score(row):
        raw=intercept+sum((float(row[k])-m)/s*b for k,m,s,b in zip(features,mean,sd,beta));v=100*(raw-low)/(high-low)
        return raw,v,cats[bisect.bisect_right(bounds,v)]
    processed=BASE/'data/processed'
    f2=read(processed/'F2_OCI_frozen_275.csv');source={r['image_id']:r for r in read(processed/'F2_features_GRDD_275.csv')}
    archive=read(processed/'inbred_OCI_archive_230.csv');retained=read(processed/'inbred_OCI_main_222.csv')
    assert len(f2)==len(source)==275 and collections.Counter(r['Set'] for r in f2)=={'train':220,'test':55}
    assert len(archive)==230 and len(retained)==222
    def verify_scores(name,rr,rawcol,scorecol,gradecol,lookup=None):
        rawerr=[];scoreerr=[];counts=collections.Counter()
        for row in rr:
            raw,val,category=score(lookup[row['image_id']] if lookup else row)
            stored_raw=float(row[rawcol]);stored_score=float(row[scorecol]);dr=abs(raw-stored_raw);ds=abs(val-stored_score)
            assert dr<=quantization_bound(stored_raw),(name,row.get('image_id',row.get('inbred_row_id')),dr)
            assert ds<=quantization_bound(stored_score),(name,row.get('image_id',row.get('inbred_row_id')),ds)
            assert category==grade(row[gradecol]),(name,category,row[gradecol]);counts[category]+=1;rawerr.append(dr);scoreerr.append(ds)
        return dict(n=len(rr),grade_counts=dict(counts),max_abs_raw_difference=max(rawerr),max_abs_score_difference=max(scoreerr))
    result['F2']=verify_scores('F2',f2,'OCI_Raw_recomputed','OCI_Score_recomputed','OCI_Grade_recomputed',source)
    result['inbred230']=verify_scores('230',archive,'OCI_Raw_frozen','OCI_Score_frozen','OCI_Grade_frozen')
    result['inbred222']=verify_scores('222',retained,'OCI_Raw_frozen','OCI_Score_frozen','OCI_Grade_frozen')
    expected=[r for r in archive if not truth(r['duplicate_source_filename'])]
    assert [r['inbred_row_id'] for r in expected]==[r['inbred_row_id'] for r in retained]
    # Derived figure CSVs may reserialize floating-point values; comparisons never alter them.
    for a,b in zip(expected,retained):
        for k in a:
            if a[k]==b[k]:continue
            try:assert math.isclose(float(a[k]),float(b[k]),rel_tol=1e-12,abs_tol=1e-10),(k,a[k],b[k])
            except ValueError:raise AssertionError((k,a[k],b[k]))
    assert [result['F2']['grade_counts'][k] for k in cats]==[137,101,37]
    assert [result['inbred230']['grade_counts'][k] for k in cats]==[107,94,29]
    assert [result['inbred222']['grade_counts'][k] for k in cats]==[102,92,28]
    result['inbred222']['outside_training_range']=sum(truth(r['any_feature_outside_F2_training_minmax']) for r in retained)
    result['inbred230']['outside_training_range']=sum(truth(r['any_feature_outside_F2_training_minmax']) for r in archive)
    assert result['inbred222']['outside_training_range']==56 and result['inbred230']['outside_training_range']==58
    predictions=BASE/'data/predictions';key=config['model_key']
    oof=[r for r in read(predictions/'Stage3B_saved_OOF_all_models.csv') if r['model']==key]
    policy=read(predictions/'Stage3C_saved_policy_OOF.csv')
    historical=[r for r in read(predictions/'Stage3B_saved_historical55_all_models.csv') if r['model']==key]
    assert len(oof)==len(policy)==660 and len(historical)==55
    for name,rr in [('Stage3B',oof),('Stage3C',policy)]:
        by=collections.defaultdict(list)
        for r in rr:by[r['outer_repeat']].append(r)
        assert len(by)==3
        for g in by.values():assert len(g)==220 and len({r['image_id'] for r in g})==220
        mm=[metrics(by[k]) for k in sorted(by)]
        result[name]=dict(complete_repeat_metrics=mm,mean_complete_repeats={k:math.fsum(m[k] for m in mm)/3 for k in mm[0]})
    result['historical55']=metrics(historical)
    expected_metrics={'Stage3B':(.4822449275011793,.6892597771705993,.5450134834665971),'Stage3C':(.478301957152443,.6918789864415332,.5485715859785251),'historical55':(.37696788392633795,.6456331343423292,.49312320057007614)}
    for name,values in expected_metrics.items():
        mm=result[name]['mean_complete_repeats'] if name!='historical55' else result[name]
        for k,v in zip(['R2','RMSE','MAE'],values):assert abs(mm[k]-v)<1e-12,(name,k,mm[k],v)
    table=read(BASE/'source_data/derived/Fig6_contingency_3x3.csv');cells=[[int(r[k]) for k in cats] for r in table]
    assert cells==[[96,33,2],[34,51,6],[7,17,29]]
    rowtot=[sum(r) for r in cells];coltot=[sum(cells[i][j] for i in range(3)) for j in range(3)];n=sum(rowtot)
    exact=sum(cells[i][i] for i in range(3))/n;chance=sum(a*b for a,b in zip(rowtot,coltot))/n**2
    result['agreement']=dict(n=n,exact_agreement=exact,kappa=(exact-chance)/(1-chance));assert exact==.64 and abs(result['agreement']['kappa']-.4148433134)<1e-10
    assert len(read(BASE/'data/validation/ROI45_planned_manifest.csv'))==45 and len(read(BASE/'data/validation/ROI15_parent_manifest.csv'))==15
    assert len(read(BASE/'data/validation/outer_curd_two_images_saved_metrics.csv'))==2
    result['image_archives']={}
    for cohort,manifest in [('F2','F2_images_275.csv'),('inbred','inbred_available_images_198.csv')]:
        rr=read(BASE/'data/image_manifests'/manifest)
        sharing=json.loads((BASE/'config/release_scope.json').read_text(encoding='utf-8')).get('image_sharing',{'F2':True,'inbred':True})
        if not sharing.get(cohort,False):
            assert all(not r['package_image_path'] for r in rr), 'Nonshared images must not be advertised as package files'
            result['image_archives'][cohort]=dict(local_collection_count=len(rr),expected_in_deposit=0,verified=0,available=False,status='NOT_SHARED_AUTHOR_INSTRUCTION')
            continue
        present=[(BASE/r['package_image_path']).exists() for r in rr]
        if args.verify_images:assert all(present),f'Missing {cohort} images: extract the image archives first'
        if any(present):
            assert all(present),'Partially extracted image archive'
            for r in rr:assert checksum(BASE/r['package_image_path'])==r['sha256'],r['package_image_path']
        result['image_archives'][cohort]=dict(expected=len(rr),verified=len(rr) if all(present) else 0,available=all(present))
    result['status']='PASS';result['validation_boundary']='No independent internal-gap accuracy, new image-acquisition repeatability or cross-environment predictive validation is established by these checks.'
    args.output_dir.mkdir(parents=True,exist_ok=True);(args.output_dir/'reproduction_result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
