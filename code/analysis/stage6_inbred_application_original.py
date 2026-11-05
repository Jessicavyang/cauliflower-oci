#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 6 retrospective, frozen transfer of F2 OCI onto 230 inbred records.

Only the F2 training set model is used. No fitting, rescaling or clustering with
inbred observations. This is NOT ground-truth external predictive validation.

Supplement Table 3 (24AU_001..24AU_230) and geometric source do not share
sample identifiers. They are linked by their row order ONLY after verifying
exact 3-feature numerical agreement for all 230 rows. Export the mapping.
"""
from __future__ import annotations
import csv, json, hashlib, sys, math
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.stats import ks_2samp, chi2_contingency, wasserstein_distance
from scipy.stats import norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
INPUT=ROOT/'input'; OUT=ROOT/'results'
FEATS=['Color_S','Struct_num_gaps','Struct_gap_ratio']
CATS=['Compact','Medium','Loose']


def rows(path):
    with open(path,newline='',encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def save(path, data, columns=None):
    data=list(data)
    if not columns:
        if not data:raise ValueError(f'No rows for {path}')
        columns=list(data[0])
    with open(path,'w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(columns),extrasaction='ignore')
        w.writeheader()
        for row in data:w.writerow({k: ('' if row.get(k) is None else row.get(k)) for k in columns})


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for x in iter(lambda:f.read(1<<20),b''):h.update(x)
    return h.hexdigest()


def mean_sd(v):return float(np.mean(v)),float(np.std(v,ddof=1))


def wilson(k,n):
    z=norm.ppf(.975)
    p=k/n;d=1+z*z/n
    c=(p+z*z/(2*n))/d
    r=z/d*np.sqrt(p*(1-p)/n+z*z/(4*n*n))
    return (c-r,c+r)


def bootstrap_diff(a,b,B=3000,seed=20261103):
    rng=np.random.default_rng(seed)
    # Pooled bootstrap within independent data tables (descriptive if rows dependent)
    diffs=np.empty(B)
    for i in range(B):
        a1=a[rng.integers(0,len(a),len(a))]
        b1=b[rng.integers(0,len(b),len(b))]
        diffs[i]=float(b1.mean()-a1.mean())
    return tuple(float(x) for x in np.quantile(diffs,[.025,.975]))


def count(rows_):return Counter(rows_)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    s=rows(INPUT/'Supplement_Table3_230_extracted.csv')
    geo=rows(INPUT/'cauliflower_geometric_results_final.csv')
    f2=rows(INPUT/'OCI_recalculated_275.csv')
    p=json.loads((INPUT/'full_training_ols_parameters.json').read_text(encoding='utf-8'))['ENet_one_se_thr70']
    frozen=json.loads((INPUT/'OCI_stage4_ab_summary.json').read_text(encoding='utf-8'))
    assert len(s)==len(geo)==230 and len(f2)==275
    assert len(set(r['image_id'] for r in s))==230
    assert set(p['features'])==set(FEATS)
    assert frozen['n_train']==220
    assert frozen['OCI_grade_score_boundaries_train_only'][0] < frozen['OCI_grade_score_boundaries_train_only'][1]
    map_err=[]
    for i,(r,g) in enumerate(zip(s,geo)):
        for feat in FEATS:
            diff=abs(float(r[feat])-float(g[feat]))
            if diff>1e-8:
                map_err.append((i+1,feat,diff))
    if map_err: raise AssertionError(f'Rows cannot be paired safely: {map_err[:10]}')
    file_counts=Counter(g['Image'] for g in geo)
    dup_names={name for name,n in file_counts.items() if n>1}
    assert len(dup_names)==4 and sum(file_counts[n] for n in dup_names)==8, 'unexpected duplicate identity records'
    cols=p['features']; x_mean=np.asarray(p['x_mean'],float); x_sd=np.asarray(p['x_sd_ddof1'],float)
    betas=np.asarray(p['coefficients_standardized'],float); intercept=float(p['intercept_standardized'])
    X=np.asarray([[float(r[k]) for k in cols] for r in s])
    raw=intercept + ((X-x_mean)/x_sd)@betas
    raw_min=float(frozen['OCI_raw_train_min']);raw_max=float(frozen['OCI_raw_train_max'])
    score=100*(raw-raw_min)/(raw_max-raw_min)
    bnds=np.asarray(frozen['OCI_grade_score_boundaries_train_only'],float)
    label_idx=np.searchsorted(bnds,score,side='right')
    grades=[CATS[i] for i in label_idx]
    assert count(grades)==Counter({'Compact':107,'Medium':94,'Loose':29}), 'grades differ from previous report; investigate'
    assert sum(g!=r['OCI_Grade'] for g,r in zip(grades,s))==0, 'inbred grades changed'
    istrain=np.array([r['Set'].strip().lower()=='train' for r in f2])
    assert istrain.sum()==220 and len(istrain)-istrain.sum()==55
    f2_features={ft: np.asarray([float(r[ft]) for r in f2])[istrain] for ft in cols}
    f2_score=np.asarray([float(r['OCI_Score_recomputed']) for r in f2])[istrain]
    ranges={ft:(float(v.min()),float(v.max())) for ft,v in f2_features.items()}
    out_range=np.stack([ (X[:,j]<ranges[ft][0])|(X[:,j]>ranges[ft][1]) for j,ft in enumerate(cols)],axis=1)
    union_out=out_range.any(axis=1)
    output=[]
    for i,(r,g) in enumerate(zip(s,geo)):
        row={'row_index_original':i+1,'inbred_row_id':r['image_id'],
             'source_image_filename':g['Image'],'duplicate_source_filename':bool(g['Image'] in dup_names),
             'Color_S':float(r['Color_S']),'Struct_num_gaps':float(r['Struct_num_gaps']),
             'Struct_gap_ratio':float(r['Struct_gap_ratio']),
             'OCI_Raw_frozen':float(raw[i]),'OCI_Score_frozen':float(score[i]),
             'OCI_Grade_frozen':grades[i],
             'OCI_Raw_supplement_original':float(r['OCI_Raw']),
             'OCI_Score_supplement_original':float(r['OCI_Score']),
             'OCI_Grade_supplement_original':r['OCI_Grade'],
             'OCI_Score_delta_new_minus_old':float(score[i]-float(r['OCI_Score'])),
             'score_outside_0_100':bool(score[i]<0 or score[i]>100),
             'any_feature_outside_F2_training_minmax':bool(union_out[i]),
             'distance_to_nearest_class_boundary':float(min(abs(score[i]-b) for b in bnds))}
        for j,ft in enumerate(cols):
            row['Ztrain_'+ft]=float((X[i,j]-x_mean[j])/x_sd[j])
            row[ft+'_outside_training_range']=bool(out_range[i,j])
        output.append(row)
    save(OUT/'inbred_230_frozen_OCI.csv',output)
    save(OUT/'duplicate_image_filename_8_records.csv',
         [r for r in output if r['duplicate_source_filename']])
    orig=Counter(r['OCI_Grade'] for r in s)
    stats=[]
    for setname,g in (
        ('F2_train_220',[r for r in f2 if r['Set']=='train']),
        ('F2_test_55',[r for r in f2 if r['Set']=='test']),
        ('F2_all_275',f2),
        ('Inbred_all_records_230',output),
        ('Inbred_without_8_ambiguous_222',[r for r in output if not r['duplicate_source_filename']])):
        if setname.startswith('F2'):
            labs=[('Compact' if r['OCI_grade_ordinal']=='0' else ('Medium' if r['OCI_grade_ordinal']=='1' else 'Loose')) for r in g]
        else:labs=[r['OCI_Grade_frozen'] for r in g]
        assert len(g)==len(labs)
        for c in CATS:
            k=sum(l==c for l in labs);lo,hi=wilson(k,len(labs))
            stats.append({'cohort':setname,'N_records':len(labs),'grade':c,'N_grade':k,'proportion':k/len(labs),
                          'wilson95_lower':lo,'wilson95_upper':hi})
    save(OUT/'grade_distribution_and_wilson95.csv',stats)
    f2_counts=[sum(r['OCI_grade_ordinal']==str(i) for r in f2 if r['Set']=='train') for i in range(3)]
    in_counts=[sum(grade==c for grade in grades) for c in CATS]
    excl_counts=[sum(r['OCI_Grade_frozen']==c for r in output if not r['duplicate_source_filename']) for c in CATS]
    chi2,pchi,dof,expect=chi2_contingency(np.array([f2_counts,in_counts]))
    chi2_ex,p_ex,dof_ex,expect_ex=chi2_contingency(np.array([f2_counts,excl_counts]))
    N=sum(f2_counts)+sum(in_counts)
    covs=[]
    vals={ft:X[:,cols.index(ft)] for ft in cols}
    vals['OCI_Score']=score
    f2_vals={**f2_features,'OCI_Score':f2_score}
    for j,ft in enumerate([*cols,'OCI_Score']):
        a=f2_vals[ft];b=vals[ft];ma,sa=mean_sd(a);mb,sb=mean_sd(b)
        ks=ks_2samp(a,b,alternative='two-sided',mode='auto')
        delta_ci=bootstrap_diff(a,b,seed=20261103+j)
        if ft!='OCI_Score':
            lo,hi=ranges[ft];nlow=int((b<lo).sum());nhigh=int((b>hi).sum())
        else:
            lo,hi=0.0,100.0;nlow=int((b<lo).sum());nhigh=int((b>hi).sum())
        pooled=np.sqrt(((len(a)-1)*sa**2+(len(b)-1)*sb**2)/(len(a)+len(b)-2))
        covs.append({'feature':ft,'train_N':len(a),'inbred_N':len(b),
                     'train_min':float(a.min()),'train_max':float(a.max()),
                     'train_mean':ma,'train_sd':sa,
                     'inbred_min':float(b.min()),'inbred_max':float(b.max()),
                     'inbred_mean':mb,'inbred_sd':sb,
                     'mean_delta_inbred_minus_F2_train':mb-ma,
                     'bootstrap95_delta_lower':delta_ci[0],
                     'bootstrap95_delta_upper':delta_ci[1],
                     'standardized_mean_difference':(mb-ma)/pooled,
                     'KS_statistic':float(ks.statistic),'KS_pvalue_exploratory':float(ks.pvalue),
                     'wasserstein':float(wasserstein_distance(a,b)),
                     'n_below_F2_train_range':nlow,'n_above_F2_train_range':nhigh,
                     'n_outside_F2_train_range':nlow+nhigh,
                     'proportion_outside_F2_train_range':(nlow+nhigh)/len(b)})
    # Holm multiplicity adjustment of three image features only
    fs=covs[:3]
    inds=np.argsort([r['KS_pvalue_exploratory'] for r in fs]);m=len(fs)
    ordered=np.array([fs[i]['KS_pvalue_exploratory'] for i in inds])
    adj=np.maximum.accumulate(np.array([(m-i)*p for i,p in enumerate(ordered)]))
    for rank,j in enumerate(inds):fs[j]['KS_pvalue_holm3']=float(min(1,adj[rank]))
    covs[-1]['KS_pvalue_holm3']='not_in_3_feature_family'
    save(OUT/'feature_shift_and_range_coverage.csv',covs)
    # Feature impact on mean OCI score relative to 220 F2 TRAIN (sum equals score mean delta)
    impacts=[]
    for j,ft in enumerate(cols):
        mean_diff=float(vals[ft].mean()-x_mean[j])
        impact=float(mean_diff/x_sd[j]*betas[j]*100/(raw_max-raw_min))
        impacts.append({'feature':ft,'beta_standardized':float(betas[j]),
                        'F2_train_mean':float(x_mean[j]),'inbred_mean':float(vals[ft].mean()),
                        'mean_delta_raw_units':mean_diff,
                        'OCI_score_mean_shift_contribution':impact})
    save(OUT/'mean_OCI_score_shift_decomposition.csv',impacts)
    # summary report with provenance and caveats
    summary={'N_inbred_rows':len(s), 'N_unique_inbred_row_ids':len(set(r['image_id'] for r in s)),
      'N_unique_source_filenames':len(file_counts),
      'N_duplicate_source_filenames':len(dup_names),
      'N_records_with_ambiguous_source_filename':sum(r['duplicate_source_filename'] for r in output),
      'all_230_three_feature_source_alignment_within_1e_8':True,
      'f2_train_N':int(istrain.sum()),'f2_test_N':int((~istrain).sum()),
      'f2_train_grade_counts':f2_counts,'inbred_grade_counts':in_counts,
      'inbred_no_ambiguous_grade_counts':excl_counts,
      'inbred_vs_F2_train_chisquare_exploratory':{'statistic':float(chi2),'pvalue':float(pchi),'df':int(dof),
                                      'cramers_v':float(np.sqrt(chi2/N))},
      'inbred_no_ambiguous_chisquare_exploratory':{'statistic':float(chi2_ex),'pvalue':float(p_ex),'df':int(dof_ex)},
      'inbred_score_range':[float(score.min()),float(score.max())],
      'inbred_mean_score':float(score.mean()),'F2_train_mean_score':float(f2_score.mean()),
      'inbred_score_below_0_count':int(sum(score<0)),'inbred_score_above_100_count':int(sum(score>100)),
      'score_below0_rows':[r['inbred_row_id'] for r in output if r['OCI_Score_frozen']<0],
      'number_inbred_with_any_feature_outside_F2_train_range':int(union_out.sum()),
      'proportion_inbred_with_any_feature_outside_F2_train_range':float(union_out.mean()),
      'max_abs_OCI_raw_difference_vs_original_supplement':max(abs(float(r['OCI_Raw_frozen'])-float(r['OCI_Raw_supplement_original'])) for r in output),
      'max_abs_OCI_score_difference_vs_original_supplement':max(abs(float(r['OCI_Score_delta_new_minus_old'])) for r in output),
      'number_grade_changes_vs_supplement':sum(r['OCI_Grade_frozen']!=r['OCI_Grade_supplement_original'] for r in output),
      'mean_score_shift_breakdown':impacts,
      'mean_score_shift_breakdown_sum':float(sum(r['OCI_score_mean_shift_contribution'] for r in impacts)),
      'model_frozen_from':'Stage3B training-only ElasticNet one_se threshold70 + training-only Stage4A/B minmax and kmeans bounds',
      'source_matching_method':'rowwise equality of all 3 features; no shared stable accession/file ID across the two 230-row inputs',
      'limitations':['No inbred GR/DD, expert grade or 3D compactness reference: cannot estimate out-of-population predictive accuracy or R².',
                     'Year, environment, genetic background and imaging protocols may be confounded.',
                     'Source filename collisions with inconsistent features need manual provenance resolution before claiming 230 independent plants.',
                     'Category-feature gradient is built into OCI and is not independent biological validation.',
                     'Current training data and historical test data have already been explored in prior analyses.'],
      'SHA256_inputs':{x.name:sha(x) for x in INPUT.iterdir() if x.is_file()}}
    (OUT/'stage6_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    # Visualization (not manuscript figures; exploratory)
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(8,4.5))
    bins=np.linspace(min(score.min(),f2_score.min())-2,max(score.max(),f2_score.max())+2,21)
    ax.hist(f2_score,bins=bins,density=True,alpha=.42,label='F2 training (n=220)')
    ax.hist(score,bins=bins,density=True,alpha=.42,label='Inbred records (n=230)')
    for b in bnds:ax.axvline(float(b),color='#505050',ls='--',lw=1)
    ax.set(xlabel='Frozen OCI score',ylabel='Density',title='Frozen OCI score distribution: F2 training vs inbred')
    ax.legend(frameon=False)
    fig.tight_layout();fig.savefig(OUT/'OCI_score_distribution_230_vs_220.png',dpi=220);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.2))
    valsimpact=[r['OCI_score_mean_shift_contribution'] for r in impacts]
    ax.barh([r['feature'] for r in impacts],valsimpact,color=['#6B7F98' if x<0 else '#3A8B7A' for x in valsimpact])
    ax.axvline(0,color='black',lw=.8)
    ax.set(xlabel='Contribution to mean OCI score shift (points)',title='Decomposition of between-cohort mean OCI change')
    for i,v in enumerate(valsimpact):ax.text(v+(.3 if v>=0 else -.3),i,f'{v:+.2f}',va='center',ha='left' if v>=0 else 'right')
    fig.tight_layout();fig.savefig(OUT/'feature_contribution_to_score_shift.png',dpi=220);plt.close(fig)
    print(json.dumps({k:summary[k] for k in ['N_inbred_rows','N_unique_source_filenames','N_records_with_ambiguous_source_filename','inbred_grade_counts','inbred_no_ambiguous_grade_counts','number_grade_changes_vs_supplement','number_inbred_with_any_feature_outside_F2_train_range','inbred_score_below_0_count','inbred_score_range','inbred_mean_score','F2_train_mean_score']},ensure_ascii=False,indent=2))
    return summary

if __name__=='__main__':main()
