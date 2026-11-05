#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OCI Stages 4A / 4B: frozen TRAIN-only score and three-grade stratification.

Input: two 275-row CSVs, plus Stage 3B full_training_ols_parameters.json.
NO model refit or scaling/clustering with TEST data.  k=3 is specified as
an ordinal-category design choice, not chosen by inspecting GR/DD outcome.
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
from scipy.stats import chi2_contingency
from sklearn import __version__ as sklearn_version
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

CATEGORIES = ["Compact", "Medium", "Loose"]
TYPES = ["Type A", "Type B", "Type C"]
LABELS = ["Tight (Type A)", "Medium (Type B)", "Loose (Type C)"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def wilson(k, n, z=1.959963984540054):
    if n == 0: return np.nan, np.nan
    p = k / n
    den = 1. + z*z/n
    center = (p + z*z/(2*n))/den
    half = z/den*np.sqrt(p*(1-p)/n + z*z/(4*n*n))
    return center-half, center+half


def main():
    p = argparse.ArgumentParser(description="Train-only OCI recalculation and 3-grade validation")
    p.add_argument("--data", default="input/cauliflower_analysis_final2.csv")
    p.add_argument("--grading", default="input/Cauliflower_Final_Grading_Auto.csv")
    p.add_argument("--parameters", default="input/full_training_ols_parameters.json")
    p.add_argument("--model-key", default="ENet_one_se_thr70")
    p.add_argument("--out", default="results")
    a = p.parse_args()
    data_path, grading_path, param_path = [Path(x).resolve() for x in (a.data,a.grading,a.parameters)]
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if not all(x.is_file() for x in (data_path,grading_path,param_path)):
        raise FileNotFoundError("Missing original input CSV or Stage3B model parameter JSON")
    features_df = pd.read_csv(data_path, dtype={"image_id":str}, encoding="utf-8-sig")
    grade_df = pd.read_csv(grading_path, dtype={"image_id":str}, encoding="utf-8-sig")
    assert len(features_df)==len(grade_df)==275, "Both inputs must contain 275 samples"
    for x in (features_df,grade_df):
        assert x.image_id.notna().all() and x.image_id.is_unique, "IDs must be complete and unique"
        x.image_id = x.image_id.str.strip()
    assert set(features_df.image_id)==set(grade_df.image_id), "Input IDs do not match"
    assert {"image_id","Set","GR","DD","OCI_Raw","OCI_Score","Grade"}.issubset(grade_df.columns)
    assert {"image_id","GR","DD"}.issubset(features_df.columns)
    d = features_df.merge(
        grade_df[["image_id","Set","GR","DD","Mech_Composite","OCI_Raw","OCI_Score","Grade"]],
        on="image_id",validate="one_to_one",suffixes=("","_old")
    )
    assert np.allclose(d.GR,d.GR_old,rtol=1e-8,atol=1e-6)
    assert np.allclose(d.DD,d.DD_old,rtol=1e-8,atol=1e-6)
    d.Set=d.Set.str.strip().str.lower()
    assert d.Set.value_counts().to_dict()=={"train":220,"test":55}, "Expected TRAIN/TEST 220/55"
    model_dict=json.loads(param_path.read_text(encoding="utf-8"))
    if a.model_key not in model_dict: raise KeyError("Missing model key "+a.model_key)
    model=model_dict[a.model_key]
    cols=model["features"]
    assert set(cols)=={"Struct_num_gaps","Struct_gap_ratio","Color_S"}, "Expected final three OCI features"
    mean=np.asarray(model["x_mean"],dtype=float)
    sd=np.asarray(model["x_sd_ddof1"],dtype=float)
    beta=np.asarray(model["coefficients_standardized"],dtype=float)
    intercept=float(model["intercept_standardized"])
    assert len(cols)==len(mean)==len(sd)==len(beta)==3 and np.all(sd>0)
    assert all(np.isfinite(d[col]).all() for col in cols)
    istrain=d.Set.eq("train").to_numpy()
    x=d[cols].to_numpy(float)
    xz=(x-mean)/sd
    raw=intercept + xz@beta
    lo=float(raw[istrain].min()); hi=float(raw[istrain].max())
    assert hi>lo
    score=100*(raw-lo)/(hi-lo) # Never rescale by TEST or clip out-of-range scores.
    for i,col in enumerate(cols):d["Z_"+col]=xz[:,i]
    tparams=model["target_parameters"]
    mcomp=0.5*((d.DD-tparams["DD_mean"])/tparams["DD_sd_ddof1"] -
               (d.GR-tparams["GR_mean"])/tparams["GR_sd_ddof1"])
    d["Mech_Composite_TRAIN_standardized"]=mcomp
    d["OCI_Raw_recomputed"]=raw
    d["OCI_Score_recomputed"]=score
    d["OCI_score_outside_training_0_100"]=(score<0)|(score>100)
    d["OCI_Score_original"]=d["OCI_Score"]
    d["OCI_Raw_original"]=d["OCI_Raw"]
    d["OCI_score_delta_new_minus_old"]=score-d["OCI_Score_original"]
    d["Grade_original"]=d["Grade"]

    # Fit 1-dimensional KMeans on the frozen TRAIN-only OCI_Score (k=3).
    kmeans=KMeans(n_clusters=3,random_state=42,n_init=10)
    kmeans.fit(score[istrain,None])
    centers=np.sort(kmeans.cluster_centers_.ravel())
    bounds=(centers[:-1]+centers[1:])/2.0
    grade_idx=np.searchsorted(bounds,score,side="right")
    grade_names=[LABELS[i] for i in grade_idx]
    d["OCI_grade_ordinal"]=grade_idx
    d["OCI_Grade_recomputed"]=grade_names
    d["OCI_grade_changed_from_original"]=d.OCI_Grade_recomputed.ne(d.Grade_original)
    d["distance_to_nearest_grade_boundary"]=np.abs(score[:,None]-bounds[None,:]).min(axis=1)

    # Audit: closest-center boundary labels must match KMeans training assignments
    # after aligning KMeans' arbitrary cluster indices with ascending OCI centers.
    center_order=np.argsort(kmeans.cluster_centers_.ravel())
    cluster_to_ord={int(label):i for i,label in enumerate(center_order)}
    labels_train=np.asarray([cluster_to_ord[int(v)] for v in kmeans.labels_])
    assert np.array_equal(labels_train,grade_idx[istrain]), "Midpoint and KMeans labels disagree"
    assert np.isclose(score[istrain].min(),0) and np.isclose(score[istrain].max(),100)
    assert np.isfinite(d["OCI_Score_recomputed"]).all()
    assert int(d.OCI_grade_changed_from_original.sum()) == sum(
        x!=y for x,y in zip(d.OCI_Grade_recomputed,d.Grade_original))

    # Order outputs by the input feature table, preserve IDs and all three structural/color measurements.
    out_cols=["image_id","Set","GR","DD","Mech_Composite_TRAIN_standardized",*cols,
              *("Z_"+c for c in cols),"OCI_Raw_recomputed","OCI_Score_recomputed",
              "OCI_grade_ordinal","OCI_Grade_recomputed","distance_to_nearest_grade_boundary",
              "OCI_score_outside_training_0_100","OCI_Raw_original","OCI_Score_original",
              "OCI_score_delta_new_minus_old","Grade_original","OCI_grade_changed_from_original"]
    d[out_cols].to_csv(out/"OCI_recalculated_275.csv",index=False,encoding="utf-8-sig",float_format="%.12g")

    boundary_rows=[]
    for i,bound in enumerate(bounds):
        boundary_rows.append({
            "boundary":f"{CATEGORIES[i]}_vs_{CATEGORIES[i+1]}",
            "lower_category":CATEGORIES[i],"upper_category":CATEGORIES[i+1],
            "OCI_Score_boundary":float(bound),
            "OCI_Raw_boundary":lo+(hi-lo)*float(bound)/100,
            "lower_cluster_center_score":float(centers[i]),
            "upper_cluster_center_score":float(centers[i+1]),
            "rule":"Assign by nearest TRAIN cluster center; values exactly on boundary go to upper category",
            "trained_on":"train_only_220",
            "k":3,"random_state":42,"n_init":10,
        })
    pd.DataFrame(boundary_rows).to_csv(out/"OCI_thresholds_train_only.csv",index=False,encoding="utf-8-sig",float_format="%.12g")

    summary_rows=[]
    for group,mask in (("train",istrain),("test",~istrain),("all",np.ones(len(d),dtype=bool))):
        group_score=score[mask];group_idx=grade_idx[mask];n=int(mask.sum())
        for grade_no,cat in enumerate(CATEGORIES):
            scores=group_score[group_idx==grade_no]
            count=len(scores);l,u=wilson(count,n)
            summary_rows.append({
                "Set":group,"n_samples":n,"Grade":cat,"Type":TYPES[grade_no],
                "count":count,"pct":100*count/n,"wilson95_lower_pct":100*l,
                "wilson95_upper_pct":100*u,
                "OCI_score_mean":float(np.mean(scores)) if count else np.nan,
                "OCI_score_sd_ddof1":float(np.std(scores,ddof=1)) if count>1 else np.nan,
                "OCI_score_min":float(np.min(scores)) if count else np.nan,
                "OCI_score_max":float(np.max(scores)) if count else np.nan,
            })
    pd.DataFrame(summary_rows).to_csv(out/"OCI_grading_validation.csv",index=False,encoding="utf-8-sig",float_format="%.12g")
    changes=d.loc[d.OCI_grade_changed_from_original,["image_id","Set","Grade_original","OCI_Grade_recomputed","OCI_Score_original","OCI_Score_recomputed"]]
    changes.to_csv(out/"OCI_changed_grades.csv",index=False,encoding="utf-8-sig")

    # TRAIN-only descriptive k diagnostics; do not choose k by GR/DD, and do not claim k3 is empirical optimum.
    kd=[]
    for k in (2,3,4,5,6):
        km=KMeans(n_clusters=k,random_state=42,n_init=10).fit(score[istrain,None])
        kd.append({"k":k,"training_silhouette_score":float(silhouette_score(score[istrain,None],km.labels_)),
                   "training_within_cluster_sse":float(km.inertia_),
                   "centers_ascending":";".join(f"{q:.8g}" for q in sorted(km.cluster_centers_.ravel()))})
    pd.DataFrame(kd).to_csv(out/"KMeans_k_sensitivity_train_only.csv",index=False,encoding="utf-8-sig")

    ctab=pd.crosstab(d["Set"],d["OCI_grade_ordinal"]).reindex(index=["train","test"],columns=[0,1,2],fill_value=0)
    chi2,pvalue,dof,exp=chi2_contingency(ctab.to_numpy(),correction=False)
    old_grade_idx=d.Grade_original.map({name:i for i,name in enumerate(LABELS)}).to_numpy()
    assert np.all(pd.notna(old_grade_idx)), "Unknown historical grade label"
    n_changed=int(d.OCI_grade_changed_from_original.sum())
    max_abs_delta=float(np.max(np.abs(d.OCI_score_delta_new_minus_old)))
    result={
        "study":"OCI stage4A and stage4B frozen-train-only recalculation",
        "model_key":a.model_key,"n_total":len(d),"n_train":int(istrain.sum()),"n_test":int((~istrain).sum()),
        "model_features":cols,"coefficients_standardized":beta.tolist(),"intercept_standardized":intercept,
        "x_mean":mean.tolist(),"x_sd_ddof1":sd.tolist(),"target_standardization":tparams,
        "OCI_raw_train_min":lo,"OCI_raw_train_max":hi,
        "OCI_score_train_min":float(score[istrain].min()),"OCI_score_train_max":float(score[istrain].max()),
        "OCI_grade_centers_train_only":centers.tolist(),"OCI_grade_score_boundaries_train_only":bounds.tolist(),
        "kmeans":{"k":3,"random_state":42,"n_init":10,"source":"training score only","k_selection":"three ordinal classes specified by research design, not selected with mechanical outcomes"},
        "n_test_scores_below_zero":int(np.sum(score[~istrain]<0)),
        "n_test_scores_above_hundred":int(np.sum(score[~istrain]>100)),
        "grade_counts_train":np.bincount(grade_idx[istrain],minlength=3).tolist(),
        "grade_counts_test":np.bincount(grade_idx[~istrain],minlength=3).tolist(),
        "grade_counts_all":np.bincount(grade_idx,minlength=3).tolist(),
        "n_grade_changed_from_original":n_changed,
        "old_new_grade_adjusted_rand_index_descriptive":float(adjusted_rand_score(old_grade_idx,grade_idx)),
        "max_absolute_OCI_score_difference_from_original":max_abs_delta,
        "train_test_grade_distribution_chi2":float(chi2),"train_test_grade_distribution_pvalue":float(pvalue),"chi2_df":int(dof),
        "disclaimer":"No independent ground-truth grading available. Same-category comparison is not predictive accuracy. The 55 historical test samples have been viewed in previous analyses. K=3 is design driven.",
        "input_sha256":{data_path.name:sha256(data_path),grading_path.name:sha256(grading_path),param_path.name:sha256(param_path)},
        "software_versions":{"python":sys.version.split()[0],"numpy":np.__version__,"pandas":pd.__version__,"scipy":scipy.__version__,"scikit_learn":sklearn_version},
    }
    (out/"OCI_stage4_ab_summary.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print("Stage4A/4B complete.")
    print("model:",a.model_key,"features:",cols)
    print("TRAIN raw min/max:",lo,hi)
    print("TRAIN KMeans centers:",centers,"boundaries:",bounds)
    print("counts TRAIN/TEST/ALL:",result["grade_counts_train"],result["grade_counts_test"],result["grade_counts_all"])
    print("changed grades:",n_changed,"max score difference:",max_abs_delta)
    print("results:",out)


if __name__=="__main__":main()
