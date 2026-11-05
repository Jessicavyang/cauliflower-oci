#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 4C/4D: frozen-parameter OCI–TA agreement and discordance audit.

Inputs: frozen Stage4A/B image scores and grades, original feature matrix.
The 220 TRAIN mechanical target and 1D KMeans TA grade model are fitted once;
train-derived mechanical thresholds are applied unchanged to all 275 F2 samples.

IMPORTANT: Discordance cannot establish that the TA measurement is 'false'.
Differences on OCI-defining features are expected by construction; the five
prespecified feature/mechanical comparisons are descriptive/associational.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy import stats
from sklearn import __version__ as sklearn_version
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, cohen_kappa_score

LABELS = ["Compact", "Medium", "Loose"]
CORE_VARS = ["GR", "DD", "Struct_num_gaps", "Struct_gap_ratio", "Color_S"]


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def wilson(k, n, z=1.959963984540054):
    if n <= 0:
        return np.nan, np.nan
    p = k / n
    denom = 1 + z*z/n
    mid = (p + z*z/(2*n))/denom
    half = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/denom
    return float(max(0, mid-half)), float(min(1, mid+half))


def kappa_from_cm(cm, kind="none"):
    """Calculate Cohen kappa using observed counts and expected independent counts."""
    cm = np.asarray(cm, dtype=float)
    n = cm.sum()
    if n == 0:
        return np.nan
    exp = np.outer(cm.sum(axis=1), cm.sum(axis=0)) / n
    ii, jj = np.indices(cm.shape)
    if kind == "linear":
        w = np.abs(ii - jj)
    elif kind == "quadratic":
        w = (ii - jj)**2
    else:
        w = (ii != jj).astype(float)
    denom = np.sum(w * exp)
    if denom <= 0:
        return np.nan
    return float(1 - np.sum(w * cm)/denom)


def bowker_test(cm):
    # Bowker symmetry test for a square matched-pairs contingency table.
    chi2 = 0.0
    df = 0
    for i in range(cm.shape[0]):
        for j in range(i + 1, cm.shape[1]):
            tot = int(cm[i, j] + cm[j, i])
            if tot > 0:
                chi2 += float((cm[i, j]-cm[j, i])**2 / tot)
                df += 1
    return chi2, df, float(stats.chi2.sf(chi2, df)) if df else np.nan


def metrics_for_cm(cm):
    n = int(cm.sum())
    correct = int(np.trace(cm))
    lo, hi = wilson(correct, n)
    chi, df, p = bowker_test(cm)
    return {
        "n": n,
        "agreement_count": correct,
        "agreement_fraction": correct/n,
        "agreement_wilson95_low": lo,
        "agreement_wilson95_high": hi,
        "discordant_count": n-correct,
        "discordant_fraction": 1 - correct/n,
        "kappa_unweighted": kappa_from_cm(cm),
        "kappa_linear": kappa_from_cm(cm, "linear"),
        "kappa_quadratic": kappa_from_cm(cm, "quadratic"),
        "bowker_chi2": chi,
        "bowker_df": df,
        "bowker_p": p,
        "oci_more_loose_n": int(sum(cm[i,j] for i in range(3) for j in range(3) if j>i)),
        "oci_more_compact_n": int(sum(cm[i,j] for i in range(3) for j in range(3) if j<i)),
        "off_by_two_grades_n": int(cm[0,2] + cm[2,0]),
    }


def bootstrap_agreement(ta, oci, rng, n_boot):
    ta = np.asarray(ta, dtype=int)
    oci = np.asarray(oci, dtype=int)
    n = len(ta)
    values = np.empty((n_boot, 4), dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        cm = np.bincount(3*ta[idx]+oci[idx], minlength=9).reshape(3,3)
        values[i,0] = kappa_from_cm(cm)
        values[i,1] = kappa_from_cm(cm, "linear")
        values[i,2] = kappa_from_cm(cm, "quadratic")
        values[i,3] = np.trace(cm)/n
    ci = {}
    for j, key in enumerate(("kappa_unweighted", "kappa_linear", "kappa_quadratic", "agreement_fraction")):
        good = values[np.isfinite(values[:,j]),j]
        ci[f"{key}_bootstrap95_low"] = float(np.quantile(good,0.025)) if len(good) else np.nan
        ci[f"{key}_bootstrap95_high"] = float(np.quantile(good,0.975)) if len(good) else np.nan
    return ci


def hedges_g(a, b):
    a, b = np.asarray(a,float),np.asarray(b,float)
    na,nb=len(a),len(b)
    sd = np.sqrt(((na-1)*np.var(a,ddof=1)+(nb-1)*np.var(b,ddof=1))/(na+nb-2))
    if sd < 1e-14:
        return np.nan
    d=(np.mean(a)-np.mean(b))/sd
    return float(d*(1-3/(4*(na+nb)-9)))


def welch_comparison(a,b):
    a,b = np.asarray(a,float),np.asarray(b,float)
    ma,mb = float(np.mean(a)),float(np.mean(b))
    va,vb = float(np.var(a,ddof=1)),float(np.var(b,ddof=1))
    na,nb=len(a),len(b)
    se2=va/na+vb/nb
    if se2 <= 0:
        return {"welch_p_raw":np.nan,"welch_df":np.nan,
                "mean_diff_welch95_low":np.nan,"mean_diff_welch95_high":np.nan}
    df=se2**2 / ((va/na)**2/(na-1)+(vb/nb)**2/(nb-1))
    t=(ma-mb)/np.sqrt(se2)
    crit=stats.t.ppf(0.975,df)
    return {"welch_p_raw":float(2*stats.t.sf(abs(t),df)),
            "welch_df":float(df),"mean_diff_welch95_low":float(ma-mb-crit*np.sqrt(se2)),
            "mean_diff_welch95_high":float(ma-mb+crit*np.sqrt(se2))}


def compare_groups(a,b, variable, group, contrast_type, rng, n_boot):
    a,b=np.asarray(a,float),np.asarray(b,float)
    if len(a)<2 or len(b)<2:
        raise ValueError(f"Insufficient sample size in {contrast_type}")
    w=welch_comparison(a,b)
    # descriptive bootstrap of mean difference, not accounting for the fact that
    # groups are defined by OCI and TA measurement variables themselves.
    aa=a[rng.integers(0,len(a),size=(n_boot,len(a)))].mean(axis=1)
    bb=b[rng.integers(0,len(b),size=(n_boot,len(b)))].mean(axis=1)
    bootdiff=aa-bb
    return {
        "comparison":contrast_type,"comparison_set":group,"variable":variable,
        "n_discordant":len(a),"n_concordant":len(b),
        "mean_discordant":float(a.mean()),"sd_discordant":float(np.std(a,ddof=1)),
        "median_discordant":float(np.median(a)),
        "mean_concordant":float(b.mean()),"sd_concordant":float(np.std(b,ddof=1)),
        "median_concordant":float(np.median(b)),
        "mean_difference_discordant_minus_concordant":float(a.mean()-b.mean()),
        "bootstrap95_diff_low":float(np.quantile(bootdiff,0.025)),
        "bootstrap95_diff_high":float(np.quantile(bootdiff,0.975)),
        "hedges_g":hedges_g(a,b),
        "mann_whitney_two_sided_p_exploratory":float(stats.mannwhitneyu(a,b,alternative="two-sided").pvalue),
        **w
    }


def holm_adjust(pvals):
    # stepdown FWER control over all prespecified comparisons
    pvals=np.asarray(pvals,float)
    ans=np.full(len(pvals),np.nan)
    valid=np.flatnonzero(np.isfinite(pvals))
    if len(valid):
        order=valid[np.argsort(pvals[valid])]
        adj=np.maximum.accumulate(pvals[order]*(len(order)-np.arange(len(order))))
        ans[order]=np.minimum(1,adj)
    return ans


def bh_adjust(pvals):
    pvals=np.asarray(pvals,float)
    ans=np.full(len(pvals),np.nan)
    valid=np.flatnonzero(np.isfinite(pvals))
    if len(valid):
        order=valid[np.argsort(pvals[valid])]
        m=len(order)
        corrected=np.minimum.accumulate((pvals[order]*(m/(np.arange(m)+1)))[::-1])[::-1]
        ans[order]=np.minimum(1,corrected)
    return ans


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--oci",default="input/OCI_recalculated_275.csv")
    p.add_argument("--features",default="input/cauliflower_analysis_final2.csv")
    p.add_argument("--ab-summary",default="input/OCI_stage4_ab_summary.json")
    p.add_argument("--grading",default="input/Cauliflower_Final_Grading_Auto.csv")
    p.add_argument("--out",default="results")
    p.add_argument("--bootstrap",type=int,default=5000)
    p.add_argument("--seed",type=int,default=42)
    a=p.parse_args()
    if a.bootstrap<100: p.error("--bootstrap must be >= 100")
    paths={k:Path(v).expanduser().resolve() for k,v in
           (("oci",a.oci),("features",a.features),("ab_summary",a.ab_summary),("grading",a.grading))}
    for k,v in paths.items():
        if not v.is_file(): raise FileNotFoundError(f"{k} not found: {v}")
    out=Path(a.out).expanduser().resolve();out.mkdir(parents=True,exist_ok=True)
    model=json.loads(paths["ab_summary"].read_text(encoding="utf-8"))
    orig=pd.read_csv(paths["oci"],dtype={"image_id":str},encoding="utf-8-sig")
    feat=pd.read_csv(paths["features"],dtype={"image_id":str},encoding="utf-8-sig")
    grading=pd.read_csv(paths["grading"],dtype={"image_id":str},encoding="utf-8-sig")
    for df in (orig,feat,grading):
        df["image_id"]=df["image_id"].astype(str).str.strip()
        if len(df)!=275 or df.image_id.duplicated().any():
            raise ValueError("All input files must contain 275 unique image IDs")
    if set(orig.image_id)!=set(feat.image_id) or set(orig.image_id)!=set(grading.image_id):
        raise ValueError("image_id sets do not agree")
    x=orig.merge(feat[["image_id"]+[c for c in feat.columns if c not in {"image_id","GR","DD"} and c not in orig.columns]],on="image_id",validate="one_to_one")
    # No change to Stage4A/B OCI grades or score.
    oci=x.OCI_grade_ordinal.to_numpy(int)
    if (oci<0).any() or (oci>2).any():raise ValueError("OCI ordinal grades must lie in [0,2]")
    if not (x.Set.value_counts().to_dict()=={"train":220,"test":55}):
        raise ValueError("Expected 220 train and 55 test")
    checked=grading[["image_id","Set","GR","DD"]].merge(x[["image_id","Set","GR","DD"]],on="image_id",suffixes=("_g","_x"),validate="one_to_one")
    if not np.allclose(checked.GR_g,checked.GR_x) or not np.allclose(checked.DD_g,checked.DD_x):
        raise ValueError("Input mechanical values do not agree")
    if not np.array_equal(checked.Set_g.str.lower(),checked.Set_x.str.lower()):
        raise ValueError("Train/test labels differ")

    tr=x.Set.eq("train").to_numpy()
    # Independently verify the frozen Stage4A/B OCI calculation before using grades.
    model_feats=model["model_features"]
    xmean=np.asarray(model["x_mean"],float)
    xsd=np.asarray(model["x_sd_ddof1"],float)
    coef=np.asarray(model["coefficients_standardized"],float)
    reconstructed_raw=((x[model_feats].to_numpy(float)-xmean)/xsd)@coef + float(model["intercept_standardized"])
    reconstructed_score=(reconstructed_raw-float(model["OCI_raw_train_min"])) / (
        float(model["OCI_raw_train_max"])-float(model["OCI_raw_train_min"])) * 100.0
    if not np.allclose(reconstructed_raw,x.OCI_Raw_recomputed.to_numpy(float),atol=1e-9,rtol=1e-9):
        raise ValueError("OCI raw values disagree with Stage4A/B frozen coefficients")
    if not np.allclose(reconstructed_score,x.OCI_Score_recomputed.to_numpy(float),atol=1e-8,rtol=1e-8):
        raise ValueError("OCI scores disagree with Stage4A/B frozen score scaling")
    reconstructed_ordinal=np.searchsorted(
        np.asarray(model["OCI_grade_score_boundaries_train_only"],float),
        reconstructed_score,side="right")
    if not np.array_equal(reconstructed_ordinal,oci):
        raise ValueError("OCI grades disagree with frozen Stage4A/B boundaries")
    target_params=model["target_standardization"]
    true_target=(((x.DD.to_numpy()-target_params["DD_mean"])/target_params["DD_sd_ddof1"])
                -((x.GR.to_numpy()-target_params["GR_mean"])/target_params["GR_sd_ddof1"]))/2
    if not np.allclose(true_target,x.Mech_Composite_TRAIN_standardized,atol=1e-10,rtol=1e-10):
        raise ValueError("Mechanical composite differs from Stage4A/B target")
    km=KMeans(n_clusters=3,random_state=42,n_init=10).fit(true_target[tr].reshape(-1,1))
    centers=np.sort(km.cluster_centers_.reshape(-1))
    thresholds=(centers[:-1]+centers[1:])/2
    ta=np.searchsorted(thresholds,true_target,side="right").astype(int)
    x["TA_grade_ordinal"]=ta
    x["TA_Grade_train_only"]=np.array(LABELS)[ta]
    x["OCI_Grade_standardized"]=np.array(LABELS)[oci]
    x["ordinal_difference_OCI_minus_TA"]=oci-ta
    x["TA_OCI_exact_agreement"]=oci==ta
    x["TA_cluster_center_train_only"]=centers[ta]
    x["TA_target_distance_to_nearest_cutoff"]=np.min(abs(true_target[:,None]-thresholds),axis=1)
    x["discordance_category"]="Other"
    g1=(ta==0)&(oci==0)
    g2=(ta==0)&(oci!=0)
    g3=(ta==2)&(oci!=2)
    g4=(ta==2)&(oci==2)
    x.loc[g1,"discordance_category"]="Concordant_Compact_GC"
    x.loc[g2,"discordance_category"]="TA_Compact_OCI_Noncompact_FH_proxy"
    x.loc[g3,"discordance_category"]="TA_Loose_OCI_Nonloose_FL_proxy"
    x.loc[g4,"discordance_category"]="Concordant_Loose_GL"
    x.to_csv(out/"OCI_TA_sample_level_275.csv",index=False,encoding="utf-8-sig")
    x.loc[g2|g3].sort_values(["discordance_category", "Set", "image_id"]).to_csv(
        out/"FH_FL_discordant_individuals_59.csv",index=False,encoding="utf-8-sig")

    rng=np.random.default_rng(a.seed)
    agreement=[];confusion=[];group_counts=[]
    for s in ("train","test","all"):
        mask=x.Set.eq(s).to_numpy() if s!="all" else np.ones(len(x),bool)
        cm=confusion_matrix(ta[mask],oci[mask],labels=[0,1,2])
        for i,label_ta in enumerate(LABELS):
            for j,label_oci in enumerate(LABELS):
                confusion.append({"Set":s,"TA_grade":label_ta,"OCI_grade":label_oci,"n":int(cm[i,j])})
        row={"Set":s,**metrics_for_cm(cm),**bootstrap_agreement(ta[mask],oci[mask],rng,a.bootstrap)}
        agreement.append(row)
        n=int(mask.sum())
        for name,num,den in (
            ("FH_proxy_TA_Compact_OCI_Noncompact",int((mask & g2).sum()),int((mask & (ta==0)).sum())),
            ("FL_proxy_TA_Loose_OCI_Nonloose",int((mask & g3).sum()),int((mask & (ta==2)).sum())),
            ("GC_Concordant_Compact",int((mask & g1).sum()),int((mask & (ta==0)).sum())),
            ("GL_Concordant_Loose",int((mask & g4).sum()),int((mask & (ta==2)).sum())),
        ):
            lo,hi=wilson(num,den)
            loall,hiall=wilson(num,n)
            group_counts.append({"Set":s,"group":name,"n":num,
                "denominator_eligible_TA_grade":den,"fraction_among_eligible":num/den if den else np.nan,
                "wilson95_low_among_eligible":lo,"wilson95_high_among_eligible":hi,
                "fraction_among_all":num/n,"wilson95_low_among_all":loall,
                "wilson95_high_among_all":hiall})

    pd.DataFrame(agreement).to_csv(out/"OCI_TA_agreement_metrics.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(confusion).to_csv(out/"OCI_TA_confusion_matrices.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(group_counts).to_csv(out/"discordance_group_counts_rates.csv",index=False,encoding="utf-8-sig")
    # Comparisons of discordance rates between train/test are descriptive only.
    split_rows=[]
    for name,indicator,denominator in (("FH_proxy",g2,ta==0),("FL_proxy",g3,ta==2)):
        pos_train=int((tr & indicator).sum());neg_train=int((tr & denominator).sum()-pos_train)
        pos_test=int((~tr & indicator).sum());neg_test=int((~tr & denominator).sum()-pos_test)
        odds,pval=stats.fisher_exact([[pos_train,neg_train],[pos_test,neg_test]])
        split_rows.append({"event":name,"train_events":pos_train,"train_non_events":neg_train,
                           "test_events":pos_test,"test_non_events":neg_test,
                           "fisher_odds_ratio":float(odds),"fisher_two_sided_p":float(pval)})
    pd.DataFrame(split_rows).to_csv(out/"train_test_discordance_rate_comparison.csv",index=False,encoding="utf-8-sig")

    core=[];explore=[]
    for s in ("train","test","all"):
        mask=x.Set.eq(s).to_numpy() if s!="all" else np.ones(len(x),bool)
        pairs=(("FH_vs_GC",g2,g1),("FL_vs_GL",g3,g4))
        for contrast,discord,concord in pairs:
            ma=mask&discord; mb=mask&concord
            for variable in CORE_VARS:
                core.append(compare_groups(x.loc[ma,variable],x.loc[mb,variable],variable,s,contrast,rng,a.bootstrap))
            # All 20 original image predictors are exploratory; q-values corrected within split/contrast.
            featcols=[c for c in feat.columns if c not in ("image_id","GR","DD")]
            for variable in featcols:
                explore.append(compare_groups(x.loc[ma,variable],x.loc[mb,variable],variable,s,contrast,rng,a.bootstrap))
    core_df=pd.DataFrame(core)
    # Adjust all ten prespecified tests in each split together (FH/GC and FL/GL x 5).
    for s in ("train","test","all"):
        sel=core_df.comparison_set.eq(s)
        core_df.loc[sel,"welch_p_holm_10_tests"] = holm_adjust(core_df.loc[sel,"welch_p_raw"])
    core_df.to_csv(out/"FH_FL_core_comparisons.csv",index=False,encoding="utf-8-sig")
    explore_df=pd.DataFrame(explore)
    for s in ("train","test","all"):
        sel=explore_df.comparison_set.eq(s)
        # 40 comparisons per split (20 predictors x two discordance contrasts).
        explore_df.loc[sel,"welch_p_BH_FDR_40_tests"] = bh_adjust(explore_df.loc[sel,"welch_p_raw"])
    explore_df.to_csv(out/"FH_FL_all20_features_exploratory.csv",index=False,encoding="utf-8-sig")

    summary={
        "analysis":"Stage4C agreement; Stage4D operational discordance groups",
        "time_utc":datetime.now(timezone.utc).isoformat(),
        "input_sha256":{k:sha256(v) for k,v in paths.items()},
        "n_total":len(x),"n_train":int(tr.sum()),"n_test":int((~tr).sum()),
        "OCI_score_and_grade":"frozen unchanged from Stage4A/B",
        "TA_target":"(-z_train(GR)+z_train(DD))/2",
        "TA_train_KMeans":{"k":3,"random_state":42,"n_init":10,
                           "centers_ordered_Compact_Medium_Loose":centers.tolist(),
                           "boundaries_ordered":thresholds.tolist()},
        "bootstrap":{"n_resamples":a.bootstrap,"seed":a.seed,
                     "method":"resample paired OCI/TA labels within each fixed split; frozen grade boundaries",
                     "limitations":"95% percentile intervals conditional on frozen grading and observed dataset; does not include threshold/model fitting uncertainty."},
        "comparisons":{"prespecified_variables":CORE_VARS,"p_adjustment":"Holm across 10 comparisons per split",
                       "exploratory_20_image_features":"Benjamini-Hochberg FDR across 40 comparisons per split"},
        "software":{"python":sys.version.split()[0],"numpy":np.__version__,
                    "pandas":pd.__version__,"scipy":scipy.__version__,
                    "sklearn":sklearn_version},
        "warnings":[
            "FH and FL are operational labels for two types of TA-OCI discordance, NOT independently verified TA errors.",
            "OCI variables and TA variables help define the compared groups, so resulting feature/GR/DD differences are not independent biological validation and may be partly mathematical by construction.",
            "The 55 historical test records have been analyzed previously; not a prospectively untouched external validation set.",
            "Complete-case 20-feature exploratory analysis includes no adjustment for selection/group-definition circularity; FDR only adjusts multiple p values.",
        ],
    }
    (out/"stage4CD_run_config_and_limitations.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8")
    print("[DONE] Stage4C/4D audit, n=",len(x))
    for s in ("train","test","all"):
        d=next(z for z in agreement if z["Set"]==s)
        mask=x.Set.eq(s).to_numpy() if s!="all" else np.ones(len(x),bool)
        fh=int(np.sum(mask & g2))
        fl=int(np.sum(mask & g3))
        print(f"{s}: agreement={d['agreement_fraction']:.3f}, "
              f"kappa={d['kappa_unweighted']:.4f}, FH-proxy={fh}, FL-proxy={fl}")


if __name__ == "__main__":
    main()
