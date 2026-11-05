#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 3C: leakage-controlled inner selection of method family and threshold.

An audit of method-choice optimism, not proof of historical preregistration.
Outer splits exactly match Stage 3B. For each outer TRAIN fold independently,
select one of the 8 complete Stage2B/3B policies (one_se vs min_mse, frequency
threshold 0.60/0.70/0.80/0.90), using 3-fold INNER-TO-OUTER CV. In each of
those 3 policy-selection folds, independently repeat the complete Stage2B
feature-selection pipeline in its training portion, then predict its held-out
portion. Minimize mean squared prediction error across the three portions.
With the resulting method choice frozen, take the corresponding outer-held-out
predictions from the completed Stage3B run, which were computed using only the
respective outer training fold. This uses no outer validation observations to
select the model. The 55 historical TEST samples are never loaded here.

The previously explored candidate family and thresholds were not preregistered;
this analysis cannot retroactively eliminate the previous data-adaptive choice.
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
for k in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[k]="1"
import numpy as np
import pandas as pd
from sklearn import __version__ as sklearn_version
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
import stage3b_nested_elasticnet as s3
import stage2b_elasticnet_stability as s2

MODELS = [f"ENet_{rule}_thr{thr}" for rule in ("one_se","min_mse") for thr in (60,70,80,90)]


def msg(text):
    print(f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {text}",flush=True)


def args_parser():
    p=argparse.ArgumentParser(description="Stage3C selection-aware method-choice CV")
    p.add_argument("--data",default="input/cauliflower_analysis_final2.csv")
    p.add_argument("--grading",default="input/Cauliflower_Final_Grading_Auto.csv")
    p.add_argument("--reference-config",default="reference_stage2b_one_se/run_config.json")
    p.add_argument("--stage3b-results",required=True)
    p.add_argument("--out",default="results_stage3c")
    p.add_argument("--outer-folds",type=int,default=5)
    p.add_argument("--outer-repeats",type=int,default=3)
    p.add_argument("--policy-cv-folds",type=int,default=3)
    p.add_argument("--stability-reps-inner",type=int,default=100)
    p.add_argument("--inner-enet-cv",type=int,default=10)
    p.add_argument("--n-jobs",type=int,default=8)
    p.add_argument("--batch-size",type=int,default=20)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--bootstrap",type=int,default=2000)
    p.add_argument("--max-outer",type=int,default=0,help="Development only: number of outer folds to run")
    p.add_argument("--resume",action="store_true")
    a=p.parse_args()
    if a.policy_cv_folds<2 or a.outer_folds<2 or a.outer_repeats<1 or a.stability_reps_inner<1 or a.inner_enet_cv<2 or a.n_jobs<1:
        p.error("Invalid folds/repetitions/jobs")
    if a.bootstrap<100:
        p.error("--bootstrap must be at least 100")
    return a


def atomic_csv(df,path):
    temp=path.with_suffix(".tmp")
    df.to_csv(temp,index=False,encoding="utf-8-sig")
    os.replace(temp,path)


def verify_stage3b(a, train, ref):
    path=Path(a.stage3b_results).resolve()
    stored_config=json.loads((path/"run_config_stage3b.json").read_text())
    if stored_config["analysis_classification"] != "FORMAL":
        raise ValueError("Stage3B results are not formal")
    if stored_config["outer_folds"] != a.outer_folds or stored_config["outer_repeats"] != a.outer_repeats or stored_config["seed"] != a.seed:
        raise ValueError("Outer CV parameters do not match completed Stage3B")
    if stored_config["stability_reps_per_outer_fold"] != 100:
        raise ValueError("Stage3B outer training checkpoints do not contain 100 repetitions")
    if stored_config["input_sha256"] != ref["input_sha256"]:
        raise ValueError("Stage3B input file hashes do not match Stage2B reference")
    if sorted(stored_config["thresholds"]) != [0.6,0.7,0.8,0.9] or sorted(stored_config["rules"]) != ["min_mse","one_se"]:
        raise ValueError("Stage3B candidate policy family mismatch")
    pred=pd.read_csv(path/"outer_oof_predictions.csv",encoding="utf-8-sig",dtype={"image_id":str})
    if set(pred.model.unique()) != set(MODELS+["Fixed_A_3features"]):
        raise ValueError("Expected eight ENet policy models and Fixed A in Stage3B predictions")
    for (repeat, model),part in pred.groupby(["outer_repeat","model"]):
        if len(part)!=220 or part.image_id.nunique()!=220:
            raise ValueError(f"Incomplete Stage3B OOF: {repeat}/{model}")
    if pred.image_id.nunique()!=220:
        raise ValueError("Unexpected Stage3B OOF image IDs")
    return pred, stored_config


def folds_for_outer(train, a):
    labels=s2.q4_labels(train["Mech_Composite"].to_numpy(float))
    kf=RepeatedStratifiedKFold(n_splits=a.outer_folds,n_repeats=a.outer_repeats,random_state=a.seed)
    return list(kf.split(np.zeros(len(train)),labels))


def policy_evaluation_one_outer(k, outer_tr, all_features, ref, a, checkpoint_root):
    """For ONE outer fold, choose model family/threshold without outer validation data."""
    r=k//a.outer_folds+1
    fold=k%a.outer_folds+1
    outer_tr=outer_tr.reset_index(drop=True).copy()
    outer_tr["Mech_Composite"]=s2.make_target(outer_tr.GR.to_numpy(),outer_tr.DD.to_numpy(),s3.target_params(outer_tr))
    stratum=s2.q4_labels(outer_tr["Mech_Composite"].to_numpy(float))
    policy_splitter=StratifiedKFold(n_splits=a.policy_cv_folds,shuffle=True,random_state=a.seed+500000+k)
    splits=list(policy_splitter.split(np.zeros(len(outer_tr)),stratum))
    inner_args=SimpleNamespace(
        outer_repeats=a.policy_cv_folds,
        outer_folds=a.policy_cv_folds,
        stability_reps=a.stability_reps_inner,
        inner_cv=a.inner_enet_cv,
        sample_fraction=0.8,
        threshold_values=[0.60,0.70,0.80,0.90],
        rule_values=["one_se","min_mse"],
        seed=a.seed+700000+10000*k,
        n_jobs=a.n_jobs,
        batch_size=a.batch_size,
        resume=a.resume,
    )
    checkpoint_dir=checkpoint_root/f"outer_r{r:02d}_f{fold:02d}"
    checkpoint_dir.mkdir(parents=True,exist_ok=True)
    pred_frames=[]
    fold_metrics_frames=[]
    for j,(tr_idx,val_idx) in enumerate(splits,1):
        msg(f"Outer {k+1}/{a.outer_folds*a.outer_repeats} - policy-inner {j}/{a.policy_cv_folds}: training pipeline on {len(tr_idx)} samples")
        feat_rows, predictions, metrics, _ = s3.analyze_fold(
            r=1,f=j,tr_idx=tr_idx,va_idx=val_idx,
            train=outer_tr,features=all_features,stage2b_cfg=ref,
            args=inner_args,checkpoint_dir=checkpoint_dir
        )
        pp=pd.DataFrame(predictions)
        mm=pd.DataFrame(metrics)
        pp["policy_inner_fold"]=j
        mm["policy_inner_fold"]=j
        pred_frames.append(pp)
        fold_metrics_frames.append(mm)
    result=pd.concat(pred_frames,ignore_index=True)
    fold_met=pd.concat(fold_metrics_frames,ignore_index=True)
    rows=[]
    for model in MODELS:
        subset=result.loc[result.model==model]
        if len(subset)!=len(outer_tr) or subset.image_id.nunique()!=len(outer_tr):
            raise ValueError(f"Missing inner policy CV predictions for {model}")
        yy=subset.y_true.to_numpy(float)
        yp=subset.y_pred.to_numpy(float)
        avg_features=float(fold_met.loc[fold_met.model==model,"n_features"].mean())
        rows.append({
            "outer_repeat":r,"outer_fold":fold,"candidate_model":model,
            "policy_cv_oof_mse":float(np.mean((yy-yp)**2)),
            "policy_cv_oof_mae":float(np.mean(np.abs(yy-yp))),
            "policy_cv_mean_features":avg_features,
        })
    scores=pd.DataFrame(rows).sort_values(["policy_cv_oof_mse","policy_cv_mean_features","candidate_model"]).reset_index(drop=True)
    winner=scores.iloc[0]
    msg(f"Outer {k+1} chosen {winner.candidate_model}; policy CV MSE={winner.policy_cv_oof_mse:.5f}; mean size={winner.policy_cv_mean_features:.2f}")
    return scores,{
        "outer_repeat":r,"outer_fold":fold,
        "selected_policy":str(winner.candidate_model),
        "selection_metric":"3-fold inner policy-CV OOF MSE; tie-break smaller model",
        "inner_policy_cv_mse":float(winner.policy_cv_oof_mse),
        "inner_policy_mean_model_size":float(winner.policy_cv_mean_features),
        "n_outer_train":len(outer_tr),
    }


def summarize_outer_selection(a, selection, original_oof, out):
    pred_rows=[]
    for choice in selection.itertuples():
        match=(original_oof.outer_repeat==choice.outer_repeat)&(original_oof.outer_fold==choice.outer_fold)
        pool=original_oof.loc[match].copy()
        selected=pool.loc[pool.model==choice.selected_policy].copy()
        if len(selected)!=220//a.outer_folds:
            raise ValueError(f"Unexpected size of outer held-out fold {choice.outer_repeat}/{choice.outer_fold}")
        selected["selected_policy"]=choice.selected_policy
        selected["selection_policy_cv_mse"]=choice.inner_policy_cv_mse
        selected["strategy"]="Policy_chosen_inside_outer_train"
        pred_rows.append(selected)
    nested=pd.concat(pred_rows,ignore_index=True)
    atomic_csv(nested,out/"selection_aware_oof_predictions.csv")
    if len(selection)!=a.outer_repeats*a.outer_folds:
        msg("PARTIAL: no full-sample performance summary until all outer folds complete")
        return

    if (nested.groupby("outer_repeat").image_id.nunique()!=220).any():
        raise ValueError("Selection-aware outer CV does not contain exactly 220 unique samples per repeat")
    summary=[]
    for (r,),part in nested.groupby(["outer_repeat"]):
        x=part.y_true.to_numpy(float); y=part.y_pred.to_numpy(float)
        summary.append({"outer_repeat":int(r),"strategy":"Policy_chosen_inside_outer_train","n_samples":len(part),
            "R2":float(r2_score(x,y)),"RMSE":float(np.sqrt(mean_squared_error(x,y))),
            "MAE":float(mean_absolute_error(x,y))})
    comp=original_oof.loc[original_oof.model=="Fixed_A_3features"].copy()
    for r,part in comp.groupby("outer_repeat"):
        x=part.y_true.to_numpy(float); y=part.y_pred.to_numpy(float)
        summary.append({"outer_repeat":int(r),"strategy":"Fixed_A_3features","n_samples":len(part),
            "R2":float(r2_score(x,y)),"RMSE":float(np.sqrt(mean_squared_error(x,y))),
            "MAE":float(mean_absolute_error(x,y))})
    reps=pd.DataFrame(summary).sort_values(["strategy","outer_repeat"])
    atomic_csv(reps,out/"selection_aware_repeat_metrics.csv")
    ss=[]
    for model,part in reps.groupby("strategy"):
        row={"strategy":model,"n_repeats":len(part)}
        for metric in ("R2","RMSE","MAE"):
            row[metric+"_mean"]=float(part[metric].mean())
            row[metric+"_sd_between_repeats"]=float(part[metric].std(ddof=1))
        ss.append(row)
    atomic_csv(pd.DataFrame(ss),out/"selection_aware_summary.csv")
    comparison(nested,comp,a.bootstrap,a.seed,out)


def comparison(nested,baseline,nboot,seed,out):
    one=nested.loc[nested.outer_repeat==1].set_index("image_id").sort_index()
    base=baseline.loc[baseline.outer_repeat==1].set_index("image_id").loc[one.index]
    if len(one)!=220 or not np.allclose(one.y_true.to_numpy(),base.y_true.to_numpy(),atol=1e-12):
        raise ValueError("OOF first-repeat paired y_true mismatch")
    y=one.y_true.to_numpy(float); p=one.y_pred.to_numpy(float); b=base.y_pred.to_numpy(float)
    rng=np.random.default_rng(seed+310001)
    rec=[]
    for metric in ("R2","RMSE","MAE"):
        def fn(yt,yp):
            if metric=="R2":return float(r2_score(yt,yp))
            if metric=="RMSE":return float(np.sqrt(mean_squared_error(yt,yp)))
            return float(mean_absolute_error(yt,yp))
        diffs=[]
        for _ in range(nboot):
            ids=rng.integers(0,len(y),len(y))
            if np.var(y[ids])<1e-12:continue
            diffs.append(fn(y[ids],p[ids])-fn(y[ids],b[ids]))
        rec.append({"metric":metric,"point_difference_selected_minus_A":fn(y,p)-fn(y,b),
                    "bootstrap95_low":float(np.quantile(diffs,.025)),
                    "bootstrap95_high":float(np.quantile(diffs,.975)),
                    "n_bootstrap":len(diffs)})
    atomic_csv(pd.DataFrame(rec),out/"paired_vs_A_first_repeat_bootstrap95.csv")


def main():
    a=args_parser()
    out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=True)
    data=Path(a.data).resolve();grade=Path(a.grading).resolve();refp=Path(a.reference_config).resolve()
    ref=json.loads(refp.read_text(encoding="utf-8"))
    hash_a={data.name:s2.sha256_file(data),grade.name:s2.sha256_file(grade)}
    if hash_a != ref["input_sha256"]:raise RuntimeError("Stage2B reference SHA256 mismatch")
    train,features,target=s2.load_training_data(data,grade)
    original,stage3b_config=verify_stage3b(a,train,ref)
    configs={
        "audit_scope":"Method policy selection among two ElasticNet rules and four frequency cutoffs, FULLY INSIDE outer training folds",
        "n_train":len(train),"test_samples_used":0,
        "outer_folds":a.outer_folds,"outer_repeats":a.outer_repeats,
        "policy_inner_folds":a.policy_cv_folds,
        "stability_reps_each_policy_inner_fold":a.stability_reps_inner,
        "elastic_net_cv_folds":a.inner_enet_cv,
        "policies":MODELS,"selection_objective":"policy inner CV OOF MSE, no outer val access",
        "run_started":datetime.now(timezone.utc).isoformat(),"seed":a.seed,
        "reference_input_hashes":hash_a,
        "source_stage3b_config_sha256":s2.sha256_file(Path(a.stage3b_results).resolve()/"run_config_stage3b.json"),
        "note":"Retrospective sensitivity analysis. Does not retroactively preregister or erase previous method exploration.",
        "software":{"python":sys.version.split()[0],"numpy":np.__version__,"pandas":pd.__version__,"sklearn":sklearn_version},
    }
    p_cfg=out/"run_config_stage3c.json"
    if p_cfg.exists():
        old=json.loads(p_cfg.read_text())
        old.pop("run_started",None); configs.pop("run_started",None)
        if old!=configs:raise RuntimeError("Output config differs; use a new directory")
        if not a.resume:raise RuntimeError("Existing output; specify --resume")
    else:
        p_cfg.write_text(json.dumps(configs,ensure_ascii=False,indent=2))
    source_splits=folds_for_outer(train,a)
    selections=[];score_blocks=[]
    checkpoint_dir=out/"policy_inner_checkpoints";checkpoint_dir.mkdir(exist_ok=True)
    for k,(tr_idx,va_idx) in enumerate(source_splits):
        r=k//a.outer_folds+1;f=k%a.outer_folds+1
        audit_file=out/f"policy_outer_r{r:02d}_f{f:02d}.csv"
        selection_file=out/f"choice_outer_r{r:02d}_f{f:02d}.json"
        if audit_file.exists() and selection_file.exists():
            if not a.resume:raise RuntimeError("Existing checkpoint; specify --resume")
            score=pd.read_csv(audit_file)
            choice=json.loads(selection_file.read_text())
        else:
            score,choice=policy_evaluation_one_outer(k,train.iloc[tr_idx].copy(),features,ref,a,checkpoint_dir)
            atomic_csv(score,audit_file)
            selection_file.write_text(json.dumps(choice,indent=2))
        score_blocks.append(score)
        selections.append(choice)
        atomic_csv(pd.DataFrame(selections),out/"selected_policies_by_outer_fold.csv")
        atomic_csv(pd.concat(score_blocks,ignore_index=True),out/"inner_policy_cv_scores.csv")
        msg(f"Selection audit completed {k+1}/{len(source_splits)} outer folds")
        if a.max_outer and len(selections)>=a.max_outer:break
    summarize_outer_selection(a,pd.DataFrame(selections),original,out)
    if len(selections)==len(source_splits): msg("STAGE 3C COMPLETE: model choice was inside each outer TRAIN fold; previous exploration remains retrospective")

if __name__=="__main__":main()
