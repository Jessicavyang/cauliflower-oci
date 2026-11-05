#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 3B: leakage-safe outer CV of the *complete* Stage 2B selection policy.

For every outer TRAIN fold independently:
  (1) define the mechanical target with outer-TRAIN GR/DD means and SDs;
  (2) screen all 20 predictors with Pearson on outer TRAIN only;
  (3) repeat 80%-without-replacement stratified subsampling;
  (4) within each subsample redo Pearson screening and tune Elastic Net via
      10-fold CV with fold-training-only screening and scaling;
  (5) select by the Stage 2B 1-SE rule and (optionally) min-MSE, count frequencies;
  (6) retain features above each stability threshold; refit the original
      standardized OCI via OLS ON OUTER TRAIN ONLY;
  (7) predict completely held-out outer samples.

This script imports the unmodified Stage 2B implementation to preserve exact
selection rules. It shares each repetition's CV grid between 1-SE and min-MSE.
Threshold comparison and fixed original three-feature OCI are provided.
The historical 55-image test set is used ONLY after all outer CV, to evaluate
full-TRAIN refits sourced from the supplied Stage 2B 1000-rep checkpoint.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import warnings

# Avoid oversubscription before loading NumPy/SciPy.
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"

import numpy as np
import pandas as pd
import scipy
from joblib import Parallel, delayed
from sklearn import __version__ as sklearn_version
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import enet_path
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedShuffleSplit

import stage2b_elasticnet_stability as s2b

ORIGINAL_A = ["Struct_num_gaps", "Struct_gap_ratio", "Color_S"]


def log(msg):
    print(f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {msg}", flush=True)


def parse_args():
    ap = argparse.ArgumentParser(
        description="Stage 3B nested CV: Stage 2B Elastic Net 1-SE/min-MSE stability selection + standardized OLS OCI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--data", default="input/cauliflower_analysis_final2.csv")
    ap.add_argument("--grading", default="input/Cauliflower_Final_Grading_Auto.csv")
    ap.add_argument("--reference-config", default="reference_stage2b_one_se/run_config.json")
    ap.add_argument("--reference-checkpoint", default="reference_stage2b_one_se/repetitions_checkpoint.csv")
    ap.add_argument("--out", default="results_stage3b")
    ap.add_argument("--outer-folds", type=int, default=5)
    ap.add_argument("--outer-repeats", type=int, default=3)
    ap.add_argument("--stability-reps", type=int, default=100)
    ap.add_argument("--inner-cv", type=int, default=10)
    ap.add_argument("--sample-fraction", type=float, default=0.8)
    ap.add_argument("--thresholds", default="0.60,0.70,0.80,0.90")
    ap.add_argument("--primary-threshold", type=float, default=0.70)
    ap.add_argument("--rules", default="one_se,min_mse", help="one_se,min_mse or one_se")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=10)
    ap.add_argument("--bootstrap-reps", type=int, default=2000)
    ap.add_argument("--max-outer-folds", type=int, default=0, help="Dev only; 0 means all")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="Single-fold fast path; NOT scientific results")
    ap.add_argument("--skip-historical-holdout", action="store_true")
    ap.add_argument("--verify-reference-only", action="store_true")
    args = ap.parse_args()
    try:
        args.threshold_values = sorted(set(float(s.strip()) for s in args.thresholds.split(",")))
        args.rule_values = list(dict.fromkeys(s.strip() for s in args.rules.split(",") if s.strip()))
    except ValueError:
        ap.error("Invalid comma-separated thresholds or rules")
    if not args.rule_values or any(v not in ("one_se", "min_mse") for v in args.rule_values):
        ap.error("--rules must contain one_se and/or min_mse")
    if not args.threshold_values or any(not (0.5 < x < 1) for x in args.threshold_values):
        ap.error("Thresholds must be >0.50 and <1.00")
    if args.primary_threshold not in args.threshold_values:
        ap.error("--primary-threshold must be in --thresholds")
    if args.n_jobs < 1 or args.batch_size < 1 or args.stability_reps < 1:
        ap.error("Invalid n-jobs/batch-size/stability-reps")
    if args.outer_folds < 2 or args.outer_repeats < 1 or args.inner_cv < 2:
        ap.error("Invalid CV folds/repeats")
    if not (0 < args.sample_fraction < 1):
        ap.error("Invalid sample fraction")
    if args.bootstrap_reps < 100:
        ap.error("--bootstrap-reps must be >=100")
    if args.smoke:
        args.outer_repeats = 1
        args.stability_reps = 3
        args.inner_cv = 3
        args.max_outer_folds = 1
        args.bootstrap_reps = 100
    return args


def stage2b_parameters(args, config, seed):
    en = config["elastic_net"]
    if en.get("selection_rule") != "one_se":
        raise ValueError("Expected uploaded reference Stage 2B selection_rule='one_se'")
    pear = config["pearson"]
    return SimpleNamespace(
        sample_fraction=args.sample_fraction,
        cv_folds=args.inner_cv,
        gr_cut=float(pear["GR_abs_r_min"]),
        dd_cut=float(pear["DD_abs_r_min"]),
        p_cut=float(pear["two_sided_raw_p_max"]),
        ratios=[float(x) for x in en["l1_ratio_grid"]],
        alphas=np.asarray(en["alpha_grid_descending"], dtype=float),
        coef_eps=float(en["nonzero_cutoff_standardized_X_beta"]),
        practical_eps=float(en["secondary_practical_cutoff"]),
        max_iter=int(en["max_iter"]),
        tol=float(en["tol"]),
        seed=int(seed),
    )


def target_params(frame):
    return {
        "GR_mean": float(frame.GR.mean()),
        "GR_sd_ddof1": float(frame.GR.std(ddof=1)),
        "DD_mean": float(frame.DD.mean()),
        "DD_sd_ddof1": float(frame.DD.std(ddof=1)),
    }


def standard_ols(train, valid, features):
    """Frozen TRAIN-only X standardization, OLS intercept, fold-defined target."""
    target = target_params(train)
    y_train = s2b.make_target(train.GR.to_numpy(), train.DD.to_numpy(), target)
    y_valid = s2b.make_target(valid.GR.to_numpy(), valid.DD.to_numpy(), target)
    if not features:
        mu_y = float(y_train.mean())
        return y_valid, np.repeat(mu_y, len(valid)), {
            "features": [], "intercept_standardized": mu_y,
            "intercept_raw": mu_y, "coefficients_standardized": [],
            "x_mean": [], "x_sd_ddof1": [], "target_parameters": target,
            "fallback_intercept_only": True,
        }
    x_train = train[features].to_numpy(float)
    x_valid = valid[features].to_numpy(float)
    mu = x_train.mean(axis=0)
    sd = x_train.std(axis=0, ddof=1)
    sd = np.where(sd < 1e-12, 1.0, sd)
    z_train = (x_train - mu) / sd
    z_valid = (x_valid - mu) / sd
    design = np.column_stack([np.ones(len(z_train)), z_train])
    fitted, *_ = np.linalg.lstsq(design, y_train, rcond=None)
    y_pred = np.column_stack([np.ones(len(z_valid)), z_valid]) @ fitted
    b0 = float(fitted[0])
    coefficients = fitted[1:]
    raw_intercept = float(b0 - np.sum(coefficients * (mu / sd)))
    return y_valid, y_pred, {
        "features": list(features), "intercept_standardized": b0,
        "intercept_raw": raw_intercept,
        "coefficients_standardized": [float(v) for v in coefficients],
        "x_mean": [float(v) for v in mu],
        "x_sd_ddof1": [float(v) for v in sd],
        "target_parameters": target, "fallback_intercept_only": False,
    }


def select_one_se(cv_records, min_choice, sub, sub_eligible, y_sub, parms):
    """Exact Stage 2B 1-SE tie-breaking rule, using one shared CV evaluation."""
    cutoff = min_choice["mean_mse"] + min_choice["se_mse"] + 1e-12
    accepted = [z for z in cv_records if z["mean_mse"] <= cutoff]
    x = sub[sub_eligible].to_numpy(float)
    xm = x.mean(axis=0)
    xs = x.std(axis=0, ddof=0)
    xs = np.where(xs < 1e-12, 1.0, xs)
    x_scaled = np.asfortranarray((x - xm) / xs, dtype=np.float64)
    ym = float(np.mean(y_sub))
    yc = np.ascontiguousarray(y_sub - ym, dtype=np.float64)
    counts = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for ratio in parms.ratios:
            alphas = sorted({z["alpha"] for z in accepted if z["l1_ratio"] == ratio}, reverse=True)
            if not alphas:
                continue
            _, path, _ = enet_path(
                x_scaled, yc, l1_ratio=float(ratio),
                alphas=np.asarray(alphas, float),
                max_iter=parms.max_iter, tol=parms.tol,
            )
            for j, alpha in enumerate(alphas):
                counts[(ratio, alpha)] = int(np.sum(np.abs(path[:, j]) > parms.coef_eps))
    return min(accepted, key=lambda z: (
        counts[(z["l1_ratio"], z["alpha"])],
        -z["alpha"], -z["l1_ratio"], z["mean_mse"],
    ))


def paired_repetition(rep, fold_train, base_pool, parms, rules):
    """ONE CV grid per repetition; exactly reconstructs Stage 2B both rules."""
    y = fold_train["Mech_Composite"].to_numpy(float)
    strata = s2b.q4_labels(y)
    splitter = StratifiedShuffleSplit(
        n_splits=1, train_size=parms.sample_fraction,
        random_state=parms.seed + rep,
    )
    idx, _ = next(splitter.split(np.zeros(len(fold_train)), strata))
    sub = fold_train.iloc[idx].reset_index(drop=True)
    y_sub = y[idx]
    screen = s2b.pearson_screen(sub, base_pool, parms)
    eligible = screen.loc[screen.eligible, "feature"].tolist()
    subhash = hashlib.sha256(";".join(sorted(sub.image_id)).encode()).hexdigest()
    if not eligible:
        return [{
            "rep": rep, "rule": rule, "n_subsample": len(sub),
            "subsample_id_sha256": subhash, "eligible": "", "selected": "",
            "n_selected": 0, "n_eligible": 0, "best_alpha": np.nan,
            "best_l1_ratio": np.nan, "best_cv_mse": np.nan,
            "n_convergence_warnings": 0,
        } for rule in rules]
    cv_records, n_cv_warn = s2b.fold_specific_cv(sub, y_sub, base_pool, parms, rep)
    best_min = min(cv_records, key=lambda z: (z["mean_mse"], -z["alpha"], -z["l1_ratio"]))
    choices = {"min_mse": best_min}
    if "one_se" in rules:
        choices["one_se"] = select_one_se(cv_records, best_min, sub, eligible, y_sub, parms)
    rows = []
    for rule in rules:
        chosen = choices[rule]
        elig, selected, coefficients, intercept, n_warn, practical = s2b.fit_full_subsample(
            sub, y_sub, base_pool, chosen["alpha"], chosen["l1_ratio"], parms,
        )
        rows.append({
            "rep": rep, "rule": rule, "n_subsample": len(sub),
            "subsample_id_sha256": subhash, "eligible": ";".join(elig),
            "n_eligible": len(elig), "selected": ";".join(sorted(selected)),
            "selected_practical": ";".join(sorted(practical)),
            "n_selected": len(selected), "best_alpha": chosen["alpha"],
            "best_l1_ratio": chosen["l1_ratio"], "best_cv_mse": chosen["mean_mse"],
            "n_convergence_warnings": n_cv_warn + n_warn,
        })
    return rows


def fold_frequency(rows, features, rule):
    sub = rows.loc[rows.rule == rule]
    sets = [set(s.split(";")) if isinstance(s, str) and s else set() for s in sub.selected]
    return {f: sum(f in s for s in sets) / len(sets) for f in features}


def atomic_csv(df, path):
    tmp = path.with_suffix(".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, path)


def metric_values(y, pred):
    return {
        "R2": float(r2_score(y, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y, pred))),
        "MAE": float(mean_absolute_error(y, pred)),
    }


def analyze_fold(r, f, tr_idx, va_idx, train, features, stage2b_cfg, args, checkpoint_dir):
    fold_train = train.iloc[tr_idx].copy().reset_index(drop=True)
    fold_valid = train.iloc[va_idx].copy().reset_index(drop=True)
    fold_train["Mech_Composite"] = s2b.make_target(
        fold_train.GR.to_numpy(), fold_train.DD.to_numpy(), target_params(fold_train)
    )
    pearson_args = stage2b_parameters(args, stage2b_cfg, args.seed + 10000*r + 1000*f)
    initial = s2b.pearson_screen(fold_train, features, pearson_args)
    pool = initial.loc[initial.eligible, "feature"].tolist()
    if not pool:
        log(f"[WARN] No outer-fold Pearson candidates in repeat={r} fold={f}")
    cp = checkpoint_dir / f"outer_r{r:02d}_f{f:02d}.csv"
    if cp.exists():
        if not args.resume:
            raise RuntimeError(f"Checkpoint {cp} already exists; use --resume")
        reps = pd.read_csv(cp).fillna("")
        finished = set(reps.loc[reps.rule == args.rule_values[0], "rep"].astype(int))
        if (reps.groupby("rep").rule.nunique() != len(args.rule_values)).any():
            raise RuntimeError(f"Partial/inconsistent rule checkpoint in {cp}")
        if finished != set(range(len(finished))):
            raise RuntimeError(f"Non-contiguous checkpoint rep indices in {cp}")
    else:
        reps = pd.DataFrame()
        finished = set()
    if len(finished) > args.stability_reps:
        raise RuntimeError(f"Checkpoint {cp} has extra repetitions")
    for start in range(len(finished), args.stability_reps, args.batch_size):
        indices = list(range(start, min(start + args.batch_size, args.stability_reps)))
        if args.n_jobs == 1:
            batch = [paired_repetition(i, fold_train, pool, pearson_args, args.rule_values) for i in indices]
        else:
            batch = Parallel(n_jobs=args.n_jobs, backend="loky")(
                delayed(paired_repetition)(i, fold_train, pool, pearson_args, args.rule_values)
                for i in indices
            )
        reps = pd.concat([reps, pd.DataFrame([row for group in batch for row in group])], ignore_index=True)
        reps = reps.sort_values(["rep", "rule"]).reset_index(drop=True)
        atomic_csv(reps, cp)
        log(f"Outer {r}/{args.outer_repeats} fold {f}/{args.outer_folds}: {min(start+len(indices),args.stability_reps)}/{args.stability_reps} stability reps")

    features_out = []
    preds_out = []
    metrics_out = []
    cfg_out = {"repeat": r, "fold": f, "n_train": len(fold_train),
               "n_validation": len(fold_valid), "initial_candidate_pool": pool,
               "target_params": target_params(fold_train), "policies": {}}
    # The fixed model is a comparator, not a nested-selected model.
    configurations = [("Fixed_A_3features", "fixed", None, ORIGINAL_A, {})]
    for rule in args.rule_values:
        freqs = fold_frequency(reps, features, rule)
        for feature in features:
            features_out.append({"outer_repeat": r, "outer_fold": f, "rule": rule,
                                 "feature": feature, "selection_frequency": freqs[feature],
                                 "n_repetitions": args.stability_reps,
                                 "outer_initial_pearson_candidate": feature in pool})
        for threshold in args.threshold_values:
            chosen = [feature for feature in features if freqs[feature] >= threshold]
            configurations.append((f"ENet_{rule}_thr{int(round(threshold*100)):02d}", rule, threshold, chosen, freqs))

    for model, rule, threshold, selected, freqs in configurations:
        y_val, predictions, fit = standard_ols(fold_train, fold_valid, selected)
        met = metric_values(y_val, predictions)
        metrics_out.append({"outer_repeat": r, "outer_fold": f, "model": model,
                            "rule": rule, "threshold": threshold,
                            "n_features": len(selected), "features": ";".join(selected),
                            "n_outer_train": len(fold_train), "n_outer_validation": len(fold_valid), **met})
        for i, (_, row) in enumerate(fold_valid.iterrows()):
            preds_out.append({"outer_repeat": r, "outer_fold": f, "model": model,
                              "image_id": row.image_id, "y_true": float(y_val[i]),
                              "y_pred": float(predictions[i]), "residual": float(y_val[i]-predictions[i])})
        cfg_out["policies"][model] = {"features": selected, "OLS_fit": fit}
    return features_out, preds_out, metrics_out, cfg_out


def finalize_results(args, out, fold_results, n_completed, n_expected, all_features):
    if not fold_results:
        return
    feature_records, pred_records, fold_metrics, cfg_rows = [], [], [], []
    for item in fold_results:
        feature_records.extend(item[0]); pred_records.extend(item[1]); fold_metrics.extend(item[2]); cfg_rows.append(item[3])
    freq_df = pd.DataFrame(feature_records)
    pred_df = pd.DataFrame(pred_records)
    fold_df = pd.DataFrame(fold_metrics)
    atomic_csv(freq_df, out / "outer_fold_feature_frequencies.csv")
    atomic_csv(pred_df, out / "outer_oof_predictions.csv")
    atomic_csv(fold_df, out / "outer_fold_metrics.csv")
    (out / "outer_fold_model_parameters.json").write_text(
        json.dumps(cfg_rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if n_completed < n_expected:
        log(f"PARTIAL: {n_completed}/{n_expected} outer folds. No pooled performance summary created.")
        return

    records = []
    models = sorted(pred_df.model.unique())
    for (repeat, model), group in pred_df.groupby(["outer_repeat", "model"], sort=True):
        if len(group) != 220 or group.image_id.nunique() != 220:
            raise RuntimeError(f"Outer OOF incomplete or duplicated: repeat={repeat} model={model}")
        row = {"outer_repeat": repeat, "model": model, "n_samples": len(group)}
        row.update(metric_values(group.y_true.to_numpy(), group.y_pred.to_numpy()))
        records.append(row)
    rep_df = pd.DataFrame(records)
    atomic_csv(rep_df, out / "outer_repeat_metrics.csv")
    summary = []
    for model, part in rep_df.groupby("model"):
        row = {"model": model, "n_outer_repeats": len(part)}
        for metric in ("R2", "RMSE", "MAE"):
            row[f"{metric}_mean"] = float(part[metric].mean())
            row[f"{metric}_sd_between_repeats"] = float(part[metric].std(ddof=1)) if len(part) > 1 else np.nan
        summary.append(row)
    atomic_csv(pd.DataFrame(summary).sort_values("model"), out / "outer_cv_summary.csv")

    feature_summary = freq_df.groupby(["rule", "feature"], as_index=False).agg(
        mean_selection_frequency=("selection_frequency", "mean"),
        sd_selection_frequency=("selection_frequency", "std"),
        n_outer_folds=("selection_frequency", "size"),
    )
    atomic_csv(feature_summary, out / "outer_feature_selection_summary.csv")

    for rule in args.rule_values:
        sub = fold_df[fold_df.rule == rule]
        th_sens = sub.groupby(["model", "threshold"], as_index=False).agg(
            mean_n_selected=("n_features", "mean"),
            median_n_selected=("n_features", "median"),
            number_empty_models=("n_features", lambda s: int((s == 0).sum())),
        )
        atomic_csv(th_sens, out / f"outer_threshold_sensitivity_{rule}.csv")

    # CI from first repeat only: one OOF prediction per observation; paired resampling.
    first = pred_df[pred_df.outer_repeat == 1].copy()
    baseline = first[first.model == "Fixed_A_3features"].set_index("image_id").sort_index()
    ids = baseline.index.to_numpy()
    n = len(ids)
    assert n == 220
    rng = np.random.default_rng(args.seed + 987654)
    draws = rng.integers(0, n, size=(args.bootstrap_reps, n))
    baseline_y = baseline.y_true.to_numpy(float)
    base_pred = baseline.y_pred.to_numpy(float)
    paired = []
    ci_rows = []
    for model in models:
        part = first[first.model == model].set_index("image_id").loc[ids]
        y = part.y_true.to_numpy(float)
        p = part.y_pred.to_numpy(float)
        if not np.allclose(y, baseline_y, atol=1e-12, rtol=1e-12):
            raise RuntimeError("Target differs across models within the same outer fold")
        obs = metric_values(y, p)
        base = metric_values(y, base_pred)
        samples = {k: [] for k in obs}
        delta = {k: [] for k in obs}
        for take in draws:
            yt = y[take]
            if np.var(yt) < 1e-12:
                continue
            va = metric_values(yt, p[take]); vb = metric_values(yt, base_pred[take])
            for key in obs:
                samples[key].append(va[key]); delta[key].append(va[key] - vb[key])
        for key in obs:
            ci_rows.append({"model": model, "metric": key,
                            "first_repeat_point": obs[key],
                            "bootstrap95_low": float(np.quantile(samples[key], .025)),
                            "bootstrap95_high": float(np.quantile(samples[key], .975)),
                            "n_bootstrap": len(samples[key])})
            if model != "Fixed_A_3features":
                paired.append({"model": model, "reference_model": "Fixed_A_3features", "metric": key,
                               "difference_model_minus_A": obs[key] - base[key],
                               "bootstrap95_low": float(np.quantile(delta[key], .025)),
                               "bootstrap95_high": float(np.quantile(delta[key], .975)),
                               "n_bootstrap": len(delta[key])})
    atomic_csv(pd.DataFrame(ci_rows), out / "outer_first_repeat_bootstrap95.csv")
    atomic_csv(pd.DataFrame(paired), out / "paired_model_differences_vs_A.csv")
    log("Full outer nested CV results and 95% bootstrap summaries written.")


def historical_refit(args, out, train, ref_config, all_features, input_hashes):
    """Secondary historical 55 TEST: frozen TRAIN-only fit; no threshold selection."""
    if args.skip_historical_holdout:
        return
    reference_path = Path(args.reference_checkpoint).resolve()
    if not reference_path.is_file():
        log("Historical refit skipped: reference checkpoint not available.")
        return
    ref = pd.read_csv(reference_path).sort_values("rep").reset_index(drop=True)
    if len(ref) != 1000 or ref.rep.tolist() != list(range(1000)):
        raise ValueError("Expected all 1000 Stage 2B one_se repetitions in reference checkpoint")
    if (ref.selection_rule != "one_se").any():
        raise ValueError("Reference checkpoint is not from Stage 2B one_se")
    if input_hashes != ref_config["input_sha256"]:
        raise ValueError("Stage 2B reference file hashes do not match current inputs")
    ss = [set(s.split(";")) if isinstance(s, str) and s else set() for s in ref.selected]
    freq = {f: sum(f in s for s in ss)/len(ss) for f in all_features}
    grading = pd.read_csv(args.grading, encoding="utf-8-sig", dtype={"image_id":str})
    d = pd.read_csv(args.data, encoding="utf-8-sig", dtype={"image_id":str})
    combined = d.merge(grading[["image_id", "Set"]], on="image_id", validate="one_to_one")
    hold = combined.loc[combined.Set.astype(str).str.lower() == "test"].copy()
    if len(hold) != 55:
        raise ValueError("Historical holdout must have exactly 55 records")
    fit_records, predictions, param_records = [], [], {}
    combos = [("Fixed_A_3features", ORIGINAL_A)]
    for th in args.threshold_values:
        combos.append((f"ENet_one_se_thr{int(round(th*100)):02d}",
                       [f for f in all_features if freq[f] >= th]))
    for model, features in combos:
        yt, yp, fit = standard_ols(train, hold, features)
        metrics = metric_values(yt, yp)
        fit_records.append({"model":model, "n_holdout":55, "n_features":len(features),
                            "features":";".join(features), **metrics})
        param_records[model] = fit
        for image_id, obs, pred in zip(hold.image_id, yt, yp):
            predictions.append({"model":model, "image_id":image_id,
                                "y_true":float(obs), "y_pred":float(pred)})
    atomic_csv(pd.DataFrame(fit_records), out / "historical_holdout_metrics.csv")
    atomic_csv(pd.DataFrame(predictions), out / "historical_holdout_predictions.csv")
    atomic_csv(pd.DataFrame([{"feature":f, "selection_frequency":freq[f]} for f in all_features]),
               out / "full_training_one_se_reference_frequencies.csv")
    (out / "full_training_ols_parameters.json").write_text(
        json.dumps(param_records, indent=2, ensure_ascii=False), encoding="utf-8")
    log("Historical (previously inspected) 55-sample holdout analysis written; NOT an independent test.")


def main():
    args = parse_args()
    data_path = Path(args.data).expanduser().resolve()
    grade_path = Path(args.grading).expanduser().resolve()
    ref_path = Path(args.reference_config).expanduser().resolve()
    if not all(p.is_file() for p in (data_path, grade_path, ref_path)):
        raise FileNotFoundError("Input CSV or Stage 2B reference config is missing")
    ref = json.loads(ref_path.read_text(encoding="utf-8"))
    hashes = {data_path.name: s2b.sha256_file(data_path), grade_path.name: s2b.sha256_file(grade_path)}
    if hashes != ref["input_sha256"]:
        raise RuntimeError("Input SHA256 does not match uploaded Stage 2B one_se run_config")
    train, features, train_params = s2b.load_training_data(data_path, grade_path)
    if features != ref["all_20_features"]:
        raise RuntimeError("Feature order differs from Stage 2B reference")
    full_args = stage2b_parameters(args, ref, args.seed)
    full_screen = s2b.pearson_screen(train, features, full_args)
    full_pool = full_screen.loc[full_screen.eligible, "feature"].tolist()
    if full_pool != ref["initial_train_candidate_pool"]:
        raise RuntimeError("Train-only candidate pool differs from Stage 2B reference")
    if args.verify_reference_only:
        log(f"REFERENCE VERIFIED: input SHA256, 220/55 partition, and {len(full_pool)} candidates: {full_pool}")
        return

    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = out / "fold_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    cfg = {
        "workflow": "Stage3B nested CV: Stage2B 1-SE/min-MSE stability selection + OLS refit",
        "analysis_classification": "SMOKE_ONLY" if args.smoke else "FORMAL",
        "input_sha256": hashes, "reference_config_sha256": s2b.sha256_file(ref_path),
        "reference_stage2b_code_sha256": s2b.sha256_file(Path(s2b.__file__).resolve()),
        "outer_folds": args.outer_folds, "outer_repeats": args.outer_repeats,
        "stability_reps_per_outer_fold": args.stability_reps,
        "inner_cv": args.inner_cv, "sample_fraction": args.sample_fraction,
        "thresholds": args.threshold_values, "primary_threshold": args.primary_threshold,
        "rules": args.rule_values, "seed": args.seed,
        "max_outer_folds": args.max_outer_folds, "bootstrap_reps": args.bootstrap_reps,
        "note": "Threshold 70% chosen after Stage2B exploration; sensitivity 60/80/90%. No formal false discovery bound for 80% subsampling.",
        "metrics": "pooled out-of-fold R2/RMSE/MAE per complete outer repeat",
        "target_standardization": "GR/DD parameters fitted separately for EACH outer training fold",
        "final_prediction_model": "OLS with TRAIN-only standardized X; empty selection -> intercept-only",
        "python": os.sys.version.split()[0], "numpy": np.__version__,
        "pandas": pd.__version__, "scipy": scipy.__version__, "sklearn": sklearn_version,
    }
    cfg_path = out / "run_config_stage3b.json"
    if cfg_path.exists():
        old = json.loads(cfg_path.read_text(encoding="utf-8"))
        if old != cfg:
            raise RuntimeError("Run configuration changed. Use new --out folder.")
        if not args.resume:
            raise RuntimeError("Output already initialized. Use --resume or a fresh --out folder.")
    else:
        cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
        full_screen.to_csv(out / "full_220_training_pearson_audit.csv", index=False, encoding="utf-8-sig")
    log(f"Verified Stage2B reference. TRAIN={len(train)}, TEST=55 unused in outer CV. Full candidate pool: {full_pool}")
    if args.smoke:
        log("*** SMOKE mode only; results are not eligible for scientific reporting ***")

    # Stratification is only for OUTER split construction. Models and training
    # target parameters are fitted from each outer TRAIN fold independently.
    outer_strata = s2b.q4_labels(train["Mech_Composite"].to_numpy(float))
    cv = RepeatedStratifiedKFold(n_splits=args.outer_folds, n_repeats=args.outer_repeats, random_state=args.seed)
    splits = list(cv.split(np.zeros(len(train)), outer_strata))
    n_expected = len(splits)
    n_run = min(n_expected, args.max_outer_folds) if args.max_outer_folds else n_expected
    fold_results = []
    for k, (tr_idx, va_idx) in enumerate(splits[:n_run]):
        r = k // args.outer_folds + 1
        f = k % args.outer_folds + 1
        log(f"Begin outer repeat {r}/{args.outer_repeats}, fold {f}/{args.outer_folds}")
        result = analyze_fold(r, f, tr_idx, va_idx, train, features, ref, args, checkpoint_dir)
        fold_results.append(result)
        log(f"Completed outer {r}/{args.outer_repeats}, fold {f}/{args.outer_folds}")
    finalize_results(args, out, fold_results, n_run, n_expected, features)

    if n_run == n_expected and not args.smoke:
        historical_refit(args, out, train, ref, features, hashes)
        log("STAGE3B COMPLETED: report primary 70% threshold and 60/80/90% sensitivity without selecting based on holdout performance.")
    else:
        log("PARTIAL OR SMOKE: intentionally did not compute full CV/holdout conclusions.")


if __name__ == "__main__":
    main()
