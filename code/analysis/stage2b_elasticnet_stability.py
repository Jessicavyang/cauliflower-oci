#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 2B: Elastic Net stability selection for cauliflower OCI.

Training-only analysis:
- 220 TRAIN samples are used; 55 TEST samples are excluded.
- The original Pearson candidate pool is derived from the 220 TRAIN samples.
- In each 80% without-replacement subsample, Pearson screening is repeated.
- In each inner CV fold, Pearson screening is re-run using that fold's training
  portion only; X scaling and Y centering are also fitted on that fold only.
- Elastic Net alpha and l1_ratio are tuned by 10-fold CV.
- No feature is forced into the model.

The inner-CV MSE is for hyperparameter tuning, not an unbiased prediction
performance estimate. Use nested outer CV for performance claims.
"""

from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

# Repetitions are parallelized by joblib; avoid nested BLAS oversubscription.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np
import pandas as pd
import scipy
from scipy.stats import pearsonr
from joblib import Parallel, delayed
from sklearn import __version__ as sklearn_version
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import enet_path
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

ID_COL = "image_id"
NON_FEATURES = {ID_COL, "GR", "DD"}
EXPECTED_TOTAL, EXPECTED_TRAIN, EXPECTED_TEST = 275, 220, 55


def log(message):
    print(f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {message}", flush=True)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_args():
    p = argparse.ArgumentParser(
        description="Stage 2B Elastic Net stability selection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data", default="input/cauliflower_analysis_final2.csv")
    p.add_argument("--grading", default="input/Cauliflower_Final_Grading_Auto.csv")
    p.add_argument("--out", default="results_stage2b_min_mse")
    p.add_argument("--reps", type=int, default=1000)
    p.add_argument("--sample-fraction", type=float, default=0.80)
    p.add_argument("--cv-folds", type=int, default=10)
    p.add_argument("--gr-cut", type=float, default=0.35)
    p.add_argument("--dd-cut", type=float, default=0.40)
    p.add_argument("--p-cut", type=float, default=0.001)
    p.add_argument("--alpha-min", type=float, default=0.003)
    p.add_argument("--alpha-max", type=float, default=1.0)
    p.add_argument("--n-alphas", type=int, default=30)
    p.add_argument("--l1-ratios", default="0.10,0.25,0.50,0.75,0.90,0.95")
    p.add_argument("--selection-rule", choices=("min_mse", "one_se"), default="min_mse")
    p.add_argument("--coef-eps", type=float, default=1e-6,
                   help="Primary nonzero coefficient cutoff on standardized X")
    p.add_argument("--practical-eps", type=float, default=0.02,
                   help="Secondary coefficient-magnitude summary cutoff")
    p.add_argument("--max-iter", type=int, default=20000)
    p.add_argument("--tol", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-jobs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=20)
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()

    if a.reps < 1 or a.n_jobs < 1 or a.batch_size < 1:
        p.error("reps, n-jobs and batch-size must be positive")
    if not (0 < a.sample_fraction < 1) or not (2 <= a.cv_folds <= 20):
        p.error("Invalid sample-fraction or cv-folds")
    if not (0 < a.alpha_min < a.alpha_max) or a.n_alphas < 3:
        p.error("Require 0 < alpha-min < alpha-max and n-alphas >= 3")
    if not (0 < a.coef_eps < a.practical_eps):
        p.error("Require 0 < coef-eps < practical-eps")
    try:
        ratios = [float(x.strip()) for x in a.l1_ratios.split(",") if x.strip()]
    except ValueError:
        p.error("l1-ratios must be comma-separated numbers")
    if not ratios or len(set(ratios)) != len(ratios) or any(x <= 0 or x > 1 for x in ratios):
        p.error("Each l1_ratio must be unique and in (0,1]")
    a.ratios = ratios
    # enet_path expects descending alpha values.
    a.alphas = np.geomspace(a.alpha_max, a.alpha_min, a.n_alphas)
    return a


def load_training_data(data_path, grading_path):
    data = pd.read_csv(data_path, encoding="utf-8-sig", dtype={ID_COL: str})
    grading = pd.read_csv(grading_path, encoding="utf-8-sig", dtype={ID_COL: str})

    if not NON_FEATURES.issubset(data.columns):
        raise ValueError(f"Feature CSV must contain {sorted(NON_FEATURES)}")
    if not {ID_COL, "Set", "GR", "DD"}.issubset(grading.columns):
        raise ValueError("Grading CSV must contain image_id, Set, GR and DD")

    features = [c for c in data.columns if c not in NON_FEATURES]
    if len(features) != 20:
        raise ValueError(f"Expected 20 image features, found {len(features)}")

    for frame in (data, grading):
        frame[ID_COL] = frame[ID_COL].astype(str).str.strip()
        if frame[ID_COL].isna().any() or frame[ID_COL].duplicated().any():
            raise ValueError("Missing or duplicated image_id")

    if len(data) != EXPECTED_TOTAL or len(grading) != EXPECTED_TOTAL:
        raise ValueError("Both input files must contain 275 rows")
    if set(data[ID_COL]) != set(grading[ID_COL]):
        raise ValueError("image_id sets do not match")

    # Verify GR/DD are identical across the two files before joining labels.
    chk = data[[ID_COL, "GR", "DD"]].merge(
        grading[[ID_COL, "GR", "DD"]], on=ID_COL, validate="one_to_one",
        suffixes=("_feature", "_grading")
    )
    for col in ("GR", "DD"):
        if not np.allclose(chk[f"{col}_feature"], chk[f"{col}_grading"],
                           rtol=1e-8, atol=1e-6):
            raise ValueError(f"{col} values differ between input files")

    joined = data.merge(grading[[ID_COL, "Set"]], on=ID_COL, validate="one_to_one")
    joined["Set"] = joined["Set"].astype(str).str.strip().str.lower()
    n_train = int((joined["Set"] == "train").sum())
    n_test = int((joined["Set"] == "test").sum())
    if (n_train, n_test) != (EXPECTED_TRAIN, EXPECTED_TEST):
        raise ValueError(f"Expected TRAIN/TEST=220/55; found {n_train}/{n_test}")

    for col in features + ["GR", "DD"]:
        joined[col] = pd.to_numeric(joined[col], errors="raise")
        if not np.isfinite(joined[col].to_numpy(float)).all():
            raise ValueError(f"Missing/non-finite numeric values in {col}")

    # The test records are not returned to any analysis function.
    train = joined.loc[joined["Set"] == "train",
                       [ID_COL] + features + ["GR", "DD"]].copy().reset_index(drop=True)
    if any(train[c].std(ddof=1) < 1e-12 for c in features + ["GR", "DD"]):
        raise ValueError("A feature or target is constant in TRAIN")

    target_params = {
        "GR_mean": float(train.GR.mean()),
        "GR_sd_ddof1": float(train.GR.std(ddof=1)),
        "DD_mean": float(train.DD.mean()),
        "DD_sd_ddof1": float(train.DD.std(ddof=1)),
    }
    train["Mech_Composite"] = make_target(
        train.GR.to_numpy(), train.DD.to_numpy(), target_params
    )
    return train, features, target_params


def make_target(gr, dd, params):
    z_gr = (np.asarray(gr, float) - params["GR_mean"]) / params["GR_sd_ddof1"]
    z_dd = (np.asarray(dd, float) - params["DD_mean"]) / params["DD_sd_ddof1"]
    return (z_dd - z_gr) / 2.0


def safe_pearson(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if len(x) < 3 or np.std(x) < 1e-14 or np.std(y) < 1e-14:
        return np.nan, np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r, p = pearsonr(x, y)
    return float(r), float(p)


def pearson_screen(frame, candidates, args):
    rows = []
    for feature in candidates:
        r_gr, p_gr = safe_pearson(frame[feature], frame["GR"])
        r_dd, p_dd = safe_pearson(frame[feature], frame["DD"])
        pass_gr = np.isfinite(r_gr) and abs(r_gr) >= args.gr_cut and p_gr < args.p_cut
        pass_dd = np.isfinite(r_dd) and abs(r_dd) >= args.dd_cut and p_dd < args.p_cut
        rows.append({
            "feature": feature, "r_GR": r_gr, "p_GR": p_gr,
            "pass_GR": bool(pass_gr), "r_DD": r_dd, "p_DD": p_dd,
            "pass_DD": bool(pass_dd), "eligible": bool(pass_gr or pass_dd),
        })
    return pd.DataFrame(rows)


def q4_labels(y):
    return pd.qcut(pd.Series(np.asarray(y)).rank(method="first"),
                   q=4, labels=False).to_numpy(dtype=int)


def standardize_train_apply_valid(x_train, x_valid):
    mu = x_train.mean(axis=0)
    sd = x_train.std(axis=0, ddof=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (x_train - mu) / sd, (x_valid - mu) / sd, mu, sd


def path_predict(x_train, y_train, x_valid, alphas, l1_ratio, args):
    """Fit an Elastic Net path using training-fold transformations only."""
    xtr, xva, _, _ = standardize_train_apply_valid(x_train, x_valid)
    y_mean = float(np.mean(y_train))
    y_centered = np.asarray(y_train, float) - y_mean
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        _, coefs, _ = enet_path(
            np.asfortranarray(xtr, dtype=np.float64),
            np.ascontiguousarray(y_centered, dtype=np.float64),
            l1_ratio=float(l1_ratio),
            alphas=np.asarray(alphas, dtype=float),
            max_iter=args.max_iter,
            tol=args.tol,
        )
    n_warn = sum(issubclass(w.category, ConvergenceWarning) for w in caught)
    preds = xva @ coefs + y_mean
    return preds, coefs, n_warn


def fold_specific_cv(sub, y_sub, base_pool, args, rep):
    """
    Inner CV: Pearson screening uses each fold's TRAINING partition only.
    X scaling and Y centering are also learned from that fold's training partition.
    """
    labels = q4_labels(y_sub)
    if np.bincount(labels).min() < args.cv_folds:
        raise ValueError("Too few samples in one target quartile for requested CV folds")

    cv = StratifiedKFold(
        n_splits=args.cv_folds, shuffle=True,
        random_state=args.seed + 100000 + rep
    )
    fold_losses = {ratio: [] for ratio in args.ratios}
    total_warnings = 0

    for tr_idx, va_idx in cv.split(np.zeros(len(sub)), labels):
        tr = sub.iloc[tr_idx].reset_index(drop=True)
        va = sub.iloc[va_idx].reset_index(drop=True)

        # Screen only with the inner-fold training observations.
        screen = pearson_screen(tr, base_pool, args)
        fold_features = screen.loc[screen.eligible, "feature"].tolist()

        # Recalculate the composite target using inner-fold TRAIN statistics.
        fold_params = {
            "GR_mean": float(tr.GR.mean()),
            "GR_sd_ddof1": float(tr.GR.std(ddof=1)),
            "DD_mean": float(tr.DD.mean()),
            "DD_sd_ddof1": float(tr.DD.std(ddof=1)),
        }
        ytr = make_target(tr.GR.to_numpy(), tr.DD.to_numpy(), fold_params)
        yva = make_target(va.GR.to_numpy(), va.DD.to_numpy(), fold_params)

        if not fold_features:
            baseline = float(np.mean(ytr))
            losses = np.repeat(np.mean((yva - baseline) ** 2), len(args.alphas))
            for ratio in args.ratios:
                fold_losses[ratio].append(losses)
            continue

        xtr = tr[fold_features].to_numpy(float)
        xva = va[fold_features].to_numpy(float)
        for ratio in args.ratios:
            preds, _, nw = path_predict(xtr, ytr, xva, args.alphas, ratio, args)
            total_warnings += nw
            losses = np.mean((yva[:, None] - preds) ** 2, axis=0)
            fold_losses[ratio].append(losses)

    records = []
    for ratio in args.ratios:
        arr = np.asarray(fold_losses[ratio], float)  # folds x alphas
        means = arr.mean(axis=0)
        ses = arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0])
        for i, alpha in enumerate(args.alphas):
            records.append({
                "mean_mse": float(means[i]),
                "se_mse": float(ses[i]),
                "alpha": float(alpha),
                "l1_ratio": float(ratio),
            })
    return records, total_warnings


def fit_full_subsample(sub, y_sub, base_pool, alpha, l1_ratio, args):
    """Re-screen full subsample, then refit selected hyperparameters."""
    screen = pearson_screen(sub, base_pool, args)
    eligible = screen.loc[screen.eligible, "feature"].tolist()
    coef_map = {f: 0.0 for f in base_pool}
    practical = []
    if not eligible:
        return eligible, [], coef_map, float(np.mean(y_sub)), 0, []

    x = sub[eligible].to_numpy(float)
    x_mean = x.mean(axis=0)
    x_sd = x.std(axis=0, ddof=0)
    x_sd = np.where(x_sd < 1e-12, 1.0, x_sd)
    x_scaled = np.asfortranarray((x - x_mean) / x_sd, dtype=np.float64)
    y_mean = float(np.mean(y_sub))
    y_centered = np.ascontiguousarray(y_sub - y_mean, dtype=np.float64)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        _, coef_path, _ = enet_path(
            x_scaled, y_centered, l1_ratio=float(l1_ratio),
            alphas=np.asarray([alpha], float),
            max_iter=args.max_iter, tol=args.tol,
        )
    n_warn = sum(issubclass(w.category, ConvergenceWarning) for w in caught)
    beta = coef_path[:, 0]
    selected = []
    for feature, b in zip(eligible, beta):
        coef_map[feature] = float(b)
        if abs(b) > args.coef_eps:
            selected.append(feature)
        if abs(b) > args.practical_eps:
            practical.append(feature)
    intercept_raw = y_mean - float(np.sum((x_mean / x_sd) * beta))
    return eligible, selected, coef_map, intercept_raw, n_warn, practical


def one_rep(rep, train, features, base_pool, target_params, args):
    y_full = train["Mech_Composite"].to_numpy(float)
    labels = q4_labels(y_full)
    splitter = StratifiedShuffleSplit(
        n_splits=1, train_size=args.sample_fraction,
        random_state=args.seed + rep
    )
    idx, _ = next(splitter.split(np.zeros(len(train)), labels))
    sub = train.iloc[idx].reset_index(drop=True)
    y_sub = y_full[idx]

    # Every 80% subsample re-runs Pearson screening from the original 220-TRAIN pool.
    sub_screen = pearson_screen(sub, base_pool, args)
    sub_eligible = sub_screen.loc[sub_screen.eligible, "feature"].tolist()

    if not sub_eligible:
        row = {
            "rep": rep, "n_subsample": len(sub),
            "subsample_id_sha256": hashlib.sha256(
                ";".join(sorted(sub[ID_COL])).encode()
            ).hexdigest(),
            "eligible": "", "n_eligible": 0, "selected": "",
            "selected_practical": "", "n_selected": 0,
            "n_selected_practical": 0, "best_alpha": np.nan,
            "best_l1_ratio": np.nan, "best_cv_mse": np.nan,
            "best_cv_mse_se": np.nan, "selection_rule": args.selection_rule,
            "coefficients_json": "{}", "intercept": np.nan,
            "n_convergence_warnings": 0,
        }
        return row

    # Hyperparameter tuning uses fold-specific screening on the current candidate pool.
    cv_records, n_cv_warn = fold_specific_cv(sub, y_sub, base_pool, args, rep)
    best_min = min(cv_records, key=lambda z: (z["mean_mse"], -z["alpha"], -z["l1_ratio"]))
    min_mse = best_min["mean_mse"]
    min_se = best_min["se_mse"]

    if args.selection_rule == "min_mse":
        chosen = best_min
    else:
        one_se_candidates = [
            z for z in cv_records if z["mean_mse"] <= min_mse + min_se + 1e-12
        ]
        # Count nonzero coefficients on the full subsample for each eligible pair.
        eligible_full = sub_eligible
        x = sub[eligible_full].to_numpy(float)
        xm = x.mean(axis=0)
        xs = x.std(axis=0, ddof=0)
        xs = np.where(xs < 1e-12, 1.0, xs)
        x_scaled = np.asfortranarray((x - xm) / xs, dtype=np.float64)
        ym = float(np.mean(y_sub))
        yc = np.ascontiguousarray(y_sub - ym, dtype=np.float64)

        counts = {}
        for ratio in args.ratios:
            alphas_here = sorted(
                {z["alpha"] for z in one_se_candidates if z["l1_ratio"] == ratio},
                reverse=True
            )
            if not alphas_here:
                continue
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("ignore", ConvergenceWarning)
                _, path, _ = enet_path(
                    x_scaled, yc, l1_ratio=float(ratio),
                    alphas=np.asarray(alphas_here, float),
                    max_iter=args.max_iter, tol=args.tol,
                )
            for j, alpha in enumerate(alphas_here):
                counts[(ratio, alpha)] = int(np.sum(np.abs(path[:, j]) > args.coef_eps))

        chosen = min(
            one_se_candidates,
            key=lambda z: (
                counts[(z["l1_ratio"], z["alpha"])],
                -z["alpha"], -z["l1_ratio"], z["mean_mse"]
            )
        )

    eligible, selected, coef_map, intercept, n_full_warn, practical = fit_full_subsample(
        sub, y_sub, base_pool, chosen["alpha"], chosen["l1_ratio"], args
    )
    row = {
        "rep": rep, "n_subsample": len(sub),
        "subsample_id_sha256": hashlib.sha256(
            ";".join(sorted(sub[ID_COL])).encode()
        ).hexdigest(),
        "eligible": ";".join(eligible), "n_eligible": len(eligible),
        "selected": ";".join(sorted(selected)),
        "selected_practical": ";".join(sorted(practical)),
        "n_selected": len(selected), "n_selected_practical": len(practical),
        "best_alpha": chosen["alpha"], "best_l1_ratio": chosen["l1_ratio"],
        "best_cv_mse": chosen["mean_mse"], "best_cv_mse_se": chosen["se_mse"],
        "best_cv_mse_global_min": min_mse, "global_min_mse_se": min_se,
        "selection_rule": args.selection_rule,
        "coefficients_json": json.dumps(coef_map, ensure_ascii=False, sort_keys=True),
        "intercept": intercept,
        "n_convergence_warnings": n_cv_warn + n_full_warn,
    }
    return row


def wilson_interval(k, n):
    if n <= 0:
        return np.nan, np.nan
    z = 1.959963984540054
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z / den * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return mid - half, mid + half


def summarize(reps, all_features, out, args):
    reps = reps.sort_values("rep").reset_index(drop=True)
    n_done = len(reps)
    checkpoints = [n for n in (100, 500, 1000) if n <= n_done]
    if n_done not in checkpoints:
        checkpoints.append(n_done)

    stability, thresholds = [], []
    for n in sorted(set(checkpoints)):
        cur = reps.iloc[:n]
        selected_sets = [
            set(s.split(";")) if isinstance(s, str) and s else set()
            for s in cur["selected"]
        ]
        eligible_sets = [
            set(s.split(";")) if isinstance(s, str) and s else set()
            for s in cur["eligible"]
        ]
        coef_dicts = [json.loads(s) for s in cur["coefficients_json"]]

        freq = {}
        for feature in all_features:
            n_pre = sum(feature in ss for ss in eligible_sets)
            n_hit = sum(feature in ss for ss in selected_sets)
            vals = np.asarray([float(d.get(feature, 0.0)) for d in coef_dicts])
            nonzero = vals[np.abs(vals) > args.coef_eps]
            lo, hi = wilson_interval(n_hit, n)
            freq[feature] = n_hit / n
            stability.append({
                "repetitions": n, "feature": feature,
                "pearson_pass_count": n_pre,
                "pearson_pass_frequency_pct": 100 * n_pre / n,
                "selected_count": n_hit,
                "selection_frequency_pct": 100 * n_hit / n,
                "wilson95_low_pct_descriptive": 100 * lo,
                "wilson95_high_pct_descriptive": 100 * hi,
                "conditional_selection_frequency_pct": (
                    100 * n_hit / n_pre if n_pre else np.nan
                ),
                "positive_count_given_selected": int((nonzero > 0).sum()),
                "negative_count_given_selected": int((nonzero < 0).sum()),
                "positive_fraction_given_selected": (
                    float((nonzero > 0).mean()) if len(nonzero) else np.nan
                ),
                "sign_consistency_given_selected": (
                    float(max((nonzero > 0).sum(), (nonzero < 0).sum()) / len(nonzero))
                    if len(nonzero) else np.nan
                ),
                "coef_mean_unconditional": float(vals.mean()),
                "coef_median_unconditional": float(np.median(vals)),
                "coef_p025_unconditional": float(np.quantile(vals, 0.025)),
                "coef_p975_unconditional": float(np.quantile(vals, 0.975)),
                "coef_median_given_selected": (
                    float(np.median(nonzero)) if len(nonzero) else np.nan
                ),
            })

        for threshold in (0.50, 0.60, 0.70):
            keep = sorted(f for f in all_features if freq[f] >= threshold)
            thresholds.append({
                "repetitions": n, "threshold": threshold,
                "n_retained": len(keep), "retained_features": ";".join(keep),
            })

    stability_df = pd.DataFrame(stability)
    stability_df.to_csv(out / "feature_stability_all.csv", index=False, encoding="utf-8-sig")
    latest_n = max(stability_df["repetitions"])
    stability_df.loc[stability_df["repetitions"] == latest_n].sort_values(
        "selection_frequency_pct", ascending=False
    ).to_csv(out / "feature_stability_final_ranked.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(thresholds).to_csv(out / "threshold_sensitivity.csv", index=False, encoding="utf-8-sig")

    reps.groupby(["best_l1_ratio", "best_alpha"], dropna=False).size().rename(
        "n_repetitions"
    ).reset_index().to_csv(out / "hyperparameter_frequencies.csv", index=False, encoding="utf-8-sig")
    reps.groupby(["n_selected", "n_selected_practical"], dropna=False).size().rename(
        "n_repetitions"
    ).reset_index().to_csv(out / "model_size_frequencies.csv", index=False, encoding="utf-8-sig")

    log(f"Completed {n_done} repetitions. Top features: " + ", ".join(
        f"{r.feature}={r.selection_frequency_pct:.1f}%"
        for r in stability_df.loc[
            stability_df.repetitions == latest_n
        ].sort_values("selection_frequency_pct", ascending=False).head(8).itertuples()
    ))


def main():
    args = parse_args()
    data_path = Path(args.data).expanduser().resolve()
    grading_path = Path(args.grading).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    if not data_path.is_file() or not grading_path.is_file():
        raise FileNotFoundError("Input CSV not found; check --data and --grading")

    train, all_features, target_params = load_training_data(data_path, grading_path)
    y = train["Mech_Composite"].to_numpy(float)
    base_screen = pearson_screen(train, all_features, args)
    base_pool = base_screen.loc[base_screen.eligible, "feature"].tolist()
    if not base_pool:
        raise RuntimeError("No feature passed the initial TRAIN-only Pearson screen")

    config = {
        "analysis": "Stage2B Elastic Net stability selection",
        "input_sha256": {
            data_path.name: sha256_file(data_path),
            grading_path.name: sha256_file(grading_path),
        },
        "n_train": len(train), "n_test_excluded": EXPECTED_TEST,
        "all_20_features": all_features, "initial_train_candidate_pool": base_pool,
        "target": "(-Z_train(GR)+Z_train(DD))/2",
        "target_parameters": target_params,
        "subsampling": "80% stratified without replacement; seed + repetition index",
        "pearson": {
            "GR_abs_r_min": args.gr_cut, "DD_abs_r_min": args.dd_cut,
            "two_sided_raw_p_max": args.p_cut,
            "logic": "(abs(r_GR)>=cut_GR AND p_GR<p_cut) OR (abs(r_DD)>=cut_DD AND p_DD<p_cut)",
            "re_screen_each_subsample": True,
            "inner_cv_fold_training_only_screen": True,
        },
        "inner_cv": {
            "folds": args.cv_folds, "stratification": "quartiles of TRAIN-only mechanical composite",
            "scaling": "X mean/SD fitted on each inner-fold training subset only",
            "target_scaling": "GR/DD means and sample SDs fitted on each inner-fold training subset only",
            "score": "mean fold MSE; hyperparameter tuning only",
        },
        "elastic_net": {
            "alpha_grid_descending": args.alphas.tolist(),
            "l1_ratio_grid": args.ratios,
            "selection_rule": args.selection_rule,
            "nonzero_cutoff_standardized_X_beta": args.coef_eps,
            "secondary_practical_cutoff": args.practical_eps,
            "max_iter": args.max_iter, "tol": args.tol,
        },
        "seed": args.seed,
        "software": {
            "python": sys.version.split()[0], "numpy": np.__version__,
            "pandas": pd.__version__, "scipy": scipy.__version__,
            "scikit_learn": sklearn_version,
        },
    }

    config_path = out / "run_config.json"
    checkpoint = out / "repetitions_checkpoint.csv"
    if config_path.exists():
        old = json.loads(config_path.read_text(encoding="utf-8"))
        if old != config:
            raise RuntimeError("Output folder contains different inputs/settings. Use a new --out folder.")
    else:
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
        base_screen.to_csv(out / "training_pearson_screen.csv", index=False, encoding="utf-8-sig")
        train[[ID_COL, "GR", "DD", "Mech_Composite"]].to_csv(
            out / "TRAIN_only_target.csv", index=False, encoding="utf-8-sig"
        )
        train[base_pool].corr().to_csv(out / "base_candidate_correlation_matrix.csv", encoding="utf-8-sig")

    if checkpoint.exists():
        if not args.resume:
            raise RuntimeError("Checkpoint exists. Add --resume to continue; do not overwrite.")
        reps = pd.read_csv(checkpoint)
        reps = reps.sort_values("rep").reset_index(drop=True)
        if reps["rep"].tolist() != list(range(len(reps))):
            raise RuntimeError("Checkpoint repetition sequence is incomplete or duplicated")
    else:
        reps = pd.DataFrame()

    if len(reps) > args.reps:
        raise RuntimeError("Checkpoint already has more repetitions than requested")

    log(f"TRAIN={len(train)}; TEST={EXPECTED_TEST} excluded; initial candidate pool ({len(base_pool)}): {base_pool}")
    log(f"Rule={args.selection_rule}; completed={len(reps)}/{args.reps}; jobs={args.n_jobs}")

    for start in range(len(reps), args.reps, args.batch_size):
        ids = list(range(start, min(start + args.batch_size, args.reps)))
        if args.n_jobs == 1:
            new_rows = [one_rep(i, train, all_features, base_pool, target_params, args) for i in ids]
        else:
            new_rows = Parallel(n_jobs=args.n_jobs, backend="loky")(
                delayed(one_rep)(i, train, all_features, base_pool, target_params, args)
                for i in ids
            )
        reps = pd.concat([reps, pd.DataFrame(new_rows)], ignore_index=True)
        tmp = checkpoint.with_suffix(".tmp")
        reps.to_csv(tmp, index=False, encoding="utf-8-sig")
        os.replace(tmp, checkpoint)
        log(f"Checkpoint saved: {len(reps)}/{args.reps}")

    summarize(reps, all_features, out, args)
    warnings_total = int(pd.to_numeric(reps["n_convergence_warnings"], errors="coerce").fillna(0).sum())
    if warnings_total:
        log(f"WARNING: {warnings_total} convergence warnings. Review solver diagnostics before reporting.")
    log("DONE. Inner-CV MSE is for hyperparameter tuning only, not an unbiased performance estimate.")


if __name__ == "__main__":
    main()
