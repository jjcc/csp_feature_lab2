#!/usr/bin/env python3
import os, json, argparse, warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss,make_scorer
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
import joblib

BASE_FEATURES = [
    "VIX","ret_2d_norm","ret_5d_norm",
    "gex_gamma_at_ul","gex_total_abs","gex_neg","gex_flip_strike","gex_distance_to_flip",
    "percentToBreakEvenBid","moneyness","delta",
    "openInterest","volume","log1p_DTE",
    "impliedVolatilityRank1y",
    "underlyingLastPrice","strike","bidPrice",
    "potentialReturnAnnual","prev_close_minus_ul_pct",
    "is_earnings_window",

    # NEW (optional) stack features—include when present:
    "tail_proba_oof",         # P[toxic] from Tail Gate (OOF or live at inference)
    "winner_proba_oof"        # P[win] from Winner (if you have OOF; else omit)
]

def build_monotone_constraints(features):
    """
    Note: due to monotone constraints cause the performance loss, the function is not used
    Return XGBoost monotone constraint string matching `features` order:
      - tail_proba_oof:  -1 (higher toxic prob => must increase P[toxic])
      - winner_proba_oof:+1 (higher win prob => must decrease P[toxic])
      - buffer-ish/safety proxies (positive monotone): moneyness, percentToBreakEvenBid (if you define it as cushion), etc.
      - leave others 0.
    """
    pos = set(["winner_proba_oof", "moneyness"])  # add more if they are strictly 'safer when larger'
    neg = set(["tail_proba_oof"])                 # strictly 'more toxic when larger'
    cons = []
    for f in features:
        if f in pos: cons.append(+1)
        elif f in neg: cons.append(-1)
        else: cons.append(0)
    return "(" + ",".join(map(str, cons)) + ")"

def parse_args():
    ap = argparse.ArgumentParser(description="Rescue trainer w/ monotone constraints + calibration + risk sweep")
    ap.add_argument("--annual-cut", type=float, default=200.0)
    ap.add_argument("--label-col", default="is_tail")
    ap.add_argument("--tailgate-label-col", default="is_tail_preda")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)

    # Calibration + class weight
    ap.add_argument("--calibration", choices=["none","isotonic","sigmoid"], default="isotonic")
    ap.add_argument("--scale-pos-weight", type=float, default=None)  # toxic (positive) class weight

    # Tail Gate guardrail
    ap.add_argument("--tail-hard-veto", type=float, default=0.45, help="NEVER rescue if tail_proba_oof >= this")
    ap.add_argument("--tail-soft-min", type=float, default=0.03, help="Ignore rescue if tail_proba_oof < this (already very safe)")

    # Threshold + utility
    ap.add_argument("--threshold-grid", type=str, default="0.02:0.30:0.01")
    ap.add_argument("--risk-weights", type=str, default="3,5,8")  # multiple curves
    return ap.parse_args()

def parse_grid(spec):
    a,b,c = [float(x) for x in spec.split(":")]
    xs=[]; v=a
    while v<=b+1e-12:
        xs.append(round(v,6)); v+=c
    return xs

def utility_metrics(y_true, p_toxic, thr, veto_mask=None):
    """Rescue if p_toxic < thr AND not vetoed."""
    if veto_mask is None:
        veto_mask = np.zeros_like(y_true, dtype=bool)
    rescue = (p_toxic < thr) & (~veto_mask)

    true_safe = (y_true==0); true_tox=(y_true==1)
    rescued_safe = int(np.sum(rescue & true_safe))
    rescued_tox  = int(np.sum(rescue & true_tox))
    total_safe   = int(np.sum(true_safe))
    total_tox    = int(np.sum(true_tox))
    safe_recovery_rate = rescued_safe/total_safe if total_safe else 0.0
    slipped_toxic_rate = rescued_tox/total_tox   if total_tox   else 0.0
    toxic_recall = 1.0 - slipped_toxic_rate
    return dict(
        rescued_safe=rescued_safe, rescued_toxic=rescued_tox,
        total_safe=total_safe, total_toxic=total_tox,
        safe_recovery_rate=float(safe_recovery_rate),
        slipped_toxic_rate=float(slipped_toxic_rate),
        toxic_recall=float(toxic_recall)
    )

def main():
    args = parse_args()
    args_input = "output/tails_train/v6b/tail_gex_v6_cut05_scores_oof.csv"
    args_outdir = os.getenv("RESCUE_OUT")

    os.makedirs(args_outdir, exist_ok=True)

    df = pd.read_csv(args_input)
    df["is_tail_preda"] = (df["tail_proba_oof"] >= args.tail_soft_min).astype(int)
    for c in ["potentialReturnAnnual", args.label_col, args.tailgate_label_col]:
        if c not in df.columns:
            raise SystemExit(f"Missing required column: {c}")

    # Candidate pool: high-reward + TG predicted toxic
    cand = df[(df["potentialReturnAnnual"] > args.annual_cut) & (df[args.tailgate_label_col]==1)].copy()
    if len(cand) == 0:
        raise SystemExit("No rows after high-reward & TG-toxic filter.")

    # Assemble features present
    features = [c for c in BASE_FEATURES if c in cand.columns]
    if "tail_proba_oof" not in cand.columns:
        warnings.warn("tail_proba_oof not found — monotone constraints still applied but rescue power will be lower.")
    X = cand[features].copy()
    y = cand[args.label_col].astype(int).values

    # Simple medians (persisted)
    med = {c: float(X[c].median()) for c in features}
    for c in features:
        X[c] = X[c].fillna(med[c])

    # Tail Gate veto band (applied in thresholding)
    tail = cand["tail_proba_oof"].values if "tail_proba_oof" in cand.columns else np.zeros(len(cand))
    veto_mask = tail >= args.tail_hard_veto
    # (Optional) we can also (soft) skip rescuing trades TG already called extremely safe
    soft_skip = tail < args.tail_soft_min

    # Class weight for toxic (=positive)
    if args.scale_pos_weight is None:
        pos_w = (np.sum(y==0) / max(np.sum(y==1),1))
    else:
        pos_w = float(args.scale_pos_weight)

    # XGB with monotone constraints
    mono = build_monotone_constraints(features)
    base = XGBClassifier(
        n_estimators=800, learning_rate=0.03, max_depth=6,
        subsample=0.9, colsample_bytree=0.9, reg_alpha=0.0, reg_lambda=1.0,
        random_state=args.seed, objective="binary:logistic", eval_metric="logloss",
        n_jobs=0, scale_pos_weight=pos_w
    )
    if args.calibration in ("isotonic","sigmoid"):
        model = CalibratedClassifierCV(base, method=args.calibration, cv=3)
    else:
        model = base

    # OOF proba via CV
    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    oof = np.zeros(len(X), float)
    for k,(tr,va) in enumerate(skf.split(X,y),1):
        pipe = Pipeline([("model", model)])
        pipe.fit(X.iloc[tr], y[tr])
        proba = pipe.predict_proba(X.iloc[va])[:,1] if hasattr(pipe,"predict_proba") else pipe.decision_function(X.iloc[va])
        oof[va] = proba

    # Basic metrics
    auc = roc_auc_score(y, oof)
    ap  = average_precision_score(y, oof)
    bs  = brier_score_loss(y, oof)

    # Threshold sweep + multi-risk utilities
    grid = parse_grid(args.threshold_grid)
    rs = []
    risk_weights = [float(x) for x in args.risk_weights.split(",")]
    for thr in grid:
        m = utility_metrics(y, oof, thr, veto_mask=veto_mask | soft_skip)
        row = {"threshold":thr, **m}
        for w in risk_weights:
            row[f"utility_rw{int(w)}"] = m["safe_recovery_rate"] - w * m["slipped_toxic_rate"]
        rs.append(row)
    thr_df = pd.DataFrame(rs)

    # Choose best by the *strictest* risk weight (last one)
    w_strict = max(risk_weights)
    best_row = thr_df.sort_values([f"utility_rw{int(w_strict)}","toxic_recall","safe_recovery_rate"],
                                  ascending=[False,False,False]).iloc[0]
    best_thr = float(best_row["threshold"])

    # Fit final on ALL
    pipe_final = Pipeline([("model", model)])
    pipe_final.fit(X, y)

    # Permutation importance (works through calibration)
    #pi = permutation_importance(pipe_final, X, y, n_repeats=10, random_state=args.seed)
    def neg_brier_scorer(estimator, X_eval, y_eval):
        # Always score on probabilities of the positive class
        if hasattr(estimator, "predict_proba"):
            p = estimator.predict_proba(X_eval)[:, 1]
        else:
            # fallback for decision_function; convert to probs
            d = estimator.decision_function(X_eval)
            p = 1.0 / (1.0 + np.exp(-d))
        return -brier_score_loss(y_eval, p)
    
    pi = permutation_importance(
        pipe_final, X, y,
        n_repeats=10,
        random_state=args.seed,
        scoring=neg_brier_scorer
    )



    pi_df = pd.DataFrame({"feature": features, "perm_importance_mean": pi.importances_mean,
                          "perm_importance_std": pi.importances_std}).sort_values("perm_importance_mean", ascending=False)

    # Save artifacts
    rep = {
        "n_candidates": int(len(cand)),
        "auc_cv": float(auc), "ap_cv": float(ap), "brier_cv": float(bs),
        "best_threshold": best_thr,
        "best_row": {k: (float(v) if isinstance(v,(int,float,np.floating)) else v) for k,v in best_row.to_dict().items()},
        "features_used": features,
        "risk_weights": risk_weights,
        "tail_hard_veto": float(args.tail_hard_veto),
        "tail_soft_min": float(args.tail_soft_min),
        "calibration": args.calibration,
        "scale_pos_weight": float(pos_w),
        "monotone_constraints": "deprecated due to performance loss"
    }
    os.makedirs(args_outdir, exist_ok=True)
    thr_df.to_csv(os.path.join(args_outdir,"threshold_metrics.csv"), index=False)
    pi_df.to_csv(os.path.join(args_outdir,"permutation_importances.csv"), index=False)
    with open(os.path.join(args_outdir,"rescue_training_report.json"), "w") as f:
        json.dump(rep, f, indent=2)

    pack = {
        "model": pipe_final, "features": features, "medians": {k: float(v) for k,v in med.items()},
        "threshold": best_thr, "label_col": args.label_col, "positive_label": 1,
        "selection_query": f"(potentialReturnAnnual > {args.annual_cut}) & ({args.tailgate_label_col} == 1)",
        "notes": "Calibrated, monotone XGB. Rescue if P[toxic]<threshold and Tail veto not triggered."
    }
    joblib.dump(pack, os.path.join(args_outdir,"rescue_model.joblib"))

    # Plots: metrics + utilities + reliability
    plt.figure(figsize=(9,5))
    plt.plot(thr_df["threshold"], thr_df["toxic_recall"], label="toxic_recall")
    plt.plot(thr_df["threshold"], thr_df["safe_recovery_rate"], label="safe_recovery_rate")
    plt.plot(thr_df["threshold"], thr_df["slipped_toxic_rate"], label="slipped_toxic_rate")
    plt.axvline(best_thr, ls="--", c="tab:blue")
    plt.xlabel("threshold (P[toxic])"); plt.ylabel("metric"); plt.title("Rescue Threshold Sweep (with Tail veto)")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(args_outdir,"threshold_sweep_no_util.png"), dpi=150); plt.close()

    # Utility curves
    plt.figure(figsize=(9,5))
    for w in risk_weights:
        plt.plot(thr_df["threshold"], thr_df[f"utility_rw{int(w)}"], label=f"utility (rw={int(w)})")
    plt.axvline(best_thr, ls="--", c="tab:blue")
    plt.xlabel("threshold (P[toxic])"); plt.ylabel("utility")
    plt.title("Utility vs Threshold (multi risk weights)")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(args_outdir,"utility_sweep.png"), dpi=150); plt.close()

    # Reliability curve (OOF)
    prob_true, prob_pred = calibration_curve(y, oof, n_bins=15, strategy="quantile")
    plt.figure(figsize=(5.5,5.5))
    plt.plot([0,1],[0,1],"--", label="perfect")
    plt.plot(prob_pred, prob_true, marker="o", label="OOF")
    plt.xlabel("Predicted P[toxic]"); plt.ylabel("Observed P[toxic]")
    plt.title(f"Reliability (calibration={args.calibration})")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(args_outdir,"reliability_curve.png"), dpi=150); plt.close()

    print(json.dumps(rep, indent=2))
    print(f"\nArtifacts saved in {args_outdir}\n")

if __name__ == "__main__":
    main()
