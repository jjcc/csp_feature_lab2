#!/usr/bin/env python3
import os, json, argparse
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV

MODEL_BACKENDS = {}
try:
    from lightgbm import LGBMClassifier
    MODEL_BACKENDS["lgbm"] = LGBMClassifier
except Exception:
    pass
try:
    from xgboost import XGBClassifier
    MODEL_BACKENDS["xgb"] = XGBClassifier
except Exception:
    pass
from sklearn.ensemble import RandomForestClassifier
MODEL_BACKENDS["rf"] = RandomForestClassifier

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_FEATURES = [
    "VIX","ret_2d_norm","ret_5d_norm",
    "gex_gamma_at_ul","gex_total_abs","gex_neg","gex_flip_strike","gex_distance_to_flip","gex_missing",
    "prev_close_minus_strike","percentToBreakEvenBid","moneyness","delta",
    "openInterest","volume","log1p_DTE","impliedVolatilityRank1y",
    "underlyingLastPrice","strike","bidPrice",
    "potentialReturnAnnual"
]

def parse_args():
    ap = argparse.ArgumentParser(description="Train High-Reward Rescue Model (toxic vs false-rejected-safe).")
    ap.add_argument("--input", default="output/tails_train/v5/tail_gex_v5_cut05_score_oof.csv")
    ap.add_argument("--outdir", default="output/rescue_tail/")
    ap.add_argument("--annual-cut", type=float, default=200.0)
    ap.add_argument("--label-col", default="is_tail")
    ap.add_argument("--tailgate-label-col", default="is_tail_preda")
    ap.add_argument("--features-file", default=None)
    ap.add_argument("--features", nargs="*", default=None)
    #ap.add_argument("--model", choices=list(MODEL_BACKENDS.keys()), default="lgbm")
    ap.add_argument("--model", choices=list(MODEL_BACKENDS.keys()), default="xgb")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--risk-weight", type=float, default=8.0)
    ap.add_argument("--threshold-grid", type=str, default="0.05:0.95:0.01")
    ap.add_argument("--calibrate", action="store_true")
    return ap.parse_args()

def make_backend(name, pos_weight, seed):
    if name == "lgbm" and "lgbm" in MODEL_BACKENDS:
        return MODEL_BACKENDS["lgbm"](
            n_estimators=500, learning_rate=0.03, num_leaves=63,
            subsample=0.9, colsample_bytree=0.9, reg_alpha=0.0, reg_lambda=1.0,
            random_state=seed, objective="binary",
            scale_pos_weight=pos_weight if pos_weight>0 else None
        )
    if name == "xgb" and "xgb" in MODEL_BACKENDS:
        return MODEL_BACKENDS["xgb"](
            n_estimators=700, learning_rate=0.03, max_depth=6,
            subsample=0.9, colsample_bytree=0.9, reg_alpha=0.0, reg_lambda=1.0,
            random_state=seed, objective="binary:logistic", eval_metric="logloss",
            scale_pos_weight=pos_weight if pos_weight>0 else 1.0, n_jobs=0
        )
    return MODEL_BACKENDS["rf"](
        n_estimators=600, max_depth=None, min_samples_split=4, min_samples_leaf=2,
        n_jobs=0, random_state=seed, class_weight={0:1.0, 1:pos_weight if pos_weight>0 else 1.0},
    )

def parse_grid(spec):
    start, stop, step = [float(x) for x in spec.split(":")]
    vals, v = [], start
    while v <= stop + 1e-12:
        vals.append(round(v, 6)); v += step
    return vals

def utility_metrics(y_true, proba_toxic, thr):
    rescue = proba_toxic < thr
    true_safe = (y_true==0); true_tox=(y_true==1)
    rescued_safe = int(np.sum(rescue & true_safe))
    rescued_tox  = int(np.sum(rescue & true_tox))
    total_safe   = int(np.sum(true_safe))
    total_tox    = int(np.sum(true_tox))
    safe_recovery_rate = rescued_safe/total_safe if total_safe>0 else 0.0
    slipped_toxic_rate = rescued_tox/total_tox if total_tox>0 else 0.0
    toxic_recall = 1.0 - slipped_toxic_rate
    return dict(rescued_safe=rescued_safe, rescued_toxic=rescued_tox, total_safe=total_safe, total_toxic=total_tox,
                safe_recovery_rate=float(safe_recovery_rate), slipped_toxic_rate=float(slipped_toxic_rate), toxic_recall=float(toxic_recall))

def main():
    import pandas as pd
    import json
    args = parse_args(); os.makedirs(args.outdir, exist_ok=True)
    args_input = "output/tails_train/v6/tail_gex_v6_cut05_scores_oof.csv"
    args_outdir = "output/rescue_tail/"
    #args.tailgate_label_col, 
    #args.label_col
    #args.annual_cut
    threshold = 0.03
    df = pd.read_csv(args_input)
    df['is_tail_preda'] = df["tail_proba_oof"].map(lambda x: 1 if x > threshold else 0)
    for col in ["potentialReturnAnnual", args.tailgate_label_col, args.label_col]:
        if col not in df.columns: raise SystemExit(f"Missing required column: {col}")
    cand = df[(df["potentialReturnAnnual"]>args.annual_cut) & (df[args.tailgate_label_col]==1)].copy()
    if len(cand)==0: raise SystemExit("No rows match the candidate filter.")
    if args.features_file:
        import json
        features = json.load(open(args.features_file))
    elif args.features:
        features = args.features
    else:
        features = DEFAULT_FEATURES
    features = [c for c in features if c in cand.columns]
    if not features: raise SystemExit("No usable features after intersection.")
    y = cand[args.label_col].astype(int).values
    X = cand[features].copy().fillna(cand[features].median())
    medians = {c: float(X[c].median()) for c in features}
    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    pos_weight = (np.sum(y==0)/max(np.sum(y==1),1))
    oof = np.zeros_like(y, float); models=[]
    for k,(tr,va) in enumerate(skf.split(X, y),1):
        base = make_backend(args.model, pos_weight, args.seed+k)
        clf = CalibratedClassifierCV(base, method="isotonic", cv=3) if args.calibrate else base
        pipe = Pipeline([("model", clf)]); pipe.fit(X.iloc[tr], y[tr])
        proba = pipe.predict_proba(X.iloc[va])[:,1] if hasattr(pipe,"predict_proba") else pipe.decision_function(X.iloc[va])
        oof[va] = proba; models.append(pipe)
    auc = roc_auc_score(y, oof); ap = average_precision_score(y, oof)
    grid = parse_grid(args.threshold_grid)
    import pandas as pd
    rows=[{"threshold":t, **utility_metrics(y,oof,t)} for t in grid]
    thr_df=pd.DataFrame(rows)
    thr_df["utility"]=thr_df["safe_recovery_rate"] - args.risk_weight*thr_df["slipped_toxic_rate"]
    best=thr_df.sort_values(["utility","toxic_recall","safe_recovery_rate"], ascending=[False,False,False]).iloc[0]
    best_thr=float(best["threshold"])
    final_base=make_backend(args.model, pos_weight, args.seed)
    final=CalibratedClassifierCV(final_base, method="isotonic", cv=3) if args.calibrate else final_base
    final_pipe=Pipeline([("model", final)]); final_pipe.fit(X, y)
    import json
    rep={
        "n_candidates":int(len(cand)),"auc_cv":float(auc),"ap_cv":float(ap),
        "best_threshold":best_thr,"best_row":{k: (float(v) if isinstance(v,(int,float,np.floating)) else v) for k,v in best.to_dict().items()},
        "features_used":features,"label_col":args.label_col,
        "tailgate_label_col":args.tailgate_label_col,"annual_cut":float(args.annual_cut),
        "risk_weight":float(args.risk_weight),"model_backend":args.model,"calibrated":bool(args.calibrate)
    }
    thr_path=os.path.join(args_outdir,"threshold_metrics.csv"); rep_path=os.path.join(args_outdir,"rescue_training_report.json"); model_path=os.path.join(args_outdir,"rescue_model.joblib")
    thr_df.to_csv(thr_path, index=False); open(rep_path,"w").write(json.dumps(rep, indent=2))
    pack={"model":final_pipe,"features":features,"medians":medians,"threshold":best_thr,"label_col":args.label_col,"positive_label":1,
          "selection_query":f"(potentialReturnAnnual > {args.annual_cut}) & ({args.tailgate_label_col} == 1)",
          "notes":"Predicts P(toxic) within Tail-Gate-rejected high-reward candidates. Rescue if proba < threshold."}
    import joblib; joblib.dump(pack, model_path)
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8,5))
        plt.plot(thr_df["threshold"], thr_df["toxic_recall"], label="toxic_recall")
        plt.plot(thr_df["threshold"], thr_df["safe_recovery_rate"], label="safe_recovery_rate")
        plt.plot(thr_df["threshold"], thr_df["slipped_toxic_rate"], label="slipped_toxic_rate")
        plt.plot(thr_df["threshold"], thr_df["utility"], label="utility")
        plt.axvline(best_thr, linestyle="--")
        plt.xlabel("threshold (P[toxic])"); plt.ylabel("metric"); plt.title("Rescue Threshold Sweep"); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(args_outdir,"threshold_sweep.png"), dpi=150); plt.close()
    except Exception: pass
    print(json.dumps(rep, indent=2))
    print(f"\nArtifacts saved:\n- {model_path}\n- {rep_path}\n- {thr_path}\n")

if __name__=="__main__":
    main()
