#!/usr/bin/env python3
import os, json, math
import numpy as np
import pandas as pd
from dotenv import load_dotenv

def load_pickle(path):
    try:
        import joblib
        return joblib.load(path)
    except Exception:
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)

def infer_pack(obj):
    if isinstance(obj, dict):
        return {
            "model": obj.get("model", obj),
            "features": obj.get("features"),
            "medians": obj.get("medians", {}) or {},
            "scaler": obj.get("scaler"),
            "calibrator": obj.get("calibrator")
        }
    feats = getattr(obj, "feature_names_in_", None)
    return {"model": obj, "features": list(feats) if feats is not None else None, "medians": {}, "scaler": None, "calibrator": None}

def build_X(df, features, medians, scaler):
    Xcols = {}
    for c in features:
        if c in df.columns:
            col = pd.to_numeric(df[c], errors="coerce")
            fill = medians.get(c)
            if fill is None or (isinstance(fill, float) and math.isnan(fill)):
                med = pd.to_numeric(df[c], errors="coerce").median()
                fill = 0.0 if (med is None or math.isnan(med)) else float(med)
            Xcols[c] = col.fillna(fill).astype(float)
        else:
            Xcols[c] = pd.Series(0.0, index=df.index)
    X = pd.DataFrame(Xcols)[features].values
    if scaler is not None:
        try:
            X = scaler.transform(X)
        except Exception:
            pass
    return X

def ensure_returns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["tradeTime"] = pd.to_datetime(out["tradeTime"], errors="coerce")
    out["expirationDate"] = pd.to_datetime(out["expirationDate"], errors="coerce")
    if "entry_credit" not in out.columns or out["entry_credit"].isna().all():
        bid = pd.to_numeric(out.get("bid"), errors="coerce")
        bidPrice = pd.to_numeric(out.get("bidPrice"), errors="coerce")
        out["entry_credit"] = np.where(~bid.isna(), bid * 100.0, np.where(~bidPrice.isna(), bidPrice * 100.0, np.nan))
    out["strike"] = pd.to_numeric(out["strike"], errors="coerce")
    out["expiry_close"] = pd.to_numeric(out["expiry_close"], errors="coerce")
    out["capital"] = out["strike"] * 100.0
    out["exit_intrinsic"] = np.maximum(0, out["strike"] - out["expiry_close"]) * 100.0
    out["total_pnl"] = out["entry_credit"] - out["exit_intrinsic"]
    out["return_pct"] = np.where(out["capital"]>0, out["total_pnl"]/out["capital"], np.nan)
    return out

def pick_threshold_from_coverage(proba, coverage):
    if len(proba) == 0: return 1.0
    k = max(1, int(round(len(proba) * coverage)))
    thr = np.partition(proba, len(proba)-k)[len(proba)-k]
    return float(thr)

def main():
    #env_file = os.environ.get("ENV_FILE", "hybrid.env")
    env_file = ".env_hybrid"
    if os.path.exists(env_file):
        load_dotenv(env_file)
    else:
        load_dotenv()

    # Inputs
    CSV_BASE = os.getenv("CSV_BASE", "labeled_trades_with_gex.csv")
    TAIL_SCORES = os.getenv("TAIL_SCORES", "tail_gex_p35_scores.csv")
    TAIL_THRESHOLD = float(os.getenv("TAIL_THRESHOLD", "0.086907"))

    WINNER_MODEL = os.getenv("WINNER_MODEL", "winner_classifier.pkl")
    TARGET_COVERAGE = os.getenv("TARGET_COVERAGE")
    THRESHOLD_WINNER = os.getenv("THRESHOLD_WINNER")
    if TARGET_COVERAGE: TARGET_COVERAGE = float(TARGET_COVERAGE)
    if THRESHOLD_WINNER: THRESHOLD_WINNER = float(THRESHOLD_WINNER)

    OUT_DIR = os.getenv("OUT_DIR", "./output_hybrid")
    os.makedirs(OUT_DIR, exist_ok=True)

    # Sim params (optional)
    RUN_SIM = os.getenv("RUN_SIM", "1").strip().lower() in {"1","true","yes","y","on"}
    INITIAL_CASH = float(os.getenv("INITIAL_CASH", "100000"))
    PER_TRADE_CAPITAL_PCT = float(os.getenv("PER_TRADE_CAPITAL_PCT", "0.30"))
    DAILY_RISK_CAP_PCT = float(os.getenv("DAILY_RISK_CAP_PCT", "1.00"))
    MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT", "10"))
    COMMISSION_PER_CONTRACT = float(os.getenv("COMMISSION_PER_CONTRACT", "0.0"))
    USE_COMPOUNDING = os.getenv("USE_COMPOUNDING", "1").strip().lower() in {"1","true","yes","y","on"}
    SORT_BY = os.getenv("SORT_BY", "ml_proba")
    SORT_DESC = int(os.getenv("SORT_DESC", "1"))
    TOP_FRAC = os.getenv("TOP_FRAC")
    TOP_FRAC = float(TOP_FRAC) if TOP_FRAC not in (None, "",) else None

    # Load
    base = pd.read_csv(CSV_BASE)
    tails = pd.read_csv(TAIL_SCORES)

    # Merge keys
    if "trade_id" in base.columns and "trade_id" in tails.columns:
        keys = ["trade_id"]
    else:
        keys = [k for k in ["baseSymbol","tradeTime","expirationDate","strike"] if k in base.columns and k in tails.columns]
        if not keys:
            raise SystemExit("No common keys to merge tail scores. Need 'trade_id' or {baseSymbol, tradeTime, expirationDate, strike}.")

    tails = tails.rename(columns={"score":"tail_score"})
    merged = base.merge(tails, on=keys, how="left")

    # Tail filter
    filtered = merged[(merged["tail_score"].isna()) | (merged["tail_score"] < TAIL_THRESHOLD)].copy()
    filtered.to_csv(os.path.join(OUT_DIR, "hybrid_tail_filtered.csv"), index=False)

    # Winner scoring
    pack = infer_pack(load_pickle(WINNER_MODEL))
    model, feats = pack["model"], pack["features"]
    if not hasattr(model, "predict_proba"):
        raise SystemExit("Winner model lacks predict_proba()")
    if not feats:
        raise SystemExit("Winner model missing features metadata.")
    X = build_X(filtered, feats, pack["medians"], pack["scaler"])
    proba = model.predict_proba(X)[:,1]
    if pack["calibrator"] is not None:
        try:
            proba = pack["calibrator"].predict_proba(proba.reshape(-1,1))[:,1]
        except Exception:
            pass

    filtered["ml_proba"] = proba
    filtered.to_csv(os.path.join(OUT_DIR, "hybrid_winner_scored.csv"), index=False)

    # Select by coverage/threshold
    if THRESHOLD_WINNER is None:
        thr = pick_threshold_from_coverage(proba, TARGET_COVERAGE if TARGET_COVERAGE is not None else 0.60)
    else:
        thr = THRESHOLD_WINNER
    selected = filtered[filtered["ml_proba"] >= thr].copy()
    selected.to_csv(os.path.join(OUT_DIR, "hybrid_winner_selected.csv"), index=False)

    # Quick metrics if 'win' exists
    metrics = {}
    if "win" in base.columns:
        total = len(base)
        kept_tail = len(filtered)
        kept_final = len(selected)
        winners_removed_tail = None
        tails_caught_tail = None
        try:
            winners_removed_tail = ( (base["win"]==1).sum() - (filtered["win"]==1).sum() ) / max(1,(base["win"]==1).sum()) * 100
        except Exception: pass
        # If you have a tail label like label=-1, fill here. Otherwise skip.
        metrics = {
            "keep_rate_after_tail_%": round(100*kept_tail/max(1,total),2),
            "keep_rate_final_%": round(100*kept_final/max(1,total),2),
            "winners_removed_after_tail_%": None if winners_removed_tail is None else round(winners_removed_tail,2)
        }

    # Optional quick sim
    sim_paths = {}
    if RUN_SIM:
        df = ensure_returns(selected.copy())
        entries = df[["tradeTime"]].dropna().rename(columns={"tradeTime":"ts"})
        exits = df[["expirationDate"]].dropna().rename(columns={"expirationDate":"ts"})
        timeline = pd.concat([entries, exits]).drop_duplicates().sort_values("ts")["ts"].tolist()
        by_entry = {ts: g.copy() for ts, g in df.groupby(df["tradeTime"])}
        equity = INITIAL_CASH; cash = INITIAL_CASH; open_positions=[]; rows=[]
        for ts in timeline:
            still=[]
            for pos in open_positions:
                if pos["expirationDate"] <= ts:
                    pnl = pos["total_pnl"]; fees = COMMISSION_PER_CONTRACT*2
                    cash += pos["capital"] + pnl - fees
                else:
                    still.append(pos)
            open_positions = still
            equity = cash
            if ts in by_entry:
                day = by_entry[ts].copy()
                equity_for_alloc = equity if USE_COMPOUNDING else INITIAL_CASH
                per_trade_cap = PER_TRADE_CAPITAL_PCT * equity_for_alloc
                daily_cap = DAILY_RISK_CAP_PCT * equity_for_alloc if DAILY_RISK_CAP_PCT is not None else float("inf")
                slots = max(0, MAX_CONCURRENT - len(open_positions))
                reserved = 0.0
                for _, row in day.iterrows():
                    need = float(row["capital"])
                    if need <= per_trade_cap and need <= cash and (reserved+need)<=daily_cap and slots>0:
                        cash -= need; reserved += need; slots -= 1
                        open_positions.append({
                            "expirationDate": row["expirationDate"],
                            "total_pnl": float(row["total_pnl"]),
                            "capital": float(row["capital"]),
                        })
            rows.append({"ts": ts, "equity": equity, "cash": cash, "open_positions": len(open_positions)})
        curve = pd.DataFrame(rows).sort_values("ts").drop_duplicates("ts")
        eq_path = os.path.join(OUT_DIR, "equity_curve.csv")
        curve.to_csv(eq_path, index=False)
        sim_paths = {"equity_curve": eq_path, "rows": int(len(curve))}

    summary = {
        "tail_threshold": TAIL_THRESHOLD,
        "winner_threshold_used": float(thr),
        "rows": {
            "base": int(len(base)),
            "after_tail": int(len(filtered)),
            "final_selected": int(len(selected))
        },
        "metrics": metrics,
        "paths": {
            "tail_filtered": os.path.join(OUT_DIR, "hybrid_tail_filtered.csv"),
            "winner_scored": os.path.join(OUT_DIR, "hybrid_winner_scored.csv"),
            "winner_selected": os.path.join(OUT_DIR, "hybrid_winner_selected.csv"),
            **({"sim": sim_paths} if RUN_SIM else {})
        }
    }
    with open(os.path.join(OUT_DIR, "hybrid_summary2.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
