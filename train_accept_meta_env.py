
import os, joblib, yaml, argparse
import numpy as np, pandas as pd
from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

# Local imports (flat, to match your run_branch_b_meta_env.py style)
from service.env_config import load_env
from service.adapters_tail import TailProbaAdapter
from service.adapters_winner import WinnerProbaAdapter
from service.identity_cal import IdentityCal
from service.branch_b_core import fit_bucket_stats

# --- helpers ---

def _bool(v, default=False):
    if v is None: return default
    if isinstance(v, bool): return v
    s = str(v).strip().lower()
    if s in ('1','true','yes','y','on'): return True
    if s in ('0','false','no','n','off'): return False
    return default

def _safe_float(v, default=None):
    try:
        return float(v)
    except:
        return default
def _ret_col(df):
    if 'return_mon' in df.columns: return 'return_mon'
    if 'return_pct' in df.columns: return 'return_pct'
    raise ValueError("Need realized return column: return_mon or return_pct")

def build_labels(df, label_kind='win', lam=0.25):
    """
    label_kind:
      - 'win'     -> y = 1[pnl_pct > 0]
      - 'utility' -> y = 1[U > 0], where U = EV - lam*|CVaR95| computed via p_win bins
    """
    ret_col = _ret_col(df)
    if label_kind == 'win':
        y = (df[ret_col] > 0).astype(int).values
        return y

    # utility-based
    if ret_col not in df.columns or df[ret_col].isna().all():
        raise ValueError("return_pct with non-NaN values is required for 'utility' labels")

    # Build bin stats on the available rows with realized return
    ev = df.dropna(subset=['p_win', ret_col]).copy()
    if len(ev) < 200:
        # trivial fallback
        wins = pd.Series({0:0.01}); non = pd.Series({0:-0.01}); cvars = pd.Series({0:-0.03})
        bins = pd.Series(0, index=df.index)
    else:
        ev['_bin'] = pd.qcut(ev['p_win'], q=10, labels=False, duplicates='drop')
        wins, non, cvars, _ = fit_bucket_stats(ev, prob_col='p_win', ret_col=ret_col, q=0.95, tau=50.0)
        bins = pd.qcut(df['p_win'], q=10, labels=False, duplicates='drop').fillna(0).astype(int)

    # map bins to expected stats
    def _map(s, m, default):
        return s.map(lambda b: m.get(int(b), default))

    # If we didn't compute wins/non/cvars above (fallback case), define defaults:
    if 'wins' not in locals():
        wins = {0:0.01}; non = {0:-0.01}; cvars = {0:-0.03}

    e_win = _map(bins, wins if isinstance(wins, dict) else wins.to_dict(), 0.01)
    e_non = _map(bins, non  if isinstance(non , dict) else non .to_dict(), -0.01)
    cvr   = _map(bins, cvars if isinstance(cvars, dict) else cvars.to_dict(), -0.03)

    EV = df['p_win'] * e_win + (1 - df['p_win']) * e_non
    U  = EV - lam * np.abs(cvr)
    y = (U > 0).astype(int).values
    return y

def pick_features(df):
    # Always include p_tail and p_win; optionally include risk/context features if present
    base = ['p_tail', 'p_win']
    extra = [c for c in [
        'log1p_DTE','moneyness','delta','impliedVolatilityRank1y','percentToBreakEvenBid','VIX',
        'openInterest','volume','breakEvenProbability','potentialReturnAnnual','gex_center_abs_strike','gex_neg','gex_pos'
    ] if c in df.columns]
    feats = base + extra
    medians = df[feats].median(numeric_only=True).to_dict()
    return feats, medians

def build_monotone_constraints(features):
    # Encourage monotonic behavior:
    #   - p_win  increasing -> +1
    #   - p_tail decreasing -> -1
    mono = {'p_win': 1, 'p_tail': -1}
    return [mono.get(f, 0) for f in features]

def train_meta_accept(X, y, features, tscv=True, folds=5, monotone=None, seed=42):
    params = {
        'objective': 'binary',
        'metric': ['auc','binary_logloss'],
        'learning_rate': 0.05,
        'num_leaves': 63,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'min_data_in_leaf': 200,
        'class_weight': 'balanced',
        'verbosity': -1,
        'deterministic': True,
        'force_row_wise': True,
        'seed': seed
    }
    if monotone is not None:
        params['monotone_constraints'] = monotone

    oof = np.zeros(len(y))
    if tscv:
        splitter = TimeSeriesSplit(n_splits=folds)
        for tr, va in splitter.split(X):
            dtr = lgb.Dataset(X[tr], label=y[tr])
            dva = lgb.Dataset(X[va], label=y[va])
            model = lgb.train(params, dtr, valid_sets=[dva], num_boost_round=5000,
                              callbacks=[lgb.early_stopping(150, verbose=False)])
            oof[va] = model.predict(X[va])
        dall = lgb.Dataset(X, label=y)
        model = lgb.train(params, dall, num_boost_round=getattr(model, 'best_iteration', 800) or 800)
    else:
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        for tr, va in skf.split(X, y):
            dtr = lgb.Dataset(X[tr], label=y[tr])
            dva = lgb.Dataset(X[va], label=y[va])
            model = lgb.train(params, dtr, valid_sets=[dva], num_boost_round=5000,
                              callbacks=[lgb.early_stopping(150, verbose=False)])
            oof[va] = model.predict(X[va])
        dall = lgb.Dataset(X, label=y)
        model = lgb.train(params, dall, num_boost_round=getattr(model, 'best_iteration', 800) or 800)

    auc = roc_auc_score(y, oof) if len(np.unique(y)) > 1 else float('nan')
    return model, oof, float(auc)

class IsoCal:
    # minimal local isotonic calibrator
    def __init__(self):
        from sklearn.isotonic import IsotonicRegression
        self.iso = IsotonicRegression(out_of_bounds='clip')
    def fit(self, scores, labels):
        scores = np.asarray(scores).reshape(-1)
        labels = np.asarray(labels).reshape(-1)
        self.iso.fit(scores, labels); return self
    def predict(self, s):
        s = np.asarray(s).reshape(-1)
        p = self.iso.predict(s)
        return np.clip(p, 1e-6, 1-1e-6)

def main():
    # Read .env
    env, _ = load_env(".env_meta1")
    input_csv   = env.get('INPUT_CSV', './data/inputs/your_oof_scores.csv')
    tail_pack   = env.get('TAIL_MODEL_PACK')
    winner_pack = env.get('WINNER_MODEL_PACK')
    out_model   = env.get('ACCEPT_MODEL_PACK', './models/accept_meta_model.pkl')
    out_cal     = env.get('ACCEPT_CAL_PACK', './models/accept_meta_cal.pkl')
    label_kind  = (env.get('LABEL_ACCEPT') or 'win').strip().lower()  # 'win' or 'utility'
    lam         = _safe_float(env.get('LAMBDA_CVAR'), 0.25)

    assert tail_pack and winner_pack, "Set TAIL_MODEL_PACK and WINNER_MODEL_PACK in .env"

    df = pd.read_csv(input_csv)
    if 'tradeTime' in df.columns:
        df['trade_date'] = pd.to_datetime(df['tradeTime'], errors='coerce').dt.date
    if 'trade_date' in df.columns:
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df = df.sort_values('trade_date').reset_index(drop=True)

    # Build p_tail and p_win using your existing models
    tail = TailProbaAdapter(tail_pack)
    win  = WinnerProbaAdapter(winner_pack)
    cal  = IdentityCal()

    df['p_tail'] = tail.predict_df(df)
    df['p_win']  = cal.predict(win.predict_df(df))

    # Pick features
    feats, med = pick_features(df)
    X = df[feats].fillna(pd.Series(med)).values

    # Labels
    y = build_labels(df, label_kind=label_kind, lam=lam)

    # Monotone constraints: p_win +1, p_tail -1
    mono = build_monotone_constraints(feats)

    # Choose CV splitter
    has_time = 'trade_date' in df.columns and df['trade_date'].notna().any()
    model, oof, auc = train_meta_accept(X, y, feats, tscv=has_time, folds=5, monotone=mono, seed=42)

    # Calibrate
    cal_iso = IsoCal().fit(oof, y)

    # Save model pack + calibrator
    pack = {
        'model': model,
        'features': feats,
        'medians': med,
        'auc_oof': float(auc),
        'label_kind': label_kind,
        'lambda_cvar_for_utility': float(lam)
    }
    joblib.dump(pack, out_model)
    joblib.dump(cal_iso.iso, out_cal)

    print(f"[meta-train] label={label_kind}, oof_auc={auc:.4f}")
    print(f"[meta-train] saved model -> {out_model}")
    print(f"[meta-train] saved calibrator -> {out_cal}")
    print(f"[meta-train] features: {feats}")

if __name__ == "__main__":
    main()
