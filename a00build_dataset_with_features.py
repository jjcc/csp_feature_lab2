#!/usr/bin/env python3
"""
Dataset builder: load raw CSP snapshots, merge GEX, add macro/price/VIX features.

Notes
- Provide an importable/callable function (`build_dataset_with_features`) for BOTH training and scoring.
- Keep the original script workflow available (env/config driven) under `main()`.
- Remove env-dependent behavior from the core builder (e.g., GEX filtering), so feature parity is controllable via args.
"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass
from datetime import time
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import pandas as pd

from service.data_prepare import add_macro_features
from service.preprocess import filter_by_dte, load_csp_files, merge_gex
from service.env_config import get_derived_file, getenv, config


def ensure_cache_dir(out_dir: str) -> str:
    """Kept for backward compatibility; some pipelines expect price_cache dir existence."""
    pc = os.path.join(out_dir, "price_cache")
    os.makedirs(pc, exist_ok=True)
    return pc


def parse_target_time(s: str) -> time:
    """Parse 'HH:MM' into datetime.time; fallback to 11:00 on error."""
    try:
        hh, mm = s.split(":")
        return time(int(hh), int(mm))
    except Exception:
        return time(11, 0)


@dataclass
class BuildOutputs:
    df: pd.DataFrame
    report: Dict[str, Any]
    paths: Dict[str, Optional[str]]


def build_dataset_with_features(
    data_dir: str,
    glob_pat: str = "coveredPut_*.csv",
    target_time: str = "11:00",
    gex_base_dir: str = "",
    gex_target_time: str = "11:00",
    vix_csv: Optional[str] = None,
    px_base_dir: Optional[str] = None,
    *,
    enforce_daily_pick: bool = True,
    gex_filter_missing: bool = False,
    out_dir: Optional[str] = None,
    basic_csv_name: Optional[str] = None,
    skip_gex_merge: bool = False,
    filter_func: Optional[callable] = None,
) -> BuildOutputs:
    """
    Build a feature dataset with:
      - raw CSP fields (from `load_csp_files`)
      - GEX merge (from `merge_gex`)
      - macro + price return features (from `add_macro_features`)

    Core invariants:
      - No environment-variable-driven filtering/behavior in this function.
      - All behavior is controlled explicitly by function parameters.

    Parameters
    ----------
    data_dir : str
        Directory containing raw CSP snapshot CSVs.
    glob_pat : str
        Filename glob pattern for CSP snapshot CSVs.
    target_time : str
        Time string 'HH:MM' used by load_csp_files to pick a daily snapshot.
    gex_base_dir : str
        Base directory for GEX CSV/SVG inputs used by merge_gex.
    gex_target_time : str
        Time string 'HH:MM' that is converted to minutes for selecting GEX snapshot.
    vix_csv : Optional[str]
        Path to VIX CSV used by add_macro_features (can be None).
    px_base_dir : Optional[str]
        Directory containing <SYMBOL>.csv with columns including Date, Close (can be None).
    enforce_daily_pick : bool
        Passed through to load_csp_files (True keeps one snapshot per day at target_time).
    gex_filter_missing : bool
        If True, filter rows where gex_missing==1 AFTER macro feature generation.
    out_dir : Optional[str]
        If provided, write intermediate and final CSVs to this directory (compatible with old behavior).
    basic_csv_name : Optional[str]
        Used to generate the derived output CSV name (via get_derived_file). Only relevant if out_dir is set.
    skip_gex_merge : bool
        If True, do not call merge_gex; expects an existing *_gex.csv written previously in out_dir.

    Returns
    -------
    BuildOutputs
        .df: final dataframe
        .report: simple JSON-friendly report
        .paths: dict of written file paths (or None)
    """
    if not gex_base_dir:
        raise ValueError("gex_base_dir must be provided (was empty).")

    # Normalize optional paths
    vix_csv = (vix_csv or "").strip() or None
    px_base_dir = (px_base_dir or "").strip() or None

    gex_target_t = parse_target_time(gex_target_time)
    gex_target_minutes = gex_target_t.hour * 60 + gex_target_t.minute

    # Step 1: Load raw CSP data
    raw = load_csp_files(
        data_dir,
        glob_pat,
        target_time=target_time,
        enforce_daily_pick=enforce_daily_pick,
    )
    if filter_func:
        raw = filter_func(raw)
    raw = raw.reset_index().rename(columns={"index": "row_id"})

    # Optional outputs
    paths: Dict[str, Optional[str]] = {"raw_csv": None, "gex_csv": None, "out_csv": None}

    # If writing outputs, ensure out_dir exists
    out_dir_path: Optional[Path] = Path(out_dir) if out_dir else None
    if out_dir_path:
        out_dir_path.mkdir(parents=True, exist_ok=True)

        # Raw copy naming: keep backward-compatible behavior:
        # write raw CSV under out_dir using the basename of basic_csv_name (or fallback).
        raw_name = os.path.basename(basic_csv_name or "trades_raw_orig.csv")
        raw_csv_path = out_dir_path / raw_name
        raw.to_csv(raw_csv_path, index=False)
        paths["raw_csv"] = str(raw_csv_path)

    trades = raw

    # Step 2: Merge GEX
    if skip_gex_merge:
        if not out_dir_path or not paths["raw_csv"]:
            raise ValueError("skip_gex_merge=True requires out_dir and a prior written raw_csv to infer *_gex.csv path.")
        gex_csv_path = Path(paths["raw_csv"]).with_suffix("").as_posix() + "_gex.csv"
        gex_csv_path = out_dir_path / os.path.basename(gex_csv_path)
        if not gex_csv_path.exists():
            raise FileNotFoundError(f"skip_gex_merge=True but expected existing file not found: {gex_csv_path}")
        gex_merged = pd.read_csv(gex_csv_path)
        paths["gex_csv"] = str(gex_csv_path)
    else:
        gex_merged = merge_gex(trades, gex_base_dir, gex_target_minutes)
        if out_dir_path:
            # Keep old naming scheme: raw basename + "_gex.csv"
            raw_basename = os.path.basename(paths["raw_csv"]) if paths["raw_csv"] else (os.path.basename(basic_csv_name or "trades_raw_orig.csv"))
            gex_csv_name = raw_basename.replace(".csv", "_gex.csv")
            gex_csv_path = out_dir_path / gex_csv_name
            gex_merged.to_csv(gex_csv_path, index=False)
            paths["gex_csv"] = str(gex_csv_path)

    # Step 3: Add macro features
    d = add_macro_features(gex_merged, vix_csv, px_base_dir)

    # Optional filter (explicit arg, NOT env var)
    if gex_filter_missing:
        if "gex_missing" not in d.columns:
            raise KeyError("gex_filter_missing=True but dataframe lacks 'gex_missing' column.")
        d = d[d["gex_missing"] == 0].copy()

    # Output naming (derived)
    if out_dir_path:
        if basic_csv_name:
            derived_name = get_derived_file(basic_csv_name)[0]
        else:
            # Fallback if caller doesn't have a "basic csv" concept.
            derived_name = "dataset_gex_macro.csv"

        out_csv_name = derived_name.replace(".csv", "_gexonly.csv") if gex_filter_missing else derived_name
        out_csv_path = out_dir_path / out_csv_name
        d.to_csv(out_csv_path, index=False)
        paths["out_csv"] = str(out_csv_path)

    # Simple report
    rep: Dict[str, Any] = {
        "rows_raw": int(len(raw)),
        "rows_gex_merged": int(len(gex_merged)),
        "gex_found": int((gex_merged.get("gex_missing", pd.Series([1] * len(gex_merged))) == 0).sum()) if len(gex_merged) else 0,
        "gex_missing": int((gex_merged.get("gex_missing", pd.Series([1] * len(gex_merged))) == 1).sum()) if len(gex_merged) else 0,
        "gex_base_dir": gex_base_dir,
        "gex_target_time": gex_target_time,
        "target_time": target_time,
        "enforce_daily_pick": bool(enforce_daily_pick),
        "rows_out": int(len(d)),
        "unique_symbols": int(d["baseSymbol"].nunique()) if "baseSymbol" in d.columns else None,
        "vix_non_null": int(d["VIX"].notna().sum()) if "VIX" in d.columns else None,
        "prev_close_non_null": int(d["prev_close"].notna().sum()) if "prev_close" in d.columns else None,
        "px_base_dir": px_base_dir,
        "vix_csv": vix_csv,
        "gex_filter_missing": bool(gex_filter_missing),
        "paths": paths,
    }

    return BuildOutputs(df=d, report=rep, paths=paths)


# Backward-compatible alias (your code currently calls this)
def build_dataset_with_feat(
    data_dir: str,
    glob_pat: str,
    target_time: str,
    out_dir: str,
    base_dir: str,
    gex_target_time_str: str,
    VIX_CSV: Optional[str],
    PX_BASE_DIR: Optional[str],
    basic_csv: str,
):
    out = build_dataset_with_features(
        data_dir=data_dir,
        glob_pat=glob_pat,
        target_time=target_time,
        gex_base_dir=base_dir,
        gex_target_time=gex_target_time_str,
        vix_csv=VIX_CSV,
        px_base_dir=PX_BASE_DIR,
        enforce_daily_pick=True,
        gex_filter_missing=False,  # IMPORTANT: no env-dependent drift; control explicitly at call sites
        out_dir=out_dir,
        basic_csv_name=basic_csv,
    )
    print(json.dumps(out.report, indent=2))
    return out.df


def main():
    # Env-driven script wrapper (fine for CLI), but calls the shared builder for parity.
    data_dir = getenv("COMMON_DATA_DIR", "")
    glob_pat = getenv("DATA_GLOB", "coveredPut_*.csv")
    target_time = getenv("DATA_TARGET_TIME", "11:00")

    # outputs
    out_dir = getenv("COMMON_OUTPUT_DIR", "output")
    out_dir = os.path.join(out_dir, "data_prep")
    os.makedirs(out_dir, exist_ok=True)

    # GEX source
    base_dir = getenv("GEX_BASE_DIR")
    gex_target_time_str = getenv("GEX_TARGET_TIME", "11:00")
    if not base_dir:
        raise SystemExit("GEX_BASE_DIR is not set in .env")

    # VIX and price sources
    VIX_CSV = getenv("MACRO_VIX_CSV", "").strip() or None
    PX_BASE_DIR = getenv("MACRO_PX_BASE_DIR", "").strip() or None  # dir with <SYMBOL>.csv, Date, Close

    # IMPORTANT:
    # This script no longer reads GEX_FILTER from env to avoid silent training/scoring drift.
    # If you still want that behavior for one-off runs, explicitly pass gex_filter_missing=True here.
    gex_filter_missing = False

    # For all
    common_configs = config.get_common_configs_raw()
    for k, v in common_configs.items():
        basic_csv = v.get("data_basic_csv", "trades_raw_orig.csv")
        data_dir_k = v.get("data_dir", data_dir)

        if k == "original":
            print(f"Skipping {k}")
            continue

        ENFORCE_DAILY_PICK = False
        out = build_dataset_with_features(
            data_dir=data_dir_k,
            glob_pat=glob_pat,
            target_time=target_time,
            gex_base_dir=base_dir,
            gex_target_time=gex_target_time_str,
            vix_csv=VIX_CSV,
            px_base_dir=PX_BASE_DIR,
            enforce_daily_pick=ENFORCE_DAILY_PICK,
            gex_filter_missing=gex_filter_missing,
            out_dir=out_dir,
            basic_csv_name=basic_csv,
            filter_func = filter_by_dte,
        )
        print(json.dumps(out.report, indent=2))
        date_str = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        log_fn = os.path.join("log", f"a00build_dataset_log_{k}_{date_str}.json".replace(" ", "_").replace(":", "-"))
        with open(log_fn, "w") as f:
            json.dump(out.report, f, indent=2)


if __name__ == "__main__":
    main()
