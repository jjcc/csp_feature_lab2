#!/usr/bin/env python3
"""
EDGAR earnings-event extractor (8-K Item 2.02 only) using YAML config.

- Reads tickers + date range from YAML
- Maps ticker -> CIK via SEC company_tickers.json
- Pulls each company's recent filings (submissions endpoint)
- Filters to 8-K filings in date window
- Fetches each 8-K primary document (HTML/text) and keeps only those that mention Item 2.02
- Outputs CSV of earnings-like events (8-K Item 2.02) with filing date + filing URL

Why this is useful
- EDGAR has no "earnings calendar" field. 8-K Item 2.02 is the closest free proxy to earnings release.
- This script avoids using 10-Q/10-K as earnings fallbacks (too laggy).

Install
  pip install requests pyyaml

Run
  python edgar_8k_item202.py

Config
  Create edgar_config.yaml next to this script (example below).
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import requests
import yaml

SEC_DATA_BASE = "https://data.sec.gov"
TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"

# Heuristic patterns for Item 2.02 in 8-K text
ITEM_2_02_RE = re.compile(r"\bitem\s*2\.?\s*02\b", re.IGNORECASE)
RESULTS_OF_OPERATIONS_RE = re.compile(r"results\s+of\s+operations\s+and\s+financial\s+condition", re.IGNORECASE)

# Some filings use HTML with entities/spaces; this helps catch variants like "Item&nbsp;2.02"
ITEM_2_02_HTMLISH_RE = re.compile(r"item(?:\s|&nbsp;|&#160;)*2(?:\s|&nbsp;|&#160;)*\.?(?:\s|&nbsp;|&#160;)*02", re.IGNORECASE)

# --- Earnings semantic rules ---

POS_PERIOD = re.compile(
    r"(three|six|nine|twelve)\s+months\s+ended|quarter\s+ended|fiscal\s+quarter\s+ended|year\s+ended|fiscal\s+year\s+ended",
    re.IGNORECASE,
)

POS_METRICS = re.compile(
    r"\b(eps|earnings\s+per\s+share|net\s+(income|loss)|revenue|operating\s+(income|loss)|adjusted\s+ebitda)\b",
    re.IGNORECASE,
)

POS_PRESS = re.compile(
    r"\bpress\s+release\b|exhibit\s+99(\.1|\.01)?\b",
    re.IGNORECASE,
)

NEG_OFFERING = re.compile(
    r"\boffering\b|registered\s+direct\s+offering|private\s+placement|rule\s+144a|capped\s+call|underwrit|convertible\s+senior\s+notes|senior\s+notes|\batm\b|at\s+the\s+market|equity\s+distribution\s+agreement|term\s+sheet|pricing\s+of\s+the\s+offering",
    re.IGNORECASE,
)

NEG_PRELIM = re.compile(
    r"not\s+yet\s+complete|will\s+not\s+be\s+available\s+until|preliminary\s+unaudited|subject\s+to\s+revision|closing\s+procedures|undue\s+reliance",
    re.IGNORECASE,
)

NEG_LIQUIDITY = re.compile(
    r"liquidity\s+update|cash\s+and\s+cash\s+equivalents|restricted\s+cash|indebtedness|borrowed\s+money|covenant",
    re.IGNORECASE,
)

NEG_SUPPLEMENT = re.compile(r"supplement(ing)?\s+and\s+updat(ing|e)\s+disclosures", re.IGNORECASE)



@dataclass
class FilingEvent:
    ticker: str
    cik: str
    filing_date: str
    report_date: str
    accession: str
    primary_document: str
    filing_url: str
    match_rule: str  # which heuristic matched


def iso_date(s: str) -> dt.date:
    return dt.datetime.strptime(s, "%Y-%m-%d").date()


def ensure_dir(p: str) -> None:
    if p:
        os.makedirs(p, exist_ok=True)


def read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def write_text_file(path: str, text: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8", errors="ignore") as f:
        f.write(text)


def load_or_fetch_json(url: str, headers: Dict[str, str], cache_path: str, sleep_s: float) -> dict:
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    data = r.json()
    #with open("output/data_prep/corp_events/company_tickers.json", "r", encoding="utf-8") as f:
    #    data = json.load(f)

    ensure_dir(os.path.dirname(cache_path))
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(data, f)

    time.sleep(sleep_s)
    return data


def fetch_url_text(url: str, headers: Dict[str, str], cache_path: str, sleep_s: float) -> str:
    if os.path.exists(cache_path):
        return read_text_file(cache_path)

    # IMPORTANT: EDGAR Archive URLs are on www.sec.gov
    h = dict(headers)
    h["Host"] = "www.sec.gov"

    r = requests.get(url, headers=h, timeout=30)
    r.raise_for_status()
    text = r.text

    write_text_file(cache_path, text)
    time.sleep(sleep_s)
    return text


def build_ticker_to_cik(headers: Dict[str, str], cache_dir: str, sleep_s: float) -> Dict[str, str]:
    cache_path = os.path.join(cache_dir, "company_tickers.json")
    data = load_or_fetch_json(TICKER_MAP_URL, headers, cache_path, sleep_s)

    out: Dict[str, str] = {}
    for rec in data.values():
        ticker = str(rec.get("ticker", "")).upper().strip()
        cik = rec.get("cik_str")
        if ticker and cik is not None:
            out[ticker] = str(cik).zfill(10)
    return out


def fetch_submissions(cik10: str, headers: Dict[str, str], cache_dir: str, sleep_s: float) -> dict:
    url = f"{SEC_DATA_BASE}/submissions/CIK{cik10}.json"
    cache_path = os.path.join(cache_dir, "submissions", f"CIK{cik10}.json")
    return load_or_fetch_json(url, headers, cache_path, sleep_s)

def build_archive_url(cik10: str, accession: str, primary_document: str) -> str:
    cik_no_zeros = str(int(cik10))  # remove leading zeros
    acc_no_dashes = accession.replace("-", "")
    return f"https://www.sec.gov/Archives/edgar/data/{cik_no_zeros}/{acc_no_dashes}/{primary_document}"


def in_range(date_str: str, start: dt.date, end: dt.date) -> bool:
    if not date_str:
        return False
    try:
        d = iso_date(date_str)
    except ValueError:
        return False
    return start <= d <= end


def looks_like_item_202(text: str) -> Optional[str]:
    """
    Return rule label if this 8-K looks like an earnings release (Item 2.02 + semantics).
    """
    # Must be Item 2.02-ish
    has_item = bool(ITEM_2_02_RE.search(text) or ITEM_2_02_HTMLISH_RE.search(text))
    if not has_item:
        return None
    
    # Compute positives once
    has_period = bool(POS_PERIOD.search(text))
    has_metrics = bool(POS_METRICS.search(text))
    has_press = bool(POS_PRESS.search(text))  # if you use it later
    
    # Hard rejects: offering/financing + prelim disclaimers
    if NEG_OFFERING.search(text) and NEG_PRELIM.search(text):
        return None
    
    # Strong reject: offering language but no earnings semantics at all
    if NEG_OFFERING.search(text) and not has_period and not has_metrics:
        return None
    
    # Liquidity-driven reject ONLY when earnings signals are weak/absent
    if NEG_LIQUIDITY.search(text) and not has_period and not has_metrics:
        return None
    
    # Require earnings-positive semantics
    if not (has_period or has_metrics):
        return None
    
    # Extra reject you added
    if NEG_OFFERING.search(text) and NEG_SUPPLEMENT.search(text):
        return None
    
    # If you return labels:
    if has_period and has_press:
        return "earnings_period_plus_press"
    if has_period and has_metrics:
        return "earnings_period_plus_metrics"
    if has_period:
        return "earnings_period"
    if has_metrics and has_press:
        return "earnings_metrics_plus_press"
    return "earnings_metrics"
 


def extract_events(sub: dict) -> List[Tuple[str, str, str, str]]:
    """
    Returns list of tuples: (filing_date, report_date, accession, primary_document) for 8-Ks.
    """
    recent = (sub.get("filings") or {}).get("recent") or {}

    forms = recent.get("form", [])
    filing_dates = recent.get("filingDate", [])
    report_dates = recent.get("reportDate", [])
    accessions = recent.get("accessionNumber", [])
    primary_docs = recent.get("primaryDocument", [])

    out: List[Tuple[str, str, str, str]] = []
    n = min(len(forms), len(filing_dates), len(accessions), len(primary_docs))
    for i in range(n):
        if str(forms[i]).strip() != "8-K":
            continue
        fdate = str(filing_dates[i]).strip()
        rdate = str(report_dates[i]).strip() if i < len(report_dates) else ""
        acc = str(accessions[i]).strip()
        doc = str(primary_docs[i]).strip()
        if fdate and acc and doc:
            out.append((fdate, rdate, acc, doc))
    return out


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Config YAML must be a mapping/object.")
    return cfg


def read_tickers(path: str) -> List[str]:
    # Accept one ticker per line, or comma/space separated
    raw = read_text_file(path)
    toks = [t.strip().upper() for t in raw.replace(",", " ").split()]
    return [t for t in toks if t]


def write_csv(events: List[FilingEvent], out_path: str) -> None:
    ensure_dir(os.path.dirname(out_path))
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "ticker",
            "cik",
            "event_type",
            "filing_date",
            "report_date",
            "accession",
            "primary_document",
            "filing_url",
            "match_rule",
        ])
        for e in sorted(events, key=lambda x: (x.ticker, x.filing_date, x.accession)):
            w.writerow([
                e.ticker,
                e.cik,
                "EARNINGS_8K_ITEM_2_02",
                e.filing_date,
                e.report_date,
                e.accession,
                e.primary_document,
                e.filing_url,
                e.match_rule,
            ])


def main() -> None:
    cfg = load_config("corp_action_config.yaml")

    user_agent = cfg.get("user_agent")
    if not user_agent or not isinstance(user_agent, str):
        raise SystemExit('Config must include user_agent: "Your Name email@domain.com"')

    headers = {
        "User-Agent": user_agent,
        "Accept-Encoding": "gzip, deflate",
        # Host is set per-request (data.sec.gov vs www.sec.gov)
    }
    dr = cfg.get("date_range") or {}

    start = iso_date(dr["start"])
    end = iso_date(dr["end"])
    sleep_s = float(cfg.get("sleep_seconds", 0.3))
    cache_dir = cfg.get("cache_dir", ".edgar_cache")
    tickers_file = cfg.get("tickers_file", "tickers.txt")
    tickers = read_tickers(tickers_file)
    # get only first 20 for testing
    #tickers = tickers[:20]

    if not tickers:
        raise SystemExit("No tickers found. Put tickers in tickers_file.")

    if end < start:
        raise SystemExit("date_range.end must be >= date_range.start")


    max_fetch = cfg.get("max_8k_fetch_per_ticker")
    max_fetch = int(max_fetch) if max_fetch is not None else 50

    out_csv = cfg.get("output_csv", "earnings_8k_item202.csv")

    # Build mapping
    start_time = time.time()
    t2c = build_ticker_to_cik(headers, cache_dir, sleep_s)

    events: List[FilingEvent] = []
    missing: List[str] = []

    count = 0
    for t in tickers:
        cik10 = t2c.get(t)
        if not cik10:
            missing.append(t)
            continue

        try:
            sub = fetch_submissions(cik10, headers=headers, cache_dir=cache_dir, sleep_s=sleep_s)
            candidates = extract_events(sub)

            # Filter by date range first (saves fetches)
            candidates = [c for c in candidates if in_range(c[0], start, end)]
            candidates = candidates[:max_fetch]

            for filing_date, report_date, accession, primary_doc in candidates:
                filing_url = build_archive_url(cik10, accession, primary_doc)

                # Cache per filing document
                cache_path = os.path.join(
                    cache_dir,
                    "filings",
                    t,
                    accession.replace("-", ""),
                    primary_doc.replace("/", "_"),
                )

                text = fetch_url_text(filing_url, headers=headers, cache_path=cache_path, sleep_s=sleep_s)
                rule = looks_like_item_202(text)
                if not rule:
                    continue

                events.append(
                    FilingEvent(
                        ticker=t,
                        cik=cik10,
                        filing_date=filing_date,
                        report_date=report_date,
                        accession=accession,
                        primary_document=primary_doc,
                        filing_url=filing_url,
                        match_rule=rule,
                    )
                )

        except requests.HTTPError as e:
            print(f"[WARN] {t} CIK={cik10}: HTTP error: {e}")
        except Exception as e:
            print(f"[WARN] {t} CIK={cik10}: error: {e}")
        count += 1
        if count % 5 == 0:
            print(f"Processed {count}/{len(tickers)} tickers...")

    elapsed = time.time() - start_time
    print(f"Processed {len(tickers)} tickers in {elapsed:.1f}) seconds.")
    write_csv(events, out_csv)

    print(f"Saved {len(events)} earnings events → {out_csv}")
    if missing:
        print(f"Tickers missing CIK mapping ({len(missing)}): {missing[:20]}")


if __name__ == "__main__":
    main()
