#!/usr/bin/env python3
"""
EDGAR corporate-event extractor using YAML config.

- Reads tickers + date range from YAML
- Fetches EDGAR filings (10-Q, 10-K, optional 8-K)
- Outputs CSV of filing events for dataset cleaning

Requirements:
  pip install requests pyyaml
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import os
import time
from dataclasses import dataclass
from typing import Dict, List

import requests
import yaml

SEC_BASE = "https://data.sec.gov"
TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"


@dataclass
class FilingEvent:
    ticker: str
    cik: str
    form: str
    filing_date: str
    report_date: str
    accession: str
    primary_document: str
    filing_url: str


def iso_date(s: str) -> dt.date:
    return dt.datetime.strptime(s, "%Y-%m-%d").date()


def ensure_dir(p: str) -> None:
    if p:
        os.makedirs(p, exist_ok=True)


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
    cache_path = os.path.join(cache_dir, "submissions", f"CIK{cik10}.json")
    url = f"{SEC_BASE}/submissions/CIK{cik10}.json"
    return load_or_fetch_json(url, headers, cache_path, sleep_s)


def extract_events(
    ticker: str,
    cik10: str,
    sub: dict,
    start: dt.date,
    end: dt.date,
    include_8k: bool,
) -> List[FilingEvent]:
    recent = (sub.get("filings") or {}).get("recent") or {}

    forms = recent.get("form", [])
    filing_dates = recent.get("filingDate", [])
    report_dates = recent.get("reportDate", [])
    accessions = recent.get("accessionNumber", [])
    primary_docs = recent.get("primaryDocument", [])

    wanted = {"10-Q", "10-K"}
    if include_8k:
        wanted.add("8-K")

    events: List[FilingEvent] = []
    n = min(len(forms), len(filing_dates), len(accessions), len(primary_docs))

    for i in range(n):
        form = str(forms[i]).strip()
        if form not in wanted:
            continue

        fdate = filing_dates[i]
        if not fdate:
            continue

        d = iso_date(fdate)
        if not (start <= d <= end):
            continue

        rdate = report_dates[i] if i < len(report_dates) else ""
        acc = accessions[i]
        doc = primary_docs[i]

        cik_no_zeros = str(int(cik10))
        acc_no_dashes = acc.replace("-", "")
        url = f"https://www.sec.gov/Archives/edgar/data/{cik_no_zeros}/{acc_no_dashes}/{doc}"

        events.append(
            FilingEvent(
                ticker=ticker,
                cik=cik10,
                form=form,
                filing_date=fdate,
                report_date=rdate,
                accession=acc,
                primary_document=doc,
                filing_url=url,
            )
        )

    return events


def write_csv(events: List[FilingEvent], path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "ticker",
            "cik",
            "form",
            "filing_date",
            "report_date",
            "accession",
            "primary_document",
            "filing_url",
        ])
        for e in sorted(events, key=lambda x: (x.ticker, x.filing_date, x.form)):
            w.writerow([
                e.ticker,
                e.cik,
                e.form,
                e.filing_date,
                e.report_date,
                e.accession,
                e.primary_document,
                e.filing_url,
            ])


def main():
    with open("corp_action_config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    headers = {
        "User-Agent": cfg["user_agent"],
        "Accept-Encoding": "gzip, deflate",
        "Host": "data.sec.gov",
    }

    start = iso_date(cfg["date_range"]["start"])
    end = iso_date(cfg["date_range"]["end"])
    include_8k = bool(cfg.get("include_8k", False))
    sleep_s = float(cfg.get("sleep_seconds", 0.2))
    cache_dir = cfg.get("cache_dir", ".edgar_cache")

    with open(cfg["tickers_file"], "r", encoding="utf-8") as f:
        tickers = [t.strip().upper() for t in f if t.strip()]
    # get only first 20 for testing
    tickers = tickers[:20]

    start_time = time.time()
    t2c = build_ticker_to_cik(headers, cache_dir, sleep_s)

    events: List[FilingEvent] = []
    missing = []

    for t in tickers:
        cik10 = t2c.get(t)
        if not cik10:
            missing.append(t)
            continue

        try:
            sub = fetch_submissions(cik10, headers, cache_dir, sleep_s)
            events.extend(extract_events(t, cik10, sub, start, end, include_8k))
        except Exception as e:
            print(f"[WARN] {t}: {e}")

    elapsed = time.time() - start_time
    print(f"Processed {len(tickers)} tickers in {elapsed:.1f}) seconds.")
    write_csv(events, cfg["output_csv"])

    print(f"Saved {len(events)} events → {cfg['output_csv']}")
    if missing:
        print(f"Tickers missing CIK mapping ({len(missing)}): {missing[:20]}")


if __name__ == "__main__":
    main()
