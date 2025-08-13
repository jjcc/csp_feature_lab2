from pathlib import Path
import pandas as pd



def parse_timestamp_from_filename(name: str):
    """
    Expecting coveredPut_YYYY-MM-DD_HH_MM.csv or similar.
    Returns (date, time) as datetime.date, datetime.time or (None,None) if not parsed.
    """
    import re, datetime as dt
    m = re.search(r'(\d{4}-\d{2}-\d{2})_(\d{2})_(\d{2})', name)
    if not m:
        return None, None
    d = dt.datetime.strptime(m.group(1), "%Y-%m-%d").date()
    t = dt.time(int(m.group(2)), int(m.group(3)))
    return d, t

def pick_daily_snapshot_files(data_dir: str, pattern: str, target_time_str: str = "11:00"):
    """
    Groups files by date and picks one file per date:
    - choose the file with time >= target_time that is closest to it
    - if none, fall back to the latest file before target_time
    Returns a sorted list of absolute paths.
    """
    import glob, os, datetime as dt
    target_h, target_m = map(int, target_time_str.split(":"))
    target = dt.time(target_h, target_m)

    #paths = sorted(glob.glob(os.path.join(data_dir, pattern)))
    paths = sorted([str(Path(p)) for p in Path(data_dir).glob(pattern)])
    by_date = {}
    for p in paths:
        name = os.path.basename(p)
        d, t = parse_timestamp_from_filename(name)
        if d is None or t is None:
            continue
        by_date.setdefault(d, []).append((t, p))

    chosen = []
    for d, lst in by_date.items():
        lst.sort()  # sort by time
        after = [x for x in lst if x[0] >= target]
        if after:
            # pick nearest after target
            chosen.append(min(after, key=lambda x: (dt.datetime.combine(d, x[0]) - dt.datetime.combine(d, target)).total_seconds())[1])
        else:
            # pick latest before target
            chosen.append(max(lst, key=lambda x: x[0])[1])
    chosen = [str(Path(p)) for p in chosen]
    return sorted(chosen)


def load_csp_files(data_dir: str, pattern: str, target_time="11:00", enforce_daily_pick=True, cut_off_date=None) -> pd.DataFrame:
    if enforce_daily_pick:
        paths = pick_daily_snapshot_files(data_dir, pattern, target_time)
    else:
        import glob, os
        paths = sorted(glob.glob(os.path.join(data_dir, pattern)))
    frames = []
    for p in paths:
        try:
            df = pd.read_csv(p)
            base_file = Path(p).name
            df["__source_file"] = base_file
            frames.append(df)
        except Exception as e:
            print(f"[WARN] Could not read {p}: {e}")
    if not frames:
        raise SystemExit(f"No files found for pattern {pattern} in {data_dir}")
    return pd.concat(frames, ignore_index=True)
