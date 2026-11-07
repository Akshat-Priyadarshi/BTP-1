#!/usr/bin/env python3
"""
Build 30-second windows (16 Hz) from Empatica EmbracePlus raw avro files + per-minute CSVs.

Per window (480 samples per channel):
    acc_x, acc_y, acc_z (g), bvp, eda, temp          -> 16 Hz values
    FIRST cell of the window (30 s level):
        act_count, met, pulse_rate, resp_rate        -> from per-minute/epoch CSVs
    FIRST cell of the 60 s block (minute level):
        activity_class, activity_intensity, body_position, sleep
        (by default written in the first row of every window, using the minute the
         window falls in; use --minute-first-only to write them only in the window that
         starts at :00 of the minute)

Outputs (in --out):
    windows_16hz.csv   long format: one row per sample (480 rows per window)
    windows_16hz.npz   X (N,480,6) float32, window_start_utc (N,), epoch (N,4), minute (N,4), names
    windows_summary.csv  one row per window (kept/NaN fractions)

Usage:
    python make_windows_16hz.py --raw  C:/data/embrace/raw_avro \
                                --minute C:/data/embrace/per_minute \
                                --out C:/data/embrace/out \
                                [--start "2026-08-14 10:30:00" --end "2026-08-14 11:30:00"]   # UTC, optional
No AWS credentials are used here: download the files first.
"""
import argparse, glob, os, re
import numpy as np
import pandas as pd
from fastavro import reader as avro_reader
from scipy.signal import butter, sosfiltfilt

FS = 16
WIN_S = 30
WIN_N = FS * WIN_S  # 480
CHANNELS = ["acc_x", "acc_y", "acc_z", "bvp", "eda", "temp", "resp"]

# BTP-ADDED: respiration as a seventh channel.
#
# A wrist device does not measure breathing, but it leaves a trace in the
# heart rhythm: the gap between beats shortens on the in-breath and
# lengthens on the out-breath. The avro files carry systolicPeaks, which is
# the time of every detected beat, so the gaps can be worked out and
# band-passed to the breathing band.
#
# What goes in the channel is the WAVEFORM, not a rate. A rate of 15 breaths
# a minute repeated 480 times is a flat line, and a flat line gives Chronos
# nothing: we measured that on skin temperature, where 98 percent of values
# landed in a single bin. The band-passed interval series oscillates at the
# breathing frequency itself, which is real shape at 16 Hz.
#
# Use --resp-mode rate for the rate in breaths per minute instead.
#
# Worth knowing what this channel is worth. Checked against WESAD's chest
# respiration belt across all fifteen subjects, a rate derived this way
# correlated at 0.185. The level is about right but the moment to moment
# tracking is poor, because the modulation is weak in real wrist PPG.
RESP_LOW, RESP_HIGH = 0.1, 0.5      # 6 to 30 breaths a minute
RESP_GRID_HZ = 4.0                  # the gaps are resampled here before filtering
MIN_IBI, MAX_IBI = 0.333, 1.5       # 40 to 180 bpm; outside is a detection error

# name -> list of filename keywords (first one that matches a file wins)
EPOCH_MAP = {
    "act_count": ["activity_counts", "acticounts"],
    "met": ["met"],
    "pulse_rate": ["pulse_rate"],
    "resp_rate": ["respiratory_rate"],
}
MINUTE_MAP = {
    "activity_class": ["activity_classification"],
    "activity_intensity": ["activity_intensity"],
    "body_position": ["body_position"],
    "sleep": ["sleep_detection"],
}


# ----------------------------------------------------------------------------- avro
def _imu_scale(params):
    if not params:
        return 1.0
    cf = params.get("conversionFactor")
    if cf:
        return cf
    try:
        return (params["physicalMax"] - params["physicalMin"]) / (params["digitalMax"] - params["digitalMin"])
    except Exception:
        return 1.0


def _in_range(path, start, end):
    """If the path contains a UTC date folder (YYYY-MM-DD), keep it only if it overlaps [start, end]."""
    m = re.search(r"(\d{4}-\d{2}-\d{2})", path.replace("\\", "/"))
    if not m or (start is None and end is None):
        return True
    d = pd.Timestamp(m.group(1))
    if start is not None and d + pd.Timedelta(days=1) <= start:
        return False
    if end is not None and d >= end:
        return False
    return True


def read_avro_folder(raw_dir, start=None, end=None, resp_mode="wave"):
    """Return dict channel -> DataFrame(t_us, value) for all avro files (sorted, deduped).
    Files in <...>/<UTC date>/... folders outside [start, end] are skipped (saves memory)."""
    files = sorted(f for f in glob.glob(os.path.join(raw_dir, "**", "*.avro"), recursive=True)
                   if _in_range(f, start, end))
    if not files:
        raise SystemExit(f"No .avro files under {raw_dir}")
    parts = {c: [] for c in CHANNELS}
    peak_us = []        # BTP-ADDED: beat times, for the respiration channel
    for f in files:
        with open(f, "rb") as fh:
            for rec in avro_reader(fh):
                raw = rec.get("rawData") or {}
                s = raw.get("accelerometer")
                if s and s.get("samplingFrequency", 0) and len(s.get("x", [])):
                    fs = s["samplingFrequency"]
                    sc = _imu_scale(s.get("imuParams"))
                    n = len(s["x"])
                    t = s["timestampStart"] + np.arange(n) * (1e6 / fs)
                    for ax in "xyz":
                        parts[f"acc_{ax}"].append(
                            pd.DataFrame({"t": t, "v": np.asarray(s[ax], dtype=float) * sc, "fs": fs}))
                # BTP-ADDED: the heartbeat times. Different firmware spells
                # this field differently, so try the ones seen in practice.
                sp = raw.get("systolicPeaks")
                if sp:
                    for key in ("peaksTimeNanos", "peaksTimeNano",
                                "peakTimeNanos", "values"):
                        v = sp.get(key)
                        if v is not None and len(v):
                            peak_us.append(np.asarray(v, dtype=float) / 1e3)
                            break

                for sensor, ch in (("bvp", "bvp"), ("eda", "eda"), ("temperature", "temp")):
                    s = raw.get(sensor)
                    if s and s.get("samplingFrequency", 0) and len(s.get("values", [])):
                        fs = s["samplingFrequency"]
                        n = len(s["values"])
                        t = s["timestampStart"] + np.arange(n) * (1e6 / fs)
                        parts[ch].append(pd.DataFrame({"t": t, "v": np.asarray(s["values"], dtype=float), "fs": fs}))
    # BTP-ADDED: turn the beat times into a breathing channel
    if peak_us:
        beats = np.unique(np.concatenate(peak_us))
        print("[info] %d heartbeats found, for the respiration channel"
              % len(beats))
        parts["resp"] = [resp_from_peaks(beats, resp_mode)]
    else:
        print("[warn] no systolicPeaks in the avro files, so respiration "
              "will be empty")

    out = {}
    for c, lst in parts.items():
        if not lst:
            print(f"[warn] no data for {c}")
            out[c] = pd.DataFrame({"t": [], "v": [], "fs": []})
            continue
        d = pd.concat(lst, ignore_index=True).sort_values("t").drop_duplicates("t")
        out[c] = d.reset_index(drop=True)
    return out


def resp_from_peaks(beat_us, mode="wave"):
    """BTP-ADDED. A breathing channel from heartbeat times.

    beat_us is the time of every detected beat, in microseconds.

        gaps between beats
          -> drop anything outside 40 to 180 bpm, which is a detection error
          -> put the gaps on a steady 4 Hz grid
          -> band-pass 0.1 to 0.5 Hz, which is 6 to 30 breaths a minute
          -> that filtered series IS the breathing waveform

    Returned in the same shape as the other channels, so resample_to_grid
    can lift it to 16 Hz alongside them.

    mode="rate" returns breaths per minute instead, counted over a rolling
    30 seconds. Flatter, and therefore worse for Chronos, but it is what a
    person would recognise as a breathing rate.
    """
    empty = pd.DataFrame({"t": [], "v": [], "fs": []})
    b = np.sort(np.asarray(beat_us, dtype=float))
    if len(b) < 20:
        return empty

    sec = b / 1e6
    ibi = np.diff(sec)
    at = sec[1:]

    ok = (ibi >= MIN_IBI) & (ibi <= MAX_IBI)
    ibi, at = ibi[ok], at[ok]
    if len(ibi) < 20:
        return empty

    # Break where the sensor stopped reading. Interpolating a breathing
    # waveform across a gap of minutes would be inventing it.
    breaks = np.where(np.diff(at) > 30.0)[0]
    bounds = [0] + (breaks + 1).tolist() + [len(at)]

    nyq = RESP_GRID_HZ / 2.0
    sos = butter(2, [RESP_LOW / nyq, min(RESP_HIGH / nyq, 0.99)],
                 btype="band", output="sos")

    ts, vs = [], []
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        seg_at, seg_ibi = at[lo:hi], ibi[lo:hi]
        if len(seg_at) < 20 or seg_at[-1] - seg_at[0] < 30:
            continue
        grid = np.arange(seg_at[0], seg_at[-1], 1.0 / RESP_GRID_HZ)
        interp = np.interp(grid, seg_at, seg_ibi)
        filt = sosfiltfilt(sos, interp)

        if mode == "rate":
            # peaks per 30 s, carried along the grid
            from scipy.signal import find_peaks
            win = int(30 * RESP_GRID_HZ)
            rate = np.full(len(filt), np.nan)
            for s in range(0, len(filt) - win, int(RESP_GRID_HZ)):
                pk, _ = find_peaks(filt[s:s + win],
                                   distance=RESP_GRID_HZ * 1.5)
                rate[s + win // 2] = len(pk) * 2.0
            good = np.isfinite(rate)
            if good.sum() < 2:
                continue
            filt = np.interp(grid, grid[good], rate[good])

        ts.append(grid * 1e6)
        vs.append(filt)

    if not ts:
        return empty
    return pd.DataFrame({"t": np.concatenate(ts), "v": np.concatenate(vs),
                         "fs": RESP_GRID_HZ})


# ----------------------------------------------------------------------------- resample
def resample_to_grid(df, grid_us, gap_s=1.0):
    """
    Resample one channel onto grid_us (µs, 16 Hz).
    - split into contiguous segments where the gap > max(gap_s, 3 sample periods)
    - if native rate > 16 Hz: zero-phase low-pass (anti-alias, cutoff 0.4*16 Hz) first
    - linear interpolation onto the grid; grid points outside any segment stay NaN
    """
    out = np.full(len(grid_us), np.nan)
    if len(df) < 2:
        return out
    t = df["t"].to_numpy()
    v = df["v"].to_numpy()
    fs = float(df["fs"].median())
    max_gap = max(gap_s, 3.0 / fs) * 1e6
    breaks = np.where(np.diff(t) > max_gap)[0] + 1
    for tt, vv in zip(np.split(t, breaks), np.split(v, breaks)):
        if len(tt) < 2:
            continue
        ok = np.isfinite(vv)
        if ok.sum() < 2:
            continue
        if fs > FS and len(vv) > 30 and ok.all():
            sos = butter(4, 0.4 * FS, btype="low", fs=fs, output="sos")
            vv = sosfiltfilt(sos, vv)
        lo, hi = tt[0], tt[-1]
        i0, i1 = np.searchsorted(grid_us, lo, "left"), np.searchsorted(grid_us, hi, "right")
        if i1 <= i0:
            continue
        out[i0:i1] = np.interp(grid_us[i0:i1], tt[ok], vv[ok])
    return out


# ----------------------------------------------------------------------------- per-minute csv
def _norm(name):
    return re.sub(r"[^a-z0-9]+", "_", os.path.basename(name).lower())


def _find_file(files, keywords):
    for kw in keywords:
        pat = re.compile(rf"(^|_){kw}(_|\.|$)")
        for f in files:
            if pat.search(_norm(f)):
                return f
    return None


def _to_utc(series):
    """Parse a time column that may be ISO strings or unix s/ms/µs/ns -> tz-naive UTC datetime64[ns]."""
    if pd.api.types.is_numeric_dtype(series):
        m = float(np.nanmedian(np.abs(series.astype(float))))
        unit = "ns" if m > 1e17 else "us" if m > 1e14 else "ms" if m > 1e11 else "s"
        return pd.to_datetime(series, unit=unit, utc=True).dt.tz_localize(None)
    return pd.to_datetime(series, utc=True, errors="coerce").dt.tz_localize(None)


TIME_COLS = ["timestamp_unix", "timestamp_iso", "timestamp", "datetime_utc", "datetime", "time", "date_iso"]
META_COLS = {"participant_full_id", "missing_value_reason", "device_serial", "device_id", "participant_id",
             "date_iso", "timestamp_iso", "timestamp_unix", "timestamp", "datetime_utc", "datetime", "time",
             "timezone", "timezone_offset", "unnamed_0"}


def read_epoch_csvs(folder, mapping):
    """Return {name: DataFrame(time, value, res_s)} using flexible column detection."""
    files = sorted(glob.glob(os.path.join(folder, "**", "*.csv"), recursive=True))
    res = {}
    for name, kws in mapping.items():
        by_file = [f for f in files if _find_file([f], kws)]
        if not by_file:
            print(f"[warn] no per-minute file found for {name} (keywords {kws})")
            continue
        frames = []
        for f in by_file:
            d = pd.read_csv(f)
            d.columns = [re.sub(r"[^a-z0-9]+", "_", c.strip().lower()) for c in d.columns]
            tcol = next((c for c in TIME_COLS if c in d.columns), None)
            vcols = [c for c in d.columns if c not in META_COLS]
            if tcol is None or not vcols:
                print(f"[warn] {os.path.basename(f)}: cannot find time/value column in {list(d.columns)}")
                continue
            vcol = vcols[-1]
            if "missing_value_reason" in d.columns:
                d.loc[d["missing_value_reason"].notna() & (d["missing_value_reason"].astype(str) != ""), vcol] = np.nan
            frames.append(pd.DataFrame({"time": _to_utc(d[tcol]), "value": d[vcol]}))
            print(f"[info] {name:18s} <- {os.path.basename(f)}  (time={tcol}, value={vcol})")
        if not frames:
            continue
        d = pd.concat(frames).dropna(subset=["time"]).sort_values("time").drop_duplicates("time")
        step = d["time"].diff().dt.total_seconds().median()
        d.attrs["res_s"] = float(step) if np.isfinite(step) and step > 0 else 60.0
        res[name] = d.reset_index(drop=True)
    return res


def lookup(series_df, starts, default_res):
    """Value whose timestamp is the latest <= window start and within one resolution step."""
    left = pd.DataFrame({"time": starts})
    res_s = series_df.attrs.get("res_s", default_res)
    m = pd.merge_asof(left, series_df.rename(columns={"time": "time"}), on="time",
                      direction="backward", tolerance=pd.Timedelta(seconds=res_s - 1e-3))
    return m["value"].to_numpy()


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="folder with raw_data .avro files")
    ap.add_argument("--minute", default=None, help="folder with aggregated_per_minute CSVs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", default=None, help="UTC start 'YYYY-MM-DD HH:MM:SS' (optional)")
    ap.add_argument("--end", default=None, help="UTC end (optional)")
    ap.add_argument("--max-nan", type=float, default=0.2, help="drop window if any channel has > this NaN fraction")
    ap.add_argument("--resp-mode", choices=["wave", "rate"], default="wave",
                    help="what the respiration channel holds. wave is the "
                         "band-passed beat-interval series, which oscillates "
                         "at the breathing frequency and so has shape for "
                         "Chronos to read. rate is breaths per minute, which "
                         "is flatter and gives a tokeniser almost nothing.")
    ap.add_argument("--minute-first-only", action="store_true",
                    help="write minute-level labels only for the window starting at :00")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    s_ts = pd.Timestamp(a.start) if a.start else None
    e_ts = pd.Timestamp(a.end) if a.end else None
    data = read_avro_folder(a.raw, s_ts, e_ts, a.resp_mode)
    tmin = min(d["t"].min() for d in data.values() if len(d))
    tmax = max(d["t"].max() for d in data.values() if len(d))
    if a.start:
        tmin = max(tmin, pd.Timestamp(a.start).value / 1e3)
    if a.end:
        tmax = min(tmax, pd.Timestamp(a.end).value / 1e3)
    win_us = WIN_S * 1e6
    w0 = np.ceil(tmin / win_us) * win_us          # aligned to :00 / :30
    n_win = int(np.floor((tmax - w0) / win_us))
    if n_win <= 0:
        raise SystemExit("No complete 30 s window in the selected range")
    grid = w0 + np.arange(n_win * WIN_N) * (1e6 / FS)
    print(f"[info] {n_win} windows from {pd.to_datetime(w0, unit='us')} UTC")

    X = np.stack([resample_to_grid(data[c], grid) for c in CHANNELS], axis=1)  # (n_win*480, 6)
    X = X.reshape(n_win, WIN_N, len(CHANNELS)).astype(np.float32)
    starts = pd.to_datetime(w0 + np.arange(n_win) * win_us, unit="us")

    nan_frac = np.isnan(X).mean(axis=1)                 # (n_win, 6)
    keep = (nan_frac <= a.max_nan).all(axis=1)

    # epoch (30 s) and minute (60 s) context
    epoch = np.full((n_win, len(EPOCH_MAP)), np.nan)
    minute = np.full((n_win, len(MINUTE_MAP)), np.nan, dtype=object)
    if a.minute:
        ep = read_epoch_csvs(a.minute, EPOCH_MAP)
        for j, name in enumerate(EPOCH_MAP):
            if name in ep:
                epoch[:, j] = pd.to_numeric(pd.Series(lookup(ep[name], starts, 60.0)), errors="coerce")
        mn = read_epoch_csvs(a.minute, MINUTE_MAP)
        minute_start = (starts.second == 0)
        for j, name in enumerate(MINUTE_MAP):
            if name in mn:
                minute[:, j] = lookup(mn[name], starts, 60.0)
        if a.minute_first_only:
            minute[~np.asarray(minute_start)] = np.nan

    # ---- save
    idx = np.where(keep)[0]
    print(f"[info] keeping {len(idx)}/{n_win} windows (max NaN fraction {a.max_nan})")
    np.savez_compressed(
        os.path.join(a.out, "windows_16hz.npz"),
        X=X[idx], window_start_utc=starts[idx].astype("datetime64[ns]").astype("int64"),
        epoch=epoch[idx], minute=minute[idx].astype(str),
        channels=np.array(CHANNELS), epoch_names=np.array(list(EPOCH_MAP)), minute_names=np.array(list(MINUTE_MAP)))

    summ = pd.DataFrame(nan_frac, columns=[f"nan_{c}" for c in CHANNELS])
    summ.insert(0, "window_start_utc", starts)
    summ["kept"] = keep
    summ.to_csv(os.path.join(a.out, "windows_summary.csv"), index=False)

    # long CSV: first row of each window carries the context
    csv_path = os.path.join(a.out, "windows_16hz.csv")
    cols = ["window_start_utc", "sample_idx", "time_utc"] + CHANNELS + list(EPOCH_MAP) + list(MINUTE_MAP)
    with open(csv_path, "w", newline="") as fh:
        first = True
        for w in idx:
            df = pd.DataFrame(X[w], columns=CHANNELS)
            df.insert(0, "time_utc", starts[w] + pd.to_timedelta(np.arange(WIN_N) / FS, unit="s"))
            df.insert(0, "sample_idx", np.arange(WIN_N))
            df.insert(0, "window_start_utc", starts[w])
            for j, name in enumerate(EPOCH_MAP):
                df[name] = np.nan
                df.loc[0, name] = epoch[w, j]
            for j, name in enumerate(MINUTE_MAP):
                df[name] = None
                df.loc[0, name] = None if pd.isna(minute[w, j]) else minute[w, j]
            df[cols].to_csv(fh, header=first, index=False)
            first = False
    print(f"[done] {csv_path}\n       {os.path.join(a.out, 'windows_16hz.npz')}")


if __name__ == "__main__":
    main()
