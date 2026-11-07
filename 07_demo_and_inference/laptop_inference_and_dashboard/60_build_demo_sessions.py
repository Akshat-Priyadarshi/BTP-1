"""
60_build_demo_sessions.py

WHAT THIS DOES
    Turns the short demo recordings downloaded for the mentor (raw avro files
    plus per-minute CSVs, in chronos-part/data_chronos_embrace/) into exactly
    the two things btp/45_score_all_with_context.py already reads:

        data/demo_sessions/user_7ch/<session>.csv      7 signals at 1 Hz
        data/demo_sessions/context_ctx8.csv            the 8 context columns,
                                                       one row per 30 seconds

    Nothing is scored here. Scoring is the next command (printed at the end).

WHY IT IS BUILT FROM THE EXISTING SCRIPTS AND NOT FROM SCRATCH
    The model was trained on columns made by btp/25 (1 Hz per-second mean) and
    btp/30_derive_context.py (RESP, ACT_COUNT, POSTURE). Re-implementing them
    would risk a quiet unit mismatch, so the avro is read into the same
    frames those scripts expect and THEIR functions are called:

        compute_resp, compute_act_count, compute_posture   from 30_derive_context
        per-second mean on floor("1s")                     as in 25_build_user_6ch

    Accelerometer is in g (imuParams conversionFactor), as in user_7ch.

THE SEVEN SIGNALS
    ACC_X/Y/Z_wrist, BVP, EDA_wrist, Temp_wrist   per-second mean of the avro
    RESP   breaths per minute from the heartbeat timing, one value per 30 s,
           held for 30 s. A staircase, like the other users. Not a belt.

THE EIGHT CONTEXT COLUMNS (and where each one comes from)
    HOUR, IS_WEEKEND           IST clock
    POSTURE, ACT_COUNT         from the accelerometer, via 30_derive_context
    ACT_CLASS                  per-minute activity-classification.csv
    BODY_POSITION              per-minute body-position.csv (body_position_left)
    SLEEP                      per-minute sleep-detection.csv
    INDOOR_OUTDOOR             "Unknown". It is a survey answer and these
                               recordings have no survey.
    No signal column enters the context; an assertion checks it.

TIMES
    sessions.csv is in UTC. Signals and context are written in IST (UTC+5:30),
    the same as every other file in the project. The per-minute CSVs are UTC
    and are joined in UTC before converting.

HOW TO RUN (from the AdaStress_BTP folder, in the test-network env)
    python3 btp/60_build_demo_sessions.py
    python3 btp/60_build_demo_sessions.py --only user14      # one session

    needs: fastavro   (pip install fastavro)
"""
import argparse
import glob
import importlib.util
import os
import sys

import numpy as np
import pandas as pd

try:
    from fastavro import reader as avro_reader
except ImportError:
    sys.exit("fastavro is missing:  pip install fastavro")

ROOT = "chronos-part/data_chronos_embrace"
OUT = "data/demo_sessions"
IST = pd.Timedelta(hours=5, minutes=30)
PAD = pd.Timedelta(seconds=90)          # read a little beyond the slot so
                                        # the 30 s windows at the edges are whole
CHANNELS = ["ACC_X_wrist", "ACC_Y_wrist", "ACC_Z_wrist",
            "BVP", "EDA_wrist", "Temp_wrist", "RESP"]
CONTEXT8 = ["HOUR", "IS_WEEKEND", "POSTURE", "ACT_COUNT",
            "ACT_CLASS", "BODY_POSITION", "SLEEP", "INDOOR_OUTDOOR"]
SIGNAL_NAMES = set(CHANNELS) | {"PR", "RESP", "MET", "SKIN_TEMP", "BVP",
                                "TILT", "MOVE_SD", "ACT_INT"}
# per-minute file keyword -> (column in file, name in the context)
PER_MINUTE = {
    "activity-classification": ("activity_class", "ACT_CLASS"),
    "body-position": ("body_position_left", "BODY_POSITION"),
    "sleep-detection": ("sleep_detection_stage", "SLEEP"),
}


def load_derive():
    """Import btp/30_derive_context.py (its name starts with a digit)."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "30_derive_context.py")
    spec = importlib.util.spec_from_file_location("derive_context", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def imu_scale(p):
    if not p:
        return 1.0
    if p.get("conversionFactor"):
        return p["conversionFactor"]
    try:
        return (p["physicalMax"] - p["physicalMin"]) / (
            p["digitalMax"] - p["digitalMin"])
    except Exception:
        return 1.0


def avro_files(user, start, end):
    """Avro files that can overlap [start, end]. Each file is ~30 minutes and
    is named <prefix>_<unix start>.avro, so skip the others without opening."""
    keep = []
    lo, hi = start.timestamp() - 2400, end.timestamp() + 60
    for f in sorted(glob.glob(os.path.join(ROOT, user, "*", "*", "raw_avro",
                                           "*.avro"))):
        try:
            u = int(os.path.basename(f).rsplit("_", 1)[1].split(".")[0])
        except ValueError:
            continue
        if lo <= u <= hi:
            keep.append(f)
    return keep


def read_avro(files, start, end):
    """Return the frames 30_derive_context expects, cut to [start-PAD, end+PAD]."""
    a, b = start - PAD, end + PAD
    lo_us, hi_us = a.value / 1e3, b.value / 1e3     # ns -> us
    acc, bvp, eda, tmp, peaks = [], [], [], [], []
    for f in files:
        with open(f, "rb") as fh:
            for rec in avro_reader(fh):
                raw = rec.get("rawData") or {}
                s = raw.get("accelerometer")
                if s and s.get("samplingFrequency") and len(s.get("x", [])):
                    n = len(s["x"])
                    t = s["timestampStart"] + np.arange(n) * (
                        1e6 / s["samplingFrequency"])
                    sc = imu_scale(s.get("imuParams"))
                    acc.append(pd.DataFrame({
                        "t": t,
                        "accelerometer_x": np.asarray(s["x"], float) * sc,
                        "accelerometer_y": np.asarray(s["y"], float) * sc,
                        "accelerometer_z": np.asarray(s["z"], float) * sc}))
                for key, store, col in (("bvp", bvp, "bvp"),
                                        ("eda", eda, "eda"),
                                        ("temperature", tmp, "temperature")):
                    s = raw.get(key)
                    if s and s.get("samplingFrequency") and len(
                            s.get("values", [])):
                        n = len(s["values"])
                        t = s["timestampStart"] + np.arange(n) * (
                            1e6 / s["samplingFrequency"])
                        store.append(pd.DataFrame(
                            {"t": t, col: np.asarray(s["values"], float)}))
                sp = raw.get("systolicPeaks")
                if sp:
                    for k in ("peaksTimeNanos", "peaksTimeNano",
                              "peakTimeNanos", "values"):
                        v = sp.get(k)
                        if v is not None and len(v):
                            peaks.append(np.asarray(v, float) / 1e3)
                            break

    def cut(parts):
        if not parts:
            return None
        d = pd.concat(parts, ignore_index=True).sort_values("t")
        d = d.drop_duplicates("t")
        d = d[(d["t"] >= lo_us) & (d["t"] <= hi_us)].reset_index(drop=True)
        if d.empty:
            return None
        d.insert(0, "datetime_utc", pd.to_datetime(d.pop("t"), unit="us"))
        return d

    syst = None
    if peaks:
        p = np.unique(np.concatenate(peaks))
        p = p[(p >= lo_us) & (p <= hi_us)]
        if len(p):
            syst = pd.DataFrame({"datetime_utc": pd.to_datetime(p, unit="us")})
    return cut(acc), cut(bvp), cut(eda), cut(tmp), syst


def per_second(df, cols, start, end):
    """Per-second mean on the UTC second, as btp/25 does on the IST second."""
    if df is None:
        return pd.DataFrame(index=pd.date_range(start, end, freq="1s",
                                                inclusive="left"),
                            columns=cols, dtype=float)
    sec = df["datetime_utc"].dt.floor("1s")
    return df[cols].groupby(sec).mean()


def thirty(df, name):
    """Snap a derive_context output onto the 30 s grid, as process_date does."""
    if df is None or not len(df):
        return None
    p = df.copy()
    p["datetime_utc"] = pd.to_datetime(p["datetime_utc"]).dt.floor("30s")
    return p.groupby("datetime_utc", as_index=False).first()


def per_minute(user, start, end):
    """The three per-minute files, joined into one frame indexed by UTC minute."""
    out = None
    for kw, (src, dst) in PER_MINUTE.items():
        frames = []
        for f in glob.glob(os.path.join(ROOT, user, "*", "*", "per_minute",
                                        "*_%s.csv" % kw)):
            d = pd.read_csv(f)
            if src in d.columns:
                frames.append(d[["timestamp_iso", src]])
        if not frames:
            continue
        d = pd.concat(frames, ignore_index=True)
        d["t"] = pd.to_datetime(d["timestamp_iso"], utc=True,
                                errors="coerce").dt.tz_localize(None)
        d = d.dropna(subset=["t"]).drop_duplicates("t").rename(
            columns={src: dst})[["t", dst]]
        out = d if out is None else out.merge(d, on="t", how="outer")
    return out.sort_values("t").reset_index(drop=True) if out is not None else None


def as_word(x):
    """Same spelling the training file uses: words stay words, numbers become
    '0.0' style strings, and a blank is the word Unknown."""
    if pd.isna(x):
        return "Unknown"
    if isinstance(x, (int, float, np.floating)):
        return str(float(x))
    return str(x)


def build_session(name, user, start, end, derive):
    print("\n%s   %s -> %s UTC" % (name, start, end))
    files = avro_files(user, start, end)
    print("    %d avro files overlap the slot" % len(files))
    if not files:
        return None
    acc, bvp, eda, tmp, syst = read_avro(files, start, end)
    print("    samples  acc %s  bvp %s  eda %s  temp %s  beats %s" % tuple(
        "-" if d is None else len(d) for d in (acc, bvp, eda, tmp, syst)))

    # ---------- 1 Hz signals ----------
    grid = pd.date_range(start, end, freq="1s", inclusive="left")
    sig = pd.DataFrame(index=grid)
    for df, cols, names in (
            (acc, ["accelerometer_x", "accelerometer_y", "accelerometer_z"],
             ["ACC_X_wrist", "ACC_Y_wrist", "ACC_Z_wrist"]),
            (bvp, ["bvp"], ["BVP"]),
            (eda, ["eda"], ["EDA_wrist"]),
            (tmp, ["temperature"], ["Temp_wrist"])):
        m = per_second(df, cols, start, end)
        m.columns = names
        sig = sig.join(m, how="left")

    # ---------- 30 s derived: RESP, ACT_COUNT, POSTURE ----------
    r = thirty(derive.compute_resp(syst), "RESP") if syst is not None else None
    ac = thirty(derive.compute_act_count(acc), "ACT_COUNT") if acc is not None else None
    po = thirty(derive.compute_posture(acc), "POSTURE") if acc is not None else None
    ctx = pd.DataFrame({"datetime_utc": pd.date_range(
        start.floor("30s"), end, freq="30s", inclusive="left")})
    for p in (r, ac, po):
        if p is not None:
            ctx = ctx.merge(p, on="datetime_utc", how="left")
    for c in ("RESP", "ACT_COUNT", "POSTURE"):
        if c not in ctx.columns:
            ctx[c] = np.nan

    # RESP is held for the 30 s it describes, like the other users
    resp = ctx.set_index("datetime_utc")["RESP"]
    sig["RESP"] = resp.reindex(grid, method="ffill", tolerance=pd.Timedelta(
        seconds=30)).values
    n_resp = int(sig["RESP"].notna().sum())
    print("    RESP present for %d of %d seconds (%.0f%%)"
          % (n_resp, len(sig), 100.0 * n_resp / max(len(sig), 1)))

    # ---------- per-minute context ----------
    pm = per_minute(user, start, end)
    if pm is not None:
        ctx = pd.merge_asof(ctx.sort_values("datetime_utc"), pm,
                            left_on="datetime_utc", right_on="t",
                            direction="backward",
                            tolerance=pd.Timedelta(seconds=60))
        ctx = ctx.drop(columns=["t"])
    for c in ("ACT_CLASS", "BODY_POSITION", "SLEEP"):
        if c not in ctx.columns:
            ctx[c] = np.nan
        ctx[c] = ctx[c].map(as_word)
    got = (ctx[["ACT_CLASS", "BODY_POSITION", "SLEEP"]] != "Unknown").any(axis=1)
    print("    per-minute context present for %.0f%% of 30 s slots"
          % (100.0 * got.mean()))

    # ---------- clock, in IST ----------
    ist = ctx["datetime_utc"] + IST
    ctx["datetime_ist"] = ist
    ctx["HOUR"] = ist.dt.hour
    ctx["IS_WEEKEND"] = (ist.dt.dayofweek >= 5).astype(int)
    ctx["INDOOR_OUTDOOR"] = "Unknown"
    ctx.insert(0, "Subject_ID", name)

    # ---------- write ----------
    sig = sig.copy()
    sig.insert(0, "datetime_ist", sig.index + IST)
    sig["date"] = sig["datetime_ist"].dt.strftime("%Y-%m-%d")
    sig["Subject_ID"] = name
    sig = sig[["datetime_ist", "date"] + CHANNELS + ["Subject_ID"]]
    return sig.reset_index(drop=True), ctx[
        ["Subject_ID", "datetime_ist"] + CONTEXT8]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", default=os.path.join(ROOT, "sessions.csv"))
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--only", default=None, help="a user id, e.g. user14")
    ap.add_argument("--max-hours", type=float, default=24.0,
                    help="rows longer than this are the earlier whole-user "
                         "downloads, not demo slots")
    args = ap.parse_args()

    derive = load_derive()
    s = pd.read_csv(args.sessions)
    s["start_utc"] = pd.to_datetime(s["start_utc"])
    s["end_utc"] = pd.to_datetime(s["end_utc"])
    s = s[(s["end_utc"] - s["start_utc"]) <= pd.Timedelta(hours=args.max_hours)]
    if args.only:
        s = s[s["user"] == args.only]
    if s.empty:
        sys.exit("no demo rows in %s" % args.sessions)

    os.makedirs(os.path.join(args.out, "user_7ch"), exist_ok=True)
    ctx_all, rows = [], []
    for _, r in s.iterrows():
        name = "%s_%s" % (r["user"], r["start_utc"].strftime("%m%d"))
        res = build_session(name, r["user"], r["start_utc"], r["end_utc"],
                            derive)
        if res is None:
            print("    nothing built")
            continue
        sig, ctx = res
        assert not (SIGNAL_NAMES & (set(ctx.columns) - {"Subject_ID",
                                                         "datetime_ist"})
                    - set(CONTEXT8)), "a signal column reached the context"
        sig.to_csv(os.path.join(args.out, "user_7ch", name + ".csv"),
                   index=False)
        ctx_all.append(ctx)
        rows.append({"session": name, "user": r["user"],
                     "account": r.get("account", ""),
                     "start_ist": r["start_utc"] + IST,
                     "end_ist": r["end_utc"] + IST,
                     "seconds": len(sig),
                     **{"%s_missing_pct" % c: round(100 * sig[c].isna().mean(), 1)
                        for c in CHANNELS}})

    if not ctx_all:
        sys.exit("Nothing built.")
    pd.concat(ctx_all).to_csv(os.path.join(args.out, "context_ctx8.csv"),
                              index=False)
    summ = pd.DataFrame(rows)
    summ.to_csv(os.path.join(args.out, "sessions_built.csv"), index=False)
    print("\n" + "=" * 70)
    print(summ.to_string(index=False))
    print("=" * 70)
    print("wrote %s/user_7ch/*.csv and %s/context_ctx8.csv" % (args.out, args.out))
    print("\nNEXT (scores them with the saved model):")
    print("  python3 btp/45_score_all_with_context.py \\")
    print("      --model runs/FINAL_ctx8_random/model.pth --run-id DEMO_SESSIONS \\")
    print("      --signal-dir %s/user_7ch --context %s/context_ctx8.csv \\"
          % (args.out, args.out))
    print("      --labelled none")


if __name__ == "__main__":
    main()
