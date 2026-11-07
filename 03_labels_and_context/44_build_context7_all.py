"""
44_build_context7_all.py

WHY THIS EXISTS
    btp/42_build_context8.py builds context for the 5,119 windows that have
    a survey answer. To score every window a person wore the device for,
    about 104,000 of them, the same context has to exist everywhere.

    This builds it from the two sources directly, rather than from the
    labelled file.

THE SEVEN COLUMNS
        HOUR            from the timestamp
        IS_WEEKEND      from the timestamp
        POSTURE         our derivation from the wrist accelerometer
        ACT_COUNT       our derivation from the wrist accelerometer
        ACT_CLASS       Empatica biomarker, one value a minute
        BODY_POSITION   Empatica biomarker, one value a minute
        SLEEP           Empatica biomarker, one value a minute

WHY NOT EIGHT
    INDOOR_OUTDOOR is left out, and it has to be.

    It is a survey answer, not a sensor reading. It exists at the 88 moments
    somebody filled in the questionnaire and nowhere else. Across 104,000
    windows that is under one percent, so the column would read "Unknown"
    almost everywhere.

    The model was trained on rows where it mostly had a real answer. Feeding
    it a column that is now almost entirely Unknown is a quiet change in what
    the input looks like, and the model has no way to tell us it is
    confused. Better to drop it and train on seven.

THE BIOMARKERS ARRIVE ONCE A MINUTE
    Windows move every 30 seconds, so each minute covers two of them. The
    value is held across both, not interpolated: these are words, and there
    is no halfway point between "walking" and "still".

    Where Empatica recorded nothing, because the watch was off or loose, the
    cell becomes "Unknown". That is honest - the state really is unknown -
    but it is also a fact about the wearer rather than about their stress,
    and this script reports how often it happens so the figure is visible
    rather than buried.

OUTPUT
    data/processed/context/all_windows_ctx7.csv
        Subject_ID, datetime_ist, and the seven columns, one row per
        30 seconds for every user

HOW TO RUN
    python3 btp/44_build_context7_all.py
"""

import os
import argparse

import numpy as np
import pandas as pd

CONTEXT_DIR = "data/processed/context"
BIOMARKER_DIR = "data/processed/biomarkers"
OUT = "data/processed/context/all_windows_ctx7.csv"

TIME_COL = "datetime_ist"
IST = pd.Timedelta(hours=5, minutes=30)

FROM_CONTEXT = ["HOUR", "IS_WEEKEND", "POSTURE", "ACT_COUNT"]
FROM_BIOMARKER = {
    "activity_class": "ACT_CLASS",
    "body_position_left": "BODY_POSITION",
    "sleep_detection_stage": "SLEEP",
}
CONTEXT7 = FROM_CONTEXT + list(FROM_BIOMARKER.values())

# A signal must never be context. Anything here reaching the output is a bug.
SIGNALS = {"PR", "RESP", "MET", "EDA_TONIC", "EDA_PHASIC_N", "EDA_PHASIC_AMP",
           "SKIN_TEMP", "PRV_RMSSD", "PRV_SDNN", "AROUSAL_AT_REST", "MOVE_SD",
           "TILT", "ACT_INT", "BVP", "EDA_wrist", "Temp_wrist"}

USERS = ["user17", "user18", "user25", "user26", "user27"]


def build_user(user, context_dir, biomarker_dir):
    cpath = os.path.join(context_dir, "%s_context.csv" % user)
    if not os.path.exists(cpath):
        print("    no context file")
        return None, None

    ctx = pd.read_csv(cpath)
    ctx[TIME_COL] = pd.to_datetime(ctx[TIME_COL], errors="coerce")
    ctx = ctx.dropna(subset=[TIME_COL]).sort_values(TIME_COL)
    ctx = ctx.reset_index(drop=True)

    missing = [c for c in FROM_CONTEXT if c not in ctx.columns]
    if missing:
        print("    context file lacks %s" % missing)
        return None, None

    out = ctx[[TIME_COL] + FROM_CONTEXT].copy()
    out.insert(0, "Subject_ID", user)

    # the biomarkers, held across the two windows each minute covers
    bpath = os.path.join(biomarker_dir, "%s_biomarkers.csv" % user)
    if os.path.exists(bpath):
        bio = pd.read_csv(bpath, usecols=lambda c: c in
                          ({"timestamp_iso"} | set(FROM_BIOMARKER)))
        bio["t"] = pd.to_datetime(bio["timestamp_iso"], errors="coerce",
                                  utc=True).dt.tz_localize(None)
        bio = bio.dropna(subset=["t"]).sort_values("t")
        bio = bio.rename(columns=FROM_BIOMARKER)

        have = [c for c in FROM_BIOMARKER.values() if c in bio.columns]
        # the biomarker clock is UTC, ours is IST
        out["_utc"] = out[TIME_COL] - IST
        out = pd.merge_asof(out.sort_values("_utc"), bio[["t"] + have],
                            left_on="_utc", right_on="t",
                            direction="backward",
                            tolerance=pd.Timedelta(seconds=60))
        out = out.drop(columns=["_utc", "t"])
        matched = int(out[have].notna().any(axis=1).sum())
    else:
        print("    no biomarker file, those three will be Unknown")
        have, matched = [], 0

    for c in FROM_BIOMARKER.values():
        if c not in out.columns:
            out[c] = np.nan

    stats = {"user": user, "windows": int(len(out)),
             "biomarker_matched": matched,
             "biomarker_pct": round(100.0 * matched / max(len(out), 1), 1)}

    # Unknown rather than blank: the model was trained with that word, and a
    # blank would be filled differently by whatever reads this next
    for c in FROM_BIOMARKER.values():
        out[c] = out[c].astype(object).where(out[c].notna(), "Unknown")

    return out[["Subject_ID", TIME_COL] + CONTEXT7], stats


def main():
    ap = argparse.ArgumentParser(
        description="Build the seven context columns for every window, so "
                    "the whole recording can be scored and not only the "
                    "parts with a survey answer.")
    ap.add_argument("--context-dir", default=CONTEXT_DIR)
    ap.add_argument("--biomarker-dir", default=BIOMARKER_DIR)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--users", default=",".join(USERS))
    args = ap.parse_args()

    users = [u.strip() for u in args.users.split(",") if u.strip()]

    print("=" * 70)
    print("Context for every window")
    print("=" * 70)
    print("seven columns: %s" % ", ".join(CONTEXT7))
    print()
    print("INDOOR_OUTDOOR is not here. It is a survey answer, so it exists")
    print("at 88 moments out of about 104,000 windows. A column that is")
    print("Unknown ninety-nine times in a hundred is not context, it is")
    print("noise with a name.")
    print("=" * 70)
    print()

    frames, summary = [], []
    for u in users:
        print("%s" % u)
        df, st = build_user(u, args.context_dir, args.biomarker_dir)
        if df is None:
            print()
            continue
        frames.append(df)
        summary.append(st)
        print("    %6d windows, %d with a biomarker value (%.0f%%)"
              % (st["windows"], st["biomarker_matched"], st["biomarker_pct"]))
        print()

    if not frames:
        raise SystemExit("Nothing produced.")

    allf = pd.concat(frames, ignore_index=True)

    leaked = SIGNALS & set(allf.columns)
    assert not leaked, "signal columns reached the output: %s" % sorted(leaked)
    seen = set(allf.columns) - {"Subject_ID", TIME_COL}
    assert seen == set(CONTEXT7), "context is %s" % sorted(seen)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    allf.to_csv(args.out, index=False)

    print("=" * 70)
    print("WHAT CAME OUT")
    print("=" * 70)
    print("  %-9s %10s %14s" % ("user", "windows", "biomarkers"))
    for s in summary:
        print("  %-9s %10d %13.0f%%"
              % (s["user"], s["windows"], s["biomarker_pct"]))
    print("  " + "-" * 36)
    print("  %-9s %10d" % ("TOTAL", len(allf)))
    print()

    print("what each column holds:")
    for c in CONTEXT7:
        v = pd.to_numeric(allf[c], errors="coerce")
        if v.notna().all():
            print("  %-16s %.1f to %.1f" % (c, v.min(), v.max()))
        else:
            vc = allf[c].astype(str).value_counts()
            top = ", ".join("%s %.0f%%" % (k, 100 * n / len(allf))
                            for k, n in vc.head(4).items())
            print("  %-16s %s" % (c, top))
    print()

    unknown = {c: 100.0 * (allf[c].astype(str) == "Unknown").mean()
               for c in FROM_BIOMARKER.values()}
    worst = max(unknown.values())
    print("  the biomarker columns are Unknown %.0f%% of the time, because"
          % worst)
    print("  the watch was off or not touching the skin. That is a fact")
    print("  about the wearer rather than about their stress, and it is")
    print("  worth saying beside any result that uses them.")
    print()
    print("saved %s  (%.1f MB)" % (args.out,
                                   os.path.getsize(args.out) / 1e6))


if __name__ == "__main__":
    main()
