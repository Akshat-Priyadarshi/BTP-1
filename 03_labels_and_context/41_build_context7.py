"""
41_build_context7.py

WHAT THIS DOES
    Rebuilds the labelled-window file so that its context is exactly the set
    the mentor asked for, and contains no signal.

        time of day      HOUR
        weekend          IS_WEEKEND
        posture          POSTURE
        tilt             TILT
        activity count   ACT_COUNT
        activity label   ACT_INT   (sedentary / light / moderate / vigorous)
        indoor/outdoor   INDOOR_OUTDOOR   (from the survey, see below)

    It writes a NEW file and leaves labelled_windows.csv untouched, so the
    twelve-column runs stay reproducible.

WHY IT EXISTS
    btp/34_train_users.py treats every column of the labelled file that is not
    in its NOT_CONTEXT set as context. The old file therefore fed PR, RESP,
    MET, EDA and SKIN_TEMP in as "context". With RESP now a signal channel in
    W7, that would use it twice, and it breaks the rule that no signal is
    context.

WHAT IS HONEST ABOUT EACH COLUMN
    ACT_INT      The mentor's notebook computes ACT_INT. The ACT_CLASS from a
                 Capture24 model that it mentions is NOT computed anywhere in
                 this project, so "activity label" here means ACT_INT.
    ACT_COUNT    agcounts was not installed locally, so this is the band-pass
                 approximation unless the notebook environment had it.
    POSTURE/TILT Derived by us in btp/30_derive_context.py, not by the mentor.
                 The tilt cut-offs there are provisional.
    INDOOR_OUTDOOR
                 Not measurable from the wristband. It is the survey's
                 "3. Current Environment" answer, so it exists only where a
                 response exists, and carries the same +-15 minute reach as
                 the label. It is answered in the same form as the STAI items,
                 so run btp/32_user_leak_check.py on it before trusting it.

HOW TO RUN
    python3 btp/41_build_context7.py
    python3 btp/34_train_users.py --config configs/w7.yaml --run-id V_x \
        --context on --labelled data/processed/labels/labelled_windows_ctx7.csv \
        --split random --seeds 5 --signal-dir data/processed/user_7ch
"""
import argparse
import os
import re
import sys

import pandas as pd

LABELLED = "data/processed/labels/labelled_windows.csv"
SURVEY = "mentor_files/Stress_and_STAI_Survey_Responses.xlsx"
OUT = "data/processed/labels/labelled_windows_ctx7.csv"

# Columns btp/34_train_users.py will NOT treat as context. Copied from its
# NOT_CONTEXT set so the two files agree.
META = ["Subject_ID", "datetime_utc", "datetime_ist", "date",
        "response_time", "STAI6", "LABEL", "LABEL_NAME"]
CONTEXT = ["HOUR", "IS_WEEKEND", "POSTURE", "TILT",
           "ACT_COUNT", "ACT_INT", "INDOOR_OUTDOOR"]

# Anything here appearing in the output is a bug, not a choice.
SIGNALS = {"PR", "RESP", "MET", "EDA_TONIC", "EDA_PHASIC_N", "EDA_PHASIC_AMP",
           "SKIN_TEMP", "PRV_RMSSD", "PRV_SDNN", "AROUSAL_AT_REST", "MOVE_SD",
           "BVP", "EDA_wrist", "Temp_wrist", "ACC_X_wrist", "ACC_Y_wrist",
           "ACC_Z_wrist", "STAI6_RAW"}


def env_name(x):
    s = str(x).strip().lower()
    if s.startswith("indoor"):
        return "Indoor"
    if s.startswith("outdoor"):
        return "Outdoor"
    if s.startswith("travel"):
        return "Travelling"
    return "Unknown"


def user_name(x):
    m = re.findall(r"\d+", str(x))
    return "user%d" % int(m[0]) if m else None


def best_match(resp, survey):
    """Match each response to a survey row, trying a few clock offsets.

    The survey clock and response_time may differ by the 5.5 h IST offset, so
    pick whichever offset matches the most responses and say which it was.
    """
    best = None
    for hours in (0.0, -5.5, 5.5):
        s = survey.copy()
        s["_t"] = s["Timestamp"] + pd.Timedelta(hours=hours)
        s = s.sort_values("_t")
        m = pd.merge_asof(resp.sort_values("response_time"),
                          s[["User_", "_t", "ENV"]],
                          left_on="response_time", right_on="_t",
                          left_by="Subject_ID", right_by="User_",
                          tolerance=pd.Timedelta("2min"),
                          direction="nearest")
        n = int(m["ENV"].notna().sum())
        if best is None or n > best[0]:
            best = (n, hours, m)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labelled", default=LABELLED)
    ap.add_argument("--survey", default=SURVEY)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    d = pd.read_csv(args.labelled)
    print("read %s: %d rows, %d columns" % (args.labelled, len(d), d.shape[1]))
    d["response_time"] = pd.to_datetime(d["response_time"])

    sv = pd.read_excel(args.survey)
    env_col = [c for c in sv.columns if "Current Environment" in str(c)]
    if not env_col or "User" not in sv.columns or "Timestamp" not in sv.columns:
        sys.exit("survey is missing Timestamp, User or the Current "
                 "Environment question")
    sv = sv.rename(columns={env_col[0]: "ENV_RAW"})
    sv["ENV"] = sv["ENV_RAW"].map(env_name)
    sv["User_"] = sv["User"].map(user_name)
    sv["Timestamp"] = pd.to_datetime(sv["Timestamp"])
    print("survey users as written :", sorted(sv["User"].astype(str).unique())[:8])

    resp = d[["Subject_ID", "response_time"]].drop_duplicates()
    n, hours, m = best_match(resp, sv)
    print("responses to match      : %d" % len(resp))
    print("matched                 : %d  (survey clock offset %+.1f h)"
          % (n, hours))
    if n < 0.95 * len(resp):
        print("\nfirst unmatched responses:")
        print(m[m["ENV"].isna()].head(8).to_string(index=False))
        sys.exit("too few matched; fix the clock/user mapping before going on")

    d = d.merge(m[["Subject_ID", "response_time", "ENV"]],
                on=["Subject_ID", "response_time"], how="left")
    d = d.rename(columns={"ENV": "INDOOR_OUTDOOR"})
    d["INDOOR_OUTDOOR"] = d["INDOOR_OUTDOOR"].fillna("Unknown")

    missing = [c for c in META + CONTEXT if c not in d.columns]
    if missing:
        sys.exit("input lacks columns: %s" % missing)
    out = d[META + CONTEXT].copy()

    # ---- guards: this is the whole point of the script ----
    leaked = SIGNALS & set(out.columns)
    assert not leaked, "signal columns in output: %s" % sorted(leaked)
    ctx_seen = set(out.columns) - set(META)
    assert ctx_seen == set(CONTEXT), "context is %s" % sorted(ctx_seen)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_csv(args.out, index=False)

    print("\nwrote %s: %d rows" % (args.out, len(out)))
    print("context columns (%d): %s" % (len(CONTEXT), ", ".join(CONTEXT)))
    print("dropped from the old file: %s"
          % ", ".join(c for c in d.columns
                      if c not in META + CONTEXT and c in SIGNALS | {"MET"}))
    print("\nINDOOR_OUTDOOR by user (windows):")
    print(pd.crosstab(out["Subject_ID"], out["INDOOR_OUTDOOR"]).to_string())
    print("\nINDOOR_OUTDOOR by label (windows):")
    print(pd.crosstab(out["INDOOR_OUTDOOR"], out["LABEL_NAME"]).to_string())


if __name__ == "__main__":
    main()
