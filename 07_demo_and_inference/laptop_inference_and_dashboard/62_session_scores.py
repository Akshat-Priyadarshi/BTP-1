"""
62_session_scores.py

WHAT THIS DOES
    One command, from the mentor's session table to the finished result.

        1  reads the session table (sessions.csv) and shows, for every
           session, its slot, its length and the interval rule that applies
        2  converts the downloaded avro + per-minute files to the model's
           input             (btp/60_build_demo_sessions.py)
        3  scores every window with the saved model
                             (btp/45_score_all_with_context.py)
        4  reports the stress score at fixed intervals, cross-checks it, and
           saves the CSVs and the figure
                             (btp/61_demo_intervals.py)

    It calls those three scripts and adds nothing of its own to the numbers,
    so the result is the same as running them one by one.

THE INTERVAL RULE (the mentor's)
        session longer than 130 min   a score every 15 minutes
        session of 100 min or less    a score every  5 minutes
        101 to 130 min                every 10 minutes (our choice; she did
                                      not say)
    By default each checkpoint shows the score AT that moment (the 60 s
    window starting there), not an average. --mode mean averages the windows
    inside each interval instead.

WHICH MODEL
    Default is runs/FINAL_ctx7_random/model.pth: the seven context columns,
    without INDOOR_OUTDOOR. INDOOR_OUTDOOR is a survey answer and these
    recordings have none.

WHAT THE NUMBERS ARE AND ARE NOT
    These sessions have no survey answers. The stress score (1 x P(Medium) +
    2 x P(High), 0 to 2) is the model's output, not a measurement, and on a
    person the model never saw it is no better than guessing. The checks in
    step 4 show the numbers are consistent with the source files, not that
    they are right.

OUTPUT (runs/<run-id>/)
    predictions.csv         every 60 s window
    interval_scores.csv     the score per checkpoint, with posture and activity
    interval_summary.csv    one row per session
    interval_checks.csv     every cross-check and its result
    interval_scores.png     the figure

HOW TO RUN (project root, test-network env)
    python3 btp/62_session_scores.py

    python3 btp/62_session_scores.py --skip-build     files already converted
    python3 btp/62_session_scores.py --skip-score     scores already there
    python3 btp/62_session_scores.py --only user14    one person
"""
import argparse
import os
import subprocess
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
IST = pd.Timedelta(hours=5, minutes=30)


def rule(minutes, short, mid, long_):
    if minutes > 130:
        return long_
    if minutes <= 100:
        return short
    return mid


def run(title, cmd):
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)
    print("$ " + " ".join(cmd) + "\n", flush=True)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        sys.exit("\nstopped: step failed (exit %d). Fix that and rerun; the "
                 "earlier steps' files are kept." % r.returncode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions",
                    default="chronos-part/data_chronos_embrace/sessions.csv")
    ap.add_argument("--model", default="runs/FINAL_ctx7_random/model.pth")
    ap.add_argument("--run-id", default="DEMO_SESSIONS_ctx7")
    ap.add_argument("--data-dir", default="data/demo_sessions")
    ap.add_argument("--only", default=None)
    ap.add_argument("--max-hours", type=float, default=24.0)
    ap.add_argument("--short", type=int, default=5)
    ap.add_argument("--mid", type=int, default=10)
    ap.add_argument("--long", type=int, default=15)
    ap.add_argument("--mode", choices=["point", "mean"], default="point",
                    help="point: the score AT each checkpoint (default). "
                         "mean: the mean of the windows in each interval.")
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--skip-score", action="store_true")
    a = ap.parse_args()
    py = sys.executable

    # ---- 1. the session table, with the rule that applies to each row ----
    s = pd.read_csv(a.sessions, parse_dates=["start_utc", "end_utc"])
    s["minutes"] = (s["end_utc"] - s["start_utc"]).dt.total_seconds() / 60
    s = s[s["minutes"] <= a.max_hours * 60]
    if a.only:
        s = s[s["user"] == a.only]
    if s.empty:
        sys.exit("no sessions in %s" % a.sessions)
    t = pd.DataFrame({
        "session": [u + "_" + d.strftime("%m%d")
                    for u, d in zip(s["user"], s["start_utc"] + IST)],
        "account": s["account"],
        "slot_ist": [(x + IST).strftime("%d %b %H:%M") + "-"
                     + (y + IST).strftime("%H:%M")
                     for x, y in zip(s["start_utc"], s["end_utc"])],
        "minutes": s["minutes"].round(0).astype(int),
        "a_score_every_min": [rule(m, a.short, a.mid, a.long)
                              for m in s["minutes"]]})
    print("SESSION TABLE")
    print(t.to_string(index=False))
    print("\nrule: > 130 min -> every %d min;  <= 100 min -> every %d min;  "
          "101-130 -> every %d min" % (a.long, a.short, a.mid))

    sig = os.path.join(a.data_dir, "user_7ch")
    ctx = os.path.join(a.data_dir, "context_ctx8.csv")
    pred = os.path.join("runs", a.run_id, "predictions.csv")
    extra = ["--only", a.only] if a.only else []

    # ---- 2. convert ----
    if not a.skip_build:
        run("STEP 1 of 3   convert the downloaded files to model input",
            [py, os.path.join(HERE, "60_build_demo_sessions.py"),
             "--sessions", a.sessions, "--out", a.data_dir,
             "--max-hours", str(a.max_hours)] + extra)
    # ---- 3. score ----
    if not a.skip_score:
        run("STEP 2 of 3   score every window with the saved model",
            [py, os.path.join(HERE, "45_score_all_with_context.py"),
             "--model", a.model, "--run-id", a.run_id,
             "--signal-dir", sig, "--context", ctx, "--labelled", "none"])
    if not os.path.exists(pred):
        sys.exit("no predictions at %s" % pred)

    # ---- 4. intervals + checks + files ----
    run("STEP 3 of 3   score at fixed intervals, cross-checks, save",
        [py, os.path.join(HERE, "61_demo_intervals.py"),
         "--pred", pred, "--out", os.path.join("runs", a.run_id),
         "--signals", sig, "--sessions", a.sessions,
         "--mode", a.mode,
         "--short", str(a.short), "--mid", str(a.mid), "--long", str(a.long)])

    out = os.path.join("runs", a.run_id)
    print("\nSAVED in %s/" % out)
    for f in ("predictions.csv", "interval_scores.csv", "interval_summary.csv",
              "interval_checks.csv", "interval_scores.png"):
        print("  %-24s %s" % (f, "ok" if os.path.exists(os.path.join(out, f))
                              else "MISSING"))
    print("\nReminder: no survey answers exist for these sessions, so these "
          "are the model's\noutput, not a measured stress level.")


if __name__ == "__main__":
    main()
