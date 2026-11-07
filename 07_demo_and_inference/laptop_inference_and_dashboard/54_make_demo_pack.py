"""
54_make_demo_pack.py

WHAT THIS DOES
    Puts everything the Jetson needs to score the five demo sessions into one
    folder, so the folder runs on its own, with no project around it.

        demo_sessions_pack/
            sessions.csv              one row per session: slot, minutes, step
            user1_0322/signals.csv    7 signals at 1 Hz
            user1_0322/context.csv    7 context columns, every 30 s
            ... (one folder per session)
            model.pth                 the classifier and the gate
            encoder.pth               the frozen encoder
            scaler.pkl                the scaling it was trained with
            reference_scores.csv      the laptop's answers, to compare with
            README.md

    It changes no existing file. It only reads the project and writes the
    new folder.

WHICH MODEL
    runs/FINAL_ctx7_random/model.pth: the seven context columns, the same
    model that made the dashboard numbers. INDOOR_OUTDOOR is not used.

THE PATH PROBLEM (the same one as before)
    The checkpoint stores the encoder and scaler as paths relative to the
    project ("runs/W7/encoder.pth"). On another machine those mean nothing.
    So both files are copied in and the paths are rewritten to bare
    filenames.

CHECKPOINT RULE (the mentor's, same as btp/61_demo_intervals.py)
    session longer than 130 min  -> every 15 min
    100 min or less              -> every  5 min
    101 to 130 min               -> every 10 min
    The step is written into sessions.csv so the Jetson does not decide it.

HOW TO RUN (project root, the env that has torch)
    python3 btp/54_make_demo_pack.py
    python3 btp/54_make_demo_pack.py --out demo_testing/demo_sessions_pack
"""
import argparse
import os
import shutil

import pandas as pd

SIGNALS = ["ACC_X_wrist", "ACC_Y_wrist", "ACC_Z_wrist", "BVP", "EDA_wrist",
           "Temp_wrist", "RESP"]
TIME_COL = "datetime_ist"


def step_for(minutes):
    if minutes > 130:
        return 15
    if minutes <= 100:
        return 5
    return 10


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    ap.add_argument("--model", default="runs/FINAL_ctx7_random/model.pth")
    ap.add_argument("--data-dir", default="data/demo_sessions")
    ap.add_argument("--reference",
                    default="runs/DEMO_SESSIONS_ctx7/interval_scores.csv")
    ap.add_argument("--out", default="demo_testing/demo_sessions_pack")
    a = ap.parse_args()

    import torch  # here, so --help works without torch

    if os.path.exists(a.out):
        raise SystemExit("%s already exists. Remove it or pick another --out "
                         "so nothing is overwritten by accident." % a.out)
    os.makedirs(a.out)

    # ---- model, with its two companions and the paths made portable
    b = torch.load(a.model, map_location="cpu", weights_only=False)
    ctx_cols = list(b["context_columns"])
    if "INDOOR_OUTDOOR" in ctx_cols:
        raise SystemExit("this model uses INDOOR_OUTDOOR, which the demo "
                         "recordings do not have. Use the 7-column model.")
    for key in ("encoder_ckpt", "scaler_ckpt"):
        src = b[key]
        if not os.path.exists(src):
            raise SystemExit("%s says %s, which does not exist from here. "
                             "Run from the project root." % (key, src))
        shutil.copy(src, os.path.join(a.out, os.path.basename(src)))
        b[key] = os.path.basename(src)
    torch.save(b, os.path.join(a.out, "model.pth"))
    print("model     : %s" % a.model)
    print("context   : %s" % ", ".join(ctx_cols))
    print("window    : %s s, stride %s s" % (b["window"], b.get("stride")))

    # ---- sessions
    ctx_all = pd.read_csv(os.path.join(a.data_dir, "context_ctx8.csv"))
    ctx_all[TIME_COL] = pd.to_datetime(ctx_all[TIME_COL])
    rows = []
    sig_dir = os.path.join(a.data_dir, "user_7ch")
    for fn in sorted(os.listdir(sig_dir)):
        if not fn.endswith(".csv"):
            continue
        sid = fn[:-4]
        sig = pd.read_csv(os.path.join(sig_dir, fn))
        sig[TIME_COL] = pd.to_datetime(sig[TIME_COL])
        sig = sig.sort_values(TIME_COL)
        miss = [c for c in SIGNALS if c not in sig.columns]
        if miss:
            raise SystemExit("%s is missing channels %s" % (fn, miss))
        t0, t1 = sig[TIME_COL].iloc[0], sig[TIME_COL].iloc[-1]
        minutes = (t1 - t0).total_seconds() / 60.0 + 1.0 / 60.0

        c = ctx_all[ctx_all["Subject_ID"] == sid].sort_values(TIME_COL)
        c = c[[TIME_COL] + ctx_cols]

        d = os.path.join(a.out, sid)
        os.makedirs(d)
        sig[[TIME_COL] + SIGNALS].to_csv(os.path.join(d, "signals.csv"),
                                         index=False)
        c.to_csv(os.path.join(d, "context.csv"), index=False)
        rows.append({"session": sid, "start_ist": str(t0), "end_ist": str(t1),
                     "seconds": len(sig), "minutes": round(minutes, 1),
                     "step_min": step_for(minutes),
                     "context_rows": len(c)})
        print("  %-12s %5d s  %6.1f min  every %d min  context rows %d"
              % (sid, len(sig), minutes, step_for(minutes), len(c)))
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "sessions.csv"),
                              index=False)

    if os.path.exists(a.reference):
        shutil.copy(a.reference, os.path.join(a.out, "reference_scores.csv"))
        print("reference : %s" % a.reference)
    else:
        print("no reference scores at %s; the comparison will be skipped"
              % a.reference)

    with open(os.path.join(a.out, "README.md"), "w") as f:
        f.write("# Demo sessions pack\n\n"
                "Five recordings, no survey answers, so no accuracy.\n"
                "Run: python3 55_session_checkpoints.py --pack <this folder>\n"
                "The score is 1 x P(Medium) + 2 x P(High), range 0 to 2.\n")
    tot = sum(os.path.getsize(os.path.join(r, f))
              for r, _, fs in os.walk(a.out) for f in fs)
    print("\nwrote %s  (%.1f MB)" % (a.out, tot / 1e6))


if __name__ == "__main__":
    main()
