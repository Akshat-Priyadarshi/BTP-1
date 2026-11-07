"""
55_session_checkpoints.py

WHAT THIS DOES
    Scores the five demo sessions on a device, at exact checkpoints.

    It scores EVERY 60-second window of every session (a window every 30 s,
    569 in all) and marks which of them are the checkpoints: the window that
    STARTS at 19:05, then 19:10, 19:15 ... (every 5 minutes for these
    sessions). Nothing is averaged and no interval is used: each row is one
    real window at one real moment. --checkpoints-only scores just the 59
    checkpoint windows.

    For each checkpoint it writes the stress score (1 x P(Medium) + 2 x
    P(High), 0 to 2), the three probabilities, the state, and what each
    context column was worth, measured by replacing that one column with
    its ordinary value for the session and scoring again.

    It also times every checkpoint (latency, memory), and finally compares
    its scores with the laptop's (reference_scores.csv in the pack).

WHAT IT DOES NOT DO
    These sessions have no survey answers. So there is no accuracy here.
    This shows that the model runs on the device, gives the same numbers as
    on the laptop, and what it costs.

HOW TO RUN (in the folder with this script, adastress/ and the pack)
    python3 55_session_checkpoints.py --pack demo_sessions_pack

    slowed down, so a live line appears every 2 s:
    python3 55_session_checkpoints.py --pack demo_sessions_pack --sleep 2

    one session:
    python3 55_session_checkpoints.py --pack demo_sessions_pack --only user1_0322

OUTPUT
    window_scores_device.csv       one row per window (all of them)
    checkpoint_scores_device.csv   the checkpoint windows only
    device_summary.txt             latency, memory, comparison with laptop
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import joblib

try:
    import psutil
    HAVE_PSUTIL = True
except ImportError:
    HAVE_PSUTIL = False


def _find_package():
    """The adastress package is in different places on different machines."""
    here = os.path.dirname(os.path.abspath(__file__))
    cands = [here,                                         # beside the script
             os.path.join(here, "vendor", "adastress"),    # project layout
             os.path.join(os.path.dirname(here), "vendor", "adastress"),
             os.path.join(os.getcwd()),
             os.path.join(os.getcwd(), "vendor", "adastress")]
    for c in cands:
        if os.path.isdir(os.path.join(c, "adastress")):
            sys.path.insert(0, c)
            return c
    raise SystemExit(
        "cannot find the adastress package. Put the adastress/ folder beside "
        "this script:\n    demo_testing/\n      55_session_checkpoints.py\n"
        "      adastress/models.py ...\n      demo_sessions_pack/\n"
        "or fetch it with git (see adastress_commit.txt in the project).")


_find_package()
from adastress.models import (ContextGateFusion, MLPClassifier,   # noqa: E402
                              StressModel, load_encoder)

STATES = ["Low", "Medium", "High"]
SIGNALS = ["ACC_X_wrist", "ACC_Y_wrist", "ACC_Z_wrist", "BVP", "EDA_wrist",
           "Temp_wrist", "RESP"]
TIME_COL = "datetime_ist"
DEVICE = torch.device("cpu")


def load_bundle(folder):
    path = os.path.join(folder, "model.pth")
    if not os.path.exists(path):
        raise SystemExit("no model.pth in %s" % folder)
    b = torch.load(path, map_location="cpu", weights_only=False)
    enc = os.path.join(folder, os.path.basename(b["encoder_ckpt"]))
    scl = os.path.join(folder, os.path.basename(b["scaler_ckpt"]))
    for p in (enc, scl):
        if not os.path.exists(p):
            raise SystemExit("%s is missing from the pack" % p)
    encoder = load_encoder(enc, b["input_channels"], b["lstm_hidden"], DEVICE)
    ctx_dim = int(b.get("context_dim", 0))
    gate = None
    if ctx_dim > 0 and b.get("gate"):
        gate = ContextGateFusion(encoder.output_dim, ctx_dim).to(DEVICE)
        gate.load_state_dict(b["gate"])
    clf = MLPClassifier(encoder.output_dim, len(b["classes"])).to(DEVICE)
    clf.load_state_dict(b["classifier"])
    model = StressModel(encoder, clf, gate).to(DEVICE)
    model.eval()
    return model, joblib.load(scl), b


def encode_context(df, fitted):
    """Same encoding the model was trained with (saved transformers)."""
    num = fitted.get("numeric_cols") or []
    cat = fitted.get("categorical_cols") or []
    pieces = []
    if num:
        pieces.append(fitted["scaler"].transform(
            df[num].apply(pd.to_numeric, errors="coerce").fillna(0.0)))
    if cat:
        pieces.append(fitted["onehot"].transform(df[cat].astype(str)))
    return np.concatenate(pieces, axis=1).astype(np.float32)


def ordinary_values(df, cols):
    out = {}
    for c in cols:
        v = pd.to_numeric(df[c], errors="coerce")
        out[c] = v.mean() if v.notna().all() else df[c].astype(str).mode().iloc[0]
    return out


def all_starts(n_seconds, window, stride=30):
    """Every window: 0, 30, 60 ... while a full 60 s window fits."""
    return list(range(0, n_seconds - window + 1, stride))


def checkpoint_starts(n_seconds, step_min, window):
    """Second offsets of the checkpoints: 0, step, 2*step ... while a full
    60 s window still fits. The first is the session start."""
    s, out = 0, []
    while s + window <= n_seconds:
        out.append(s)
        s += step_min * 60
    return out


def rss_mb():
    if not HAVE_PSUTIL:
        return float("nan")
    return psutil.Process(os.getpid()).memory_info().rss / 1e6


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    ap.add_argument("--pack", required=True)
    ap.add_argument("--out", default=".")
    ap.add_argument("--only", default=None, help="one session name")
    ap.add_argument("--sleep", type=float, default=0.0)
    ap.add_argument("--checkpoints-only", action="store_true",
                    help="score only the checkpoint windows, not every window")
    ap.add_argument("--no-explain", action="store_true",
                    help="skip the per-column measurement")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    torch.set_grad_enabled(False)
    rss0 = rss_mb()
    t_load = time.perf_counter()
    model, scaler, b = load_bundle(a.pack)
    t_load = time.perf_counter() - t_load
    rss1 = rss_mb()
    ctx_cols = list(b["context_columns"])
    fitted = b["context_fitted"]
    window = int(b["window"])

    sess = pd.read_csv(os.path.join(a.pack, "sessions.csv"))
    if a.only:
        sess = sess[sess["session"] == a.only]
    if sess.empty:
        raise SystemExit("no such session")

    print("model loaded in %.2f s, memory +%.0f MB" % (t_load, rss1 - rss0))
    print("context columns: %s\n" % ", ".join(ctx_cols))

    rows = []
    for _, s in sess.iterrows():
        sid, step = s["session"], int(s["step_min"])
        sig = pd.read_csv(os.path.join(a.pack, sid, "signals.csv"))
        sig[TIME_COL] = pd.to_datetime(sig[TIME_COL])
        sig = sig.sort_values(TIME_COL).reset_index(drop=True)
        ctx = pd.read_csv(os.path.join(a.pack, sid, "context.csv"))
        ctx[TIME_COL] = pd.to_datetime(ctx[TIME_COL])
        ctx = ctx.sort_values(TIME_COL)
        arr = pd.DataFrame(sig[SIGNALS].to_numpy(float)) \
                .interpolate(limit_direction="both").fillna(0.0).to_numpy()
        usual = ordinary_values(ctx, ctx_cols)

        ck = set(checkpoint_starts(len(sig), step, window))
        starts = (sorted(ck) if a.checkpoints_only
                  else all_starts(len(sig), window))
        print("%s: %d s -> %d windows scored, %d of them checkpoints "
              "(every %d min)" % (sid, len(sig), len(starts), len(ck), step))

        for k, st in enumerate(starts):
            t_a = time.perf_counter()
            tc = sig[TIME_COL].iloc[st]
            x = arr[st:st + window]                        # (60, 7)
            x = scaler.transform(x).T[None, :, :]          # (1, 7, 60)
            x = np.nan_to_num(x, nan=0.0, posinf=0.0,
                              neginf=0.0).astype(np.float32)

            row = pd.merge_asof(pd.DataFrame({TIME_COL: [tc]}),
                                ctx[[TIME_COL] + ctx_cols], on=TIME_COL,
                                direction="nearest",
                                tolerance=pd.Timedelta(seconds=30))
            for c in ctx_cols:
                if row[c].isna().any():
                    row[c] = usual[c]
            C = encode_context(row, fitted)
            t_b = time.perf_counter()

            xb, cb = torch.tensor(x), torch.tensor(C)
            P = torch.softmax(model(xb, cb), dim=1).numpy()[0]
            t_c = time.perf_counter()

            score = float(P[1] + 2 * P[2])
            rec = {"session": sid, "block": k + 1,
                   "is_checkpoint": st in ck,
                   "datetime_ist": tc.strftime("%Y-%m-%d %H:%M:%S"),
                   "time_ist": tc.strftime("%H:%M"),
                   "window_covers": "%s-%s" % (
                       tc.strftime("%H:%M"),
                       (tc + pd.Timedelta(seconds=window)).strftime("%H:%M")),
                   "score_0_2": score, "P_Low": float(P[0]),
                   "P_Medium": float(P[1]), "P_High": float(P[2]),
                   "state": STATES[int(P.argmax())],
                   "uncertainty": float(1 - P.max()),
                   "prep_ms": (t_b - t_a) * 1e3,
                   "model_ms": (t_c - t_b) * 1e3,
                   "latency_ms": (t_c - t_a) * 1e3,
                   "rss_mb": rss_mb()}
            for c in ctx_cols:
                rec["ctx_" + c] = row[c].iloc[0]

            if not a.no_explain:
                for c in ctx_cols:
                    r2 = row.copy()
                    r2[c] = usual[c]
                    P2 = torch.softmax(model(xb, torch.tensor(
                        encode_context(r2, fitted))), dim=1).numpy()[0]
                    rec["move_" + c] = abs(score - float(P2[1] + 2 * P2[2]))
            rows.append(rec)

            print(" %s %s  score %.2f  %-6s  (L %.2f  M %.2f  H %.2f)  %.1f ms"
                  % ("*" if rec["is_checkpoint"] else " ",
                     tc.strftime("%H:%M:%S"), score, rec["state"], P[0], P[1], P[2],
                     rec["latency_ms"]))
            if a.sleep:
                time.sleep(a.sleep)
        print()

    df = pd.DataFrame(rows)
    out_csv = os.path.join(a.out, "checkpoint_scores_device.csv")
    df.to_csv(os.path.join(a.out, "window_scores_device.csv"), index=False)
    df[df["is_checkpoint"]].to_csv(out_csv, index=False)

    lat = df["latency_ms"]
    lines = ["windows scored     : %d across %d sessions (%d are checkpoints)"
             % (len(df), df["session"].nunique(), int(df["is_checkpoint"].sum())),
             "latency median     : %.1f ms" % lat.median(),
             "latency first      : %.1f ms (nothing warm yet)"
             % lat.iloc[0],
             "latency worst      : %.1f ms" % lat.max(),
             "model load         : %.2f s" % t_load,
             "memory after load  : %.0f MB (+%.0f MB for the model)"
             % (rss1, rss1 - rss0),
             "model on disk      : %.2f MB" % (sum(
                 os.path.getsize(os.path.join(a.pack, f))
                 for f in ("model.pth", "encoder.pth", "scaler.pkl")
                 if os.path.exists(os.path.join(a.pack, f))) / 1e6)]

    def compare(label, ref, keys, ref_score="score_0_2", ref_state="state"):
        m = df.merge(ref, on=keys, suffixes=("", "_ref"))
        lines.append("")
        lines.append("%s: %d of %d found in the laptop file"
                     % (label, len(m), len(df)))
        if not len(m):
            return
        d = (m["score_0_2"] - m[ref_score]).abs()
        same = (m["state"] == m[ref_state]).mean()
        lines.append("  largest score difference : %.5f" % d.max())
        lines.append("  same state               : %.0f%%" % (100 * same))
        lines.append("  RESULT                   : %s"
                     % ("PASS, the device gives the laptop's numbers"
                        if d.max() < 1e-3 and same == 1.0
                        else "CHECK, differences above 0.001"))

    p = os.path.join(a.pack, "reference_windows.csv")
    if os.path.exists(p):
        rw = pd.read_csv(p).rename(columns={"Subject_ID": "session"})
        rw["state_ref_"] = np.array(STATES)[
            rw[["P_Low", "P_Medium", "P_High"]].to_numpy().argmax(1)]
        rw["score_ref_"] = rw["stress_score"]
        compare("every window", rw[["session", "datetime_ist", "score_ref_",
                                    "state_ref_"]],
                ["session", "datetime_ist"], "score_ref_", "state_ref_")
    p = os.path.join(a.pack, "reference_scores.csv")
    if os.path.exists(p):
        ck = df[df["is_checkpoint"]]
        m = ck.merge(pd.read_csv(p), on=["session", "time_ist"],
                     suffixes=("", "_ref"))
        lines.append("")
        lines.append("checkpoints only: %d of %d found in the laptop file"
                     % (len(m), len(ck)))
        if len(m):
            d = (m["score_0_2"] - m["score_0_2_ref"]).abs()
            lines.append("  largest score difference : %.5f" % d.max())

    if not a.no_explain:
        mv = [c for c in df.columns if c.startswith("move_")]
        lines.append("\nwhat each context column moved the score, on 0-2")
        lines.append("  %-16s %8s %8s" % ("column", "average", "largest"))
        for c in sorted(mv, key=lambda c: -df[c].mean()):
            lines.append("  %-16s %8.4f %8.4f"
                         % (c[5:], df[c].mean(), df[c].max()))

    text = "\n".join(lines)
    print("=" * 60)
    print(text)
    print("=" * 60)
    with open(os.path.join(a.out, "device_summary.txt"), "w") as f:
        f.write(text + "\n")
    print("wrote %s and device_summary.txt" % out_csv)


if __name__ == "__main__":
    main()
