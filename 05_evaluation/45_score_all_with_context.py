"""
45_score_all_with_context.py

WHAT THIS DOES
    Scores every window of every user with a model that has a context gate,
    and writes one row per window.

    btp/26_apply_to_users.py cannot do this: it was written for a model with
    no gate, and has no way to feed one.

WHAT IT NEEDS
    a model saved by btp/34_train_users.py with --save-model, which carries
        the gate and, importantly, the fitted context transformers
    the 1 Hz signals, from data/processed/user_7ch/
    the context for every window, from btp/44_build_context7_all.py

    The transformers matter. Encoding context with freshly fitted ones would
    give different numbers for the same words, and the gate would be reading
    something it had never seen.

THREE THINGS TO SAY ABOUT THESE SCORES

    These users were in the training data. The model was fitted on all five,
    so scoring them is not prediction, it is the model repeating what it was
    taught. Useful for seeing the pipeline work end to end; not evidence
    that it works on anybody new.

    The split was pooled and random. Windows overlap by 30 seconds and each
    survey answer labels about 60 of them, so near identical windows sat on
    both sides of the split. That is why the training score is 0.841 and the
    score on a person never seen is 0.286, against a chance level of 0.333.

    The context is thinner here than in training. Across the labelled
    windows the Empatica biomarkers were present 83 percent of the time,
    because people answer surveys while wearing the watch. Across every
    window it is nearer 39 percent, and the rest read "Unknown". The model
    meets that word far more often now than it did while learning.

OUTPUT
    runs/<run-id>/predictions.csv       one row per window
    runs/<run-id>/per_user.csv          one row per user
    runs/<run-id>/per_response.csv      one row per survey answer, with what
                                        the person actually reported

    The per-response file is the one worth looking at. It is the only view
    where there is a truth to compare against.

HOW TO RUN
    python3 btp/45_score_all_with_context.py \\
        --model runs/FINAL_ctx7_random/model.pth \\
        --run-id SCORES_ctx7
"""

import os
import sys
import json
import argparse

import numpy as np
import pandas as pd
import torch
import joblib

VENDOR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "vendor", "adastress")
sys.path.insert(0, VENDOR)

from adastress.models import (ContextGateFusion, MLPClassifier,     # noqa: E402
                              StressModel, load_encoder)

SIGNAL_DIR = "data/processed/user_7ch"
CONTEXT_ALL = "data/processed/context/all_windows_ctx7.csv"
LABELLED = "data/processed/labels/labelled_windows.csv"
LABEL_NAMES = ["LowStress", "MediumStress", "HighStress"]
SHORT = {"LowStress": "Low", "MediumStress": "Medium", "HighStress": "High"}
TIME_COL = "datetime_ist"
DEVICE = torch.device("cpu")


def load_model(path):
    b = torch.load(path, map_location="cpu", weights_only=False)
    for k in ("encoder_ckpt", "scaler_ckpt"):
        if not os.path.exists(b[k]):
            raise SystemExit("%s points at %s, which is not there. Run from "
                             "the project root." % (k, b[k]))

    encoder = load_encoder(b["encoder_ckpt"], b["input_channels"],
                           b["lstm_hidden"], DEVICE)
    ctx_dim = int(b.get("context_dim", 0))
    gate = None
    if ctx_dim > 0:
        if not b.get("gate"):
            raise SystemExit("this model says it has context but no gate was "
                             "saved with it")
        gate = ContextGateFusion(encoder.output_dim, ctx_dim).to(DEVICE)
        gate.load_state_dict(b["gate"])

    clf = MLPClassifier(encoder.output_dim, len(b["classes"])).to(DEVICE)
    clf.load_state_dict(b["classifier"])
    model = StressModel(encoder, clf, gate).to(DEVICE)
    model.eval()
    return model, joblib.load(b["scaler_ckpt"]), b


def encode_context(df, fitted):
    """Context to numbers, using the transformers fitted during training."""
    num = fitted.get("numeric_cols") or []
    cat = fitted.get("categorical_cols") or []
    pieces = []
    if num:
        pieces.append(fitted["scaler"].transform(
            df[num].apply(pd.to_numeric, errors="coerce").fillna(0.0)))
    if cat:
        pieces.append(fitted["onehot"].transform(df[cat].astype(str)))
    return np.concatenate(pieces, axis=1).astype(np.float32)


def windows_for_user(path, cols, window, stride):
    """Cut the signal into windows, inside each day so none spans a gap."""
    df = pd.read_csv(path)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print("    missing channels %s" % missing)
        return None, None

    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
    df = df.dropna(subset=[TIME_COL]).sort_values(TIME_COL)
    df = df.reset_index(drop=True)

    arr = df[cols].to_numpy(dtype=float)
    arr = pd.DataFrame(arr).interpolate(limit_direction="both") \
            .fillna(0.0).to_numpy()

    groups = (df.groupby("date").groups.items() if "date" in df.columns
              else [(None, df.index)])
    starts = []
    for _d, idx in groups:
        lo, hi = int(min(idx)), int(max(idx)) + 1
        starts += list(range(lo, hi - window + 1, stride))
    if not starts:
        return None, None

    X = np.stack([arr[s:s + window].T for s in starts]).astype(np.float32)
    meta = pd.DataFrame({
        TIME_COL: [df[TIME_COL].iloc[s] for s in starts],
        "mid_time": [df[TIME_COL].iloc[s + window // 2] for s in starts],
        "date": [str(df[TIME_COL].iloc[s].date()) for s in starts],
    })
    return X, meta


def main():
    ap = argparse.ArgumentParser(
        description="Score every window with a model that has a context "
                    "gate.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--signal-dir", default=SIGNAL_DIR)
    ap.add_argument("--context", default=CONTEXT_ALL)
    ap.add_argument("--labelled", default=LABELLED)
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()

    run_dir = os.path.join("runs", args.run_id)
    os.makedirs(run_dir, exist_ok=True)

    model, scaler, b = load_model(args.model)
    cols = list(b["signal_columns"])
    ctx_cols = list(b.get("context_columns", []))
    fitted = b.get("context_fitted")
    ctx_dim = int(b.get("context_dim", 0))
    window = int(b["window"])
    stride = int(b.get("stride", window // 2))

    print("=" * 74)
    print("Scoring every window, with context")
    print("=" * 74)
    print("model    : %s" % args.model)
    print("           %s" % b.get("trained_on", "unknown"))
    print("channels : %d  %s" % (len(cols), cols))
    print("context  : %d columns -> %d values" % (len(ctx_cols), ctx_dim))
    if ctx_dim and fitted is None:
        raise SystemExit("the fitted context transformers were not saved "
                         "with this model, so context cannot be encoded the "
                         "way it was in training. Retrain with --save-model.")
    print("window   : %d s, stride %d s" % (window, stride))
    print("=" * 74)
    print()

    ctx_all = None
    if ctx_dim:
        if not os.path.exists(args.context):
            raise SystemExit("no context at %s. Run "
                             "btp/44_build_context7_all.py first."
                             % args.context)
        ctx_all = pd.read_csv(args.context)
        ctx_all[TIME_COL] = pd.to_datetime(ctx_all[TIME_COL], errors="coerce")
        ctx_all = ctx_all.dropna(subset=[TIME_COL])
        print("context rows: %d" % len(ctx_all))
        print()

    users = sorted(f[:-4] for f in os.listdir(args.signal_dir)
                   if f.endswith(".csv") and f != "users_all.csv")

    out_frames = []
    for user in users:
        print("%s" % user)
        X, meta = windows_for_user(
            os.path.join(args.signal_dir, "%s.csv" % user), cols, window,
            stride)
        if X is None:
            print()
            continue

        flat = X.transpose(0, 2, 1).reshape(-1, X.shape[1])
        X = scaler.transform(flat).reshape(X.shape[0], X.shape[2],
                                           X.shape[1]).transpose(0, 2, 1)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0,
                          neginf=0.0).astype(np.float32)

        if ctx_dim:
            cu = ctx_all[ctx_all["Subject_ID"] == user].sort_values(TIME_COL)
            joined = pd.merge_asof(meta.sort_values(TIME_COL),
                                   cu[[TIME_COL] + ctx_cols],
                                   on=TIME_COL, direction="nearest",
                                   tolerance=pd.Timedelta(seconds=stride))
            got = int(joined[ctx_cols].notna().any(axis=1).sum())
            # a window with no context row at all gets the words the model
            # saw most often, rather than blanks it never saw
            for c in ctx_cols:
                if joined[c].isna().any():
                    v = pd.to_numeric(cu[c], errors="coerce")
                    fill = (v.mean() if v.notna().all()
                            else cu[c].astype(str).mode().iloc[0]
                            if len(cu) else "Unknown")
                    joined[c] = joined[c].fillna(fill)
            C = encode_context(joined, fitted)
            meta = joined
            print("    %6d windows, %d matched a context row (%.0f%%)"
                  % (len(X), got, 100.0 * got / max(len(X), 1)))
        else:
            C = np.zeros((len(X), 1), dtype=np.float32)
            print("    %6d windows" % len(X))

        probs, gates = [], []
        with torch.no_grad():
            for s in range(0, len(X), args.batch):
                xb = torch.tensor(X[s:s + args.batch])
                cb = torch.tensor(C[s:s + args.batch]) if ctx_dim else None
                probs.append(torch.softmax(model(xb, cb), dim=1).numpy())
                if ctx_dim:
                    h = model.embed(xb)
                    gates.append(model.gate.gate_values(h, cb)
                                 .mean(dim=1).numpy())
        P = np.concatenate(probs)

        r = meta.copy()
        r.insert(0, "Subject_ID", user)
        r["P_Low"] = P[:, 0]
        r["P_Medium"] = P[:, 1]
        r["P_High"] = P[:, 2]
        r["predicted"] = [LABEL_NAMES[i] for i in P.argmax(1)]
        r["stress_score"] = P[:, 1] + 2 * P[:, 2]
        r["uncertainty"] = 1 - P.max(1)
        if gates:
            r["gate"] = np.concatenate(gates)
        out_frames.append(r)

        print("    %s" % dict(pd.Series(r["predicted"]).value_counts()))
        print("    score mean %.2f, median %.2f"
              % (r["stress_score"].mean(), r["stress_score"].median()))
        print()

    if not out_frames:
        raise SystemExit("Nothing scored.")

    allr = pd.concat(out_frames, ignore_index=True)
    allr.to_csv(os.path.join(run_dir, "predictions.csv"), index=False)

    # ---------------- per user ----------------
    pu = allr.groupby("Subject_ID").agg(
        windows=("stress_score", "size"),
        hours=("stress_score", lambda s: round(len(s) * stride / 3600.0, 1)),
        score_mean=("stress_score", "mean"),
        score_median=("stress_score", "median"),
        uncertainty=("uncertainty", "mean"),
        pct_low=("predicted", lambda s: 100.0 * (s == "LowStress").mean()),
        pct_medium=("predicted", lambda s: 100.0 * (s == "MediumStress").mean()),
        pct_high=("predicted", lambda s: 100.0 * (s == "HighStress").mean()),
    ).round(3).reset_index()
    pu.to_csv(os.path.join(run_dir, "per_user.csv"), index=False)

    print("=" * 74)
    print("PER USER")
    print("=" * 74)
    print(pu.to_string(index=False))
    print()

    # ---------------- per survey answer ----------------
    if os.path.exists(args.labelled):
        lab = pd.read_csv(args.labelled)
        lab[TIME_COL] = pd.to_datetime(lab[TIME_COL], errors="coerce")
        lab = lab.dropna(subset=[TIME_COL, "LABEL"])

        j = pd.merge_asof(
            lab.sort_values(TIME_COL),
            allr[["Subject_ID", TIME_COL, "stress_score", "predicted",
                  "P_Low", "P_Medium", "P_High"]].sort_values(TIME_COL),
            on=TIME_COL, by="Subject_ID", direction="nearest",
            tolerance=pd.Timedelta(seconds=stride)).dropna(
                subset=["predicted"])

        if len(j):
            pr = j.groupby(["Subject_ID", "response_time"]).agg(
                windows=("stress_score", "size"),
                reported=("LABEL_NAME", "first"),
                stai6=("STAI6", "first"),
                score_mean=("stress_score", "mean"),
                predicted=("predicted",
                           lambda s: s.value_counts().index[0]),
            ).round(3).reset_index()
            pr["correct"] = pr["reported"] == pr["predicted"]
            pr.to_csv(os.path.join(run_dir, "per_response.csv"), index=False)

            print("=" * 74)
            print("PER SURVEY ANSWER")
            print("=" * 74)
            print("%d answers matched a scored window" % len(pr))
            print("the prediction matched what the person said %.0f%% of "
                  "the time" % (100 * pr["correct"].mean()))
            print()
            print(pd.crosstab(pr["reported"], pr["predicted"]).to_string())
            print()
            rho = pr[["score_mean", "stai6"]].corr(method="spearman"
                                                   ).iloc[0, 1]
            print("rank correlation between the score and the STAI-6 "
                  "number: %+.2f" % rho)
            print()
            print("Read that number carefully. These five users were all in")
            print("the training data, and the split was random, so near")
            print("identical windows sat on both sides. On a person the")
            print("model had never seen, balanced accuracy was 0.286 against")
            print("a chance level of 0.333.")
            print()

    json.dump({"model": args.model,
               "trained_on": b.get("trained_on"),
               "context_columns": ctx_cols, "context_dim": ctx_dim,
               "channels": cols, "window": window, "stride": stride,
               "windows_scored": int(len(allr)),
               "users": pu.to_dict("records")},
              open(os.path.join(run_dir, "summary.json"), "w"), indent=2)

    print("saved %s/predictions.csv   (%d rows)" % (run_dir, len(allr)))
    print("saved %s/per_user.csv" % run_dir)
    if os.path.exists(os.path.join(run_dir, "per_response.csv")):
        print("saved %s/per_response.csv   <- the one worth sending"
              % run_dir)


if __name__ == "__main__":
    main()
