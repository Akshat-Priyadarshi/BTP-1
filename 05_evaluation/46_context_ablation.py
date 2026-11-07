"""
46_context_ablation.py

WHAT THIS ANSWERS
    Which context columns are doing anything, and which are along for the
    ride.

    Two questions, asked separately, because they are not the same:

        drop one      everything except this column. If the score barely
                      moves, the column was contributing nothing beyond what
                      the others already say.

        keep one      this column and nothing else. A column can look
                      useless when dropped because another carries the same
                      information; alone, you see what it actually holds.

    A column that looks pointless under "drop one" but useful under "keep
    one" is duplicated elsewhere. One that looks useless under both is
    genuinely empty.

HOW MANY RUNS
    With eight columns:

         1   all eight, the reference
         8   each one dropped
         8   each one alone
         1   no context at all, the floor
        --
        18   runs, each over five seeds

    Every run uses the same windows, the same frozen encoder and the same
    seeds, so the only thing changing is which context the gate can see.

HOW TO READ IT
    The split is pooled random 70/10/20, as asked.

    One thing to carry while reading: on this split context looks far more
    useful than it is. Windows overlap by 30 seconds and each survey answer
    labels about 60 of them, so near identical windows sit on both sides.
    Measured earlier on the same data:

        split                     no context    eight columns
        pooled random                  0.550            0.871
        survey answers kept whole      0.468            0.398

    Context adds 0.32 on one split and costs 0.07 on the other. So what
    follows is a ranking of how much each column helps the model recognise
    moments it has already seen. That is a fair question, and it is not the
    same as which column predicts stress.

OUTPUT
    runs/<run-id>/ablation.csv
    docs/64_context_ablation.md

HOW TO RUN
    python3 btp/46_context_ablation.py \\
        --config configs/w7.yaml \\
        --labelled data/processed/labels/labelled_windows_ctx8.csv \\
        --signal-dir data/processed/user_7ch \\
        --run-id ABLATE_ctx8

    About forty minutes for eighteen runs at five seeds.
"""

import os
import sys
import json
import argparse
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib

warnings.filterwarnings("ignore")

VENDOR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "vendor", "adastress")
sys.path.insert(0, VENDOR)

from adastress.config import load_config                            # noqa: E402
from adastress.models import (ContextGateFusion, MLPClassifier,     # noqa: E402
                              StressModel, load_encoder)
from adastress.utils import set_seed                                # noqa: E402

LABELLED = "data/processed/labels/labelled_windows_ctx8.csv"
SIGNAL_DIR = "data/processed/user_7ch"
REPORT = "docs/64_context_ablation.md"

NOT_CONTEXT = {"Subject_ID", "datetime_utc", "datetime_ist", "date",
               "response_time", "STAI6", "STAI6_RAW", "LABEL", "LABEL_NAME"}
TIME_COL = "datetime_ist"
DEVICE = torch.device("cpu")


def load_windows(labelled_path, signal_dir, signal_cols, window, context_cols):
    """Pair each labelled window with the signal beneath it. Done once."""
    lab = pd.read_csv(labelled_path)
    lab[TIME_COL] = pd.to_datetime(lab[TIME_COL], errors="coerce")
    lab = lab.dropna(subset=[TIME_COL])

    Xs, Cs, ys, subs, resp = [], [], [], [], []
    for user, g in lab.groupby("Subject_ID", sort=True):
        spath = os.path.join(signal_dir, "%s.csv" % user)
        if not os.path.exists(spath):
            continue
        sig = pd.read_csv(spath)
        sig[TIME_COL] = pd.to_datetime(sig[TIME_COL], errors="coerce")
        sig = sig.dropna(subset=[TIME_COL]).sort_values(TIME_COL)

        arr = sig[signal_cols].to_numpy(dtype=float)
        arr = pd.DataFrame(arr).interpolate(limit_direction="both") \
                .fillna(0.0).to_numpy()
        tidx = sig[TIME_COL].to_numpy()
        starts = np.searchsorted(tidx, g[TIME_COL].to_numpy(), side="left")

        for pos, (_i, row) in zip(starts, g.iterrows()):
            if pos + window > len(arr):
                continue
            span = (tidx[pos + window - 1] - tidx[pos]) / np.timedelta64(1, "s")
            if not (window - 5 <= span <= window + 5):
                continue
            Xs.append(arr[pos:pos + window].T)
            ys.append(int(row["LABEL"]))
            subs.append(user)
            resp.append("%s|%s" % (user, row.get("response_time", "")))
            Cs.append(row[context_cols].to_dict())

    if not Xs:
        raise SystemExit("No labelled window could be paired with signal.")
    return (np.stack(Xs).astype(np.float32), np.asarray(ys),
            pd.DataFrame(Cs), np.asarray(subs), np.asarray(resp))


def build_context(C_train, C_all, cols):
    """Context to numbers, fitted on the training rows only."""
    if not cols:
        return np.zeros((len(C_all), 1), dtype=np.float32), 0
    num = [c for c in cols if pd.api.types.is_numeric_dtype(C_all[c])]
    cat = [c for c in cols if c not in num]
    pieces = []
    if num:
        sc = StandardScaler().fit(C_train[num].fillna(0.0))
        pieces.append(sc.transform(C_all[num].fillna(0.0)))
    if cat:
        oh = OneHotEncoder(handle_unknown="ignore",
                           sparse_output=False).fit(C_train[cat].astype(str))
        pieces.append(oh.transform(C_all[cat].astype(str)))
    out = np.concatenate(pieces, axis=1).astype(np.float32)
    return out, out.shape[1]


def one_run(X, y, C, cols, cfg, seeds, seed0, ratios, epochs, classes):
    """Train and test with one set of context columns, over several seeds."""
    tr_pct, va_pct, te_pct = [float(v) for v in ratios.split("/")]
    tot = tr_pct + va_pct + te_pct
    tr_pct, va_pct = tr_pct / tot, va_pct / tot
    n = len(y)

    accs, bals, f1s, bases, gates, dims = [], [], [], [], [], []

    for si in range(seeds):
        rng = np.random.default_rng(seed0 + si)
        order = rng.permutation(n)
        n_tr, n_va = int(round(tr_pct * n)), int(round(va_pct * n))
        tr = np.zeros(n, dtype=bool); te = np.zeros(n, dtype=bool)
        tr[order[:n_tr]] = True
        te[order[n_tr + n_va:]] = True

        set_seed(seed0)
        Call, ctx_dim = build_context(C[tr], C, cols)
        dims.append(ctx_dim)

        lab, cnt = np.unique(y[tr], return_counts=True)
        bases.append(float((y[te] == lab[cnt.argmax()]).mean()))

        tr_dl = DataLoader(TensorDataset(torch.tensor(X[tr]),
                                         torch.tensor(Call[tr]),
                                         torch.tensor(y[tr])),
                           batch_size=cfg.classifier.batch_size, shuffle=True)
        te_dl = DataLoader(TensorDataset(torch.tensor(X[te]),
                                         torch.tensor(Call[te]),
                                         torch.tensor(y[te])),
                           batch_size=cfg.classifier.batch_size)

        encoder = load_encoder(cfg.paths.encoder_ckpt, X.shape[1],
                               cfg.ssl.lstm_hidden, DEVICE)
        gate = (ContextGateFusion(encoder.output_dim, ctx_dim).to(DEVICE)
                if ctx_dim > 0 else None)
        clf = MLPClassifier(encoder.output_dim, len(classes),
                            dropout=cfg.classifier.dropout).to(DEVICE)
        model = StressModel(encoder, clf, gate).to(DEVICE)

        crit = nn.CrossEntropyLoss()
        opt = optim.Adam(model.adaptable_parameters(), lr=cfg.classifier.lr)
        for _ in range(epochs):
            model.classifier.train()
            if model.gate is not None:
                model.gate.train()
            for xb, cb, yb in tr_dl:
                loss = crit(model(xb, cb if ctx_dim else None), yb)
                opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        pr, tu, gv = [], [], []
        with torch.no_grad():
            for xb, cb, yb in te_dl:
                out = model(xb, cb if ctx_dim else None)
                if gate is not None:
                    gv.append(gate.gate_values(model.embed(xb), cb)
                              .mean(dim=1).numpy())
                pr.append(out.argmax(1).numpy()); tu.append(yb.numpy())
        p, t = np.concatenate(pr), np.concatenate(tu)

        accs.append(float(accuracy_score(t, p)))
        bals.append(float(balanced_accuracy_score(t, p)))
        f1s.append(float(f1_score(t, p, average="macro", zero_division=0)))
        if gv:
            gates.append(float(np.concatenate(gv).mean()))

    return {"accuracy": float(np.mean(accs)),
            "balanced": float(np.mean(bals)),
            "balanced_sd": float(np.std(bals)),
            "macro_f1": float(np.mean(f1s)),
            "baseline": float(np.mean(bases)),
            "gate": float(np.mean(gates)) if gates else None,
            "context_dim": int(np.mean(dims))}


def main():
    ap = argparse.ArgumentParser(
        description="Drop each context column in turn, then use each alone, "
                    "and see what changes.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--labelled", default=LABELLED)
    ap.add_argument("--signal-dir", default=SIGNAL_DIR)
    ap.add_argument("--ratios", default="70/10/20")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    window = cfg.window.window_sec * cfg.window.fs
    epochs = args.epochs or cfg.classifier.epochs
    signal_cols = list(cfg.data.signal_columns)

    run_dir = os.path.join("runs", args.run_id)
    os.makedirs(run_dir, exist_ok=True)

    head = pd.read_csv(args.labelled, nrows=1)
    ctx_cols = [c for c in head.columns if c not in NOT_CONTEXT]

    print("=" * 76)
    print("CONTEXT ABLATION")
    print("=" * 76)
    print("channels : %d  %s" % (len(signal_cols), signal_cols))
    print("context  : %d  %s" % (len(ctx_cols), ", ".join(ctx_cols)))
    print("split    : pooled random %s, %d seeds" % (args.ratios, args.seeds))
    print("runs     : 1 with all, %d dropping one, %d keeping one, 1 with "
          "none = %d" % (len(ctx_cols), len(ctx_cols), 2 * len(ctx_cols) + 2))
    print("=" * 76)
    print()

    X, y, C, subs, resp = load_windows(args.labelled, args.signal_dir,
                                       signal_cols, window, ctx_cols)
    print("windows: %s   users: %s" % (X.shape, ", ".join(sorted(set(subs)))))

    scaler = joblib.load(cfg.paths.scaler_ckpt)
    flat = X.transpose(0, 2, 1).reshape(-1, X.shape[1])
    X = scaler.transform(flat).reshape(X.shape[0], X.shape[2],
                                       X.shape[1]).transpose(0, 2, 1)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    classes = sorted(set(y))
    print()

    def run(label, cols, kind):
        r = one_run(X, y, C, cols, cfg, args.seeds, args.seed, args.ratios,
                    epochs, classes)
        r["label"] = label
        r["kind"] = kind
        r["n_columns"] = len(cols)
        print("  %-26s %d cols  balanced %.3f  (sd %.3f)  gate %s"
              % (label, len(cols), r["balanced"], r["balanced_sd"],
                 "%.3f" % r["gate"] if r["gate"] is not None else "   -"))
        return r

    rows = []

    print("reference")
    full = run("all %d columns" % len(ctx_cols), ctx_cols, "all")
    rows.append(full)
    none = run("no context", [], "none")
    rows.append(none)
    print()

    print("dropping one at a time")
    for c in ctx_cols:
        rows.append(run("without %s" % c, [x for x in ctx_cols if x != c],
                        "drop"))
    print()

    print("keeping one at a time")
    for c in ctx_cols:
        rows.append(run("only %s" % c, [c], "only"))
    print()

    df = pd.DataFrame(rows)
    df["vs_all"] = df["balanced"] - full["balanced"]
    df["vs_none"] = df["balanced"] - none["balanced"]
    df.to_csv(os.path.join(run_dir, "ablation.csv"), index=False)

    # ---------------- the two tables ----------------
    print("=" * 76)
    print("WHAT EACH COLUMN CONTRIBUTES  (dropped from the full set)")
    print("=" * 76)
    print("A large fall means the column was carrying something the others")
    print("do not. A rise means the model was better off without it.")
    print()
    d = df[df["kind"] == "drop"].copy()
    d["column"] = d["label"].str.replace("without ", "", regex=False)
    d = d.sort_values("vs_all")
    print("  %-18s %10s %10s  %s" % ("column removed", "balanced",
                                     "change", ""))
    print("  " + "-" * 62)
    print("  %-18s %10.3f %10s" % ("(none removed)", full["balanced"], "-"))
    for _, r in d.iterrows():
        bar = ("#" * min(int(abs(r["vs_all"]) * 300), 26)) if r["vs_all"] < 0 \
            else ("+" * min(int(r["vs_all"] * 300), 26))
        print("  %-18s %10.3f %+10.3f  %s"
              % (r["column"], r["balanced"], r["vs_all"], bar))
    print()

    print("=" * 76)
    print("WHAT EACH COLUMN HOLDS ON ITS OWN")
    print("=" * 76)
    print("Against no context at all. A column can look useless when dropped")
    print("because another says the same thing; alone, you see what it has.")
    print()
    o = df[df["kind"] == "only"].copy()
    o["column"] = o["label"].str.replace("only ", "", regex=False)
    o = o.sort_values("vs_none", ascending=False)
    print("  %-18s %10s %10s  %s" % ("column alone", "balanced",
                                     "vs none", ""))
    print("  " + "-" * 62)
    print("  %-18s %10.3f %10s" % ("(no context)", none["balanced"], "-"))
    for _, r in o.iterrows():
        bar = "#" * min(int(max(r["vs_none"], 0) * 60), 26)
        print("  %-18s %10.3f %+10.3f  %s"
              % (r["column"], r["balanced"], r["vs_none"], bar))
    print()

    # ---------------- what it adds up to ----------------
    print("=" * 76)
    print("READING IT")
    print("=" * 76)
    worst_drop = d.iloc[0]
    best_only = o.iloc[0]
    print("  all %d columns      %.3f" % (len(ctx_cols), full["balanced"]))
    print("  no context         %.3f" % none["balanced"])
    print("  context is worth   %+.3f on this split"
          % (full["balanced"] - none["balanced"]))
    print()
    print("  costliest to lose  %s  (%+.3f)"
          % (worst_drop["column"], worst_drop["vs_all"]))
    print("  best on its own    %s  (%+.3f over nothing)"
          % (best_only["column"], best_only["vs_none"]))
    print()

    helped = d[d["vs_all"] > 0.005]
    if len(helped):
        print("  Removing these IMPROVED the score, so they were getting in")
        print("  the way: %s" % ", ".join(helped["column"]))
        print()

    dead = d[d["vs_all"].abs() < 0.005]
    if len(dead):
        print("  Removing these changed almost nothing: %s"
              % ", ".join(dead["column"]))
        print("  Either they hold nothing, or another column says the same.")
        print("  The table above separates those two cases.")
        print()

    print("  Careful with all of this. The split is pooled random, so near")
    print("  identical windows sit on both sides and the model can lean on")
    print("  recognising moments it has seen. Measured on the same data,")
    print("  context is worth +0.32 on this split and -0.07 when survey")
    print("  answers are kept whole. What is ranked above is how much each")
    print("  column helps that recognition, which is not the same question")
    print("  as which column predicts stress.")
    print()

    # ---------------- report ----------------
    L = []
    L.append("# Which context columns matter")
    L.append("")
    L.append("Written by `btp/46_context_ablation.py`.")
    L.append("")
    L.append("Eighteen runs on the same windows, the same frozen encoder and")
    L.append("the same %d seeds. Only the context changes." % args.seeds)
    L.append("")
    L.append("| | balanced accuracy |")
    L.append("|---|---:|")
    L.append("| all %d columns | %.3f |" % (len(ctx_cols), full["balanced"]))
    L.append("| no context | %.3f |" % none["balanced"])
    L.append("| guessing | %.3f |" % full["baseline"])
    L.append("")
    L.append("## Dropping one at a time")
    L.append("")
    L.append("| column removed | balanced | change |")
    L.append("|---|---:|---:|")
    L.append("| *(none removed)* | %.3f | — |" % full["balanced"])
    for _, r in d.iterrows():
        L.append("| `%s` | %.3f | %+.3f |"
                 % (r["column"], r["balanced"], r["vs_all"]))
    L.append("")
    L.append("## Each column on its own")
    L.append("")
    L.append("| column alone | balanced | vs no context |")
    L.append("|---|---:|---:|")
    L.append("| *(no context)* | %.3f | — |" % none["balanced"])
    for _, r in o.iterrows():
        L.append("| `%s` | %.3f | %+.3f |"
                 % (r["column"], r["balanced"], r["vs_none"]))
    L.append("")
    L.append("## How to read it")
    L.append("")
    L.append("A column that barely matters when dropped but scores well")
    L.append("alone is saying the same thing as another column. One that is")
    L.append("flat in both tables is genuinely empty.")
    L.append("")
    L.append("The split is pooled random. Windows overlap by 30 seconds and")
    L.append("each survey answer labels about 60 of them, so near identical")
    L.append("windows sit on both sides. Measured on the same data, context")
    L.append("is worth **+0.32** on this split and **-0.07** when survey")
    L.append("answers are kept whole.")
    L.append("")
    L.append("So this ranks how much each column helps the model recognise")
    L.append("moments it has already seen. That is a fair question and worth")
    L.append("answering, but it is not the same as which column predicts")
    L.append("stress.")
    L.append("")

    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    open(REPORT, "w").write("\n".join(L) + "\n")
    json.dump({"config": args.config, "labelled": args.labelled,
               "seeds": args.seeds, "ratios": args.ratios,
               "context_columns": ctx_cols, "rows": rows},
              open(os.path.join(run_dir, "ablation.json"), "w"), indent=2)

    print("saved %s/ablation.csv" % run_dir)
    print("report: %s" % REPORT)


if __name__ == "__main__":
    main()
