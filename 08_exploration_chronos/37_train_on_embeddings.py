"""
37_train_on_embeddings.py

WHAT THIS IS
    The last step of the SensorLLM path, stopped short of the language
    model.

        signal → tokenise → Chronos T5 encoder → embeddings → classifier

    SensorLLM projects the embeddings into a language model's hidden size
    and hands them to LLaMA. That needs an A100. We give them instead to
    AdaStress's own gate and classifier, which keeps the numbers comparable
    with everything else measured in this project.

WHAT REPLACES WHAT
    In every earlier run the AdaStress encoder read the signal and produced
    256 numbers per window. Here the Chronos T5 encoder reads tokens and
    produces 3072 (six channels of 512). Everything after that point is the
    same code:

        adastress.models.ContextGateFusion   the gate
        adastress.models.MLPClassifier       the classifier

    So the comparison isolates one thing: which encoder produced the
    representation.

ON STANDARDISING FIRST
    The embeddings come out with a very small spread, about 0.02, and a
    range of roughly -0.16 to 0.19. A classifier trained on numbers that
    small learns slowly and unevenly, so each dimension is standardised
    before training.

    Fitted on the training rows only. Fitting on everything would let the
    held-out person's own statistics shape the scaling, which is a quiet
    way of leaking.

WHAT TO EXPECT
    The SensorLLM paper runs Chronos embeddings with a simple head as one
    of its baselines, and it came out worst or nearly worst on both of its
    datasets. They write that "Chronos embeddings alone are of limited
    utility", and say they picked Chronos precisely because it had not been
    pretrained on motion sensor data.

    There is a second reason to expect little here. Tokenisation normalises
    each window on its own, which removes the level that distinguishes one
    window from another. We measured that cost separately: dequantised skin
    temperature correlated with the original at -0.021, and training on
    those signals lost about 8 points of balanced accuracy on every split.
    The T5 encoder reads the same normalised tokens.

HOW TO RUN
    From the AdaStress_BTP folder, after btp/36_chronos_embeddings.py:

        for S in random response user; do
          python3 btp/37_train_on_embeddings.py --run-id E_${S} \\
              --split $S --seeds 5 --context off
        done
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             f1_score, confusion_matrix)
from sklearn.preprocessing import StandardScaler, OneHotEncoder

VENDOR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "vendor", "adastress")
sys.path.insert(0, VENDOR)

from adastress.models import ContextGateFusion, MLPClassifier    # noqa: E402
from adastress.utils import set_seed                             # noqa: E402

EMBED_DIR = "data/processed/chronos_embed"
LABELLED = "data/processed/labels/labelled_windows.csv"
LABEL_NAMES = ["LowStress", "MediumStress", "HighStress"]

NOT_CONTEXT = {"Subject_ID", "datetime_utc", "datetime_ist", "date",
               "response_time", "STAI6", "STAI6_RAW", "LABEL", "LABEL_NAME"}

TIME_COL = "datetime_ist"
TOLERANCE_SEC = 30          # a labelled window and its embedding must line up


def load_pairs(embed_dir, labelled_path, context_cols):
    """Match each labelled window to the embedding of the same window."""
    lab = pd.read_csv(labelled_path)
    lab[TIME_COL] = pd.to_datetime(lab[TIME_COL], errors="coerce")
    lab = lab.dropna(subset=[TIME_COL])

    Es, ys, subs, Cs, resp = [], [], [], [], []

    for user, g in lab.groupby("Subject_ID", sort=True):
        epath = os.path.join(embed_dir, "%s_embed.npy" % user)
        ipath = os.path.join(embed_dir, "%s_index.csv" % user)
        if not (os.path.exists(epath) and os.path.exists(ipath)):
            print("  %-8s no embeddings, skipped" % user)
            continue

        E = np.load(epath)
        idx = pd.read_csv(ipath)
        idx[TIME_COL] = pd.to_datetime(idx[TIME_COL], errors="coerce")
        idx = idx.dropna(subset=[TIME_COL]).reset_index(drop=True)

        # match on window start time, nearest within half a window
        left = g[[TIME_COL, "LABEL"]].sort_values(TIME_COL).copy()
        if "response_time" in g.columns:
            left["response_time"] = g.sort_values(TIME_COL)["response_time"]
        if context_cols:
            for c in context_cols:
                left[c] = g.sort_values(TIME_COL)[c]

        right = idx.reset_index().rename(columns={"index": "row"})
        right = right[[TIME_COL, "row"]].sort_values(TIME_COL)

        merged = pd.merge_asof(left, right, on=TIME_COL, direction="nearest",
                               tolerance=pd.Timedelta(seconds=TOLERANCE_SEC))
        hit = merged[merged["row"].notna()]

        rows = hit["row"].astype(int).to_numpy()
        Es.append(E[rows])
        ys.append(hit["LABEL"].to_numpy().astype(int))
        subs.append(np.full(len(rows), user))
        resp.append(np.array(["%s|%s" % (user, t)
                              for t in hit.get("response_time",
                                               hit[TIME_COL])]))
        if context_cols:
            Cs.append(hit[context_cols])

        print("  %-8s %5d of %5d labelled windows matched an embedding"
              % (user, len(rows), len(g)))

    if not Es:
        raise SystemExit("Nothing matched. Run btp/36_chronos_embeddings.py "
                         "first.")

    return (np.concatenate(Es).astype(np.float32),
            np.concatenate(ys),
            np.concatenate(subs),
            (pd.concat(Cs, ignore_index=True) if context_cols else None),
            np.concatenate(resp))


def build_context(C_train, C_all, cols):
    """Context to numbers, fitted on the training rows only."""
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
    return np.concatenate(pieces, axis=1).astype(np.float32)


def make_folds(args, subs, resp, n, users):
    """Same three splits as btp/34_train_users.py, so results line up."""
    if args.split == "user":
        return [(u, ~(subs == u), subs == u) for u in users]

    tr_pct, va_pct, te_pct = [float(x) for x in args.ratios.split("/")]
    tot = tr_pct + va_pct + te_pct
    tr_pct, va_pct = tr_pct / tot, va_pct / tot

    folds = []
    for si in range(args.seeds):
        rng = np.random.default_rng(args.seed + si)
        if args.split == "random":
            order = rng.permutation(n)
        else:
            groups = [np.where(resp == r)[0] for r in np.unique(resp)]
            rng.shuffle(groups)
            order = np.concatenate(groups)
        n_tr, n_va = int(round(tr_pct * n)), int(round(va_pct * n))
        tr = np.zeros(n, dtype=bool); te = np.zeros(n, dtype=bool)
        tr[order[:n_tr]] = True
        te[order[n_tr + n_va:]] = True
        folds.append(("seed%d" % (args.seed + si), tr, te))
    return folds


def main():
    ap = argparse.ArgumentParser(
        description="Train the AdaStress gate and classifier on Chronos T5 "
                    "embeddings.")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--embed-dir", default=EMBED_DIR)
    ap.add_argument("--labelled", default=LABELLED)
    ap.add_argument("--context", choices=["on", "off"], default="off")
    ap.add_argument("--split", choices=["user", "random", "response"],
                    default="user")
    ap.add_argument("--ratios", default="70/10/20")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    run_dir = os.path.join("runs", args.run_id)
    if os.path.exists(run_dir):
        raise SystemExit("Run folder exists: %s" % run_dir)
    os.makedirs(run_dir)

    head = pd.read_csv(args.labelled, nrows=1)
    context_cols = ([c for c in head.columns if c not in NOT_CONTEXT]
                    if args.context == "on" else [])

    print("=" * 72)
    print("RUN      : %s" % args.run_id)
    print("input    : Chronos T5 embeddings from %s" % args.embed_dir)
    print("           the AdaStress encoder is NOT used in this run")
    print("context  : %s" % (", ".join(context_cols) if context_cols
                             else "none"))
    print("split    : %s%s" % (args.split,
                               "  %s, %d seed(s)" % (args.ratios, args.seeds)
                               if args.split != "user" else
                               "  one person held out at a time"))
    print("=" * 72)
    print()

    print("matching labelled windows to embeddings:")
    E, y, subs, C, resp = load_pairs(args.embed_dir, args.labelled,
                                     context_cols)
    users = sorted(set(subs))
    print()
    print("embeddings: %s   users: %s" % (E.shape, ", ".join(users)))
    print("labels    :", {LABEL_NAMES[c]: int((y == c).sum())
                          for c in sorted(set(y))})
    print("values    : mean %.4f, sd %.4f, range %.3f to %.3f"
          % (E.mean(), E.std(), E.min(), E.max()))
    print()

    classes = sorted(set(y))
    device = torch.device("cpu")
    results = []

    for held, tr, te in make_folds(args, subs, resp, len(y), users):
        if te.sum() < 10 or tr.sum() < 50:
            print("fold %s skipped, too few windows" % held)
            continue
        set_seed(args.seed)

        # standardise, fitted on the training rows only. The raw embeddings
        # have a spread of about 0.02, which trains poorly as-is.
        sc = StandardScaler().fit(E[tr])
        Es = np.nan_to_num(sc.transform(E), nan=0.0,
                           posinf=0.0, neginf=0.0).astype(np.float32)

        if context_cols:
            Call = build_context(C[tr], C, context_cols)
            ctx_dim = Call.shape[1]
        else:
            Call = np.zeros((len(y), 1), dtype=np.float32)
            ctx_dim = 0

        lab, cnt = np.unique(y[tr], return_counts=True)
        baseline = float((y[te] == lab[cnt.argmax()]).mean())

        tr_dl = DataLoader(TensorDataset(torch.tensor(Es[tr]),
                                         torch.tensor(Call[tr]),
                                         torch.tensor(y[tr])),
                           batch_size=args.batch_size, shuffle=True)
        te_dl = DataLoader(TensorDataset(torch.tensor(Es[te]),
                                         torch.tensor(Call[te]),
                                         torch.tensor(y[te])),
                           batch_size=args.batch_size)

        # the repo's own gate and classifier, on top of Chronos embeddings
        # instead of the AdaStress encoder's output
        gate = (ContextGateFusion(E.shape[1], ctx_dim).to(device)
                if ctx_dim > 0 else None)
        clf = MLPClassifier(E.shape[1], len(classes),
                            dropout=args.dropout).to(device)

        params = list(clf.parameters())
        if gate is not None:
            params += list(gate.parameters())
        opt = optim.Adam(params, lr=args.lr)
        criterion = nn.CrossEntropyLoss()

        for _ep in range(args.epochs):
            clf.train()
            if gate is not None:
                gate.train()
            for eb, cb, yb in tr_dl:
                h = gate(eb, cb) if gate is not None else eb
                loss = criterion(clf(h), yb)
                opt.zero_grad()
                loss.backward()
                opt.step()

        clf.eval()
        if gate is not None:
            gate.eval()
        preds, trues, gates = [], [], []
        with torch.no_grad():
            for eb, cb, yb in te_dl:
                h = gate(eb, cb) if gate is not None else eb
                if gate is not None:
                    gates.append(gate.gate_values(eb, cb).mean(dim=1).numpy())
                preds.append(clf(h).argmax(1).numpy())
                trues.append(yb.numpy())
        pred, true = np.concatenate(preds), np.concatenate(trues)

        acc = float(accuracy_score(true, pred))
        bal = float(balanced_accuracy_score(true, pred))
        mf1 = float(f1_score(true, pred, average="macro", zero_division=0))
        gm = float(np.concatenate(gates).mean()) if gates else None

        n_cls = len(set(true))
        note = "  <- one class only, read balanced accuracy" if n_cls == 1 else ""
        print("%-9s test %4d  acc %.3f  balanced %.3f  macroF1 %.3f  "
              "(guessing %.3f)%s"
              % (held, te.sum(), acc, bal, mf1, baseline, note))
        if gm is not None:
            print("          gate %.3f" % gm)

        results.append({"fold": held, "n_test": int(te.sum()),
                        "n_train": int(tr.sum()), "accuracy": acc,
                        "balanced_accuracy": bal, "macro_f1": mf1,
                        "majority_baseline": baseline, "gate_mean": gm,
                        "classes_in_test": n_cls,
                        "confusion_matrix": confusion_matrix(
                            true, pred, labels=classes).tolist()})

    if not results:
        raise SystemExit("No fold could be run.")

    accs = [r["accuracy"] for r in results]
    bals = [r["balanced_accuracy"] for r in results]
    f1s = [r["macro_f1"] for r in results]
    base = [r["majority_baseline"] for r in results]

    print()
    print("=" * 72)
    print("ACROSS THE FOLDS")
    print("=" * 72)
    print("  %-24s %9s %10s %9s %10s"
          % ("", "accuracy", "balanced", "macroF1", "guessing"))
    print("  %-24s %9.3f %10.3f %9.3f %10.3f"
          % ("MEAN", np.mean(accs), np.mean(bals), np.mean(f1s),
             np.mean(base)))
    print("  %-24s %9.3f %10.3f %9.3f"
          % ("SPREAD (best - worst)", max(accs) - min(accs),
             max(bals) - min(bals), max(f1s) - min(f1s)))
    print()
    print("  Compare against the same split with the AdaStress encoder.")
    print("  Chance for three classes is 0.333.")
    print()

    json.dump({"run_id": args.run_id, "input": "chronos_t5_embeddings",
               "embed_dir": args.embed_dir, "embed_dim": int(E.shape[1]),
               "context": args.context, "context_columns": context_cols,
               "split": args.split, "ratios": args.ratios,
               "seeds": args.seeds, "epochs": args.epochs,
               "mean_accuracy": float(np.mean(accs)),
               "mean_balanced_accuracy": float(np.mean(bals)),
               "mean_macro_f1": float(np.mean(f1s)),
               "mean_baseline": float(np.mean(base)),
               "per_fold": results},
              open(os.path.join(run_dir, "metrics.json"), "w"), indent=2)
    print("saved %s/metrics.json" % run_dir)


if __name__ == "__main__":
    main()
