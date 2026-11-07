"""
36_chronos_embeddings.py

WHAT THIS IS
    The path SensorLLM actually takes, applied to our data.

        signal ──→ tokenise ──→ Chronos T5 encoder ──→ embeddings
                                     (pretrained, frozen)

    Not an embedding layer we invent and train. The pretrained Chronos
    encoder, which is the whole point of using a foundation model.

WHERE THIS COMES FROM, IN THEIR SOURCE
    sensorllm/model/utils.py:

        pipeline = ChronosPipeline.from_pretrained(ckpt, tc=...)
        self.pt_encoder_backbone = pipeline.model

        @torch.no_grad()
        def ts_embed(self, token_ids, attention_mask):
            ...
            return self.pt_encoder_backbone.encode(token_ids, attention_mask)

    sensorllm/model/chronos_model/chronos_model.py:

        def encode(self, input_ids, attention_mask):
            return self.model.encoder(
                input_ids=input_ids, attention_mask=attention_mask
            ).last_hidden_state

    So: token ids into the T5 encoder, `last_hidden_state` out, shape
    (batch, sequence_length, d_model). Frozen throughout, marked
    @torch.no_grad() and switched to eval.

    They then project it with `ts_proj` into the language model's hidden
    size and hand it to the LLM. We stop before that, because the LLM half
    needs LLaMA3-8B and an A100. We take the embeddings and give them to
    AdaStress's own gate and classifier, which keeps the result comparable
    with everything else we have measured.

WHAT WE HAD TO DECIDE, THAT THEY DECIDE DIFFERENTLY

    Which tokeniser.
        Their default is MeanScaleUniformBins, which divides by the mean
        absolute value. That suits accelerometer data centred on zero.

        It does not suit ours. Measured on our five users, mean scaling puts
        98.2 percent of skin temperature in a single bin, because the
        channel sits around 31 degrees and dividing by its own average gives
        about 1.0 every time. Pulse rate compresses to 413 bins of 4093.
        StanNormalizeUniformBins, which SensorLLM added and upstream Chronos
        does not ship, subtracts the mean first and gives 3341 and 2999.

        Default here is therefore StanNormalize. Use --tokeniser to compare.

    How to pool.
        encode() returns one vector per timestep. SensorLLM keeps the whole
        sequence because an LLM can read a sequence. We need one vector per
        window for the classifier, so it has to be reduced.

        --pool mean   average over the 60 timesteps. The default.
        --pool last   the final timestep, which their config also supports
                      through its `last_token` flag.

WHAT TO EXPECT, WRITTEN DOWN BEFOREHAND
    The SensorLLM paper runs Chronos embeddings with an MLP head as one of
    its baselines, and it came out worst or nearly worst on both datasets:
    44.2 F1 on CAPTURE-24 against 60.2 for the best method, and 82.2 on
    UCI-HAR against 93.2. They say plainly that "Chronos embeddings alone
    are of limited utility", and that they chose Chronos precisely because
    it had NOT been pretrained on motion sensor data, which made it a fair
    test of their own alignment stage.

    So this is their control condition, not their result. Worth running
    because it is the faithful version of the idea, and worth reporting
    with that context attached.

    There is a second reason for caution here. Tokenisation normalises each
    window on its own, which throws away the level that separates one window
    from another. We measured what that costs: dequantised skin temperature
    correlated with the original at -0.021, and training on those signals
    dropped balanced accuracy by about 8 points on every split. The T5
    encoder sees the same normalised tokens, so the same information is
    already gone before it starts.

OUTPUT
    data/processed/chronos_embed/<user>_embed.npy    float32 [N, 6*d_model]
    data/processed/chronos_embed/<user>_index.csv    which window each row is
    docs/59_chronos_embeddings.md

HOW TO RUN
    From the AdaStress_BTP folder:

        pip install "transformers>=4.30"
        python3 btp/36_chronos_embeddings.py

    About 11 minutes on CPU for 5000 windows, with chronos-t5-small.
    Use --model amazon/chronos-t5-mini if that is too slow.
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import torch

SIGNAL_DIR = "data/processed/user_6ch"
OUT_DIR = "data/processed/chronos_embed"
REPORT = "docs/59_chronos_embeddings.md"

# Straight from the Chronos config SensorLLM vendors.
N_TOKENS = 4096
N_SPECIAL = 2
LOW, HIGH = -15.0, 15.0

WINDOW = 60
STRIDE = 30

SIGNAL_COLS = ["ACC_X_wrist", "ACC_Y_wrist", "ACC_Z_wrist",
               "BVP", "EDA_wrist", "Temp_wrist"]

TIME_COL = "datetime_ist"

CENTERS = np.linspace(LOW, HIGH, N_TOKENS - N_SPECIAL - 1)
BOUNDARIES = np.concatenate([[-1e20], (CENTERS[1:] + CENTERS[:-1]) / 2,
                             [1e20]])


def tokenise_batch(x, method):
    """Tokenise a batch of windows of one channel. x is [n, WINDOW].

    Reimplements SensorLLM's two tokenisers. torch.bucketize with right=True
    is numpy searchsorted with side='right'. Our token ids were checked
    against theirs and came out identical on signals shaped like BVP, skin
    temperature and pulse rate, under both methods.
    """
    x = np.asarray(x, dtype=float)
    mask = ~np.isnan(x)

    if method == "MeanScaleUniformBins":
        denom = np.nansum(mask, axis=1)
        scale = np.nansum(np.abs(x) * mask, axis=1) / np.maximum(denom, 1)
        scale[~(scale > 0)] = 1.0
        scaled = x / scale[:, None]
    else:                                    # StanNormalizeUniformBins
        denom = np.maximum(np.nansum(mask, axis=1), 1)
        mu = np.nansum(x * mask, axis=1) / denom
        var = np.nansum((x - mu[:, None]) ** 2 * mask, axis=1) / denom
        sd = np.sqrt(var + 1e-8)
        scaled = (x - mu[:, None]) / sd[:, None]

    scaled = np.nan_to_num(scaled, nan=0.0)
    ids = np.searchsorted(BOUNDARIES, scaled, side="right") + N_SPECIAL
    ids = np.clip(ids, N_SPECIAL, N_TOKENS - 1)
    ids[~mask] = 0                           # pad_token_id
    return ids.astype(np.int64), mask


def windows_for_user(path):
    """Cut 60-second windows, inside each day so none spans a gap."""
    df = pd.read_csv(path)
    missing = [c for c in SIGNAL_COLS if c not in df.columns]
    if missing:
        print("    missing channels %s, skipped" % missing)
        return None, None

    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
    df = df.dropna(subset=[TIME_COL]).sort_values(TIME_COL)
    df = df.reset_index(drop=True)

    arr = df[SIGNAL_COLS].to_numpy(dtype=float)
    starts, rows = [], []
    for date, g in df.groupby("date", sort=True):
        lo, hi = g.index.min(), g.index.max() + 1
        for s in range(lo, hi - WINDOW + 1, STRIDE):
            starts.append(s)
            rows.append({"date": date, "window_start": int(s),
                         TIME_COL: df[TIME_COL].iloc[s]})
    if not starts:
        return None, None

    X = np.stack([arr[s:s + WINDOW] for s in starts])   # [n, WINDOW, 6]
    return X, pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(
        description="Embed the user windows with the pretrained Chronos T5 "
                    "encoder, the way SensorLLM does.")
    ap.add_argument("--signal-dir", default=SIGNAL_DIR)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--model", default="amazon/chronos-t5-small",
                    help="chronos-t5-mini or -tiny are smaller and faster")
    ap.add_argument("--tokeniser",
                    choices=["StanNormalizeUniformBins",
                             "MeanScaleUniformBins"],
                    default="StanNormalizeUniformBins",
                    help="SensorLLM's default is MeanScale, but that puts "
                         "98 percent of our skin temperature in one bin.")
    ap.add_argument("--pool", choices=["mean", "last"], default="mean",
                    help="encode() gives one vector per timestep; this is "
                         "how they are reduced to one per window")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--users", default="user17,user18,user25,user26,user27")
    args = ap.parse_args()

    try:
        from transformers import AutoModel
    except ImportError:
        raise SystemExit("transformers is not installed.\n"
                         "  pip install \"transformers>=4.30\"")

    os.makedirs(args.out_dir, exist_ok=True)
    users = [u.strip() for u in args.users.split(",") if u.strip()]

    print("=" * 72)
    print("Chronos T5 embeddings, the SensorLLM path")
    print("model     : %s" % args.model)
    print("tokeniser : %s" % args.tokeniser)
    print("pooling   : %s over the %d timesteps" % (args.pool, WINDOW))
    print("channels  : %d, each tokenised and embedded separately"
          % len(SIGNAL_COLS))
    print("=" * 72)
    print()
    print("Loading the encoder. First run downloads about 200 MB.")

    model = AutoModel.from_pretrained(args.model).encoder
    model.eval()
    d_model = model.config.d_model
    n_params = sum(p.numel() for p in model.parameters())
    print("  d_model %d, %.1fM parameters, frozen" % (d_model,
                                                      n_params / 1e6))
    print()

    summary, all_embed_stats = [], []

    for user in users:
        path = os.path.join(args.signal_dir, "%s.csv" % user)
        if not os.path.exists(path):
            print("%s: no 1 Hz file, skipped" % user)
            continue

        print("%s" % user)
        X, index = windows_for_user(path)
        if X is None:
            continue
        n = len(X)
        print("    %d windows" % n)

        # one embedding per channel, concatenated. SensorLLM tokenises each
        # channel separately too; their dataset loop is `for context in ts:`.
        per_channel = []
        for ci, col in enumerate(SIGNAL_COLS):
            ids, mask = tokenise_batch(X[:, :, ci], args.tokeniser)
            att = torch.tensor(mask.astype(np.int64))
            ids_t = torch.tensor(ids)

            outs = []
            with torch.no_grad():
                for b in range(0, n, args.batch):
                    h = model(input_ids=ids_t[b:b + args.batch],
                              attention_mask=att[b:b + args.batch]
                              ).last_hidden_state      # [B, WINDOW, d_model]
                    if args.pool == "mean":
                        m = att[b:b + args.batch].unsqueeze(-1).float()
                        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
                    else:
                        pooled = h[:, -1, :]
                    outs.append(pooled.numpy())
            emb = np.concatenate(outs).astype(np.float32)
            per_channel.append(emb)
            print("      %-12s %s" % (col, emb.shape))

        E = np.concatenate(per_channel, axis=1)
        np.save(os.path.join(args.out_dir, "%s_embed.npy" % user), E)
        index.to_csv(os.path.join(args.out_dir, "%s_index.csv" % user),
                     index=False)

        print("    saved %s_embed.npy  %s  (%.1f MB)"
              % (user, E.shape, E.nbytes / 1e6))
        print()
        summary.append({"user": user, "windows": int(n),
                        "dim": int(E.shape[1])})
        all_embed_stats.append(E)

    if not summary:
        raise SystemExit("Nothing produced.")

    E = np.concatenate(all_embed_stats)
    print("=" * 72)
    print("WHAT CAME OUT")
    print("=" * 72)
    print("  windows     : %d" % len(E))
    print("  dimensions  : %d  (%d channels x %d)"
          % (E.shape[1], len(SIGNAL_COLS), d_model))
    print("  values      : mean %.3f, sd %.3f, range %.2f to %.2f"
          % (E.mean(), E.std(), E.min(), E.max()))

    # a dead embedding would have almost no variation between windows
    per_dim_sd = E.std(axis=0)
    dead = int((per_dim_sd < 0.01).sum())
    print("  dimensions with almost no variation: %d of %d"
          % (dead, E.shape[1]))
    if dead > E.shape[1] * 0.5:
        print("    More than half carry nothing. The encoder is seeing")
        print("    very similar input for every window.")
    print()

    print("Next, train on these and compare against the other inputs:")
    print()
    print("  python3 btp/37_train_on_embeddings.py --embed-dir %s \\"
          % args.out_dir)
    print("      --run-id E01_random --split random --seeds 5")
    print()

    L = []
    L.append("# Chronos T5 embeddings")
    L.append("")
    L.append("Written by `btp/36_chronos_embeddings.py`, following the path")
    L.append("SensorLLM takes: tokenise, then run the pretrained Chronos T5")
    L.append("encoder and keep `last_hidden_state`.")
    L.append("")
    L.append("| setting | value |")
    L.append("|---|---|")
    L.append("| model | `%s` |" % args.model)
    L.append("| parameters | %.1fM, frozen |" % (n_params / 1e6))
    L.append("| tokeniser | `%s` |" % args.tokeniser)
    L.append("| pooling | %s over %d timesteps |" % (args.pool, WINDOW))
    L.append("| output | %d dimensions per window |" % E.shape[1])
    L.append("")
    L.append("## Why not their default tokeniser")
    L.append("")
    L.append("SensorLLM defaults to `MeanScaleUniformBins`, which suits")
    L.append("accelerometer data centred on zero. On our channels it does")
    L.append("not: skin temperature sits around 31 degrees, so dividing by")
    L.append("its own average gives about 1.0 every time and **98.2 percent")
    L.append("of values land in a single bin**. Pulse rate compresses to 413")
    L.append("bins of 4093. Subtracting the mean first gives 3341 and 2999.")
    L.append("")
    L.append("## Per user")
    L.append("")
    L.append("| user | windows | dimensions |")
    L.append("|---|---:|---:|")
    for r in summary:
        L.append("| %s | %d | %d |" % (r["user"], r["windows"], r["dim"]))
    L.append("")
    L.append("## What to expect from this")
    L.append("")
    L.append("The SensorLLM paper runs Chronos embeddings with an MLP head as")
    L.append("a **baseline**, and it came out worst or nearly worst on both")
    L.append("datasets: 44.2 F1 on CAPTURE-24 against 60.2 for the best, and")
    L.append("82.2 on UCI-HAR against 93.2. They write that \"Chronos")
    L.append("embeddings alone are of limited utility\", and that they chose")
    L.append("Chronos precisely because it had not been pretrained on motion")
    L.append("sensor data.")
    L.append("")
    L.append("So this is their control condition rather than their result.")
    L.append("")
    L.append("There is a second reason for caution. Tokenisation normalises")
    L.append("each window on its own, which removes the level that separates")
    L.append("one window from another. We measured that cost directly:")
    L.append("dequantised skin temperature correlated with the original at")
    L.append("-0.021, and training on those signals lost about 8 points of")
    L.append("balanced accuracy on every split. The T5 encoder reads the same")
    L.append("normalised tokens, so that information is gone before it")
    L.append("starts.")
    L.append("")

    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    open(REPORT, "w").write("\n".join(L) + "\n")
    json.dump({"model": args.model, "tokeniser": args.tokeniser,
               "pool": args.pool, "d_model": int(d_model),
               "embed_dim": int(E.shape[1]),
               "dead_dimensions": dead, "per_user": summary},
              open(os.path.join(args.out_dir, "embed_stats.json"), "w"),
              indent=2)
    print("report: %s" % REPORT)


if __name__ == "__main__":
    main()
