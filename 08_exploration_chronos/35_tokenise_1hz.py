"""
35_tokenise_1hz.py

WHAT THIS DOES
    Applies Chronos tokenisation to the 1 Hz signal channels, and writes the
    result back out as floats so the existing frozen encoder can read it.

    Two files per user:

      <user>_tokens.npz      the integer token IDs, 2 to 4095
      <user>_dequant.csv     the same thing turned back into numbers, in the
                             exact shape of data/processed/user_6ch/<user>.csv

    The second is the one the model trains on. The first is kept because the
    token IDs are what a language-model-style approach would consume, and
    because counting distinct tokens per channel is a useful measurement in
    its own right.

WHY WE CONVERT BACK TO FLOATS
    The repository's encoder starts with Conv1d(6, 64, kernel=7). It expects
    six continuous channels. Token IDs are integers from a 4096-word
    vocabulary, and feeding them directly would need an embedding layer in
    front, which changes the input width and means the WESAD-trained encoder
    can no longer be used.

    So this script does the honest version of the experiment: tokenise, then
    map each token back to the centre of its bin. The signal keeps Chronos's
    per-window normalisation and its 4096-level quantisation, but stays six
    float channels. Exactly one thing changes, and the comparison against the
    existing results is like for like.

    The version with an embedding layer is a separate experiment, and it
    needs the self-supervised stage re-run from scratch.

WHICH TOKENISER
    StanNormalizeUniformBins by default: subtract the mean, divide by the
    standard deviation, then bin.

    Measured on the raw user data, upstream Chronos's mean scaling put
    98.2 percent of skin temperature into a single bin, because the channel
    sits around 31 degrees and dividing by its own average size gives about
    1.0 every time. Pulse rate had the same problem around 80 bpm. Subtracting
    the mean first fixed both. This is very likely why SensorLLM added the
    second tokeniser, which upstream Chronos does not ship.

    --method MeanScaleUniformBins runs the upstream version for comparison.

WHAT IS KEPT FROM SENSORLLM, EXACTLY
    Verified against their source, token for token:

      centers    = linspace(-15, 15, 4096 - 2 - 1)
      boundaries = [-1e20, midpoints of centers, +1e20]
      token_id   = bucketize(scaled, boundaries, right=True) + 2

    0 is PAD and 1 is EOS, so real values start at 2. Each channel is
    tokenised on its own, as they do.

ONE LIMITATION, STATED UP FRONT
    The 1 Hz channels have already lost the pulse waveform. Measured earlier:
    raw 64 Hz BVP uses 4,058 of the 4,094 available tokens, and the same
    signal averaged to one value a second uses about 10. Tokenising will not
    bring that back.

    It does not disadvantage this experiment, because the encoder only ever
    reads 1 Hz anyway. But it does mean a low token count for BVP here is
    expected, and is a property of the averaging rather than of Chronos.

HOW TO RUN
    From the AdaStress_BTP folder:

        python3 btp/35_tokenise_1hz.py
        python3 btp/35_tokenise_1hz.py --method MeanScaleUniformBins

    A minute or two.
"""

import os
import glob
import json
import argparse
import numpy as np
import pandas as pd

SIGNAL_DIR = "data/processed/user_6ch"
OUT_DIR = "data/processed/tokens_1hz"
REPORT = "docs/58_tokens_1hz.md"

# Chronos vocabulary, exactly as the released config has it.
N_TOKENS = 4096
N_SPECIAL = 2           # 0 is PAD, 1 is EOS, real values start at 2
LOW, HIGH = -15.0, 15.0

WINDOW = 60             # the model's window, in seconds at 1 Hz
STRIDE = 30

SIGNAL_COLS = ["ACC_X_wrist", "ACC_Y_wrist", "ACC_Z_wrist",
               "BVP", "EDA_wrist", "Temp_wrist"]

TIME_COL = "datetime_ist"

# The five labelled users. The others have no survey answers, so nothing can
# be trained or measured on them.
LABELLED = ["user17", "user18", "user25", "user26", "user27"]


def make_bins():
    """Chronos's bin centres and boundaries, as in their source."""
    centers = np.linspace(LOW, HIGH, N_TOKENS - N_SPECIAL - 1)
    boundaries = np.concatenate([[-1e20],
                                 (centers[1:] + centers[:-1]) / 2,
                                 [1e20]])
    return centers, boundaries


CENTERS, BOUNDARIES = make_bins()


def tokenise(x, method):
    """One window of one channel to token IDs.

    Both methods scale by something worked out from the window itself, which
    is what makes the result independent of the device's units.
    """
    x = np.asarray(x, dtype=float)
    mask = ~np.isnan(x)
    if not mask.any():
        return np.zeros(len(x), dtype=np.int16), (0.0, 1.0)

    if method == "MeanScaleUniformBins":
        scale = float(np.nansum(np.abs(x) * mask) / np.nansum(mask))
        scale = scale if scale > 0 else 1.0
        scaled = np.where(mask, x / scale, 0.0)
        state = (0.0, scale)
    else:
        mu = float(np.nansum(x * mask) / np.nansum(mask))
        var = float(np.nansum((x - mu) ** 2 * mask) / np.nansum(mask))
        sd = float(np.sqrt(var + 1e-8))
        scaled = np.where(mask, (x - mu) / sd, 0.0)
        state = (mu, sd)

    # bucketize(..., right=True) is searchsorted with side='right'
    ids = np.searchsorted(BOUNDARIES, scaled, side="right") + N_SPECIAL
    ids = np.clip(ids, N_SPECIAL, N_TOKENS - 1)
    ids[~mask] = 0                                    # PAD
    return ids.astype(np.int16), state


def dequantise(ids, state, method, keep_normalised=True):
    """Token IDs back to numbers.

    Two ways, and only one of them is an experiment.

    keep_normalised=True   stop at the bin centre. Each window is left on
                           its own normalised scale, which is what Chronos
                           actually does to a series. THIS is the change we
                           are testing.

    keep_normalised=False  multiply back by the scale and add the mean, as
                           their output_transform does. This puts the signal
                           back exactly where it started.

    The second was the first thing this script did, and testing showed why
    that was wrong: the dequantised signal correlated with the original at
    1.000 on every channel. With 4,093 bins spread over plus or minus 15
    standard deviations, the binning is far too fine to change anything. The
    only thing Chronos really does to a window is normalise it, and undoing
    that leaves nothing behind.

    So the default keeps the normalised values. Each 60 second window then
    arrives at the encoder centred on zero with unit spread, per channel,
    instead of carrying whatever the global WESAD scaler produced. That is a
    real difference and worth measuring.
    """
    idx = np.clip(ids.astype(int) - N_SPECIAL - 1, 0, len(CENTERS) - 1)
    vals = CENTERS[idx]
    if keep_normalised:
        return vals
    mu, sd = state
    if method == "MeanScaleUniformBins":
        return vals * sd
    return vals * sd + mu


def process_user(user, path, method, keep_normalised=True):
    """Tokenise one user's 1 Hz file, window by window."""
    df = pd.read_csv(path)
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
    df = df.dropna(subset=[TIME_COL]).sort_values(TIME_COL).reset_index(drop=True)

    missing = [c for c in SIGNAL_COLS if c not in df.columns]
    if missing:
        print("    missing channels: %s" % missing)
        return None, None, None

    out = df.copy()
    tokens = {c: [] for c in SIGNAL_COLS}
    stats = {c: {"windows": 0, "distinct": set(), "pad": 0}
             for c in SIGNAL_COLS}

    # Windows move 30 seconds at a time, so each second belongs to two of
    # them. We tokenise each window on its own, then write back the values
    # from whichever window covers that second last. The alternative,
    # averaging the overlap, would blur the quantisation we are trying to
    # measure.
    for _date, g in df.groupby("date", sort=True):
        idx = g.index.to_numpy()
        arr = g[SIGNAL_COLS].to_numpy(dtype=float)
        n = len(idx)

        for s in range(0, n - WINDOW + 1, STRIDE):
            block = arr[s:s + WINDOW]
            for ci, col in enumerate(SIGNAL_COLS):
                ids, state = tokenise(block[:, ci], method)
                tokens[col].append(ids)
                out.loc[idx[s:s + WINDOW], col] = dequantise(
                    ids, state, method, keep_normalised)
                stats[col]["windows"] += 1
                stats[col]["distinct"].update(np.unique(ids[ids >= N_SPECIAL])
                                              .tolist())
                stats[col]["pad"] += int((ids < N_SPECIAL).sum())

    if not tokens[SIGNAL_COLS[0]]:
        return None, None, None

    packed = {c: np.stack(v).astype(np.int16) for c, v in tokens.items()}
    return out, packed, stats


def main():
    ap = argparse.ArgumentParser(
        description="Chronos tokenisation of the 1 Hz channels, written back "
                    "as floats so the frozen encoder can read them.")
    ap.add_argument("--signal-dir", default=SIGNAL_DIR)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--method", default="StanNormalizeUniformBins",
                    choices=["StanNormalizeUniformBins",
                             "MeanScaleUniformBins"])
    ap.add_argument("--users", default=",".join(LABELLED))
    ap.add_argument("--undo-scaling", action="store_true",
                    help="put the values back on their original scale after "
                         "binning. Off by default, because doing so makes "
                         "the output correlate with the input at 1.000 and "
                         "the experiment measures nothing. Left in so that "
                         "can be shown rather than asserted.")
    args = ap.parse_args()
    keep_norm = not args.undo_scaling

    os.makedirs(args.out_dir, exist_ok=True)
    users = [u.strip() for u in args.users.split(",") if u.strip()]

    print("=" * 70)
    print("Chronos tokenisation of the 1 Hz channels")
    print("method : %s" % args.method)
    print("vocab  : %d tokens, %d reserved, real values from %d"
          % (N_TOKENS, N_SPECIAL, N_SPECIAL))
    print("window : %d s, stride %d s" % (WINDOW, STRIDE))
    print("values : %s" % ("kept on Chronos's normalised scale, per window"
                           if keep_norm else
                           "put back on the original scale (changes almost "
                           "nothing)"))
    print("=" * 70)
    print()

    all_stats, summary = {}, []

    for user in users:
        path = os.path.join(args.signal_dir, "%s.csv" % user)
        if not os.path.exists(path):
            print("%s: no 1 Hz file" % user)
            continue

        print("%s" % user)
        deq, packed, stats = process_user(user, path, args.method, keep_norm)
        if deq is None:
            print("    nothing produced")
            print()
            continue

        dpath = os.path.join(args.out_dir, "%s_dequant.csv" % user)
        tpath = os.path.join(args.out_dir, "%s_tokens.npz" % user)
        deq.to_csv(dpath, index=False)
        np.savez_compressed(tpath, **packed)

        nwin = stats[SIGNAL_COLS[0]]["windows"]
        print("    %d rows, %d windows" % (len(deq), nwin))
        print("    saved %s (%.1f MB) and %s (%.1f MB)"
              % (os.path.basename(dpath), os.path.getsize(dpath) / 1e6,
                 os.path.basename(tpath), os.path.getsize(tpath) / 1e6))
        print()

        all_stats[user] = stats
        summary.append({"user": user, "rows": len(deq), "windows": nwin})

    if not all_stats:
        raise SystemExit("Nothing produced.")

    # ---- the measurement ----
    print("=" * 70)
    print("DISTINCT TOKENS USED PER CHANNEL, OF 4,093 POSSIBLE")
    print("=" * 70)
    print("A channel using a handful has nothing in it to read. One using")
    print("hundreds is alive.")
    print()
    print("  %-14s %10s %10s %10s  %s"
          % ("channel", "windows", "distinct", "pad", "verdict"))
    print("  " + "-" * 62)

    merged = {}
    for col in SIGNAL_COLS:
        w = sum(all_stats[u][col]["windows"] for u in all_stats)
        d = set()
        for u in all_stats:
            d |= all_stats[u][col]["distinct"]
        p = sum(all_stats[u][col]["pad"] for u in all_stats)
        verdict = ("DEAD" if len(d) < 20 else
                   "weak" if len(d) < 100 else "alive")
        merged[col] = {"windows": w, "distinct": len(d), "pad": p,
                       "verdict": verdict}
        print("  %-14s %10d %10d %10d  %s" % (col, w, len(d), p, verdict))
    print()

    bvp = merged.get("BVP", {}).get("distinct", 0)
    print("For comparison, the same BVP at its raw 64 Hz rate used 4,058")
    print("distinct tokens. Here it uses %d. The averaging to 1 Hz is what" % bvp)
    print("costs that, not the tokenisation: a pulse swings above and below")
    print("zero every beat, so a one-second mean cancels it.")
    print()

    # ---- what changed in the numbers ----
    print("=" * 70)
    print("WHAT QUANTISATION DID TO THE VALUES")
    print("=" * 70)
    print("Comparing each dequantised channel against the original, so the")
    print("size of the change is visible rather than assumed.")
    print()
    print("  %-14s %12s %12s %10s" % ("channel", "orig sd", "dequant sd",
                                      "correlation"))
    print("  " + "-" * 52)
    for col in SIGNAL_COLS:
        o, q = [], []
        for user in all_stats:
            a = pd.read_csv(os.path.join(args.signal_dir, "%s.csv" % user),
                            usecols=[col])[col].to_numpy(dtype=float)
            b = pd.read_csv(os.path.join(args.out_dir,
                                         "%s_dequant.csv" % user),
                            usecols=[col])[col].to_numpy(dtype=float)
            m = min(len(a), len(b))
            o.append(a[:m])
            q.append(b[:m])
        o, q = np.concatenate(o), np.concatenate(q)
        ok = ~np.isnan(o) & ~np.isnan(q)
        r = float(np.corrcoef(o[ok], q[ok])[0, 1]) if ok.sum() > 10 else np.nan
        print("  %-14s %12.4f %12.4f %10.3f"
              % (col, np.nanstd(o), np.nanstd(q), r))
    print()
    if keep_norm:
        print("The spread should now be about 1.0 on every channel: each")
        print("window has been centred and scaled on its own. The correlation")
        print("shows how much of the shape survived, and should stay high.")
        print("That is the change being tested: per-window normalisation")
        print("instead of one global scaler.")
    else:
        print("Correlation near 1.000 on every channel means this run changed")
        print("essentially nothing. 4,093 bins over plus or minus 15 standard")
        print("deviations is too fine to matter, and un-scaling undid the one")
        print("thing that would have. Drop --undo-scaling to run the real")
        print("experiment.")
    print()

    # ---- report ----
    L = []
    L.append("# Chronos tokens from the 1 Hz channels")
    L.append("")
    L.append("Written by `btp/35_tokenise_1hz.py`.")
    L.append("")
    L.append("| setting | value |")
    L.append("|---|---|")
    L.append("| method | `%s` |" % args.method)
    L.append("| vocabulary | %d tokens, %d reserved |" % (N_TOKENS, N_SPECIAL))
    L.append("| window | %d s, stride %d s |" % (WINDOW, STRIDE))
    L.append("| users | %s |" % ", ".join(sorted(all_stats)))
    L.append("| values | %s |" % ("normalised per window"
                                  if keep_norm else "original scale"))
    L.append("")
    L.append("## Why the tokens are converted back to floats")
    L.append("")
    L.append("The encoder starts with `Conv1d(6, 64, kernel=7)` and expects")
    L.append("six continuous channels. Feeding token IDs directly would need")
    L.append("an embedding layer, which changes the input width and means the")
    L.append("WESAD-trained encoder could not be reused.")
    L.append("")
    L.append("So the signal is tokenised and then mapped back to the centre of")
    L.append("each bin. It keeps Chronos's per-window normalisation and its")
    L.append("4096-level quantisation, and stays six float channels, so")
    L.append("exactly one thing changes against the existing results.")
    L.append("")
    L.append("## Distinct tokens per channel")
    L.append("")
    L.append("| channel | windows | distinct | pad | verdict |")
    L.append("|---|---:|---:|---:|---|")
    for col, s in merged.items():
        L.append("| `%s` | %d | %d | %d | %s |"
                 % (col, s["windows"], s["distinct"], s["pad"], s["verdict"]))
    L.append("")
    L.append("The same BVP at its raw 64 Hz rate used 4,058 distinct tokens.")
    L.append("Here it uses %d. That is the cost of averaging to 1 Hz, not of" % bvp)
    L.append("tokenisation: a pulse swings above and below zero every beat, so")
    L.append("a one-second mean cancels it.")
    L.append("")
    L.append("## Per user")
    L.append("")
    L.append("| user | rows | windows |")
    L.append("|---|---:|---:|")
    for r in summary:
        L.append("| %s | %d | %d |" % (r["user"], r["rows"], r["windows"]))
    L.append("")

    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    open(REPORT, "w").write("\n".join(L) + "\n")
    json.dump({"method": args.method, "channels": merged, "users": summary},
              open(os.path.join(args.out_dir, "token_stats.json"), "w"),
              indent=2)

    print("report: %s" % REPORT)
    print()
    print("Next: train on the dequantised files and compare.")
    print("  python3 btp/34_train_users.py --config configs/w6.yaml \\")
    print("      --run-id T01_tok_random --context off --split random "
          "--seeds 5 \\")
    print("      --signal-dir %s --signal-suffix _dequant" % args.out_dir)


if __name__ == "__main__":
    main()
