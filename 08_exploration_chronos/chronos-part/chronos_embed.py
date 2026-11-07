#!/usr/bin/env python3
"""
Run the (frozen) Chronos encoder on the 30 s / 16 Hz windows from make_windows_16hz.py.

Input : windows_16hz.npz  (X: N x 480 x 6, channels = acc_x, acc_y, acc_z, bvp, eda, temp)
Output: one .npy per channel in --out:
            emb_<channel>.npy  shape (N, 481, d_model)  float16   (480 tokens + 1 end token)
        plus  window_start_utc.npy  (same order)
        --pool also writes emb_<channel>_mean.npy (N, d_model): mean over the 481 tokens (small, for quick checks)

Usage:
    pip install chronos-forecasting torch
    python chronos_embed.py --npz windows_16hz.npz --out emb --model amazon/chronos-t5-small
    python chronos_embed.py --npz windows_16hz.npz --out emb --model amazon/chronos-t5-large --channels eda bvp acc_x --batch 16

Size warning: large model = 481 x 1024 x 2 bytes ~ 1 MB per window per channel. Use --channels / --limit, or --pool.
"""
import argparse, os
import numpy as np
import torch

CH = ["acc_x", "acc_y", "acc_z", "bvp", "eda", "temp", "resp"]

# resp is added by add_resp_channel.py. It is the band-passed
# beat-interval series, which oscillates at the breathing
# frequency, rather than a rate in breaths per minute. A rate
# repeated 480 times is a flat line, and a flat line gives the
# tokeniser nothing to work with.


def zscore(x, eps=1e-6):
    """Per-window z-score, NaN-aware (as SensorLLM does per segment). Flat window -> zeros."""
    m = np.nanmean(x, axis=1, keepdims=True)
    s = np.nanstd(x, axis=1, keepdims=True)
    s = np.where(s < eps, 1.0, s)
    return (x - m) / s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="amazon/chronos-t5-small")
    ap.add_argument("--channels", nargs="+", default=CH, choices=CH)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None, help="only the first N windows (for testing)")
    ap.add_argument("--no-zscore", action="store_true")
    ap.add_argument("--pool", action="store_true", help="also save mean-pooled (N, d_model)")
    a = ap.parse_args()

    from chronos import ChronosPipeline

    z = np.load(a.npz, allow_pickle=False)
    X = z["X"]
    names = list(z["channels"])
    if a.limit:
        X = X[: a.limit]
    N = len(X)
    os.makedirs(a.out, exist_ok=True)
    np.save(os.path.join(a.out, "window_start_utc.npy"), z["window_start_utc"][:N])

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if dev == "cuda" else torch.float32
    pipe = ChronosPipeline.from_pretrained(a.model, device_map=dev, torch_dtype=dtype)
    print(f"model={a.model} device={dev} windows={N}")

    for ch in a.channels:
        c = names.index(ch)
        x = X[:, :, c].astype(np.float32)
        if not a.no_zscore:
            x = zscore(x)
        path = os.path.join(a.out, f"emb_{ch}.npy")
        mm = None
        pooled = []
        for i in range(0, N, a.batch):
            ctx = torch.tensor(x[i:i + a.batch])                      # (B, 480)  NaN allowed (masked by tokenizer)
            with torch.no_grad():
                emb, _ = pipe.embed(ctx)                              # (B, 481, d_model)
            emb = emb.float().cpu().numpy()
            if mm is None:
                mm = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=(N,) + emb.shape[1:])
            mm[i:i + len(emb)] = emb.astype(np.float16)
            if a.pool:
                pooled.append(emb.mean(axis=1))
        mm.flush()
        if a.pool:
            np.save(os.path.join(a.out, f"emb_{ch}_mean.npy"), np.concatenate(pooled))
        print(f"  {ch}: saved {path} {mm.shape}")


if __name__ == "__main__":
    main()
