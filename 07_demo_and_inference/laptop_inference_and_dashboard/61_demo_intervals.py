"""
61_demo_intervals.py

WHAT THIS DOES
    Takes the scored demo sessions (runs/<run-id>/predictions.csv, from
    btp/45_score_all_with_context.py) and reports the stress score at fixed
    checkpoints, the way the mentor asked:

        session longer than 130 min  ->  a checkpoint every 15 minutes
        session of 100 min or less   ->  a checkpoint every  5 minutes
        in between (101-130 min)     ->  every 10 minutes   [our choice,
                                         the mentor did not say]

    DEFAULT  --mode point
        The score AT each checkpoint: the 60-second window that starts at
        19:05, 19:10, 19:15 ... and its own probabilities, state, posture and
        activity. Nothing is averaged. The first checkpoint is the session
        start, and the last is the last checkpoint that still has a full
        window.

    OPTIONAL --mode mean
        The mean of the window scores inside each interval, as an earlier
        version did. Smoother, and hides short swings.

    A single 60 s window is noisy: the grey curve in the figure shows the
    score moving a lot between neighbouring windows, so a checkpoint score
    can sit well away from the scores around it. That is what the point mode
    shows, and is why the mean mode exists.

WHAT IT DOES NOT DO
    These sessions have no survey answers, so nothing here is accuracy. The
    stress score is ours: 1 x P(Medium) + 2 x P(High), range 0 to 2.

CROSS-CHECKS (printed, and saved to interval_checks.csv)
    probabilities sum to 1; the score really is P(Medium)+2P(High); in 0-2
    windows 30 s apart with no gap
    first/last window match the mentor's slot in sessions.csv (UTC -> IST)
    window count matches what the slot length implies
    the 1 Hz signal file has the slot's number of seconds, few missing
    values, and sensible channel ranges
    point mode: every checkpoint is one real window and its score equals the
    value looked up straight from the predictions file; no checkpoint missing
    mean mode: each interval equals an independent pandas resample
    A FAIL means do not use that session's numbers until it is understood.

OUTPUT (in --out, default runs/DEMO_SESSIONS/)
    interval_scores.csv, interval_summary.csv, interval_checks.csv,
    interval_scores.png

HOW TO RUN (project root)
    python3 btp/61_demo_intervals.py
    python3 btp/61_demo_intervals.py --mode mean
"""
import argparse
import os

import numpy as np
import pandas as pd

STATES = ["Low", "Medium", "High"]
SIGNALS = ["ACC_X_wrist", "ACC_Y_wrist", "ACC_Z_wrist", "BVP", "EDA_wrist",
           "Temp_wrist", "RESP"]
RANGES = {"ACC_X_wrist": (-3, 3), "ACC_Y_wrist": (-3, 3),
          "ACC_Z_wrist": (-3, 3), "Temp_wrist": (25, 40), "RESP": (4, 40),
          "EDA_wrist": (-1, 60), "BVP": (-500, 500)}
IST = pd.Timedelta(hours=5, minutes=30)


def interval_minutes(length_min, short, long_, mid):
    if length_min > 130:
        return long_
    if length_min <= 100:
        return short
    return mid


def mode_of(s):
    s = s.dropna().astype(str)
    return s.value_counts().index[0] if len(s) else "-"


def hhmm(t):
    return t.strftime("%H:%M")


def build(pred, short, long_, mid, mode="point", min_windows=3):
    pred = pred.copy()
    pred["datetime_ist"] = pd.to_datetime(pred["datetime_ist"])
    rows, summ = [], []
    for sid, g in pred.sort_values("datetime_ist").groupby("Subject_ID",
                                                           sort=False):
        t0 = g["datetime_ist"].iloc[0]
        last = g["datetime_ist"].iloc[-1]
        length = (last + pd.Timedelta(seconds=60) - t0).total_seconds() / 60.0
        step = interval_minutes(length, short, long_, mid)
        n0 = len(rows)

        if mode == "point":
            by_t = g.set_index("datetime_ist")
            k = 0
            while t0 + pd.Timedelta(minutes=k * step) <= last:
                tc = t0 + pd.Timedelta(minutes=k * step)
                if tc in by_t.index:
                    w = by_t.loc[tc]
                    pm = w[["P_Low", "P_Medium", "P_High"]].astype(float)
                    rows.append({
                        "session": sid, "interval_min": step, "block": k + 1,
                        "time_ist": hhmm(tc),
                        "window_covers": "%s-%s" % (hhmm(tc),
                                                   hhmm(tc + pd.Timedelta(seconds=60))),
                        "windows": 1,
                        "score_0_2": float(w["stress_score"]),
                        "P_Low": float(pm["P_Low"]),
                        "P_Medium": float(pm["P_Medium"]),
                        "P_High": float(pm["P_High"]),
                        "state": STATES[int(np.argmax(pm.to_numpy()))],
                        "uncertainty": float(w["uncertainty"]),
                        "posture": str(w["POSTURE"]),
                        "activity": str(w["ACT_CLASS"]),
                        "act_count": float(pd.to_numeric(w["ACT_COUNT"],
                                                         errors="coerce")),
                        "t_minutes": k * step})
                else:
                    print("  %s: no window starts at %s, checkpoint skipped"
                          % (sid, hhmm(tc)))
                k += 1
        else:
            g = g.assign(bin=((g["datetime_ist"] - t0).dt.total_seconds()
                              // (60 * step)).astype(int))
            for b, h in g.groupby("bin"):
                if len(h) < min_windows:
                    continue
                start = t0 + pd.Timedelta(minutes=int(b) * step)
                pm = h[["P_Low", "P_Medium", "P_High"]].mean()
                rows.append({
                    "session": sid, "interval_min": step, "block": int(b) + 1,
                    "time_ist": hhmm(start),
                    "window_covers": "%s-%s" % (hhmm(start), hhmm(
                        start + pd.Timedelta(minutes=step))),
                    "windows": len(h),
                    "score_0_2": float(h["stress_score"].mean()),
                    "P_Low": float(pm["P_Low"]),
                    "P_Medium": float(pm["P_Medium"]),
                    "P_High": float(pm["P_High"]),
                    "state": STATES[int(np.argmax(pm.to_numpy()))],
                    "uncertainty": float(h["uncertainty"].mean()),
                    "posture": mode_of(h["POSTURE"]),
                    "activity": mode_of(h["ACT_CLASS"]),
                    "act_count": float(pd.to_numeric(h["ACT_COUNT"],
                                                     errors="coerce").mean()),
                    "t_minutes": int(b) * step})
        sc = [r["score_0_2"] for r in rows[n0:]]
        summ.append({"session": sid, "length_min": round(length, 1),
                     "interval_min": step, "mode": mode, "points": len(sc),
                     "mean": float(np.mean(sc)), "min": float(np.min(sc)),
                     "max": float(np.max(sc)),
                     "range": float(np.max(sc) - np.min(sc))})
    return pd.DataFrame(rows), pd.DataFrame(summ)


def run_checks(pred, iv, sessions_path, signal_dir, min_windows, mode):
    out = []

    def add(sess, name, ok, detail, warn=False):
        out.append({"session": sess, "check": name,
                    "status": "PASS" if ok else ("WARN" if warn else "FAIL"),
                    "detail": detail})

    slots = None
    if sessions_path and os.path.exists(sessions_path):
        slots = pd.read_csv(sessions_path, parse_dates=["start_utc", "end_utc"])
        slots["start_ist"] = slots["start_utc"] + IST
        slots["end_ist"] = slots["end_utc"] + IST

    p = pred.copy()
    p["datetime_ist"] = pd.to_datetime(p["datetime_ist"])
    for sid, g in p.sort_values("datetime_ist").groupby("Subject_ID",
                                                        sort=False):
        t = g["datetime_ist"]
        ps = g[["P_Low", "P_Medium", "P_High"]].sum(axis=1)
        add(sid, "probabilities sum to 1", (ps - 1).abs().max() < 1e-3,
            "worst |sum-1| = %.2e" % (ps - 1).abs().max())
        d = (g["P_Medium"] + 2 * g["P_High"] - g["stress_score"]).abs().max()
        add(sid, "score = P(Medium)+2P(High)", d < 1e-4, "worst diff %.2e" % d)
        add(sid, "score within 0-2",
            g["stress_score"].between(-1e-6, 2 + 1e-6).all(),
            "min %.3f, max %.3f" % (g["stress_score"].min(),
                                    g["stress_score"].max()))
        gaps = int((t.diff().dropna() != pd.Timedelta(seconds=30)).sum())
        add(sid, "windows 30 s apart, no gap", gaps == 0,
            "%d irregular steps in %d windows" % (gaps, len(g)))

        v = iv[iv["session"] == sid]
        step = int(v["interval_min"].iloc[0])
        t0 = t.iloc[0]
        if mode == "point":
            exp_n = int((t.iloc[-1] - t0).total_seconds() // (60 * step)) + 1
            add(sid, "no checkpoint missing", len(v) == exp_n,
                "%d checkpoints, expected %d" % (len(v), exp_n))
            lookup = g.set_index("datetime_ist")["stress_score"]
            worst = 0.0
            for _, r in v.iterrows():
                tc = t0 + pd.Timedelta(minutes=int(r["t_minutes"]))
                worst = max(worst, abs(float(lookup.loc[tc]) - r["score_0_2"]))
            add(sid, "checkpoint score = value in predictions file",
                worst < 1e-9, "worst diff %.2e over %d checkpoints"
                % (worst, len(v)))
            on_grid = all(int(m) % step == 0 for m in v["t_minutes"])
            add(sid, "checkpoints exactly %d min apart" % step, on_grid,
                "from %s, every %d min" % (hhmm(t0), step))
        else:
            r = (g.set_index("datetime_ist")["stress_score"]
                 .resample("%dmin" % step, origin=t0).agg(["mean", "size"]))
            r = r[r["size"] >= min_windows]
            diff = (np.abs(r["mean"].to_numpy() - v["score_0_2"].to_numpy())
                    if len(r) == len(v) else np.array([np.inf]))
            add(sid, "interval scores = independent resample",
                diff.max() < 1e-6, "%d blocks, worst diff %.2e"
                % (len(v), diff.max()))
            kept = int(v["windows"].sum())
            add(sid, "no window lost between file and table",
                kept <= len(g) and len(g) - kept < min_windows * 2,
                "%d of %d windows in blocks" % (kept, len(g)))

        name_user, mmdd = sid.rsplit("_", 1)
        m = None
        if slots is not None:
            m = slots[(slots["user"] == name_user)
                      & (slots["start_ist"].dt.strftime("%m%d") == mmdd)]
            if len(m):
                row = m.iloc[0]
                slot_s = (row["end_ist"] - row["start_ist"]).total_seconds()
                add(sid, "first window = slot start", t0 == row["start_ist"],
                    "window %s, slot %s IST" % (t0.strftime("%H:%M:%S"),
                                               row["start_ist"].strftime("%H:%M:%S")))
                last_end = t.iloc[-1] + pd.Timedelta(seconds=60)
                add(sid, "last window ends at slot end",
                    abs((row["end_ist"] - last_end).total_seconds()) <= 30,
                    "ends %s, slot %s IST" % (last_end.strftime("%H:%M:%S"),
                                              row["end_ist"].strftime("%H:%M:%S")))
                exp = int((slot_s - 60) // 30 + 1)
                add(sid, "window count matches slot length", len(g) == exp,
                    "%d windows, slot implies %d" % (len(g), exp))
            else:
                add(sid, "slot found in sessions.csv", False,
                    "no row for %s on %s" % (name_user, mmdd), warn=True)

        sp = os.path.join(signal_dir, sid + ".csv") if signal_dir else None
        if sp and os.path.exists(sp):
            sg = pd.read_csv(sp)
            if m is not None and len(m):
                add(sid, "signal file seconds = slot seconds",
                    len(sg) == int(slot_s),
                    "%d rows, slot %d s" % (len(sg), slot_s))
            miss = sg[SIGNALS].isna().mean() * 100
            add(sid, "signal missing values small (<2%)", miss.max() < 2,
                "worst %s %.1f%%" % (miss.idxmax(), miss.max()), warn=True)
            bad = [c for c, (lo, hi) in RANGES.items()
                   if not sg[c].dropna().between(lo, hi).all()]
            add(sid, "signal ranges sensible", not bad,
                "outside range: %s" % (", ".join(bad) if bad else "none"))
        elif sp:
            add(sid, "signal file present", False, sp + " not found", warn=True)
    return pd.DataFrame(out)


def plot(pred, iv, out, mode):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False})
    col = {"Low": "#4C9F70", "Medium": "#E0A030", "High": "#C8553D"}
    pcol = {"sitting": "#9AA0A6", "standing": "#2F6DB5", "walking": "#7B4EA3"}
    order = list(dict.fromkeys(iv["session"]))
    n = len(order)
    fig = plt.figure(figsize=(11.5, 4.6 * n))
    gs = GridSpec(2 * n, 1, figure=fig, height_ratios=[3, 1.15] * n,
                  hspace=0.62)
    for k, sid in enumerate(order):
        g = pred[pred["Subject_ID"] == sid].copy()
        g["t"] = pd.to_datetime(g["datetime_ist"])
        t0 = g["t"].min()
        g["m"] = (g["t"] - t0).dt.total_seconds() / 60.0
        v = iv[iv["session"] == sid].reset_index(drop=True)
        step = int(v["interval_min"].iloc[0])
        pt = mode == "point"
        xs = v["t_minutes"].to_numpy(dtype=float)
        xc = xs if pt else xs + step / 2          # where the marker sits
        xmax = float(g["m"].max()) + 1 if pt else xs[-1] + step
        top = fig.add_subplot(gs[2 * k])
        bot = fig.add_subplot(gs[2 * k + 1], sharex=top)

        top.plot(g["m"], g["stress_score"], color="#C9CDD2", lw=1,
                 label="every 60 s window")
        if pt:
            top.plot(xc, v["score_0_2"], color="k", lw=1.2, alpha=.55)
        else:
            top.step(list(xs) + [xmax],
                     list(v["score_0_2"]) + [v["score_0_2"].iloc[-1]],
                     where="post", color="k", lw=1.6)
        for x, s_, st in zip(xc, v["score_0_2"], v["state"]):
            top.scatter(x, s_, s=80 if pt else 70, color=col[st], zorder=3,
                        edgecolor="white")
            top.text(x, s_ + 0.09, "%.2f" % s_, ha="center", fontsize=9)
        ymax = max(0.6, float(max(v["score_0_2"].max(),
                                  g["stress_score"].max())) * 1.15 + 0.1)
        top.set_ylim(0, min(2.3, max(ymax, 0.6) + 0.1))
        top.set_ylabel("score (0-2)")
        what = ("score at each %d-min checkpoint" % step if pt
                else "mean over each %d min" % step)
        top.set_title("%s    %s-%s IST    %s"
                      % (sid, v["time_ist"].iloc[0],
                         (t0 + pd.Timedelta(minutes=xmax)).strftime("%H:%M"),
                         what), loc="left", fontsize=11.5)
        plt.setp(top.get_xticklabels(), visible=False)

        width = step * 0.8 if pt else step * 0.94
        left = xs - width / 2 if pt else xs
        bottom = np.zeros(len(v))
        for name, c in (("P_Low", col["Low"]), ("P_Medium", col["Medium"]),
                        ("P_High", col["High"])):
            bot.bar(left, v[name], width=width, bottom=bottom, align="edge",
                    color=c, edgecolor="white", lw=.4)
            bottom += v[name].to_numpy()
        for x, ps in zip(left, v["posture"]):
            bot.add_patch(plt.Rectangle((x, -0.30), width, 0.16,
                                        color=pcol.get(ps, "#DDD"),
                                        clip_on=False))
        bot.set_ylim(0, 1)
        bot.set_yticks([0, 1])
        bot.set_ylabel("P(state)")
        top.set_xlim(-step * 0.5 if pt else 0, xmax + (step * 0.3 if pt else 0))
        ticks = list(range(0, int(xmax) + 1, step * (2 if step == 5 else 1)))
        bot.set_xticks(ticks)
        bot.tick_params(axis="x", pad=17)
        bot.set_xticklabels([(t0 + pd.Timedelta(minutes=m_)).strftime("%H:%M")
                             for m_ in ticks], fontsize=9)
        if k == n - 1:
            bot.set_xlabel("clock time (IST)")

    h = [plt.Line2D([], [], marker="o", ls="", color=c,
                    label="%s (largest probability)" % k_)
         for k_, c in col.items()]
    h += [plt.Rectangle((0, 0), 1, 1, color=c, label="posture: " + k_)
          for k_, c in pcol.items()]
    fig.legend(handles=h, loc="lower center", ncol=3, frameon=False,
               fontsize=9)
    fig.suptitle("Model stress score at fixed checkpoints\n"
                 "no survey answers exist for these sessions, so this shows "
                 "how the score is built, not how right it is",
                 y=0.995, fontsize=12.5)
    fig.subplots_adjust(top=0.955, bottom=0.07, left=0.07, right=0.98)
    fig.savefig(out, dpi=190)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="runs/DEMO_SESSIONS/predictions.csv")
    ap.add_argument("--out", default="runs/DEMO_SESSIONS")
    ap.add_argument("--mode", choices=["point", "mean"], default="point",
                    help="point: the score at each checkpoint (default). "
                         "mean: the mean of the windows in each interval.")
    ap.add_argument("--short", type=int, default=5)
    ap.add_argument("--mid", type=int, default=10)
    ap.add_argument("--long", type=int, default=15)
    ap.add_argument("--signals", default="data/demo_sessions/user_7ch")
    ap.add_argument("--sessions",
                    default="chronos-part/data_chronos_embrace/sessions.csv")
    ap.add_argument("--min-windows", type=int, default=3,
                    help="mean mode: drop an end block with fewer windows")
    a = ap.parse_args()

    pred = pd.read_csv(a.pred)
    iv, sm = build(pred, a.short, a.long, a.mid, a.mode, a.min_windows)
    os.makedirs(a.out, exist_ok=True)
    chk = run_checks(pred, iv, a.sessions, a.signals, a.min_windows, a.mode)
    iv.drop(columns=["t_minutes"]).round(4).to_csv(
        os.path.join(a.out, "interval_scores.csv"), index=False)
    sm.round(4).to_csv(os.path.join(a.out, "interval_summary.csv"), index=False)
    chk.to_csv(os.path.join(a.out, "interval_checks.csv"), index=False)
    try:
        plot(pred, iv, os.path.join(a.out, "interval_scores.png"), a.mode)
        print("figure: %s/interval_scores.png" % a.out)
    except ImportError:
        print("matplotlib missing, figure skipped")

    pd.set_option("display.width", 200)
    print("\nmode: %s" % ("score AT each checkpoint (no averaging)"
                          if a.mode == "point"
                          else "mean of the windows in each interval"))
    for sid, v in iv.groupby("session", sort=False):
        print("\n%s  (every %d min)" % (sid, v["interval_min"].iloc[0]))
        print(v[["time_ist", "score_0_2", "state", "P_Low", "P_Medium",
                 "P_High", "posture", "activity", "act_count"]]
              .round(2).to_string(index=False))
    print("\nSUMMARY")
    print(sm.round(2).to_string(index=False))

    print("\nCROSS-CHECKS")
    for sid, c in chk.groupby("session", sort=False):
        print("  %s" % sid)
        for _, r in c.iterrows():
            print("    [%s] %-46s %s" % (r["status"], r["check"], r["detail"]))
    bad = chk[chk["status"] == "FAIL"]
    warn = chk[chk["status"] == "WARN"]
    print("\n  %d checks: %d pass, %d warn, %d FAIL"
          % (len(chk), (chk["status"] == "PASS").sum(), len(warn), len(bad)))
    if len(bad):
        print("  FAILED - do not use these numbers until understood:")
        print(bad[["session", "check", "detail"]].to_string(index=False))


if __name__ == "__main__":
    main()
