"""
device_dashboard.py

A dashboard for the results of the Jetson run.

Reads (from --dir, default run_output):
    window_scores_device.csv   one row per window, from 55_session_checkpoints.py
    device_summary.txt         latency, memory, comparison with the laptop
    power_summary.txt          idle, running and the difference (optional)
    power_trace.csv            power every half second (optional)
and (from --pack, default demo_sessions_pack):
    <session>/signals.csv      the 1 Hz signals, to show what the body was
                               doing in the 60 s behind each score

Run (in the Final_Demo folder):
    pip3 install streamlit altair          # once, needs internet
    streamlit run device_dashboard.py -- --dir run_output --pack demo_sessions_pack
Then open the address it prints (http://localhost:8501) in the browser.
"""
import argparse
import os

import altair as alt
import pandas as pd
import streamlit as st

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="run_output")
ap.add_argument("--pack", default="demo_sessions_pack")
args, _ = ap.parse_known_args()
D, PACK = args.dir, args.pack

st.set_page_config(page_title="ContextGate on the Jetson", layout="wide")
GREEN, GOLD, RED = "#006633", "#CC9900", "#C00000"
COL = {"Low": GREEN, "Medium": GOLD, "High": RED}

st.markdown("""
<style>
.chiprow{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:2px 0 10px 0}
.chiplabel{font-weight:700;min-width:200px;color:#333}
.chip{padding:3px 11px;border-radius:14px;font-size:0.9rem;border:1px solid}
.sig{background:#e8f3ec;border-color:#006633;color:#006633}
.ctx{background:#fbf4dd;border-color:#CC9900;color:#7a5c00}
</style>""", unsafe_allow_html=True)

SIGNALS = ["ACC x", "ACC y", "ACC z", "BVP (pulse wave)",
           "EDA (skin conductance)", "Skin temperature", "Respiration",
           "Pulse rate (bpm, per minute)"]
CONTEXTS = ["Hour", "Weekend", "Posture", "Activity count", "Activity class",
            "Body position", "Sleep"]


def chips(label, items, cls):
    h = "".join('<span class="chip %s">%s</span>' % (cls, i) for i in items)
    st.markdown('<div class="chiprow"><span class="chiplabel">%s</span>%s'
                '</div>' % (label, h), unsafe_allow_html=True)


st.title("ContextGate on a Jetson Orin Nano")
chips("Physiological signals (1 Hz)", SIGNALS, "sig")
chips("Context (every 30 s)", CONTEXTS, "ctx")
st.caption("Score = 1 x P(Medium) + 2 x P(High), range 0 to 2. These "
           "recordings have no survey answers, so there is no accuracy. "
           "Pulse rate is the wristband's own per-minute estimate, shown "
           "for reference: it is not one of the model's seven inputs. "
           "Units, sampling rates and ranges are in the Guide tab.")

p = os.path.join(D, "window_scores_device.csv")
if not os.path.exists(p):
    st.error("No %s. Run 55_session_checkpoints.py first." % p)
    st.stop()
df = pd.read_csv(p)
df["t"] = pd.to_datetime(df["datetime_ist"])
df["is_checkpoint"] = df["is_checkpoint"].astype(bool)


@st.cache_data
def load_signals(session):
    f = os.path.join(PACK, session, "signals.csv")
    if not os.path.exists(f):
        return None
    s = pd.read_csv(f)
    s["datetime_ist"] = pd.to_datetime(s["datetime_ist"])
    return s.set_index("datetime_ist").sort_index()


@st.cache_data
def load_pulse(session):
    f = os.path.join(PACK, session, "pulse_rate.csv")
    if not os.path.exists(f):
        return None
    p = pd.read_csv(f)
    p["datetime_ist"] = pd.to_datetime(p["datetime_ist"])
    return p.set_index("datetime_ist")["pulse_rate_bpm"].sort_index()


def pulse_at(pulse, start):
    """The wristband's pulse rate for the minute that starts at `start`."""
    if pulse is None:
        return None
    v = pulse.get(start.floor("min"))
    return None if v is None or pd.isna(v) else float(v)


def window_signals(sig, start):
    """What the body signals did in the 60 s that start at `start`."""
    w = sig.loc[start:start + pd.Timedelta(seconds=59)]
    if w.empty:
        return {}
    return {"EDA": w["EDA_wrist"].mean(), "BVP mean": w["BVP"].mean(),
            "BVP spread": w["BVP"].std(),
            "ACC x": w["ACC_X_wrist"].mean(), "ACC y": w["ACC_Y_wrist"].mean(),
            "ACC z": w["ACC_Z_wrist"].mean(),
            "Temp": w["Temp_wrist"].mean(), "Resp": w["RESP"].mean()}


# ---------------------------------------------------------------- top numbers
c = st.columns(5)
c[0].metric("Windows scored", f"{len(df):,}")
c[1].metric("Checkpoints (every 5 min)", int(df["is_checkpoint"].sum()))
c[2].metric("Latency, median", f"{df['latency_ms'].median():.1f} ms")
c[3].metric("Latency, worst", f"{df['latency_ms'].max():.1f} ms")
c[4].metric("Mean score", f"{df['score_0_2'].mean():.2f}")

tab1, tab2, tab3, tab4 = st.tabs(["Scores", "Latency", "Power and summary",
                                  "Guide: what the columns mean"])

# ---------------------------------------------------------------- scores
with tab1:
    sessions = list(df["session"].unique())
    s = st.selectbox("Recording", sessions)
    d = df[df["session"] == s].copy()
    d["minute"] = (d["t"] - d["t"].iloc[0]).dt.total_seconds() / 60.0
    sig = load_signals(s)
    pulse = load_pulse(s)

    t0, t1 = d["t"].iloc[0], d["t"].iloc[-1] + pd.Timedelta(seconds=60)
    k = st.columns(4)
    k[0].metric("Date", t0.strftime("%d %b %Y"))
    k[1].metric("Time (IST)", "%s to %s" % (t0.strftime("%H:%M"),
                                            t1.strftime("%H:%M")))
    k[2].metric("Length", "%.0f min" % ((t1 - t0).total_seconds() / 60))
    k[3].metric("Windows / checkpoints",
                "%d / %d" % (len(d), int(d["is_checkpoint"].sum())))

    base = alt.Chart(d).encode(
        x=alt.X("minute:Q", title="minutes into the recording"),
        y=alt.Y("score_0_2:Q", title="stress score (0 to 2)",
                scale=alt.Scale(domain=[0, 2])))
    line = base.mark_line(color="#bbbbbb", strokeWidth=1.2)
    pts = alt.Chart(d[d["is_checkpoint"]]).mark_circle(size=90).encode(
        x="minute:Q", y="score_0_2:Q",
        color=alt.Color("state:N", scale=alt.Scale(
            domain=list(COL), range=list(COL.values())), title="state"),
        tooltip=["time_ist", "score_0_2", "state"])
    st.altair_chart((line + pts).properties(height=320),
                    use_container_width=True)
    st.caption("Grey line: every 30 s window. Dots: the checkpoint windows, "
               "the score at that exact minute (not an average).")

    # ---- the table: score, then what the body and the situation were
    ck = d[d["is_checkpoint"]].copy()
    rows = []
    for _, r in ck.iterrows():
        row = {"Time": r["time_ist"], "Score": round(r["score_0_2"], 2),
               "State": r["state"],
               "Pulse (bpm)": pulse_at(pulse, r["t"])}
        if sig is not None:
            w = window_signals(sig, r["t"])
            row.update({"EDA": w.get("EDA"), "BVP mean": w.get("BVP mean"),
                        "ACC x": w.get("ACC x"), "ACC y": w.get("ACC y"),
                        "ACC z": w.get("ACC z")})
        row.update({"Act count": r.get("ctx_ACT_COUNT"),
                    "Act class": r.get("ctx_ACT_CLASS"),
                    "Posture": r.get("ctx_POSTURE")})
        rows.append(row)
    tbl = pd.DataFrame(rows)

    st.subheader("Checkpoints: the score, and the signals and context behind it")
    if sig is None:
        st.warning("No signals found in %s/%s/. Signals are left out; copy "
                   "the demo_sessions_pack folder here to show them."
                   % (PACK, s))
    st.dataframe(
        tbl, hide_index=True, use_container_width=True,
        column_config={
            "Score": st.column_config.ProgressColumn(
                "Score (0-2)", min_value=0, max_value=2, format="%.2f"),
            "Pulse (bpm)": st.column_config.NumberColumn(format="%.0f"),
            "EDA": st.column_config.NumberColumn(format="%.3f"),
            "BVP mean": st.column_config.NumberColumn(format="%.5f"),
            "ACC x": st.column_config.NumberColumn(format="%.2f"),
            "ACC y": st.column_config.NumberColumn(format="%.2f"),
            "ACC z": st.column_config.NumberColumn(format="%.2f"),
            "Act count": st.column_config.NumberColumn(format="%.1f")})
    st.caption("Signals are averages over the 60 s window that starts at the "
               "checkpoint. Context is what the model was given for that "
               "window. EDA in microsiemens, ACC in g.")

    # ---- one checkpoint in detail
    st.subheader("One checkpoint in detail")
    pick = st.selectbox("Checkpoint", list(ck["time_ist"]), key="ckpick")
    r = ck[ck["time_ist"] == pick].iloc[0]
    a, b, cc = st.columns([1, 1.2, 1.2])
    a.metric("Stress score", "%.2f" % r["score_0_2"], r["state"],
             delta_color="off")
    a.caption("Window %s, on %s" % (r["window_covers"],
                                    r["t"].strftime("%d %b %Y")))
    b.markdown("**Signals in that minute**")
    if sig is not None:
        w = window_signals(sig, r["t"])
        pv = pulse_at(pulse, r["t"])
        b.write("Pulse rate: **%s**  \nEDA: **%.3f**  \n"
                "BVP mean: **%.5f**  \n"
                "ACC x / y / z: **%.2f / %.2f / %.2f**  \n"
                "Skin temperature: **%.1f**  \nRespiration: **%.1f**"
                % ("%.0f bpm" % pv if pv is not None else "n/a",
                   w["EDA"], w["BVP mean"], w["ACC x"],
                   w["ACC y"], w["ACC z"], w["Temp"], w["Resp"]))
    else:
        b.write("signals not available")
    cc.markdown("**Context the model was given**")
    cc.write("Body position: **%s**  \nPosture: **%s**  \n"
             "Activity class: **%s**  \nActivity count: **%.1f**  \n"
             "Hour: **%s**, weekend: **%s**, asleep: **%s**"
             % (r["ctx_BODY_POSITION"], r["ctx_POSTURE"], r["ctx_ACT_CLASS"],
                float(r["ctx_ACT_COUNT"]), r["ctx_HOUR"],
                "yes" if int(r["ctx_IS_WEEKEND"]) else "no",
                "yes" if float(r["ctx_SLEEP"]) >= 0.5 else "no"))

    with st.expander("All windows of this recording"):
        st.dataframe(d[["time_ist", "score_0_2", "state", "latency_ms",
                        "ctx_POSTURE", "ctx_ACT_CLASS", "is_checkpoint"]]
                     .round(3), hide_index=True, use_container_width=True)

# ---------------------------------------------------------------- latency
with tab2:
    h = alt.Chart(df).mark_bar(color=GREEN).encode(
        x=alt.X("latency_ms:Q", bin=alt.Bin(maxbins=40),
                title="time to score one window (ms)"),
        y=alt.Y("count()", title="windows"))
    st.altair_chart(h.properties(height=300), use_container_width=True)
    st.write("The first window is the slowest because nothing is warm yet "
             "(%.1f ms). The median window takes %.1f ms."
             % (df["latency_ms"].iloc[0], df["latency_ms"].median()))
    if "rss_mb" in df.columns:
        st.write("Memory held by the process: about %.0f MB."
                 % df["rss_mb"].iloc[-1])

# ---------------------------------------------------------------- text files
with tab3:
    for fn, title in [("device_summary.txt", "Device summary"),
                      ("power_summary.txt", "Power: idle against running")]:
        fp = os.path.join(D, fn)
        st.subheader(title)
        if os.path.exists(fp):
            st.code(open(fp).read(), language=None)
        else:
            st.info("%s not found yet." % fn)
    tp = os.path.join(D, "power_trace.csv")
    if os.path.exists(tp):
        t = pd.read_csv(tp)
        rail = "VDD_IN" if "VDD_IN" in t.columns else (
            "POM_5V_IN" if "POM_5V_IN" in t.columns else None)
        if rail:
            t["seconds"] = t["unix_time"] - t["unix_time"].iloc[0]
            ch = alt.Chart(t).mark_line().encode(
                x=alt.X("seconds:Q", title="seconds"),
                y=alt.Y(rail + ":Q", title="board power (mW)",
                        scale=alt.Scale(zero=False)),
                color=alt.Color("phase:N", title="phase"))
            st.subheader("Board power over time")
            st.altair_chart(ch.properties(height=300),
                            use_container_width=True)

# ---------------------------------------------------------------- guide
with tab4:
    st.markdown("""
**How the data was recorded.** An Empatica EmbracePlus wristband, worn on the
wrist. Its sensors record at different rates; every signal is brought to **1
sample per second** (the average of each second) before the model sees it.
Ranges are what these five recordings contain. The bands are rough guides from
general physiology and from our own data, not medical limits.

| Column | Unit | Recorded at (native) | Used at | Range here | What the range means |
|---|---|---|---|---|---|
| **EDA** | microsiemens (uS) | 4 Hz | 1 Hz | about 0 to 4.9, mostly 0.2 to 3.7 | Skin conductance: how active the sweat glands are. Below 0.5 calm or dry skin. 0.5 to 2 usual. Above 2 to 3 higher arousal, or warm sweaty skin. Rises with stress, but also with heat and movement. A tiny negative value is sensor noise, read it as 0. |
| **BVP mean** | relative optical units (the wristband does not give a physical unit) | 64 Hz | 1 Hz | -0.025 to +0.065; 98% of windows within 0.001 of zero | The pulse wave swings above and below zero, so its average is about 0 and the sign means nothing. A larger value only marks a window with movement noise. |
| **Pulse (bpm)** | beats per minute | 1 value per minute (wristband's own estimate) | per minute | 64 to 125 | At rest in adults 60 to 100 is usual. Below 60 low. 100 to 130 raised by movement, stress or caffeine. Not a model input. |
| **ACC x, y, z** | g (1 g = gravity, 9.8 m/s2) | 64 Hz (range of the sensor +-16 g) | 1 Hz | -1.1 to +1.1 | With the wrist still, the three add up to about 1 g and show the tilt of the wrist. A negative value only means that axis points the opposite way to gravity. Fast changes from window to window mean movement. |
| **Skin temperature** | degrees Celsius | 1 Hz | 1 Hz | 29.8 to 34.6 | Cooler skin: cold surroundings or a loose strap. 32 to 35 is typical when worn well. |
| **Respiration** | breaths per minute | worked out from the pulse, every 30 s | 1 Hz (held for 30 s) | 4 to 28 | A weak channel: it agrees only 0.19 with a real breathing belt. |
| **Act count** | movement counts per 30 s (no physical unit) | from the accelerometer, every 30 s | per 30 s | 1.9 to 248, median 15 | Below 5 still. 5 to 30 light movement (typing, gestures). 30 to 100 active (walking). Above 100 vigorous. |
| **Act class** | label | wristband, per minute | per 30 s | still, generic, walking (running not seen) | still: no movement. generic: movement that is not walking or running. walking: walking. |
| **Posture** | label | from wrist tilt and movement, every 30 s | per 30 s | sitting, standing, walking | The situation the body signals sit in. |
| **Body position** | label | wristband, per minute | per 30 s | sitting or reclining or lying, standing, miscellaneous | Lying and reclining often go with rest or sleep. |
| **Hour, Weekend, Sleep** | hour 0 to 23, yes/no, yes/no | clock and wristband | per 30 s | hours 15 to 20 here, all weekdays | Time of day matters to the model because stress and routine follow the clock. |
| **Score** | none (0 to 2) | one per 60 s window, every 30 s | per window | 0 to 2 | 1 x P(Medium) + 2 x P(High). Near 0: model is sure of Low. Near 1: Medium. Near 2: High. The model's estimate, not a measurement. |
| **State** | label | per window | per window | Low, Medium, High | Chosen by the highest probability, so score ranges overlap: Low up to 0.79, Medium 0.52 to 1.49, High 1.08 to 2.0 in this run. |

In the checkpoint table, signal values are averages over the 60 s window that
starts at the checkpoint.
""")
    st.info("A rise in score together with a rise in EDA and a high pulse rate "
            "is a pattern, not a proof of stress. Movement, heat and caffeine "
            "move the same signals, which is why the model also gets the "
            "context.")