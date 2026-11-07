"""
dashboard.py

Stress monitoring from a wrist wearable (ContextGate, 7 context columns).

    pip install streamlit altair
    streamlit run dashboard.py

Reads, from the project root:

    runs/SCORES_ctx7/predictions.csv        one row per window, the five labelled people
    runs/SCORES_ctx7/per_user.csv           one row per person
    runs/SCORES_ctx7/per_response.csv       one row per survey answer
    runs/ABLATE_ctx7/ablation.csv           context ablation (random split, 5 seeds)
    runs/DEMO_SESSIONS_ctx7/*.csv           the five demo recordings, checkpoint scores

Every page that needs a file shows a message instead of failing when it is
missing, so the pages work one at a time.

The honest numbers are on the "How much to trust it" page and in the sidebar.
They are there on purpose: a dashboard that shows only the flattering figure
falls apart under the first question.
"""

import os

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

alt.data_transformers.disable_max_rows()

SCORES = "runs/SCORES_ctx7"
ABLATION = "runs/ABLATE_ctx7/ablation.csv"
DEMO = "runs/DEMO_SESSIONS_ctx7"

# --------------------------------------------------------------------------
# palette
INK = "#1d2733"
MUTED = "#6b7785"
ACCENT = "#2f6db5"
GREY = "#9aa0a6"
LOW, MED, HIGH = "#2e8b6a", "#e0a030", "#c8553d"
BAND = {"LowStress": LOW, "MediumStress": MED, "HighStress": HIGH}
SHORT = {"LowStress": "Low", "MediumStress": "Moderate",
         "HighStress": "High"}
STATE_COL = {"Low": LOW, "Medium": MED, "High": HIGH}
POSTURE_COL = {"sitting": "#8fb3d9", "standing": "#f0c987",
               "lying": "#b9a6d6", "walking": "#e49b86",
               "unknown": "#d9dde2"}

# --------------------------------------------------------------------------
# Final results, from runs/C0_* (signals only) and runs/C7_random, C7b_response,
# C7b_user, C7_temporal (signals + 7 context). 5 seeds on random, response,
# user; one fold on temporal. Answer level = one prediction per survey answer.
SPLITS = [
    ("random", "Random 70/10/20",
     "Windows from the same survey answer sit on both sides"),
    ("response", "Answers kept whole",
     "Each survey answer is wholly train or wholly test"),
    ("user", "A person never seen",
     "Leave one person out"),
    ("temporal", "Earlier train, later test",
     "First 70% of each person's answers train, the rest test"),
]
RES = {"random": {"off": {"bal": .54, "ans_bal": .595, "ans_ci": [.496, .687], "rho": .349, "rho_ci": [.148, .516], "n": 88, "conf": [[25, 5, 3], [7, 20, 0], [11, 9, 8]]}, "guess": .383, "on": {"bal": .836, "ans_bal": .976, "ans_ci": [.94, 1.0], "rho": .866, "rho_ci": [.824, .889], "n": 88, "conf": [[33, 0, 0], [0, 27, 0], [1, 1, 26]]}},
       "response": {"off": {"bal": .436, "ans_bal": .477, "ans_ci": [.354, .599], "rho": -.039, "rho_ci": [-.322, .243], "n": 57, "conf": [[10, 8, 2], [4, 12, 2], [9, 5, 5]]}, "guess": .309, "on": {"bal": .36, "ans_bal": .372, "ans_ci": [.246, .503], "rho": -.18, "rho_ci": [-.446, .116], "n": 57, "conf": [[6, 5, 9], [6, 9, 3], [10, 3, 6]]}},
       "user": {"off": {"bal": .24, "ans_bal": .15, "ans_ci": [.086, .222], "rho": -.564, "rho_ci": [-.711, -.385], "n": 88, "conf": [[10, 12, 11], [24, 2, 1], [19, 7, 2]]}, "guess": .258, "on": {"bal": .244, "ans_bal": .139, "ans_ci": [.076, .21], "rho": -.456, "rho_ci": [-.608, -.269], "n": 88, "conf": [[9, 9, 15], [14, 2, 11], [18, 8, 2]]}},
       "temporal": {"off": {"bal": .524, "ans_bal": .558, "ans_ci": [.367, .74], "rho": .119, "rho_ci": [-.387, .516], "n": 26, "conf": [[5, 3, 0], [1, 6, 1], [2, 5, 3]]}, "guess": .317, "on": {"bal": .488, "ans_bal": .5, "ans_ci": [.301, .683], "rho": .289, "rho_ci": [-.126, .63], "n": 26, "conf": [[5, 1, 2], [2, 3, 3], [2, 3, 5]]}}}

st.set_page_config(page_title="Stress monitoring", layout="wide",
                   page_icon="📈", initial_sidebar_state="expanded")

st.markdown(f"""
<style>
.block-container {{ padding-top: 2.2rem; max-width: 1250px; }}
h1, h2, h3 {{ letter-spacing: -0.01em; }}
.hero {{ padding: 1.4rem 1.6rem; border-radius: 16px; margin-bottom: 1.2rem;
        background: linear-gradient(120deg, {ACCENT} 0%, #1f4a80 100%);
        color: white; }}
.hero h1 {{ color: white; margin: 0 0 .25rem 0; font-size: 2rem; }}
.hero p  {{ margin: 0; opacity: .88; font-size: 1.02rem; }}
.card {{ border: 1px solid rgba(128,128,128,.25); border-radius: 14px;
        padding: 1rem 1.15rem; height: 100%;
        background: rgba(128,128,128,.06); }}
.card .k {{ font-size: .78rem; text-transform: uppercase; letter-spacing: .06em;
           color: {MUTED}; margin-bottom: .2rem; }}
.card .v {{ font-size: 1.9rem; font-weight: 650; line-height: 1.1; }}
.card .s {{ font-size: .85rem; color: {MUTED}; margin-top: .25rem; }}
.step {{ border-left: 4px solid {ACCENT}; padding: .1rem 0 .1rem .8rem;
        margin-bottom: .7rem; }}
.step b {{ display: block; }}
.chip {{ display: inline-block; padding: .12rem .6rem; border-radius: 999px;
        font-size: .82rem; font-weight: 600; color: white; }}
.note {{ border-left: 4px solid {MED}; background: rgba(224,160,48,.10);
        padding: .7rem 1rem; border-radius: 0 10px 10px 0;
        margin: .6rem 0 1rem 0; font-size: .95rem; }}
.bad  {{ border-left-color: {HIGH}; background: rgba(200,85,61,.09); }}
.good {{ border-left-color: {LOW};  background: rgba(46,139,106,.09); }}
.bar {{ height: 12px; border-radius: 6px;
       background: linear-gradient(90deg, {LOW}, {MED}, {HIGH}); }}
</style>
""", unsafe_allow_html=True)


def card(k, v, s=""):
    return ('<div class="card"><div class="k">%s</div><div class="v">%s</div>'
            '<div class="s">%s</div></div>' % (k, v, s))


def cards(items):
    cols = st.columns(len(items))
    for c, it in zip(cols, items):
        c.markdown(card(*it), unsafe_allow_html=True)


def note(text, kind=""):
    st.markdown('<div class="note %s">%s</div>' % (kind, text),
                unsafe_allow_html=True)


def hero(title, sub):
    st.markdown('<div class="hero"><h1>%s</h1><p>%s</p></div>'
                % (title, sub), unsafe_allow_html=True)


def chip(state):
    return ('<span class="chip" style="background:%s">%s</span>'
            % (STATE_COL.get(state, GREY), state))


def tidy(ch):
    return (ch.configure_axis(labelColor=MUTED, titleColor=MUTED,
                              gridColor="#8883", domainColor="#8885")
              .configure_view(strokeWidth=0)
              .configure_legend(labelColor=MUTED, titleColor=MUTED))


def show(ch, **kw):
    st.altair_chart(tidy(ch), use_container_width=True, **kw)


# --------------------------------------------------------------------------
@st.cache_data
def load_scores():
    p = os.path.join(SCORES, "predictions.csv")
    if not os.path.exists(p):
        return None, None, None
    d = pd.read_csv(p)
    d["datetime_ist"] = pd.to_datetime(d["datetime_ist"], errors="coerce")
    d = d.dropna(subset=["datetime_ist"])
    d["day"] = d["datetime_ist"].dt.date
    d["hour"] = d["datetime_ist"].dt.hour

    def maybe(name):
        q = os.path.join(SCORES, name)
        return pd.read_csv(q) if os.path.exists(q) else None

    pu = maybe("per_user.csv")
    pr = maybe("per_response.csv")
    if pr is not None and "response_time" in pr.columns:
        pr["response_time"] = pd.to_datetime(pr["response_time"],
                                             errors="coerce")
        pr["day"] = pr["response_time"].dt.date
    return d, pu, pr


@st.cache_data
def load_ablation():
    return pd.read_csv(ABLATION) if os.path.exists(ABLATION) else None


@st.cache_data
def load_demo():
    f = os.path.join(DEMO, "interval_scores.csv")
    if not os.path.exists(f):
        return None, None, None
    iv = pd.read_csv(f)
    sm = os.path.join(DEMO, "interval_summary.csv")
    sm = pd.read_csv(sm) if os.path.exists(sm) else None
    p = os.path.join(DEMO, "predictions.csv")
    pw = None
    if os.path.exists(p):
        pw = pd.read_csv(p)
        pw["datetime_ist"] = pd.to_datetime(pw["datetime_ist"])
    return iv, sm, pw


d, per_user, per_resp = load_scores()
ablation = load_ablation()
demo_iv, demo_sum, demo_win = load_demo()

# --------------------------------------------------------------------------
st.sidebar.markdown("### 📈 Stress monitoring")
st.sidebar.caption("ContextGate · wrist wearable · BTP, IIT Kharagpur")
page = st.sidebar.radio("Page", ["Start here", "Demo sessions", "A day",
                                 "Everyone", "What the context did",
                                 "How much to trust it"],
                        label_visibility="collapsed")
st.sidebar.divider()
st.sidebar.markdown("**The honest numbers**")
st.sidebar.caption("Balanced accuracy, signals + 7 context")
for key, name, _ in SPLITS:
    v = RES[key]["on"]["bal"]
    g = RES[key]["guess"]
    st.sidebar.markdown(
        "%s &nbsp; **%.2f** <span style='color:%s'>(guess %.2f)</span>"
        % (name, v, MUTED, g), unsafe_allow_html=True)
st.sidebar.caption(
    "Only the first row is high, and it is recall: the model has already "
    "seen near-identical windows of every survey answer it is tested on.")


def need(df, what):
    if df is None:
        st.info("%s not found. See the docstring at the top of "
                "`dashboard.py` for what to run." % what)
        st.stop()


# ============================================================== START
if page == "Start here":
    hero("Stress from a wristband",
         "Seven wrist signals, seven pieces of context, one score "
         "between 0 and 2.")

    n_h = (len(d) * 30 / 3600) if d is not None else 0
    cards([("Labelled people", "5", "answered the STAI-6 survey"),
           ("Survey answers", "88", "the only ground truth we have"),
           ("Hours scored", "%.0f" % n_h if d is not None else "-",
            "30 s stride, 60 s windows"),
           ("Context columns", "7", "hour, weekend, posture, activity, "
            "sleep, body position")])

    st.write("")
    c1, c2 = st.columns([3, 2])
    with c1:
        st.subheader("How a score is made")
        st.markdown("""
<div class="step"><b>1. Seven signals at 1 Hz</b>
ACC x/y/z, BVP, EDA, skin temperature, respiration (pulse-derived), cut into 60 s windows every 30 s.</div>
<div class="step"><b>2. A frozen encoder</b>
Pretrained on WESAD and left alone, so it turns each window into a fixed vector.</div>
<div class="step"><b>3. Gate + classifier, trained on our five people</b>
The 7 context columns join here. The classifier gives P(Low), P(Moderate), P(High).</div>
<div class="step"><b>4. Our own score</b>
<code>stress_score = 1 × P(Moderate) + 2 × P(High)</code>, so 0 is confidently low and 2 is confidently high.</div>
""", unsafe_allow_html=True)
    with c2:
        st.subheader("Reading a score")
        st.markdown('<div class="bar"></div>', unsafe_allow_html=True)
        a, b, c = st.columns(3)
        a.markdown("**0**<br><span style='color:%s'>low</span>" % MUTED,
                   unsafe_allow_html=True)
        b.markdown("<div style='text-align:center'><b>1</b><br>"
                   "<span style='color:%s'>moderate, or unsure</span></div>"
                   % MUTED, unsafe_allow_html=True)
        c.markdown("<div style='text-align:right'><b>2</b><br>"
                   "<span style='color:%s'>high</span></div>" % MUTED,
                   unsafe_allow_html=True)
        st.caption("A score near 1 can mean confident Moderate or an even "
                   "split across all three, so uncertainty is shown beside "
                   "it.")
        note("Where it works and where it does not is on <b>How much to "
             "trust it</b>. Short version: it follows a person's own "
             "survey answers over time (0.49 against 0.32 for guessing), "
             "and it does not transfer to someone it has never seen.")

    st.subheader("The five demo recordings")
    st.write("Five recordings with no survey answers, scored at checkpoints "
             "(every 5 min for sessions up to 100 min). They show how a "
             "score is calculated; they cannot be graded.")
    if demo_sum is not None:
        t = demo_sum[["session", "length_min", "points", "mean", "min",
                      "max"]].copy()
        t.columns = ["session", "minutes", "checkpoints", "mean score",
                     "lowest", "highest"]
        st.dataframe(t, hide_index=True, use_container_width=True)


# ============================================================== DEMO
elif page == "Demo sessions":
    hero("Demo sessions, checkpoint by checkpoint",
         "Score at a point in time, every 5 minutes. Not an average "
         "over the interval.")
    need(demo_iv, "runs/DEMO_SESSIONS_ctx7/interval_scores.csv")

    note("<b>Rule.</b> Over 130 min: a checkpoint every 15 min. "
         "100 min or less: every 5 min. In between: every 10 min. These "
         "recordings have <b>no survey answers</b>, so no accuracy exists "
         "for them. The score shows what the model does, not whether it "
         "is right.")

    sessions = list(dict.fromkeys(demo_iv["session"]))
    sess = st.selectbox("Recording", sessions)
    iv = demo_iv[demo_iv["session"] == sess].sort_values("block").copy()
    every = int(iv["interval_min"].iloc[0])

    day0 = None
    pw = None
    if demo_win is not None:
        pw = demo_win[demo_win["Subject_ID"] == sess].sort_values(
            "datetime_ist")
        if len(pw):
            day0 = str(pw["datetime_ist"].dt.date.iloc[0])
    if day0 is None:
        day0 = "2000-01-01"
    iv["t"] = pd.to_datetime(day0 + " " + iv["time_ist"])

    length = demo_sum[demo_sum["session"] == sess]["length_min"].iloc[0] \
        if demo_sum is not None else np.nan
    cards([("Length", "%d min" % length if np.isfinite(length) else "-",
            "%s to %s IST" % (iv["time_ist"].iloc[0], iv["time_ist"].iloc[-1])),
           ("Checkpoints", "%d" % len(iv), "one every %d min" % every),
           ("Mean of checkpoints", "%.2f" % iv["score_0_2"].mean(), "0 to 2"),
           ("Range", "%.2f to %.2f" % (iv["score_0_2"].min(),
                                       iv["score_0_2"].max()),
            "lowest to highest checkpoint")])
    st.write("")

    layers = []
    if pw is not None and len(pw):
        win = alt.Chart(pw).mark_line(strokeWidth=1, color=GREY,
                                      opacity=.55).encode(
            x=alt.X("datetime_ist:T", title=None,
                    axis=alt.Axis(format="%H:%M")),
            y=alt.Y("stress_score:Q", title="stress score (0 to 2)",
                    scale=alt.Scale(domain=[0, 2])))
        layers.append(win)
    pts = alt.Chart(iv).encode(
        x=alt.X("t:T", title=None, axis=alt.Axis(format="%H:%M")),
        y=alt.Y("score_0_2:Q", scale=alt.Scale(domain=[0, 2])))
    layers.append(pts.mark_line(color=ACCENT, strokeWidth=2.2))
    layers.append(pts.mark_circle(size=170, opacity=1).encode(
        color=alt.Color("state:N", legend=alt.Legend(title="checkpoint"),
                        scale=alt.Scale(domain=list(STATE_COL),
                                        range=list(STATE_COL.values()))),
        tooltip=[alt.Tooltip("time_ist:N", title="time"),
                 alt.Tooltip("score_0_2:Q", title="score", format=".2f"),
                 alt.Tooltip("P_Low:Q", title="P(Low)", format=".2f"),
                 alt.Tooltip("P_Medium:Q", title="P(Moderate)", format=".2f"),
                 alt.Tooltip("P_High:Q", title="P(High)", format=".2f"),
                 alt.Tooltip("posture:N", title="posture"),
                 alt.Tooltip("activity:N", title="activity")]))
    main = alt.layer(*layers).properties(height=300)

    if pw is not None and len(pw) and "POSTURE" in pw.columns:
        st_ = pw.copy()
        st_["end"] = st_["datetime_ist"] + pd.Timedelta(seconds=30)
        st_["posture"] = st_["POSTURE"].astype(str).str.lower()
        strip = alt.Chart(st_).mark_rect().encode(
            x=alt.X("datetime_ist:T", title=None, axis=None),
            x2="end:T",
            color=alt.Color("posture:N",
                            scale=alt.Scale(domain=list(POSTURE_COL),
                                            range=list(POSTURE_COL.values())),
                            legend=alt.Legend(title="posture")),
            tooltip=["posture:N"]).properties(height=18)
        show(alt.vconcat(main, strip, spacing=6).resolve_scale(color="independent"))
    else:
        show(main)
    st.caption("Big dots are the checkpoints, the figures to quote. The thin "
               "grey line is every 30 s window: it jumps about, which is "
               "why one window is not a reading.")

    st.subheader("The checkpoints")
    t = iv[["time_ist", "score_0_2", "state", "P_Low", "P_Medium", "P_High",
            "uncertainty", "posture", "activity"]].copy()
    t.columns = ["time (IST)", "score", "state", "P(Low)", "P(Moderate)",
                 "P(High)", "uncertainty", "posture", "activity"]
    st.dataframe(
        t, hide_index=True, use_container_width=True,
        column_config={
            "score": st.column_config.ProgressColumn(
                "score", min_value=0.0, max_value=2.0, format="%.2f"),
            "P(Low)": st.column_config.NumberColumn(format="%.2f"),
            "P(Moderate)": st.column_config.NumberColumn(format="%.2f"),
            "P(High)": st.column_config.NumberColumn(format="%.2f"),
            "uncertainty": st.column_config.NumberColumn(format="%.2f")})

    note("<b>Account differences.</b> BVP spread is about 0.007 to 0.03 "
         "for user12, user13 and user14 and 0.13 to 0.16 for user1 and "
         "user25. The model was trained on the second group, so scores "
         "across these recordings are not strictly comparable.")


# ============================================================== A DAY
elif page == "A day":
    hero("One person, one day", "The score across the day, and where the "
         "person said how they felt.")
    need(d, "runs/SCORES_ctx7/predictions.csv")

    c1, c2 = st.columns(2)
    user = c1.selectbox("Person", sorted(d["Subject_ID"].unique()))
    du = d[d["Subject_ID"] == user]
    days = sorted(du["day"].unique())

    labelled = set()
    if per_resp is not None:
        labelled = set(per_resp[per_resp["Subject_ID"] == user]["day"])
    opts = [("%s  (%d answers)" % (x, int(((per_resp["day"] == x) &
                                           (per_resp["Subject_ID"] == user)).sum()))
             if x in labelled else str(x)) for x in days]
    pick = c2.selectbox("Day", opts,
                        index=next((i for i, x in enumerate(days)
                                    if x in labelled), 0))
    day = days[opts.index(pick)]

    dd = du[du["day"] == day].sort_values("datetime_ist")
    if dd.empty:
        st.warning("Nothing scored that day.")
        st.stop()

    top = dd["predicted"].value_counts().index[0]
    cards([("Hours scored", "%.1f" % (len(dd) * 30 / 3600),
            "%d windows" % len(dd)),
           ("Average score", "%.2f" % dd["stress_score"].mean(),
            "0 low, 2 high"),
           ("Most of the day", SHORT[top], "most common class"),
           ("Peak", "%.2f" % dd["stress_score"].max(),
            "at %s" % dd.loc[dd["stress_score"].idxmax(),
                             "datetime_ist"].strftime("%H:%M"))])
    st.write("")

    base = alt.Chart(dd).encode(
        x=alt.X("datetime_ist:T", title=None, axis=alt.Axis(format="%H:%M")))
    area = base.mark_area(opacity=.18, color=ACCENT).encode(
        y=alt.Y("stress_score:Q", title="stress score",
                scale=alt.Scale(domain=[0, 2])))
    line = base.mark_line(strokeWidth=1.3, color=ACCENT).encode(
        y="stress_score:Q",
        tooltip=[alt.Tooltip("datetime_ist:T", title="time",
                             format="%H:%M:%S"),
                 alt.Tooltip("stress_score:Q", title="score", format=".2f"),
                 alt.Tooltip("predicted:N", title="class"),
                 alt.Tooltip("uncertainty:Q", title="uncertainty",
                             format=".2f")])
    layers = [area, line]
    if per_resp is not None:
        pr = per_resp[(per_resp["Subject_ID"] == user) &
                      (per_resp["day"] == day)]
        if len(pr):
            layers.append(alt.Chart(pr).mark_rule(strokeWidth=3).encode(
                x="response_time:T",
                color=alt.Color("reported:N",
                                scale=alt.Scale(domain=list(BAND),
                                                range=list(BAND.values())),
                                legend=alt.Legend(title="they reported")),
                tooltip=[alt.Tooltip("response_time:T", title="answered at",
                                     format="%H:%M"),
                         alt.Tooltip("reported:N", title="they said"),
                         alt.Tooltip("predicted:N", title="model said"),
                         alt.Tooltip("stai6:Q", title="STAI-6",
                                     format=".0f")]))
    show(alt.layer(*layers).properties(height=300))
    st.caption("Vertical lines are the moments they filled in the "
               "questionnaire, coloured by what they said.")

    g1, g2 = st.columns([3, 2])
    with g1:
        st.subheader("By hour")
        hr = dd.groupby("hour")["stress_score"].mean().reset_index()
        show(alt.Chart(hr).mark_bar(color=ACCENT, cornerRadiusEnd=3).encode(
            x=alt.X("hour:O", title="hour of day"),
            y=alt.Y("stress_score:Q", title="average score",
                    scale=alt.Scale(domain=[0, 2])),
            tooltip=["hour", alt.Tooltip("stress_score:Q", format=".2f")]
        ).properties(height=240))
        night = hr[hr["hour"].between(3, 8)]["stress_score"].mean()
        aft = hr[hr["hour"].between(12, 18)]["stress_score"].mean()
        if np.isfinite(night) and np.isfinite(aft) and night > aft + 0.1:
            note("Higher overnight (%.2f between 3 and 8) than in the "
                 "afternoon (%.2f). At night the body is still and skin "
                 "temperature shifts, and in WESAD those patterns meant "
                 "the stress task. Treat overnight scores with suspicion."
                 % (night, aft), "bad")
    with g2:
        st.subheader("How the day split")
        vc = dd["predicted"].value_counts().reset_index()
        vc.columns = ["predicted", "windows"]
        vc["label"] = vc["predicted"].map(SHORT)
        show(alt.Chart(vc).mark_arc(innerRadius=60).encode(
            theta="windows:Q",
            color=alt.Color("predicted:N", legend=None,
                            scale=alt.Scale(domain=list(BAND),
                                            range=list(BAND.values()))),
            tooltip=["label", "windows"]).properties(height=240))
        for _, r in vc.iterrows():
            st.markdown("<span style='color:%s'>●</span> **%s** — %d windows "
                        "(%.0f%%)" % (BAND[r["predicted"]], r["label"],
                                      r["windows"],
                                      100 * r["windows"] / len(dd)),
                        unsafe_allow_html=True)


# ============================================================== EVERYONE
elif page == "Everyone":
    hero("All five people", "How the score looks across everyone we have "
         "survey answers for.")
    need(per_user, "runs/SCORES_ctx7/per_user.csv")

    cards([("People", "%d" % len(per_user), ""),
           ("Windows scored", format(len(d), ",") if d is not None else "-",
            ""),
           ("Hours", "%.0f" % per_user["hours"].sum(), "")])
    st.write("")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Average score")
        show(alt.Chart(per_user).mark_bar(color=ACCENT,
                                          cornerRadiusEnd=4).encode(
            y=alt.Y("Subject_ID:N", sort="-x", title=None),
            x=alt.X("score_mean:Q", title="average stress score",
                    scale=alt.Scale(domain=[0, 2])),
            tooltip=["Subject_ID", alt.Tooltip("score_mean:Q", format=".2f"),
                     alt.Tooltip("windows:Q", format=","),
                     alt.Tooltip("hours:Q", format=".0f")]
        ).properties(height=240))
    with c2:
        st.subheader("How each person's time was classified")
        m = per_user.melt(id_vars="Subject_ID",
                          value_vars=["pct_low", "pct_medium", "pct_high"],
                          var_name="band", value_name="pct")
        m["band"] = m["band"].map({"pct_low": "Low", "pct_medium": "Moderate",
                                   "pct_high": "High"})
        show(alt.Chart(m).mark_bar().encode(
            y=alt.Y("Subject_ID:N", title=None),
            x=alt.X("pct:Q", title="share of windows", stack="normalize",
                    axis=alt.Axis(format="%")),
            color=alt.Color("band:N", legend=alt.Legend(title=None,
                                                       orient="top"),
                            scale=alt.Scale(domain=["Low", "Moderate", "High"],
                                            range=[LOW, MED, HIGH])),
            tooltip=["Subject_ID", "band", alt.Tooltip("pct:Q", format=".1f")]
        ).properties(height=240))

    with st.expander("The table"):
        st.dataframe(per_user, use_container_width=True, hide_index=True)

    if per_resp is not None and len(per_resp):
        st.subheader("Against what people reported")
        k1, k2 = st.columns([2, 3])
        with k1:
            st.markdown(card("Answers matched", "%d of %d" % (
                per_resp["correct"].sum(), len(per_resp)),
                "%.0f%% of survey answers" % (100 * per_resp["correct"].mean())),
                unsafe_allow_html=True)
        with k2:
            ct = pd.crosstab(per_resp["reported"].map(SHORT),
                             per_resp["predicted"].map(SHORT))
            ct.index.name = "reported"
            ct.columns.name = "model said"
            st.dataframe(ct, use_container_width=True)
        note("This is <b>recall, not prediction</b>. These five people were "
             "in the training data and the split was random, so the model "
             "has already seen near-identical windows of these answers. "
             "On answers it had not seen, balanced accuracy was 0.36 with "
             "context and 0.44 without, against about 0.31 for guessing. "
             "For a person it had never met it was 0.24 against 0.26.",
             "bad")


# ============================================================== CONTEXT
elif page == "What the context did":
    hero("What the context did",
         "Seven context columns: HOUR, IS_WEEKEND, POSTURE, ACT_COUNT, "
         "ACT_CLASS, BODY_POSITION, SLEEP.")

    st.subheader("Context off and on, four ways of testing")
    rows = []
    for key, name, _ in SPLITS:
        r = RES[key]
        rows += [dict(split=name, model="Signals only", bal=r["off"]["bal"]),
                 dict(split=name, model="Signals + 7 context",
                      bal=r["on"]["bal"])]
    cmp = pd.DataFrame(rows)
    guess = pd.DataFrame([dict(split=n, guess=RES[k]["guess"])
                          for k, n, _ in SPLITS])
    order = [n for _, n, _ in SPLITS]
    bars = alt.Chart(cmp).mark_bar(cornerRadiusEnd=3).encode(
        x=alt.X("split:N", sort=order, title=None,
                axis=alt.Axis(labelAngle=0, labelLimit=170)),
        xOffset=alt.XOffset("model:N"),
        y=alt.Y("bal:Q", title="balanced accuracy",
                scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("model:N", scale=alt.Scale(
            domain=["Signals only", "Signals + 7 context"],
            range=[GREY, ACCENT]), legend=alt.Legend(title=None,
                                                      orient="top")),
        tooltip=["split", "model", alt.Tooltip("bal:Q", format=".3f")])
    labels = bars.mark_text(dy=-8, color=MUTED).encode(
        text=alt.Text("bal:Q", format=".2f"))
    ticks = alt.Chart(guess).mark_tick(color=INK, thickness=2, width=150,
                                       strokeDash=[4, 3]).encode(
        x=alt.X("split:N", sort=order), y="guess:Q",
        tooltip=[alt.Tooltip("guess:Q", title="guessing", format=".3f")])
    show(alt.layer(bars, labels, ticks).properties(height=340))
    st.caption("Dashed tick: what always guessing the commonest class "
               "would score on that split.")

    note("Context adds <b>+0.30</b> on the random split and nothing "
         "reliable on the three fair ones. The gain does not carry over to "
         "answers the model has not seen, so we do not claim that context "
         "improves detection.", "bad")

    st.subheader("Which column carries it (random split)")
    if ablation is None:
        st.info("Run `btp/46_context_ablation.py` to fill this section.")
    else:
        full = ablation[ablation["kind"] == "all"].iloc[0]
        none = ablation[ablation["kind"] == "none"].iloc[0]
        cards([("All 7 columns", "%.3f" % full["balanced"], ""),
               ("No context", "%.3f" % none["balanced"], ""),
               ("Context is worth", "%+.3f" % (full["balanced"] -
                                               none["balanced"]),
                "on this split")])
        st.write("")
        c1, c2 = st.columns(2)
        dr = ablation[ablation["kind"] == "drop"].copy()
        dr["column"] = dr["label"].str.replace("without ", "", regex=False)
        on = ablation[ablation["kind"] == "only"].copy()
        on["column"] = on["label"].str.replace("only ", "", regex=False)
        with c1:
            st.markdown("**Remove one column**")
            show(alt.Chart(dr).mark_bar(color=HIGH,
                                        cornerRadiusEnd=3).encode(
                y=alt.Y("column:N", sort="x", title=None),
                x=alt.X("vs_all:Q", title="change in balanced accuracy"),
                tooltip=["column", alt.Tooltip("balanced:Q", format=".3f"),
                         alt.Tooltip("vs_all:Q", format="+.3f")]
            ).properties(height=280))
        with c2:
            st.markdown("**Use one column alone**")
            show(alt.Chart(on).mark_bar(color=LOW,
                                        cornerRadiusEnd=3).encode(
                y=alt.Y("column:N", sort="-x", title=None),
                x=alt.X("vs_none:Q", title="gain over no context"),
                tooltip=["column", alt.Tooltip("balanced:Q", format=".3f"),
                         alt.Tooltip("vs_none:Q", format="+.3f")]
            ).properties(height=280))
        st.caption("A column can look useless when removed because another "
                   "says the same thing; alone, you see what it holds.")
        note("<b>The clock does most of the work.</b> HOUR and IS_WEEKEND "
             "cost the most when removed and give the most alone. On a "
             "random split, knowing the hour is close to knowing which "
             "moment it is, and near-identical windows sit on both sides. "
             "The body-derived columns barely register.")


# ============================================================== TRUST
else:
    hero("How much to trust it",
         "The same data and the same code, tested four ways.")

    st.subheader("Window level and answer level")
    rows = []
    for key, name, how in SPLITS:
        r = RES[key]
        rows.append({"test": name + (" (recall)" if key == "random" else ""),
                     "guessing": r["guess"],
                     "signals only": r["off"]["bal"],
                     "+ 7 context": r["on"]["bal"],
                     "answers": r["on"]["n"],
                     "answer level, signals only":
                         "%.2f [%.2f, %.2f]" % (r["off"]["ans_bal"],
                                                *r["off"]["ans_ci"]),
                     "answer level, + context":
                         "%.2f [%.2f, %.2f]" % (r["on"]["ans_bal"],
                                                *r["on"]["ans_ci"])})
    st.dataframe(pd.DataFrame(rows), hide_index=True,
                 use_container_width=True,
                 column_config={
                     "guessing": st.column_config.NumberColumn(format="%.3f"),
                     "signals only": st.column_config.NumberColumn(
                         format="%.3f"),
                     "+ 7 context": st.column_config.NumberColumn(
                         format="%.3f")})
    st.caption("Balanced accuracy. Answer level is one prediction per "
               "survey answer with a 95% bootstrap interval over answers. "
               "WESAD reference, held-out subjects, wrist only: 0.935.")

    c1, c2 = st.columns(2)
    with c1:
        note("<b>What holds up.</b> Following a person's own answers over "
             "time (earlier answers train, later test): 0.52 from signals "
             "alone, against 0.32 for guessing. But only 26 test answers, "
             "so the interval is 0.37 to 0.74.", "good")
    with c2:
        note("<b>What does not.</b> A person never seen: below guessing, "
             "and the ordering is reversed (rank correlation with STAI-6 "
             "is −0.56 without context and −0.46 with it, both intervals "
             "exclude zero).", "bad")

    st.subheader("Reported against detected, on answers not trained on")
    pick = st.radio("Test", [n for _, n, _ in SPLITS[1:]], horizontal=True)
    key = next(k for k, n, _ in SPLITS if n == pick)
    st.caption(next(h for k, n, h in SPLITS if k == key))
    m1, m2 = st.columns(2)
    names = ["Low", "Moderate", "High"]
    for col, t, title in ((m1, "off", "Signals only"),
                          (m2, "on", "Signals + 7 context")):
        conf = np.array(RES[key][t]["conf"])
        cm = pd.DataFrame([(names[i], names[j], int(conf[i, j]))
                           for i in range(3) for j in range(3)],
                          columns=["reported", "detected", "n"])
        base = alt.Chart(cm).encode(
            x=alt.X("detected:N", sort=names, title="model detected",
                    axis=alt.Axis(labelAngle=0)),
            y=alt.Y("reported:N", sort=names, title="person reported"))
        heat = base.mark_rect(cornerRadius=4).encode(
            color=alt.Color("n:Q", legend=None,
                            scale=alt.Scale(scheme="blues")))
        txt = base.mark_text(fontSize=18, fontWeight="bold").encode(
            text="n:Q", color=alt.condition(alt.datum.n > conf.max() * .55,
                                            alt.value("white"),
                                            alt.value(INK)))
        col.markdown("**%s**" % title)
        with col:
            show((heat + txt).properties(height=250))

    st.subheader("What does not work")
    st.markdown(
        "- **Transfer to a new person.** Five people used the self-report "
        "scale in five different parts of its range. One reported low "
        "stress on all 18 occasions; another never reported low at all.\n"
        "- **Context on fair splits.** It is the clock, and it does not "
        "survive a split where answers are kept whole.\n"
        "- **The gate never chose a side.** It stayed between 0.42 and "
        "0.50, the halfway point.\n"
        "- **Overnight scores read as stress.** At night the body is still "
        "and skin temperature shifts, and in WESAD those patterns meant the "
        "stress task.\n"
        "- **Respiration is derived.** It comes from the pulse at 30 s "
        "resolution; its correlation with the belt is 0.185.")

    st.subheader("What the score is")
    st.code("stress_score = 1 x P(Moderate) + 2 x P(High)      range 0 to 2")
    st.caption("That formula is ours. The model outputs three probabilities "
               "and nothing more. A score of 1.0 means either confidence in "
               "Moderate or an even split across all three, which is why "
               "uncertainty is reported beside it.")
