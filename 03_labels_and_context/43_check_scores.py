"""
43_check_scores.py

WHAT THIS DOES
    Takes the scores written by btp/26_apply_to_users.py and answers three
    questions the scoring run itself cannot:

    1. A score on a readable scale.
       stai_equiv = P_Low*m0 + P_Medium*m1 + P_High*m2, where m0..m2 are the
       mean STAI-6 of the survey answers in each class. It lives on the same
       20-80 scale as the questionnaire. Because it is an expected value of
       class means, it can never go below the Low mean or above the High mean.
       Also written: p_high_minus_low, which says which way the model leans
       without pretending a score near the middle means "Medium".

    2. Does it follow the clock when the watch was actually worn?
       The model also scores hours when the watch was off. This joins
       Empatica's wearing_detection_percentage and prints the hour profile
       for all windows and for worn windows only.

    3. Does it agree with what people reported?
       At every window a survey answer labels, compare the prediction with the
       answer: balanced accuracy against the 0.333 chance level, and rank
       correlation between stai_equiv and the STAI-6 score. The model never
       saw user labels, so this is a real zero-shot test. The unit that
       counts is the survey answer (88 of them), not the 5,000+ windows
       that share them.

TIME
    Predictions carry an IST mid_time. Biomarkers are UTC, so the join shifts
    mid_time back by 5 h 30 min. Labelled windows are matched to the nearest
    prediction centre within 30 seconds.

HOW TO RUN
    python3 btp/43_check_scores.py --pred runs/SCORES_W7_allusers/predictions.csv
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

LABELLED = "data/processed/labels/labelled_windows.csv"
SUMMARY = "data/processed/labels/label_summary.json"
BIO = "data/processed/biomarkers/%s_biomarkers.csv"
IST = pd.Timedelta(hours=5, minutes=30)
NAMES = {"LowStress": 0, "MediumStress": 1, "HighStress": 2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--labelled", default=LABELLED)
    ap.add_argument("--worn-threshold", type=float, default=50.0,
                    help="wearing_detection_percentage at or above this counts "
                         "as worn")
    args = ap.parse_args()

    p = pd.read_csv(args.pred)
    p["mid_time"] = pd.to_datetime(p["mid_time"])
    p["hour"] = p["mid_time"].dt.hour
    lab = pd.read_csv(args.labelled, usecols=[
        "Subject_ID", "datetime_ist", "response_time", "STAI6", "LABEL"])
    lab["datetime_ist"] = pd.to_datetime(lab["datetime_ist"])
    lab = lab.dropna(subset=["LABEL"])
    lab["LABEL"] = lab["LABEL"].astype(int)

    # ---- 1. a readable score ----
    resp = lab.drop_duplicates(["Subject_ID", "response_time"])
    means = resp.groupby("LABEL")["STAI6"].mean().reindex([0, 1, 2])
    if means.isna().any():
        raise SystemExit("a class has no survey answers: %s" % means.to_dict())
    cuts = json.load(open(SUMMARY))["cut_points"]
    p["stai_equiv"] = (p["P_Low"] * means[0] + p["P_Medium"] * means[1]
                       + p["P_High"] * means[2])
    p["p_high_minus_low"] = p["P_High"] - p["P_Low"]
    p["band"] = pd.cut(p["stai_equiv"], [0, cuts[0], cuts[1], 100],
                       labels=["Low", "Medium", "High"])
    print("class weights (mean STAI-6 over survey answers): "
          "Low %.1f  Medium %.1f  High %.1f   cut points %.1f / %.1f"
          % (means[0], means[1], means[2], cuts[0], cuts[1]))
    print("predicted share: %s" % p["predicted"].value_counts(normalize=True)
          .round(3).to_dict())
    print("share of windows with P_High-P_Low between -0.5 and 0.5: %.2f "
          "(low means the model is bimodal, so a mid score is a mixture)"
          % (p["p_high_minus_low"].abs().lt(0.5).mean()))

    # ---- 2. wear flag ----
    p["worn"] = np.nan
    for u in sorted(p["Subject_ID"].unique()):
        path = BIO % u
        if not os.path.exists(path):
            continue
        b = pd.read_csv(path, usecols=lambda c: c in
                        {"timestamp_iso", "wearing_detection_percentage"})
        if "wearing_detection_percentage" not in b.columns:
            print("no wearing_detection_percentage in %s" % path)
            continue
        b["_min"] = pd.to_datetime(b["timestamp_iso"], utc=True) \
            .dt.tz_localize(None).dt.floor("min")
        m = b.drop_duplicates("_min").set_index("_min")["wearing_detection_percentage"]
        idx = u == p["Subject_ID"]
        key = (p.loc[idx, "mid_time"] - IST).dt.floor("min")
        p.loc[idx, "worn"] = key.map(m).values
    have = p["worn"].notna()
    print("\nwearing_detection_percentage joined for %.0f%% of windows; values:"
          % (100 * have.mean()))
    print(p["worn"].describe().round(1).to_dict())
    is_worn = p["worn"] >= args.worn_threshold
    print("windows worn (>= %.0f): %.0f%%" % (args.worn_threshold,
                                              100 * is_worn.mean()))

    print("\nHOUR (IST): mean stai_equiv, all windows vs worn only")
    t = pd.DataFrame({"all": p.groupby("hour")["stai_equiv"].mean(),
                      "worn": p[is_worn].groupby("hour")["stai_equiv"].mean(),
                      "n_worn": p[is_worn].groupby("hour").size()}).round(1)
    print(t.to_string())

    # ---- 3. agreement with the survey ----
    p = p.sort_values("mid_time")
    lab["_c"] = lab["datetime_ist"] + pd.Timedelta(seconds=30)
    lab = lab.sort_values("_c")
    cols = ["Subject_ID", "mid_time", "P_Low", "P_Medium", "P_High", "predicted",
            "stai_equiv", "p_high_minus_low", "worn"]
    m = pd.merge_asof(lab, p[cols], left_on="_c", right_on="mid_time",
                      by="Subject_ID", direction="nearest",
                      tolerance=pd.Timedelta(seconds=30)).dropna(subset=["predicted"])
    m["pred"] = m["predicted"].map(NAMES)
    print("\nlabelled windows matched to a prediction: %d" % len(m))

    def report(df, tag):
        if len(df) < 30 or df["LABEL"].nunique() < 2:
            print("  %-12s too few windows or one class" % tag)
            return
        bal = balanced_accuracy_score(df["LABEL"], df["pred"])
        rho_w = spearmanr(df["stai_equiv"], df["STAI6"])[0]
        r = df.groupby(["Subject_ID", "response_time"]).agg(
            s=("stai_equiv", "mean"), y=("STAI6", "first")).reset_index()
        rho_r = spearmanr(r["s"], r["y"])
        print("  %-12s windows %5d  answers %3d  balanced acc %.3f  "
              "rho(windows) %+.2f  rho(answers) %+.2f (p=%.2f)"
              % (tag, len(df), len(r), bal, rho_w, rho_r[0], rho_r[1]))

    print("chance for balanced accuracy is 0.333; rho 0 means no agreement")
    report(m, "all")
    report(m[m["worn"] >= args.worn_threshold], "worn only")
    for u, g in m.groupby("Subject_ID"):
        report(g, u)
    print("\nconfusion (rows true Low/Med/High, columns predicted):")
    print(confusion_matrix(m["LABEL"], m["pred"], labels=[0, 1, 2]))

    out = args.pred.replace("predictions.csv", "predictions_scored.csv")
    p.to_csv(out, index=False)
    print("\nsaved %s" % out)


if __name__ == "__main__":
    main()
